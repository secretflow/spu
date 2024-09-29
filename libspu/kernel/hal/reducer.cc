// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/kernel/hal/reducer.h"

#include <stack>

#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/utils.h"

namespace spu::kernel::hal {

std::vector<spu::MemRef> TreeReduce(SPUContext *ctx,
                                    absl::Span<const spu::MemRef> inputs,
                                    int64_t axis,
                                    const BatchedMemRefBinaryFn &reducer,
                                    const BroadcastCallbackFcn &bcaster) {
  const int64_t nargs = inputs.size();

  std::vector<spu::MemRef> outputs(inputs.begin(), inputs.end());

  std::vector<spu::MemRef> lhs(nargs);
  std::vector<spu::MemRef> rhs(nargs);

  std::stack<std::vector<spu::MemRef>> tails;

  Index slice_begin(inputs.back().shape().size(), 0);
  Shape slice_size = inputs.back().shape();
  Strides slice_strides(inputs.back().shape().size(), 1);
  int64_t len = outputs[0].shape()[axis];
  while (len > 1) {
    const int64_t half = len / 2;
    slice_size[axis] = half;

    // lhs & rhs
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      slice_begin[axis] = 0;

      lhs[idx] = kernel::hal::_extract_slice(ctx, outputs[idx], slice_begin,
                                             slice_size, slice_strides);

      slice_begin[axis] = half;
      rhs[idx] = kernel::hal::_extract_slice(ctx, outputs[idx], slice_begin,
                                             slice_size, slice_strides);
    }

    bcaster(slice_size);

    // tail
    if (len % 2 == 1) {
      slice_begin[axis] = 2 * half;
      slice_size[axis] = len - slice_begin[axis];
      std::vector<spu::MemRef> &tail = tails.emplace(nargs);
      for (size_t idx = 0; idx < outputs.size(); ++idx) {
        tail[idx] = kernel::hal::_extract_slice(ctx, outputs[idx], slice_begin,
                                                slice_size, slice_strides);
      }
    }

    outputs = reducer(lhs, rhs);
    len /= 2;

    SPU_ENFORCE(outputs[0].shape()[axis] == len);
  }

  // TODO: this may cause at worst 2*lg(n) time of reducer call, compare the
  // best case lg(n) times.
  //
  // consider len = 63, will iterate 5 (31, 15, 7, 3, 1), and generate
  // len(tails) = 5, the total number is 5 + 5 = 10 times.
  //
  // Optimize ME.
  bcaster(outputs.front().shape());
  while (!tails.empty()) {
    outputs = reducer(outputs, tails.top());
    tails.pop();
  }

  return outputs;
}

std::vector<spu::MemRef> Reduce(SPUContext *ctx,
                                absl::Span<const spu::MemRef> inputs,
                                absl::Span<const spu::MemRef> init_values,
                                const Axes &dims_to_reduce,
                                const BatchedMemRefBinaryFn &reducer,
                                const BroadcastCallbackFcn &bcaster,
                                bool ignore_init_values) {
  // Reduce multiple dimension
  //
  // The straight-forward method iterates dimension_to_reduce with each dim a
  // TreeReduce kernel. In SPU, we tries to minimize the reducer call.
  //
  // The algorithm is summarized below:
  //
  // Input:
  //   shape       2 3 4 5 6
  //   dims          X   X
  //
  // Steps:
  //   perm        2 4 6 3 5         0 2 4 1 3
  //   flatten     2 4 6 15
  //   reduce      2 4 6 1
  //   result      2 1 4 1 6
  //
  // Note(jint), theoretically, this method will reduce number of reducer calls,
  // in this example, from
  //   ceil(lg(3)) + ceil(lg(5)) = 2 + 3 = 5
  // to
  //   ceil(lg(3 * 5)) = 4
  //
  // But in current TreeReduce (unoptimized) implementation, this method is
  // slower.
  //
  // Note(jint): this `lowering` progress is easy to be ported to
  // compile-time.

  const auto in_shape = inputs[0].shape();

  Axes perm(in_shape.size(), 0);
  std::iota(perm.begin(), perm.end(), 0);
  // swap axes, move the dims to reduce to inner most.
  std::stable_partition(perm.begin(), perm.end(), [&](int64_t axis) {
    return std::find(dims_to_reduce.begin(), dims_to_reduce.end(), axis) ==
           dims_to_reduce.end();
  });

  Shape flat_shape;
  int64_t numel_to_reduce = 1;
  for (size_t axis = 0; axis < in_shape.size(); axis++) {
    if (std::find(dims_to_reduce.begin(), dims_to_reduce.end(), axis) ==
        dims_to_reduce.end()) {
      flat_shape.push_back(in_shape[axis]);
    } else {
      numel_to_reduce *= in_shape[axis];
    }
  }
  flat_shape.push_back(numel_to_reduce);

  std::vector<spu::MemRef> flattened;
  for (const auto &input : inputs) {
    flattened.push_back(kernel::hal::_reshape(
        ctx, kernel::hal::_transpose(ctx, input, perm), flat_shape));
  }

  // reduce the inner most axis
  auto results = TreeReduce(ctx, flattened, flattened[0].shape().size() - 1,
                            reducer, bcaster);

  // broadcast to origin shape.
  Shape out_shape = inputs[0].shape();
  for (const auto &axis : dims_to_reduce) {
    out_shape[axis] = 1;
  }

  for (auto &result : results) {
    result = kernel::hal::_reshape(ctx, result, out_shape);
  }

  if (ignore_init_values) {
    return results;
  }

  std::vector<spu::MemRef> broadcasted_init_values;
  // init_values are scalars, broadcast to return shape first.
  for (const auto &v : init_values) {
    broadcasted_init_values.push_back(
        kernel::hal::_broadcast(ctx, v, out_shape, {}));
  }

  bcaster(out_shape);

  return reducer(results, broadcasted_init_values);
}

}  // namespace spu::kernel::hal
