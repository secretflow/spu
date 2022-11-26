// Copyright 2022 Ant Group Co., Ltd.
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

#include "spu/kernel/hlo/reduce.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stack>
#include <vector>

#include "spu/core/parallel_utils.h"
#include "spu/core/shape_util.h"
#include "spu/kernel/hal/constants.h"
#include "spu/kernel/hal/shape_ops.h"
#include "spu/kernel/hlo/geometrical.h"
#include "spu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

std::vector<spu::Value> TreeReduce(HalContext *ctx,
                                   absl::Span<const spu::Value> inputs,
                                   int64_t axis,
                                   const BatchedValueBinaryFn &reducer) {
  const int64_t nargs = inputs.size();

  std::vector<spu::Value> outputs(inputs.begin(), inputs.end());

  std::vector<spu::Value> lhs(nargs);
  std::vector<spu::Value> rhs(nargs);

  std::stack<std::vector<spu::Value>> tails;

  std::vector<int64_t> slice_begin(inputs.back().shape().size(), 0);
  std::vector<int64_t> slice_end = inputs.back().shape();
  std::vector<int64_t> slice_strides(inputs.back().shape().size(), 1);
  int64_t len = outputs[0].shape()[axis];
  while (len > 1) {
    const int64_t half = len / 2;

    // lhs & rhs
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      slice_begin[axis] = 0;
      slice_end[axis] = half;

      lhs[idx] = hal::slice(
          ctx, outputs[idx],
          absl::MakeSpan(slice_begin.data(), outputs[idx].shape().size()),
          absl::MakeSpan(slice_end.data(), outputs[idx].shape().size()),
          absl::MakeSpan(slice_strides.data(), outputs[idx].shape().size()));

      slice_begin[axis] = half;
      slice_end[axis] = 2 * half;
      rhs[idx] = hal::slice(
          ctx, outputs[idx],
          absl::MakeSpan(slice_begin.data(), outputs[idx].shape().size()),
          absl::MakeSpan(slice_end.data(), outputs[idx].shape().size()),
          absl::MakeSpan(slice_strides.data(), outputs[idx].shape().size()));
    }

    // tail
    if (len % 2 == 1) {
      slice_begin[axis] = 2 * half;
      slice_end[axis] = len;
      std::vector<spu::Value> &tail = tails.emplace(nargs);
      for (size_t idx = 0; idx < outputs.size(); ++idx) {
        tail[idx] = hal::slice(
            ctx, outputs[idx],
            absl::MakeSpan(slice_begin.data(), outputs[idx].shape().size()),
            absl::MakeSpan(slice_end.data(), outputs[idx].shape().size()),
            absl::MakeSpan(slice_strides.data(), outputs[idx].shape().size()));
      }
    }

    outputs = reducer(lhs, rhs);
    len /= 2;

    YACL_ENFORCE(outputs[0].shape()[axis] == len);
  }

  // TODO: this may cause at worst 2*lg(n) time of reducer call, compare the
  // best case lg(n) times.
  //
  // consider len = 63, will iterate 5 (31, 15, 7, 3, 1), and generate
  // len(tails) = 5, the total number is 5 + 5 = 10 times.
  //
  // Optimize ME.
  while (!tails.empty()) {
    outputs = reducer(outputs, tails.top());
    tails.pop();
  }

  return outputs;
}

spu::Value ExpandStridedWindow(
    HalContext *ctx, const spu::Value &base,
    absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
  const auto &base_shape = base.shape();
  const size_t ndim = base_shape.size();

  YACL_ENFORCE(ndim == window_shape.size() &&    //
               ndim == window_strides.size() &&  //
               ndim == padding.size());

  // calculate output shape
  std::vector<int64_t> expanded_shape(ndim, 0);
  for (size_t dim = 0; dim < ndim; dim++) {
    int64_t padded_size =
        padding[dim].first + padding[dim].second + base_shape[dim];
    YACL_ENFORCE((padded_size - window_shape[dim]) % window_strides[dim] == 0);
    expanded_shape[dim] =
        ((padded_size - window_shape[dim]) / window_strides[dim] + 1) *
        window_shape[dim];
  }

  const std::vector<int64_t> window_dilations(window_shape.size(), 1);
  const std::vector<int64_t> base_dilations(base.shape().size(), 1);
  // expand it, assume padding & dialation element is zero.
  spu::Value expanded =
      hal::zeros(ctx, base.vtype(), base.dtype(), expanded_shape);

  auto numel = calcNumel(expanded_shape);

  yacl::parallel_for(
      0, numel, computeTaskSize(numel), [&](int64_t begin, int64_t end) {
        std::vector<int64_t> expanded_index =
            unflattenIndex(begin, expanded_shape);

        std::vector<int64_t> window_count_index(ndim, 0);
        std::vector<int64_t> window_index(ndim, 0);

        for (int64_t idx = begin; idx < end; ++idx) {
          for (size_t dim = 0; dim < ndim; dim++) {
            window_index[dim] = expanded_index[dim] % window_shape[dim];
            window_count_index[dim] = expanded_index[dim] / window_shape[dim];
          }

          std::vector<int64_t> base_index(ndim, 0);
          bool out_of_bound = getBaseIndexFromWindowIndex(
              window_shape, window_strides, window_dilations, padding,
              base_shape, base_dilations, window_count_index, window_index,
              absl::MakeSpan(base_index));
          if (!out_of_bound) {
            expanded.copyElementFrom(base, base_index, expanded_index);
          }
          if (!bumpIndices<int64_t>(expanded_shape,
                                    absl::MakeSpan(expanded_index))) {
            break;
          }
        }
      });

  return expanded;
}

spu::Value ConvertToTiledLayout(HalContext *ctx, const spu::Value &in,
                                absl::Span<const int64_t> block_shape) {
  // Note(jint): use pad+reshape+transpose to convert from column layout to
  // tiled layout.
  //
  // For example, in shape = [6, 12], window = [2, 3]
  // The result is [3, 4, 2, 3]
  YACL_ENFORCE(in.shape().size() == block_shape.size());
  std::vector<int64_t> tiled_shape;
  for (size_t dim = 0; dim < in.shape().size(); dim++) {
    YACL_ENFORCE(in.shape()[dim] % block_shape[dim] == 0);
    tiled_shape.push_back(in.shape()[dim] / block_shape[dim]);
    tiled_shape.push_back(block_shape[dim]);
  }
  std::vector<int64_t> perm(tiled_shape.size(), 0);
  std::iota(perm.begin(), perm.end(), 0);
  std::stable_partition(perm.begin(), perm.end(),
                        [](int64_t x) { return x % 2 == 0; });

  auto out = hal::reshape(ctx, in, tiled_shape);
  return hal::transpose(ctx, out, perm);
}

// So idea here..
// When windows size is 2x2, tile and run parallel on window element level has
// way to much overhead (both memory and computation).
// Just do a window level parallel is good enough
// And without dilation and padding, this can be achieved through just slicing
// FIXME: This is a super special case...consider generalize it a little bit
std::vector<spu::Value>
ReduceWindow1x2x2x1NoPaddingOneStrideWithoutDilationWithWindowMask(
    HalContext *ctx, absl::Span<const spu::Value> inputs,
    absl::Span<const spu::Value> init_values,
    absl::Span<const int64_t> ret_shape, const BatchedValueBinaryFn &reducer) {
  std::vector<int64_t> start_indices = {0, 0, 0, 0};
  auto input_shape = inputs[0].shape();

  std::vector<spu::Value> input_slices(4);
  std::vector<spu::Value> mask_slices(4);
  input_slices[0] = hal::slice(
      ctx, inputs[0], {0, 0, 0, 0},
      {input_shape[0], input_shape[1] - 1, input_shape[2] - 1, input_shape[3]},
      {1, 1, 1, 1});
  input_slices[1] = hal::slice(
      ctx, inputs[0], {0, 0, 1, 0},
      {input_shape[0], input_shape[1] - 1, input_shape[2], input_shape[3]},
      {1, 1, 1, 1});
  input_slices[2] = hal::slice(
      ctx, inputs[0], {0, 1, 0, 0},
      {input_shape[0], input_shape[1], input_shape[2] - 1, input_shape[3]},
      {1, 1, 1, 1});
  input_slices[3] = hal::slice(
      ctx, inputs[0], {0, 1, 1, 0},
      {input_shape[0], input_shape[1], input_shape[2], input_shape[3]},
      {1, 1, 1, 1});

  std::vector<int64_t> mask_shape = {inputs[0].shape()[0], ret_shape[1],
                                     ret_shape[2], inputs[0].shape().back(), 4};
  for (int64_t mask_idx = 0; mask_idx < 4; ++mask_idx) {
    mask_slices[mask_idx] = hal::slice(ctx, inputs.back(), {mask_idx, 0},
                                       {mask_idx + 1, 4}, {1, 1});
    mask_slices[mask_idx] =
        hal::reshape(ctx, mask_slices[mask_idx], {1, 1, 1, 1, 4});
    mask_slices[mask_idx] =
        hal::broadcast_to(ctx, mask_slices[mask_idx], mask_shape);
  }

  std::vector<spu::Value> ret = {input_slices[0], mask_slices[0]};
  for (size_t i = 1; i < input_slices.size(); ++i) {
    ret = reducer({input_slices[i], mask_slices[i]}, ret);
  }

  return ret;
}

std::vector<spu::Value> ReduceWindowWithoutDilation(
    HalContext *ctx, absl::Span<const spu::Value> inputs,
    absl::Span<const spu::Value> init_values,
    absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    bool last_operand_is_window_mask, bool ignore_init_value,
    absl::Span<const int64_t> ret_shape, const BatchedValueBinaryFn &reducer) {
  // Add a fast 1x2x2x1, no padding fast reduce
  auto no_padding = std::all_of(window_padding.begin(), window_padding.end(),
                                [](const std::pair<int64_t, int64_t> &p) {
                                  return p.first == 0 && p.second == 0;
                                });
  auto one_stride = std::all_of(window_strides.begin(), window_strides.end(),
                                [](auto v) { return v == 1; });
  if (window_shape == absl::Span<const int64_t>{1, 2, 2, 1} &&
      inputs.size() == 2 && last_operand_is_window_mask && no_padding &&
      one_stride) {
    return ReduceWindow1x2x2x1NoPaddingOneStrideWithoutDilationWithWindowMask(
        ctx, inputs, init_values, ret_shape, reducer);
  }
  const size_t nargs =
      last_operand_is_window_mask ? inputs.size() - 1 : inputs.size();

  auto window_size = std::accumulate(window_shape.begin(), window_shape.end(),
                                     1, std::multiplies<>());

  // expand the operand, simplify following actions without strides and padding.
  std::vector<spu::Value> expanded;
  for (size_t idx = 0; idx < nargs; ++idx) {
    const auto &input = inputs[idx];
    auto x = ExpandStridedWindow(ctx, input, window_shape, window_strides,
                                 window_padding);
    expanded.emplace_back(ConvertToTiledLayout(ctx, x, window_shape));
  }

  if (last_operand_is_window_mask) {
    auto mask = inputs.back();
    std::vector<int64_t> shape(expanded.back().shape().size() + 1, 1);
    shape.back() = window_size;

    auto mask_idx = shape.size() - 2;
    for (auto iter = window_shape.rbegin(); iter != window_shape.rend();
         ++iter) {
      shape[mask_idx--] = *iter;
    }

    mask = hal::reshape(ctx, mask, shape);

    for (size_t idx = 0; idx < expanded.back().shape().size(); ++idx) {
      shape[idx] = expanded.back().shape()[idx];
    }

    expanded.emplace_back(hal::broadcast_to(ctx, mask, shape, {}));
  }

  // Flatten the window, to maximize parallel processing.
  std::vector<int64_t> tiled_1d_shape(ret_shape.begin(), ret_shape.end());
  tiled_1d_shape.push_back(window_size);
  for (size_t idx = 0; idx < nargs; ++idx) {
    // reshape to tiled_1d
    expanded[idx] = hal::reshape(ctx, expanded[idx], tiled_1d_shape);
  }

  if (last_operand_is_window_mask) {
    std::vector<int64_t> tiled_1d_shape_mask(tiled_1d_shape.begin(),
                                             tiled_1d_shape.end());
    tiled_1d_shape_mask.emplace_back(window_size);
    expanded.back() = hal::reshape(ctx, expanded.back(), tiled_1d_shape_mask);
  }

  // reduce the last axis
  auto outputs = TreeReduce(ctx, expanded, tiled_1d_shape.size() - 1, reducer);

  // reduce the last axis
  for (size_t idx = 0; idx < nargs; idx++) {
    outputs[idx] = hal::reshape(ctx, outputs[idx], ret_shape);
  }
  if (last_operand_is_window_mask) {
    std::vector<int64_t> mask_ret_shape(ret_shape.begin(), ret_shape.end());
    mask_ret_shape.emplace_back(window_size);
    outputs.back() = hal::reshape(ctx, outputs.back(), mask_ret_shape);
  }

  if (!ignore_init_value) {
    // init_values are scalars, broadcast to return shape first.
    std::vector<spu::Value> broadcasted_init_values;
    for (const auto &v : init_values) {
      broadcasted_init_values.push_back(hal::broadcast_to(ctx, v, ret_shape));
    }

    return reducer(outputs, broadcasted_init_values);
  }
  return outputs;
}

std::vector<spu::Value> ReduceWindow(HalContext *ctx,
                                     absl::Span<const spu::Value> inputs,
                                     absl::Span<const spu::Value> init_values,
                                     absl::Span<const int64_t> ret_shape,
                                     const ReduceWindowConfig &config,
                                     const BatchedValueBinaryFn &reducer) {
  if (std::all_of(config.window_dilations.begin(),
                  config.window_dilations.end(),
                  [](const int64_t x) { return x == 1; }) &&
      std::all_of(config.base_dilations.begin(), config.base_dilations.end(),
                  [](const int64_t x) { return x == 1; })) {
    return ReduceWindowWithoutDilation(
        ctx, inputs, init_values, config.window_shape, config.window_strides,
        config.window_padding, config.last_operand_is_window_mask,
        config.ignore_init_value, ret_shape, reducer);
  }

  YACL_ENFORCE(config.last_operand_is_window_mask == false);

  const int64_t ndims = inputs[0].shape().size();
  std::vector<int64_t> window_index(ndims, 0);
  int64_t nargs = inputs.size();

  // Init...
  std::vector<spu::Value> rets(nargs);
  for (int64_t idx = 0; idx < nargs; ++idx) {
    rets[idx] = hal::expand(ctx, init_values[idx], ret_shape);
  }

  // For each resulting dimension, calculate and assign computed value.
  auto evaluate_impl =
      [&](absl::Span<int64_t> output_index) -> std::vector<spu::Value> {
    std::vector<spu::Value> ret;
    RunOnWindowIndex(
        config.window_shape, config.window_strides, config.window_dilations,
        config.window_padding, inputs[0].shape(), config.base_dilations,
        output_index, window_index,
        [&](absl::Span<const int64_t> operand_index) {
          for (int64_t idx = 0; idx < nargs; ++idx) {
            ret.emplace_back(inputs[idx].getElementAt(operand_index));
          }
        });
    return ret;
  };

  // For each window index
  std::vector<spu::Value> batchs(nargs);
  for (int64_t idx = 0; idx < nargs; ++idx) {
    batchs[idx] = hal::expand(ctx, inputs[idx].getElementAt({}), ret_shape);
  }

  do {
    // Collect one element from each window
    std::vector<int64_t> output_index(ret_shape.size(), 0);
    do {
      auto r = evaluate_impl(absl::MakeSpan(output_index));
      if (!r.empty()) {
        for (int64_t idx = 0; idx < nargs; ++idx) {
          batchs[idx].copyElementFrom(r[idx], {}, output_index);
        }
      }
    } while (
        bumpIndices(absl::MakeSpan(ret_shape), absl::MakeSpan(output_index)));

    // Now run the batch
    rets = reducer(rets, batchs);

  } while (
      bumpIndices<int64_t>(config.window_shape, absl::MakeSpan(window_index)));

  return rets;
}

std::vector<spu::Value> Reduce(HalContext *ctx,
                               absl::Span<const spu::Value> inputs,
                               absl::Span<const spu::Value> init_values,
                               absl::Span<const int64_t> dims_to_reduce,
                               const BatchedValueBinaryFn &reducer) {
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

  const std::vector<int64_t> in_shape = inputs[0].shape();

  std::vector<int64_t> perm(in_shape.size(), 0);
  std::iota(perm.begin(), perm.end(), 0);
  // swap axes, move the dims to reduce to inner most.
  std::stable_partition(perm.begin(), perm.end(), [&](int64_t axis) {
    return std::find(dims_to_reduce.begin(), dims_to_reduce.end(), axis) ==
           dims_to_reduce.end();
  });

  std::vector<int64_t> flat_shape;
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

  std::vector<spu::Value> flattened;
  for (const auto &input : inputs) {
    flattened.push_back(
        hal::reshape(ctx, hal::transpose(ctx, input, perm), flat_shape));
  }

  // reduce the inner most axis
  auto results =
      TreeReduce(ctx, flattened, flattened[0].shape().size() - 1, reducer);

  // broadcast to origin shape.
  std::vector<int64_t> out_shape = inputs[0].shape();
  for (const auto &axis : dims_to_reduce) {
    out_shape[axis] = 1;
  }

  for (auto &result : results) {
    result = hal::reshape(ctx, result, out_shape);
  }

  // init_values are scalars, broadcast to return shape first.
  std::vector<spu::Value> broadcasted_init_values;
  for (const auto &v : init_values) {
    broadcasted_init_values.push_back(hal::broadcast_to(ctx, v, out_shape));
  }

  return reducer(results, broadcasted_init_values);
}

}  // namespace spu::kernel::hlo
