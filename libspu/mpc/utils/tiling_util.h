// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except x compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to x writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <future>

#include "libspu/core/context.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"

namespace spu::mpc {

// Intuition: some of the ops has complicated (compute, communication, compute,
// communication ...) behaviour, when the input is large, CPU will wait for comm
// interleavely. Pipeline is a software instruction level parallelism that use
// tiling+concurrent to reduce the waiting time.
//
// Tiling+concurrent could be treated as the opposite of fusion+vectorization.

template <typename Fn, typename... Args>
Value tiled(Fn&& fn, SPUContext* ctx, const Value& x, Args&&... args) {
  const int64_t kBlockSize = kMinTaskSize;
  if (!ctx->config().experimental_enable_intra_op_par()  //
      || !ctx->prot()->hasLowCostFork()                  //
      || x.numel() <= kBlockSize                         //
  ) {
    return fn(ctx, x, std::forward<Args>(args)...);
  }

  // from inner to outer, find an outermost dimension whose all inner
  // dimensions has elements less than kBlockSize
  int64_t slicing_dim = -1;
  int64_t slice_numel = 1;
  for (int64_t dim = x.shape().size() - 1; dim >= 0; dim--) {
    slice_numel *= x.shape()[dim];
    if (slice_numel > kBlockSize) {
      slice_numel /= x.shape()[dim];
      slicing_dim = dim;
      break;
    }
  }

  // get the slice stride in the slicing dimension
  int64_t slice_stride = std::lround(kBlockSize / slice_numel);

  if (slice_stride == 0) {
    return fn(ctx, x, std::forward<Args>(args)...);
  }

  // get the slice num in the slicing dimension
  int64_t num_slice_dim =
      x.shape()[slicing_dim] / slice_stride +
      ((x.shape()[slicing_dim] % slice_stride) != 0 ? 1 : 0);

  // get the slice num in the left outer dimensions
  int64_t num_slice = 1;
  for (int64_t dim = 0; dim < slicing_dim; dim++) {
    num_slice *= x.shape()[dim];
  }

  std::vector<std::unique_ptr<SPUContext>> sub_ctxs;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    sub_ctxs.push_back(ctx->fork());
  }

  std::vector<std::future<Value>> futures;

  // initialize slice indices
  std::vector<int64_t> start_indices(x.shape().size());
  std::vector<int64_t> end_indices(x.shape());
  end_indices[slicing_dim] = slice_stride;
  for (int64_t dim = slicing_dim - 1; dim >= 0; dim--) {
    end_indices[dim] = 1;
  }

  auto data = x.data();
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    auto async_res = std::async(
        [&](int64_t index, std::vector<int64_t> s_indices,
            std::vector<int64_t> e_indices) {
          NdArrayRef slice_data = data.slice(s_indices, e_indices, {});

          auto ret =
              fn(sub_ctxs[index].get(), Value(slice_data, DT_INVALID), args...);

          return ret;
        },
        slice_idx, start_indices, end_indices);
    futures.push_back(std::move(async_res));

    // update indices
    if (end_indices[slicing_dim] == x.shape()[slicing_dim]) {  // carry_out
      start_indices[slicing_dim] = 0;
      end_indices[slicing_dim] = slice_stride;
      for (int64_t dim = slicing_dim - 1; dim >= 0; dim--) {
        start_indices[dim] = (start_indices[dim] + 1) % data.shape()[dim];
        end_indices[dim] = (end_indices[dim] + 1) % data.shape()[dim] + 1;
        if (end_indices[dim] != 1) break;
      }
    } else {
      start_indices[slicing_dim] += slice_stride;
      end_indices[slicing_dim] += slice_stride;
      if (end_indices[slicing_dim] > x.shape()[slicing_dim])
        end_indices[slicing_dim] = x.shape()[slicing_dim];
    }
  }

  std::vector<Value> out_slices;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    out_slices.push_back(futures[slice_idx].get());
  }

  // Assume out.shape = x.shape
  NdArrayRef out(out_slices[0].storage_type(), x.shape());
  int64_t offset = 0;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    std::memcpy(static_cast<std::byte*>(out.data()) + offset,
                out_slices[slice_idx].data().data(),
                out_slices[slice_idx].numel() * out.elsize());
    offset += out_slices[slice_idx].numel() * out.elsize();
  }

  return Value(out, DT_INVALID);
}

template <typename Fn, typename... Args>
Value tiled(Fn&& fn, SPUContext* ctx, const Value& x, const Value& y,
            Args&&... args) {
  SPU_ENFORCE(x.shape() == y.shape());

  const int64_t kBlockSize = kMinTaskSize;
  if (!ctx->config().experimental_enable_intra_op_par()  //
      || !ctx->prot()->hasLowCostFork()                  //
      || x.numel() <= kBlockSize                         //
  ) {
    return fn(ctx, x, y, std::forward<Args>(args)...);
  }

  // from inner to outer, find an outermost dimension whose all inner
  // dimensions has elements less than kBlockSize
  int64_t slicing_dim = -1;
  int64_t slice_numel = 1;
  for (int64_t dim = x.shape().size() - 1; dim >= 0; dim--) {
    slice_numel *= x.shape()[dim];
    if (slice_numel > kBlockSize) {
      slice_numel /= x.shape()[dim];
      slicing_dim = dim;
      break;
    }
  }

  // get the slice stride in the slicing dimension
  int64_t slice_stride = std::lround(kBlockSize / slice_numel);

  if (slice_stride == 0) {
    return fn(ctx, x, y, std::forward<Args>(args)...);
  }

  // get the slice num in the slicing dimension
  int64_t num_slice_dim =
      x.shape()[slicing_dim] / slice_stride +
      ((x.shape()[slicing_dim] % slice_stride) != 0 ? 1 : 0);

  // get the slice num in the left outer dimensions
  int64_t num_slice = 1;
  for (int64_t dim = 0; dim < slicing_dim; dim++) {
    num_slice *= x.shape()[dim];
  }

  std::vector<std::unique_ptr<SPUContext>> sub_ctxs;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    sub_ctxs.push_back(ctx->fork());
  }

  std::vector<std::future<Value>> futures;

  // initialize slice indices
  std::vector<int64_t> start_indices(x.shape().size());
  std::vector<int64_t> end_indices(x.shape());
  end_indices[slicing_dim] = slice_stride;
  for (int64_t dim = slicing_dim - 1; dim >= 0; dim--) {
    end_indices[dim] = 1;
  }

  auto data_x = x.data();
  auto data_y = y.data();
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    auto async_res = std::async(
        [&](int64_t index, std::vector<int64_t> s_indices,
            std::vector<int64_t> e_indices) {
          NdArrayRef slice_data_x = data_x.slice(s_indices, e_indices, {});
          NdArrayRef slice_data_y = data_y.slice(s_indices, e_indices, {});

          auto ret = fn(sub_ctxs[index].get(), Value(slice_data_x, DT_INVALID),
                        Value(slice_data_y, DT_INVALID), args...);

          return ret;
        },
        slice_idx, start_indices, end_indices);
    futures.push_back(std::move(async_res));

    // update indices
    if (end_indices[slicing_dim] == x.shape()[slicing_dim]) {  // carry_out
      start_indices[slicing_dim] = 0;
      end_indices[slicing_dim] = slice_stride;
      for (int64_t dim = slicing_dim - 1; dim >= 0; dim--) {
        start_indices[dim] = (start_indices[dim] + 1) % data_x.shape()[dim];
        end_indices[dim] = (end_indices[dim] + 1) % data_x.shape()[dim] + 1;
        if (end_indices[dim] != 1) break;
      }
    } else {
      start_indices[slicing_dim] += slice_stride;
      end_indices[slicing_dim] += slice_stride;
      if (end_indices[slicing_dim] > x.shape()[slicing_dim])
        end_indices[slicing_dim] = x.shape()[slicing_dim];
    }
  }

  std::vector<Value> out_slices;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    out_slices.push_back(futures[slice_idx].get());
  }

  // Assume out.shape = x.shape
  NdArrayRef out(out_slices[0].storage_type(), x.shape());
  int64_t offset = 0;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    std::memcpy(static_cast<std::byte*>(out.data()) + offset,
                out_slices[slice_idx].data().data(),
                out_slices[slice_idx].numel() * out.elsize());

    offset += out_slices[slice_idx].numel() * out.elsize();
  }

  return Value(out, DT_INVALID);
}

}  // namespace spu::mpc
