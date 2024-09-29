// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
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
MemRef tiled(Fn&& fn, SPUContext* ctx, const MemRef& x, Args&&... args) {
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

  std::vector<std::future<MemRef>> futures;

  // initialize slice indices
  Index start_indices(x.shape().size());
  Shape slice_shape(x.shape().begin(), x.shape().end());
  slice_shape[slicing_dim] = slice_stride;
  for (int64_t dim = 0; dim < slicing_dim; ++dim) {
    slice_shape[dim] = 1;
  }

  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    MemRef slice_data = x.slice(start_indices, slice_shape, {});
    auto async_res = std::async(
        [&](int64_t index, MemRef slice_data) {
          return fn(sub_ctxs[index].get(), MemRef(std::move(slice_data)),
                    args...);
        },
        slice_idx, slice_data);
    futures.push_back(std::move(async_res));

    // update indices
    if (slice_shape[slicing_dim] == x.shape()[slicing_dim]) {  // carry_out
      start_indices[slicing_dim] = 0;
      slice_shape[slicing_dim] = slice_stride;
    } else {
      start_indices[slicing_dim] += slice_stride;
      if (start_indices[slicing_dim] + slice_shape[slicing_dim] >
          x.shape()[slicing_dim]) {
        slice_shape[slicing_dim] =
            x.shape()[slicing_dim] - start_indices[slicing_dim];
      }
    }
  }

  std::vector<MemRef> out_slices;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    out_slices.push_back(futures[slice_idx].get());
  }

  // Assume out.shape = x.shape
  MemRef out(out_slices[0].eltype(), x.shape());
  int64_t offset = 0;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    std::memcpy(out.data<std::byte>() + offset, out_slices[slice_idx].data(),
                out_slices[slice_idx].numel() * out.elsize());
    offset += out_slices[slice_idx].numel() * out.elsize();
  }

  return MemRef(out);
}

template <typename Fn, typename... Args>
MemRef tiled(Fn&& fn, SPUContext* ctx, const MemRef& x, const MemRef& y,
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

  std::vector<std::future<MemRef>> futures;

  // initialize slice indices
  Index start_indices(x.shape().size());
  Shape slice_shape(x.shape().begin(), x.shape().end());
  slice_shape[slicing_dim] = slice_stride;
  for (int64_t dim = 0; dim < slicing_dim; ++dim) {
    slice_shape[dim] = 1;
  }

  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    MemRef slice_data_x = x.slice(start_indices, slice_shape, {});
    MemRef slice_data_y = y.slice(start_indices, slice_shape, {});
    auto async_res = std::async(
        [&](int64_t index, MemRef slice_data_x, MemRef slice_data_y) {
          return fn(sub_ctxs[index].get(), MemRef(std::move(slice_data_x)),
                    MemRef(std::move(slice_data_y)), args...);
        },
        slice_idx, slice_data_x, slice_data_y);
    futures.push_back(std::move(async_res));

    // update indices
    if (slice_shape[slicing_dim] == x.shape()[slicing_dim]) {  // carry_out
      start_indices[slicing_dim] = 0;
      slice_shape[slicing_dim] = slice_stride;
    } else {
      start_indices[slicing_dim] += slice_stride;
      if (start_indices[slicing_dim] + slice_shape[slicing_dim] >
          x.shape()[slicing_dim]) {
        slice_shape[slicing_dim] =
            x.shape()[slicing_dim] - start_indices[slicing_dim];
      }
    }
  }

  std::vector<MemRef> out_slices;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    out_slices.push_back(futures[slice_idx].get());
  }

  // Assume out.shape = x.shape
  MemRef out(out_slices[0].eltype(), x.shape());
  int64_t offset = 0;
  for (int64_t slice_idx = 0; slice_idx < num_slice_dim * num_slice;
       slice_idx++) {
    std::memcpy(out.data<std::byte>() + offset, out_slices[slice_idx].data(),
                out_slices[slice_idx].numel() * out.elsize());

    offset += out_slices[slice_idx].numel() * out.elsize();
  }

  return MemRef(out);
}

}  // namespace spu::mpc
