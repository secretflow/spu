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

// Inituition: some of the ops has complicated (compute, communication, compute,
// communication ...) behaviour, when the input is large, CPU will wait for comm
// interleavely. Pipeline is a software instruction level parallelism that use
// tiling+concurrent to reduce the waiting time.
//
// Tiling+concurrent could be treated as the opposite of fusion+vectorization.

namespace detail {

template <typename Fn, typename... Args>
ArrayRef tiled(Fn&& fn, SPUContext* ctx, const ArrayRef& x, Args&&... args) {
  const int64_t kBlockSize = kMinTaskSize;
  if (!ctx->config().experimental_enable_intra_op_par()  //
      || !ctx->prot()->hasLowCostFork()                  //
      || x.numel() <= kBlockSize                         //
  ) {
    return fn(ctx, x, std::forward<Args>(args)...);
  }

  std::vector<std::unique_ptr<SPUContext>> sub_ctxs;

  const int64_t numBlocks =
      x.numel() / kBlockSize + ((x.numel() % kBlockSize) != 0 ? 1 : 0);

  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    sub_ctxs.push_back(ctx->fork());
  }

  std::vector<std::future<ArrayRef>> futures;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    auto async_res = std::async(
        [&](int64_t index) {
          int64_t begin = index * kBlockSize;
          int64_t end = std::min(begin + kBlockSize, x.numel());

          return fn(sub_ctxs[index].get(), x.slice(begin, end), args...);
        },
        blk_idx);
    futures.push_back(std::move(async_res));
  }

  std::vector<ArrayRef> out_slices;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    out_slices.push_back(futures[blk_idx].get());
  }

  // Assume out.numel = x.numel
  ArrayRef out(out_slices[0].eltype(), x.numel());
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    int64_t begin = blk_idx * kBlockSize;
    int64_t end = std::min(begin + kBlockSize, x.numel());
    std::memcpy(&out.at(begin), &out_slices[blk_idx].at(0),
                (end - begin) * out.elsize());
  }

  return out;
}

template <typename Fn, typename... Args>
ArrayRef tiled_2(Fn&& fn, SPUContext* ctx, const ArrayRef& x, const ArrayRef& y,
                 Args&&... args) {
  SPU_ENFORCE(x.numel() == y.numel());
  const int64_t kBlockSize = kMinTaskSize;
  if (!ctx->config().experimental_enable_intra_op_par()  //
      || !ctx->prot()->hasLowCostFork()                  //
      || x.numel() <= kBlockSize                         //
  ) {
    return fn(ctx, x, y, std::forward<Args>(args)...);
  }

  const int64_t numel = x.numel();

  std::vector<std::unique_ptr<SPUContext>> sub_ctxs;

  const int64_t numBlocks =
      numel / kBlockSize + ((numel % kBlockSize) != 0 ? 1 : 0);

  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    sub_ctxs.push_back(ctx->fork());
  }

  std::vector<std::future<ArrayRef>> futures;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    auto async_res = std::async(
        [&](int64_t index) {
          int64_t begin = index * kBlockSize;
          int64_t end = std::min(begin + kBlockSize, numel);

          return fn(sub_ctxs[index].get(), x.slice(begin, end),
                    y.slice(begin, end), args...);
        },
        blk_idx);
    futures.push_back(std::move(async_res));
  }

  std::vector<ArrayRef> out_slices;
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    out_slices.push_back(futures[blk_idx].get());
  }

  // Assume out.numel = numel
  ArrayRef out(out_slices[0].eltype(), numel);
  for (int64_t blk_idx = 0; blk_idx < numBlocks; blk_idx++) {
    int64_t begin = blk_idx * kBlockSize;
    int64_t end = std::min(begin + kBlockSize, numel);
    std::memcpy(&out.at(begin), &out_slices[blk_idx].at(0),
                (end - begin) * out.elsize());
  }

  return out;
}

}  // namespace detail

template <typename Fn, typename... Args>
Value tiled(Fn&& fn, SPUContext* ctx, const Value& x, Args&&... args) {
  // TODO: using nd-slice for tiling.
  ArrayRef flat_x = flatten(x.data());

  auto wrap_fn = [fn](SPUContext* wctx, const ArrayRef& wx,
                      Args&&... wargs) -> ArrayRef {
    auto res =
        fn(wctx, WrapValue(wx, {wx.numel()}), std::forward<Args>(wargs)...);
    auto [res_arr, res_shape, res_dtype] = UnwrapValue(res);
    return res_arr;
  };

  auto flat_z =
      detail::tiled(wrap_fn, ctx, flat_x, std::forward<Args>(args)...);
  return WrapValue(flat_z, x.shape());
}

template <typename Fn, typename... Args>
Value tiled(Fn&& fn, SPUContext* ctx, const Value& x, const Value& y,
            Args&&... args) {
  SPU_ENFORCE(x.shape() == y.shape());

  // TODO: using nd-slice for tiling.
  ArrayRef flat_x = flatten(x.data());
  SPU_ENFORCE(flat_x.numel() == x.numel());
  ArrayRef flat_y = flatten(y.data());
  SPU_ENFORCE(flat_y.numel() == y.numel());

  auto wrap_fn = [fn](SPUContext* wctx, const ArrayRef& wx, const ArrayRef& wy,
                      Args&&... wargs) -> ArrayRef {
    auto res = fn(wctx, WrapValue(wx, {wx.numel()}),
                  WrapValue(wy, {wy.numel()}), std::forward<Args>(wargs)...);
    auto [res_arr, res_shape, res_dtype] = UnwrapValue(res);
    return res_arr;
  };

  // FIXME: if not rename to tile_2, it will select the first (less specialized)
  // candidate.
  auto flat_z = detail::tiled_2(wrap_fn, ctx, flat_x, flat_y,
                                std::forward<Args>(args)...);
  return WrapValue(flat_z, x.shape());
}

}  // namespace spu::mpc
