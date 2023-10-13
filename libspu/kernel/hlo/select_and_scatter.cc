
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

#include "libspu/kernel/hlo/select_and_scatter.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hlo/const.h"  // iota
#include "libspu/kernel/hlo/reduce.h"
#include "libspu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

/// The simplified scatter function
static spu::Value ScatterWindow(
    SPUContext *ctx,                    //
    const spu::Value &source,           // scatter source, shape = num_window
    const spu::Value &scatter_indices,  // the one-hot encoded scatter index
    const spu::Value &init,         // scalar value for non-scattered positions.
    const Shape &base_shape,        //
    const Shape &window_shape,      //
    const Strides &window_strides,  //
    const ValueBinaryFn &scatter_fn) {
  // alias shapes, use B,W,N.
  const Shape &B = base_shape;                // base shape
  const Shape &W = window_shape;              // window shape
  const Shape &N = source.shape();            // number of window
  const Shape NW2d = {N.numel(), W.numel()};  // flat N x W

  // sanity check.
  const size_t ndim = source.shape().size();
  SPU_ENFORCE_EQ(ndim, window_shape.size());
  SPU_ENFORCE_EQ(ndim, window_strides.size());
  SPU_ENFORCE(init.shape().isScalar());

  // scatter_indices is the one-hot encoding for each window.
  // win0: [0, 0, 0, 1, 0, 0]  // position 3 is selected.
  // win1: [0, 1, 0, 0, 0, 0]  // position 1 is selected.
  // ...
  SPU_ENFORCE_EQ(ndim + 1, scatter_indices.shape().size());
  SPU_ENFORCE_EQ(N, Shape(scatter_indices.shape().begin(),
                          scatter_indices.shape().begin() + ndim));
  SPU_ENFORCE_EQ(W.numel(), scatter_indices.shape()[ndim]);
  auto scatter_indices_2d = hal::reshape(ctx, scatter_indices, NW2d);

  auto source2d =
      hal::broadcast_to(ctx, hal::reshape(ctx, source, {N.numel()}), NW2d, {0});

  // One hot selected value per-window.
  // win0: [0, 0, 0, X, 0, 0]  // position 3 is selected.
  // win1: [0, Y, 0, 0, 0, 0]  // position 1 is selected.
  auto selected = hal::mul(ctx, source2d, scatter_indices_2d);
  SPU_ENFORCE_EQ(selected.shape(), NW2d);

  // selected value per-window index.
  std::vector<spu::Value> base_per_widx(W.numel());
  for (int64_t widx = 0; widx < W.numel(); widx++) {
    // for the i-th index in window, find all selected values.
    // win0: _, [0], _, _, _, _
    // win1: _, [Y], _, _, _, _
    // ..
    auto sel_pw = hal::slice(ctx, selected, {0, widx}, {N.numel(), widx + 1});
    SPU_ENFORCE_EQ(sel_pw.shape(), Shape({N.numel(), 1}));
    sel_pw = hal::reshape(ctx, sel_pw, N);

    // scatter it from num_window space to base space.
    Index window_index = unflattenIndex(widx, W);
    Sizes padding_lo(ndim, 0);
    Sizes padding_hi(ndim, 0);
    Sizes padding_in(ndim, 0);
    for (size_t dim = 0; dim < ndim; dim++) {
      padding_lo[dim] = window_index[dim];
      padding_hi[dim] = window_shape[dim] - window_index[dim] - 1;
      padding_in[dim] = window_strides[dim] - 1;
    }

    base_per_widx[widx] =
        hal::pad(ctx, sel_pw, init, padding_lo, padding_hi, padding_in);
    SPU_ENFORCE_EQ(base_per_widx[widx].shape(), B);
  }

  // last step, stack and reduce it.
  auto res = hal::concatenate(ctx, base_per_widx, 0);
  Shape WflatB = {W.numel()};
  WflatB.insert(WflatB.end(), B.begin(), B.end());
  res = hal::reshape(ctx, res, WflatB);

  res = TreeReduce(
      ctx, {res}, /* axis */ 0,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        return std::vector<spu::Value>{scatter_fn(lhs[0], rhs[0])};
      })[0];

  // TODO: if the reshape failed, that maybe the right edge is not sampled by
  // the window, we should add a padding operation here.
  return hal::reshape(ctx, res, B);
}

spu::Value MaxPoolScatter(
    SPUContext *ctx, const spu::Value &scatter_indices,
    const spu::Value &source, const Shape &window_shape,
    const Shape &base_shape, const Strides &window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding) {
  auto no_padding = std::all_of(window_padding.begin(), window_padding.end(),
                                [](const std::pair<int64_t, int64_t> &p) {
                                  return p.first == 0 && p.second == 0;
                                });
  SPU_ENFORCE(no_padding, "Expect padding to be removed by previous pass");

  // In MaxPoolScatter, one-hot scatter_indices is carried from the 'forward'
  // reduce window operation. So we can avoid on equal test here.

  auto init = hal::zeros(ctx, source.dtype(), {});
  auto scatter_fn = [&ctx](spu::Value const &lhs,
                           spu::Value const &rhs) -> spu::Value {
    return hal::add(ctx, lhs, rhs);
  };

  return ScatterWindow(ctx, source, scatter_indices, init, base_shape,
                       window_shape, window_strides, scatter_fn);
}

spu::Value SelectAndScatter(
    SPUContext *ctx, const spu::Value &base, const spu::Value &source,
    const spu::Value &init_val, const Shape &window_shape,
    const Strides &window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    const ValueBinaryFn &select_fn, const ValueBinaryFn &scatter_fn) {
  // sanity check.
  const size_t ndim = base.shape().size();
  SPU_ENFORCE_EQ(ndim, window_shape.size());
  SPU_ENFORCE_EQ(ndim, window_strides.size());
  SPU_ENFORCE(init_val.shape().isScalar());

  // alias shapes, use B,W,N.
  const Shape &W = window_shape;
  const Shape &N = source.shape();

  // clang-format off
  //
  // The algorithm:
  // tiled = win_count x window          : (N,W)
  // index = iota(0, num_window)         : (_,W)
  // sel_pos = reduce(tiled, index)      : (N,)   # find selected position of each window
  // onehot = sel_pos == index           : (N,_)->(_,W)->(N,W)  # each window ia one-hot position
  // sel_val = sel(onehot, source, init) : (N,W)->(N,)->()->(N,W)
  // sel_val = reduce(sel_val, 1)        : (N,W)->(N)
  //
  // clang-format on

  // Expand the base, simplify further actions without strides and padding.
  // Now tiled shaped is (N0, N1, ..., Nn, W0, W1, ..., Wn) where
  // window_count = (N0, N1, ..., Nn), where Ni = (Bi-Wi)/Strides{i} + 1
  auto tiled = expandWindow(ctx, base, W, window_strides, padding, init_val);
  SPU_ENFORCE_EQ(tiled.shape().size(), 2 * ndim);
  SPU_ENFORCE_EQ(N, Shape(tiled.shape().begin(), tiled.shape().begin() + ndim));
  SPU_ENFORCE_EQ(W, Shape(tiled.shape().begin() + ndim, tiled.shape().end()));

  // Use 2k, (N, W) to (N.numel(), W.numel()) to make future processing simpler.
  const Shape NW2d = {N.numel(), W.numel()};
  auto tiled2d = hal::reshape(ctx, tiled, NW2d);

  // indices is the iota for each window.
  // win0: [0, 1, 2, 3, 4, 5]
  // win1: [0, 1, 2, 3, 4, 5]
  // ...
  auto indices = hal::broadcast_to(ctx, Iota(ctx, DT_I64, W.numel()), NW2d);
  SPU_ENFORCE_EQ(indices.shape(), NW2d);

  // Apply the reduce with indices.
  auto reduced = TreeReduce(
      ctx, {tiled2d, indices}, 1,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        SPU_ENFORCE(lhs.size() == 2 && rhs.size() == 2);
        auto pred = select_fn(lhs[0], rhs[0]);
        std::vector<spu::Value> rets;
        for (size_t idx = 0; idx < lhs.size(); idx++) {
          // TODO: if reduce window does not require lhs[0].shape ==
          // lhs[1].shape, then we could avoid the later comparison.
          rets.push_back(hal::select(ctx, pred, lhs[idx], rhs[idx]));
        }
        return rets;
      });

  // indices is the iota for each window.
  // win0: [3]              // position 3 is selected.
  // win1: [1]              // position 1 is selected.
  // ...
  auto sel_pos = reduced[1];
  SPU_ENFORCE_EQ(sel_pos.shape(), Shape({N.numel(), 1}));

  // win0: [3, 3, 3, 3, 3, 3]
  // win1: [1, 1, 1, 1, 1, 1]
  sel_pos = hal::broadcast_to(ctx, sel_pos, NW2d, {0});

  // one hot encoding for each window
  // win0: [0, 0, 0, 1, 0, 0]  // position 3 is selected.
  // win1: [0, 1, 0, 0, 0, 0]  // position 1 is selected.
  // ...
  auto onehot = hal::equal(ctx, sel_pos, indices);
  SPU_ENFORCE_EQ(onehot.shape(), NW2d);

  Shape N_W1d = N;
  N_W1d.push_back(W.numel());

  return ScatterWindow(ctx, source, hal::reshape(ctx, onehot, N_W1d), init_val,
                       base.shape(), window_shape, window_strides, scatter_fn);
}

}  // namespace spu::kernel::hlo
