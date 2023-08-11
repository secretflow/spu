
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

#include <cstdint>
#include <future>
#include <iostream>
#include <vector>

#include "yacl/utils/parallel.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/kernel/hal/polymorphic.h"  // for select
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/reduce.h"
#include "libspu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

spu::Value MaxPoolScatter1x2x2x1NoPaddingNoDilation(
    SPUContext *ctx, const spu::Value &scatter_indices,
    const spu::Value &source, const Strides &window_strides) {
  std::vector<spu::Value> slices(4);
  for (int64_t idx = 0; idx < 4; ++idx) {
    slices[idx] = hal::slice(
        ctx, scatter_indices, {0, 0, 0, 0, idx},
        {scatter_indices.shape()[0], scatter_indices.shape()[1],
         scatter_indices.shape()[2], scatter_indices.shape()[3], idx + 1},
        {1, 1, 1, 1, 1});
    slices[idx] =
        hal::mul(ctx, hal::reshape(ctx, slices[idx], source.shape()), source);

    // FIXME(jint), handle int type promotion
    slices[idx] = hal::dtype_cast(ctx, slices[idx], source.dtype());
  }

  // Improvement idea: If window strides is >= window size (no overlap), we
  // should be able to compute scatter result with just one multiply
  auto z = hal::zeros(ctx, slices[0].dtype());
  if (slices[0].isSecret()) {
    z = hal::seal(ctx, z);
  }

  std::vector<std::future<spu::Value>> f_slices(4);
  f_slices[0] =
      std::async(std::launch::async, hal::pad, ctx, slices[0], z,
                 Sizes{0, 0, 0, 0}, Sizes{0, 1, 1, 0},
                 Sizes{0, window_strides[1] - 1, window_strides[2] - 1, 0});
  f_slices[1] =
      std::async(std::launch::async, hal::pad, ctx, slices[1], z,
                 Sizes{0, 0, 1, 0}, Sizes{0, 1, 0, 0},
                 Sizes{0, window_strides[1] - 1, window_strides[2] - 1, 0});
  f_slices[2] =
      std::async(std::launch::async, hal::pad, ctx, slices[2], z,
                 Sizes{0, 1, 0, 0}, Sizes{0, 0, 1, 0},
                 Sizes{0, window_strides[1] - 1, window_strides[2] - 1, 0});
  f_slices[3] =
      std::async(std::launch::async, hal::pad, ctx, slices[3], z,
                 Sizes{0, 1, 1, 0}, Sizes{0, 0, 0, 0},
                 Sizes{0, window_strides[1] - 1, window_strides[2] - 1, 0});

  spu::Value ret = f_slices[0].get();
  for (size_t idx = 1; idx < 4; ++idx) {
    ret = hal::add(ctx, ret, f_slices[idx].get());
  }

  return ret;
};

spu::Value MaxPoolScatter(
    SPUContext *ctx, const spu::Value &scatter_indices,
    const spu::Value &source, const Shape &window_shape,
    const Shape &base_shape, const Strides &window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding) {
  // Add a fast 1x2x2x1, no padding fast reduce
  auto no_padding = std::all_of(window_padding.begin(), window_padding.end(),
                                [](const std::pair<int64_t, int64_t> &p) {
                                  return p.first == 0 && p.second == 0;
                                });
  if (window_shape == absl::Span<const int64_t>{1, 2, 2, 1} && no_padding) {
    return MaxPoolScatter1x2x2x1NoPaddingNoDilation(ctx, scatter_indices,
                                                    source, window_strides);
  }
  //  source_shape * window_numel
  auto tiled_1d_shape = source.shape();
  const int64_t window_numel = std::accumulate(
      window_shape.begin(), window_shape.end(), 1, std::multiplies<>());
  tiled_1d_shape.push_back(window_numel);

  Axes broadcast_dims(source.shape().size(), 0);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);

  auto tiled_1d_source =
      hal::broadcast_to(ctx, source, tiled_1d_shape, broadcast_dims);

  // selected_pos is the one hot encoding for each window.
  auto selected = hal::mul(ctx, tiled_1d_source, scatter_indices);

  Shape tiled_shape(source.shape().begin(), source.shape().end());
  tiled_shape.insert(tiled_shape.end(), window_shape.begin(),
                     window_shape.end());

  selected = hal::reshape(ctx, selected, tiled_shape);

  const size_t ndim = base_shape.size();
  auto base_x_window_shape = base_shape;
  base_x_window_shape.insert(base_x_window_shape.end(), window_shape.begin(),
                             window_shape.end());
  auto output = hal::zeros(ctx, source.dtype(), base_x_window_shape);
  if (source.isSecret()) {
    output = hal::seal(ctx, output);
  }

  const std::vector<int64_t> window_dilations(window_shape.size(), 1);
  const std::vector<int64_t> base_dilations(source.shape().size(), 1);
  std::vector<int64_t> window_index(ndim, 0);

  do {
    yacl::parallel_for(
        0, source.numel(), 2048, [&](int64_t begin, int64_t end) {
          Index tiled_index(2 * ndim, 0);
          Index base_x_window_index(2 * ndim, 0);
          std::copy(window_index.begin(), window_index.end(),
                    base_x_window_index.begin() + ndim);
          std::copy(window_index.begin(), window_index.end(),
                    tiled_index.begin() + ndim);
          auto source_index = unflattenIndex(begin, source.shape());
          for (int64_t idx = begin; idx < end; ++idx) {
            bool out_of_bound = getBaseIndexFromWindowIndex(
                window_shape, window_strides, window_dilations, window_padding,
                base_shape, base_dilations,
                absl::MakeSpan(tiled_index).subspan(0, ndim), window_index,
                absl::MakeSpan(base_x_window_index).subspan(0, ndim));
            if (!out_of_bound) {
              // TODO: anti-pattern, do not use .data(), use ops instead.
              output.data().update_slice(
                  selected.data().slice_scalar_at(tiled_index),
                  base_x_window_index);
            }
            bumpIndices(source.shape(),
                        absl::MakeSpan(tiled_index).subspan(0, ndim));
          }
        });
  } while (bumpIndices(window_shape, absl::MakeSpan(window_index)));

  auto base_1d_shape = base_shape;
  base_1d_shape.push_back(window_numel);
  output = hal::reshape(ctx, output, base_1d_shape);

  output = TreeReduce(
      ctx, {output}, base_1d_shape.size() - 1,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        return std::vector<spu::Value>{hal::add(ctx, lhs[0], rhs[0])};
      })[0];

  return hal::reshape(ctx, output, base_shape);
}

spu::Value SelectAndScatter(
    SPUContext *ctx, const spu::Value &base, const spu::Value &source,
    const spu::Value &init_val, const Shape &window_shape,
    const Strides &window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    const ValueBinaryFn &select_fn, const ValueBinaryFn &scatter_fn) {
  const size_t ndim = base.shape().size();

  // expand the base, simplify following actions without strides and padding.
  auto tiled =
      expandWindow(ctx, base, window_shape, window_strides, window_padding);

  // collapse the tile to 1d for better reduce performance
  auto tiled_1d_shape = source.shape();
  const int64_t window_numel = std::accumulate(
      window_shape.begin(), window_shape.end(), 1, std::multiplies<>());
  tiled_1d_shape.push_back(window_numel);
  auto tiled_1d = hal::reshape(ctx, tiled, tiled_1d_shape);

  //
  auto indices = Iota(ctx, DT_I64, window_numel);
  indices = hal::broadcast_to(ctx, indices, tiled_1d_shape);

  // Apply the reduce with indices.
  // total number of select_fn call is log2(window_numel)
  auto reduced = TreeReduce(
      ctx, {tiled_1d, indices}, tiled_1d_shape.size() - 1,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        auto pred = select_fn(lhs[0], rhs[0]);
        pred = hal::_prefer_a(ctx, pred);

        std::vector<spu::Value> rets;
        for (size_t idx = 0; idx < lhs.size(); idx++) {
          rets.push_back(hal::select(ctx, pred, lhs[idx], rhs[idx]));
        }
        return rets;
      });

  Axes broadcast_dims(source.shape().size(), 0);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);

  // selected_pos is the one hot encoding for each window.
  auto selected_pos = hal::equal(
      ctx, hal::broadcast_to(ctx, reduced[1], tiled_1d_shape), indices);
  auto selected = hal::select(
      ctx, selected_pos,
      hal::broadcast_to(ctx, source, tiled_1d_shape, broadcast_dims),
      hal::broadcast_to(ctx, init_val, tiled_1d_shape));

  // last step, collapse expanded shape to strided window
  // build a tensor with [base.shape() x window_shape], so each
  // [base.shape(), window_index] does not overlap with each other.
  selected = hal::reshape(ctx, selected, tiled.shape());

  auto base_x_window_shape = base.shape();
  base_x_window_shape.insert(base_x_window_shape.end(), window_shape.begin(),
                             window_shape.end());
  auto output = hal::expand(ctx, init_val, base_x_window_shape);

  const std::vector<int64_t> window_dilations(window_shape.size(), 1);
  const std::vector<int64_t> base_dilations(source.shape().size(), 1);
  std::vector<int64_t> window_index(ndim, 0);
  Index tiled_index(2 * ndim, 0);
  Index base_x_window_index(2 * ndim, 0);
  std::vector<int64_t> base_index(ndim, 0);
  do {
    std::copy(window_index.begin(), window_index.end(),
              base_x_window_index.begin() + ndim);
    std::fill(tiled_index.begin(), tiled_index.begin() + ndim, 0);
    std::copy(window_index.begin(), window_index.end(),
              tiled_index.begin() + ndim);

    do {
      bool out_of_bound = getBaseIndexFromWindowIndex(
          window_shape, window_strides, window_dilations, window_padding,
          base.shape(), base_dilations,
          absl::MakeSpan(tiled_index).subspan(0, ndim), window_index,
          absl::MakeSpan(base_x_window_index).subspan(0, ndim));
      if (!out_of_bound) {
        output.data().update_slice(selected.data().slice_scalar_at(tiled_index),
                                   base_x_window_index);
      }

    } while (bumpIndices(source.shape(),
                         absl::MakeSpan(tiled_index).subspan(0, ndim)));
  } while (bumpIndices(window_shape, absl::MakeSpan(window_index)));

  auto base_1d_shape = base.shape();
  base_1d_shape.push_back(window_numel);
  output = hal::reshape(ctx, output, base_1d_shape);

  output = TreeReduce(
      ctx, {output}, base_1d_shape.size() - 1,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        return std::vector<spu::Value>{scatter_fn(lhs[0], rhs[0])};
      })[0];

  return hal::reshape(ctx, output, base.shape());
}

}  // namespace spu::kernel::hlo
