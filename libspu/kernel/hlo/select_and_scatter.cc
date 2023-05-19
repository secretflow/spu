
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
#include "libspu/core/shape_util.h"
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

spu::Value MaxPoolScatter1x2x2x1NoPaddingNoDialation(
    SPUContext *ctx, const spu::Value &scatter_indices,
    const spu::Value &source, absl::Span<const int64_t> window_strides) {
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
  f_slices[0] = std::async(
      std::launch::async, hal::pad, ctx, slices[0], z,
      std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 1, 1, 0},
      std::vector<int64_t>{0, window_strides[1] - 1, window_strides[2] - 1, 0});
  f_slices[1] = std::async(
      std::launch::async, hal::pad, ctx, slices[1], z,
      std::vector<int64_t>{0, 0, 1, 0}, std::vector<int64_t>{0, 1, 0, 0},
      std::vector<int64_t>{0, window_strides[1] - 1, window_strides[2] - 1, 0});
  f_slices[2] = std::async(
      std::launch::async, hal::pad, ctx, slices[2], z,
      std::vector<int64_t>{0, 1, 0, 0}, std::vector<int64_t>{0, 0, 1, 0},
      std::vector<int64_t>{0, window_strides[1] - 1, window_strides[2] - 1, 0});
  f_slices[3] = std::async(
      std::launch::async, hal::pad, ctx, slices[3], z,
      std::vector<int64_t>{0, 1, 1, 0}, std::vector<int64_t>{0, 0, 0, 0},
      std::vector<int64_t>{0, window_strides[1] - 1, window_strides[2] - 1, 0});

  spu::Value ret = f_slices[0].get();
  for (size_t idx = 1; idx < 4; ++idx) {
    ret = hal::add(ctx, ret, f_slices[idx].get());
  }

  return ret;
};

spu::Value MaxPoolScatter(
    SPUContext *ctx, const spu::Value &scatter_indices,
    const spu::Value &source, absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> base_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding) {
  // Add a fast 1x2x2x1, no padding fast reduce
  auto no_padding = std::all_of(window_padding.begin(), window_padding.end(),
                                [](const std::pair<int64_t, int64_t> &p) {
                                  return p.first == 0 && p.second == 0;
                                });
  if (window_shape == absl::Span<const int64_t>{1, 2, 2, 1} && no_padding) {
    return MaxPoolScatter1x2x2x1NoPaddingNoDialation(ctx, scatter_indices,
                                                     source, window_strides);
  }
  //  source_shape * window_numel
  std::vector<int64_t> tiled_1d_shape = source.shape();
  const int64_t window_numel = std::accumulate(
      window_shape.begin(), window_shape.end(), 1, std::multiplies<int64_t>());
  tiled_1d_shape.push_back(window_numel);

  std::vector<int64_t> broadcast_dims(source.shape().size(), 0);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);

  auto tiled_1d_source =
      hal::broadcast_to(ctx, source, tiled_1d_shape, broadcast_dims);

  // selected_pos is the one hot encoding for each window.
  auto selected = hal::mul(ctx, tiled_1d_source, scatter_indices);

  std::vector<int64_t> tiled_shape(source.shape().begin(),
                                   source.shape().end());
  tiled_shape.insert(tiled_shape.end(), window_shape.begin(),
                     window_shape.end());

  selected = hal::reshape(ctx, selected, tiled_shape);

  const size_t ndim = base_shape.size();
  std::vector<int64_t> base_x_window_shape(base_shape.begin(),
                                           base_shape.end());
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
          std::vector<int64_t> tiled_index(2 * ndim, 0);
          std::vector<int64_t> base_x_window_index(2 * ndim, 0);
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
            bumpIndices<int64_t>(source.shape(),
                                 absl::MakeSpan(tiled_index).subspan(0, ndim));
          }
        });
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  std::vector<int64_t> base_1d_shape(base_shape.begin(), base_shape.end());
  base_1d_shape.push_back(window_numel);
  output = hal::reshape(ctx, output, base_1d_shape);

  output = TreeReduce(
      ctx, {output}, base_1d_shape.size() - 1,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        return std::vector<spu::Value>{hal::add(ctx, lhs[0], rhs[0])};
      })[0];

  return hal::reshape(ctx, output, base_shape);
}

spu::Value SelectAndScatterExpanded(
    SPUContext *ctx, const spu::Value &base, const spu::Value &source,
    const spu::Value &init_val, absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    const ValueBinaryFn &select_fn, const ValueBinaryFn &scatter_fn) {
  const size_t ndim = base.shape().size();

  // expand the base, simplify following actions without strides and padding.
  auto expanded = ExpandStridedWindow(ctx, base, window_shape, window_strides,
                                      window_padding);

  // sanity check, make (source x window == expanded)
  for (size_t dim = 0; dim < ndim; dim++) {
    SPU_ENFORCE(expanded.shape()[dim] % window_shape[dim] == 0);
    SPU_ENFORCE(expanded.shape()[dim] / window_shape[dim] ==
                source.shape()[dim]);
  }

  auto tiled = ConvertToTiledLayout(ctx, expanded, window_shape);

  // collapse the tile to 1d for better reduce performance
  std::vector<int64_t> tiled_1d_shape = source.shape();
  const int64_t window_numel = std::accumulate(
      window_shape.begin(), window_shape.end(), 1, std::multiplies<int64_t>());
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

  std::vector<int64_t> broadcast_dims(source.shape().size(), 0);
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

  std::vector<int64_t> base_x_window_shape = base.shape();
  base_x_window_shape.insert(base_x_window_shape.end(), window_shape.begin(),
                             window_shape.end());
  auto output = hal::expand(ctx, init_val, base_x_window_shape);

  const std::vector<int64_t> window_dilations(window_shape.size(), 1);
  const std::vector<int64_t> base_dilations(source.shape().size(), 1);
  std::vector<int64_t> window_index(ndim, 0);
  std::vector<int64_t> tiled_index(2 * ndim, 0);
  std::vector<int64_t> base_x_window_index(2 * ndim, 0);
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

    } while (bumpIndices<int64_t>(
        source.shape(), absl::MakeSpan(tiled_index).subspan(0, ndim)));
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  std::vector<int64_t> base_1d_shape = base.shape();
  base_1d_shape.push_back(window_numel);
  output = hal::reshape(ctx, output, base_1d_shape);

  output = TreeReduce(
      ctx, {output}, base_1d_shape.size() - 1,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        return std::vector<spu::Value>{scatter_fn(lhs[0], rhs[0])};
      })[0];

  return hal::reshape(ctx, output, base.shape());
}

spu::Value SelectAndScatterNaive(
    SPUContext *ctx, const spu::Value &operand, const spu::Value &source,
    const spu::Value &init_val, absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    const ValueBinaryFn &select_fn, const ValueBinaryFn &scatter_fn) {
  // Create an index matrix
  auto idx_matrix = hal::iota(ctx, DT_I64, operand.numel());
  if (operand.isSecret()) {
    idx_matrix = hal::seal(ctx, idx_matrix);
  }
  idx_matrix = hal::reshape(ctx, idx_matrix, operand.shape());

  const auto rank = window_shape.size();
  std::vector<int64_t> window_index(rank, 0);

  spu::Value selected_val;
  spu::Value selected_idx;
  bool first_iter = true;

  std::vector<int64_t> dummy_window_dilation(window_shape.size(), 1);
  std::vector<int64_t> dummy_base_dilation(operand.shape().size(), 1);

  // Select part
  {
    spu::Value current_val(NdArrayRef(operand.data().eltype(), source.shape()),
                           operand.dtype());
    spu::Value current_idx(
        NdArrayRef(idx_matrix.data().eltype(), source.shape()),
        idx_matrix.dtype());

    do {
      std::vector<int64_t> output_index(source.shape().size(), 0);
      do {
        RunOnWindowIndex(
            window_shape, window_strides, dummy_window_dilation, window_padding,
            operand.shape(), dummy_base_dilation, output_index, window_index,
            [&](absl::Span<const int64_t> operand_index) {
              current_val.data().update_slice(
                  operand.data().slice_scalar_at(operand_index), output_index);
              current_idx.data().update_slice(
                  idx_matrix.data().slice_scalar_at(operand_index),
                  output_index);
            });
      } while (
          bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));

      if (first_iter) {
        // First iter, don't do the real compute, just copy to selected
        selected_val = current_val;
        selected_idx = current_idx;
        first_iter = false;
      } else {
        auto ret = select_fn(selected_val, current_val);
        selected_val = hal::select(ctx, ret, selected_val, current_val);
        selected_idx = hal::select(ctx, ret, selected_idx, current_idx);
      }
    } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));
  }

  // Scatter
  auto result = hal::expand(ctx, init_val, operand.shape());
  std::fill(window_index.begin(), window_index.end(), 0);

  spu::Value idx_slice(NdArrayRef(idx_matrix.data().eltype(), source.shape()),
                       idx_matrix.dtype());

  spu::Value result_slice(NdArrayRef(result.data().eltype(), source.shape()),
                          result.dtype());

  do {
    std::vector<int64_t> output_index(source.shape().size(), 0);
    do {
      RunOnWindowIndex(
          window_shape, window_strides, dummy_window_dilation, window_padding,
          operand.shape(), dummy_base_dilation, output_index, window_index,
          [&](absl::Span<const int64_t> operand_index) {
            idx_slice.data().update_slice(
                idx_matrix.data().slice_scalar_at(operand_index), output_index);
            result_slice.data().update_slice(
                result.data().slice_scalar_at(operand_index), output_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));

    auto mask = hal::equal(ctx, selected_idx, idx_slice);

    auto computed = scatter_fn(result_slice, source);

    result_slice = hal::select(ctx, mask, computed, result_slice);

    // Reset, copy window again...
    std::fill(output_index.begin(), output_index.end(), 0);
    do {
      RunOnWindowIndex(window_shape, window_strides, dummy_window_dilation,
                       window_padding, operand.shape(), dummy_base_dilation,
                       output_index, window_index,
                       [&](absl::Span<const int64_t> operand_index) {
                         result.data().update_slice(
                             result_slice.data().slice_scalar_at(output_index),
                             operand_index);
                       });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  return result;
}

}  // namespace spu::kernel::hlo
