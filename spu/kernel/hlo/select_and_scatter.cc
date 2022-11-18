
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

#include "spu/kernel/hlo/select_and_scatter.h"

#include <cstdint>
#include <iostream>

#include "yasl/utils/parallel.h"

#include "spu/core/shape_util.h"
#include "spu/kernel/context.h"
#include "spu/kernel/hal/constants.h"
#include "spu/kernel/hal/debug.h"
#include "spu/kernel/hal/polymorphic.h"  // for select
#include "spu/kernel/hal/shape_ops.h"
#include "spu/kernel/hlo/const.h"
#include "spu/kernel/hlo/reduce.h"
#include "spu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

spu::Value MaxPoolScatter(
    HalContext *ctx, const spu::Value &scatter_indices,
    const spu::Value &source, absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> base_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding) {
  // source_shape * window_numel
  std::vector<int64_t> tiled_1d_shape = source.shape();
  const int64_t window_numel = std::accumulate(
      window_shape.begin(), window_shape.end(), 1, std::multiplies<int64_t>());
  tiled_1d_shape.push_back(window_numel);

  auto tiled_1d_select = hal::reshape(ctx, scatter_indices, tiled_1d_shape);

  std::vector<int64_t> broadcast_dims(source.shape().size(), 0);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);

  auto tiled_1d_source =
      hal::broadcast_to(ctx, source, tiled_1d_shape, broadcast_dims);

  // selected_pos is the one hot encoding for each window.
  auto selected = hal::mul(ctx, tiled_1d_source, tiled_1d_select);

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
  auto output =
      hal::zeros(ctx, source.vtype(), source.dtype(), base_x_window_shape);

  const std::vector<int64_t> window_dilations(window_shape.size(), 1);
  const std::vector<int64_t> base_dilations(source.shape().size(), 1);
  std::vector<int64_t> window_index(ndim, 0);

  do {
    yasl::parallel_for(
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
              output.copyElementFrom(selected, tiled_index,
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
    HalContext *ctx, const spu::Value &base, const spu::Value &source,
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
    YASL_ENFORCE(expanded.shape()[dim] % window_shape[dim] == 0);
    YASL_ENFORCE(expanded.shape()[dim] / window_shape[dim] ==
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
  auto indices = Iota<int64_t>(ctx, window_numel, VIS_PUBLIC);
  indices = hal::broadcast_to(ctx, indices, tiled_1d_shape);

  // Apply the reduce with indices.
  // total number of select_fn call is log2(window_numel)
  auto reduced = TreeReduce(
      ctx, {tiled_1d, indices}, tiled_1d_shape.size() - 1,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        auto pred = select_fn(lhs[0], rhs[0]);
        pred =
            hal::mul(ctx, pred,
                     hal::constant(ctx, 1U, pred.shape()));  // noop, to ashare

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
        output.copyElementFrom(selected, tiled_index, base_x_window_index);
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
    HalContext *ctx, const spu::Value &operand, const spu::Value &source,
    const spu::Value &init_val, absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    const ValueBinaryFn &select_fn, const ValueBinaryFn &scatter_fn) {
  // Create an index matrix
  auto idx_matrix =
      hal::reshape(ctx, Iota<int64_t>(ctx, operand.numel(), operand.vtype()),
                   operand.shape());

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
              current_val.copyElementFrom(operand, operand_index, output_index);
              current_idx.copyElementFrom(idx_matrix, operand_index,
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
            idx_slice.copyElementFrom(idx_matrix, operand_index, output_index);
            result_slice.copyElementFrom(result, operand_index, output_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));

    auto mask = hal::equal(ctx, selected_idx, idx_slice);

    auto computed = scatter_fn(result_slice, source);

    result_slice = hal::select(ctx, mask, computed, result_slice);

    // Reset, copy window again...
    std::fill(output_index.begin(), output_index.end(), 0);
    do {
      RunOnWindowIndex(
          window_shape, window_strides, dummy_window_dilation, window_padding,
          operand.shape(), dummy_base_dilation, output_index, window_index,
          [&](absl::Span<const int64_t> operand_index) {
            result.copyElementFrom(result_slice, output_index, operand_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  return result;
}

}  // namespace spu::kernel::hlo
