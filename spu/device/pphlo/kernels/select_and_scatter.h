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

#pragma once

#include <utility>

#include "spu/core/ndarray_ref.h"
#include "spu/device/pphlo/kernels/basic_binary.h"
#include "spu/device/pphlo/kernels/basic_ternary.h"
#include "spu/device/pphlo/kernels/const.h"
#include "spu/device/pphlo/kernels/geometrical.h"
#include "spu/device/pphlo/kernels/utils.h"
#include "spu/hal/shape_ops.h"

namespace spu::device::pphlo::kernel {

struct SelectAndScatterConfig {
  absl::Span<const int64_t> window_shape;
  absl::Span<const int64_t> window_strides;
  absl::Span<const std::pair<int64_t, int64_t>> window_padding;
  absl::Span<const int64_t> window_dilations;
  absl::Span<const int64_t> base_dilations;
};

template <typename SelectComp>
hal::Value SelectAndScatter(HalContext *ctx, const hal::Value &operand,
                            const hal::Value &source,
                            const hal::Value &init_val,
                            const SelectComp &select_body,
                            const SelectAndScatterConfig &config) {

  auto result = hal::expand(ctx, init_val, operand.shape());

  auto idx_matrix =
      Reshape(ctx, Iota<int64_t>(ctx, operand.numel(), operand.vtype()),
              operand.shape());

  const auto rank = operand.shape().size();
  std::vector<int64_t> window_index(rank, 0);

  hal::Value current_val(NdArrayRef(operand.data().eltype(), source.shape()),
                         operand.dtype());
  hal::Value current_idx(NdArrayRef(idx_matrix.data().eltype(), source.shape()),
                         idx_matrix.dtype());
  hal::Value selected_val;
  hal::Value selected_idx;
  bool first_iter = true;

  do {
    std::vector<int64_t> output_index(source.shape().size(), 0);
    do {
      RunOnWindowIndex(
          config.window_shape, config.window_strides, config.window_dilations,
          config.window_padding, operand.shape(), config.base_dilations,
          output_index, window_index,
          [&](absl::Span<const int64_t> operand_index) {
            current_val.copyElementFrom(operand, operand_index, output_index);
            current_idx.copyElementFrom(idx_matrix, operand_index,
                                        output_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));
    if (first_iter) {
      // First iter, don't do the real compute, just copy to selected
      selected_val = current_val.clone();
      selected_idx = current_idx.clone();
      first_iter = false;
    } else {
      auto ret = select_body(selected_val, current_val);
      selected_val = select(ctx, ret, selected_val, current_val);
      selected_idx = select(ctx, ret, selected_idx, current_idx);
    }
  } while (
      bumpIndices<int64_t>(config.window_shape, absl::MakeSpan(window_index)));

  // Scatter
  std::fill(window_index.begin(), window_index.end(), 0);
  hal::Value idx_slice(NdArrayRef(idx_matrix.data().eltype(), source.shape()),
                       idx_matrix.dtype());
  hal::Value result_slice(NdArrayRef(result.data().eltype(), source.shape()),
                          result.dtype());

  do {
    std::vector<int64_t> output_index(source.shape().size(), 0);
    do {
      RunOnWindowIndex(
          config.window_shape, config.window_strides, config.window_dilations,
          config.window_padding, operand.shape(), config.base_dilations,
          output_index, window_index,
          [&](absl::Span<const int64_t> operand_index) {
            idx_slice.copyElementFrom(idx_matrix, operand_index, output_index);
            result_slice.copyElementFrom(result, operand_index, output_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));

    auto mask = Equal(ctx, selected_idx, idx_slice);

    auto added = Add(ctx, result_slice, source);
    result_slice = Select(ctx, mask, added, result_slice);

    // Reset, copy window again...
    std::fill(output_index.begin(), output_index.end(), 0);
    do {
      RunOnWindowIndex(
          config.window_shape, config.window_strides, config.window_dilations,
          config.window_padding, operand.shape(), config.base_dilations,
          output_index, window_index,
          [&](absl::Span<const int64_t> operand_index) {
            result.copyElementFrom(result_slice, output_index, operand_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));
  } while (
      bumpIndices<int64_t>(config.window_shape, absl::MakeSpan(window_index)));

  return result;
}

} // namespace spu::device::pphlo::kernel
