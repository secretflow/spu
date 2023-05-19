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
#include <vector>

#include "absl/types/span.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hlo {

using ValueBinaryFn =
    std::function<spu::Value(spu::Value const &lhs, spu::Value const &rhs)>;

spu::Value SelectAndScatterExpanded(
    SPUContext *ctx, const spu::Value &base, const spu::Value &source,
    const spu::Value &init_val, absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    const ValueBinaryFn &select_fn, const ValueBinaryFn &scatter_fn);

spu::Value SelectAndScatterNaive(
    SPUContext *ctx, const spu::Value &operand, const spu::Value &source,
    const spu::Value &init_val, absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    const ValueBinaryFn &select_fn, const ValueBinaryFn &scatter_fn);

/**
 * @brief This is a special implementation of MaxPooling's scatter part
 *
 * @param ctx HAL Context
 * @param scatter_indices Selected indices by ArgMax step
 * @param update Update values
 * @param window_shape
 * @param base_shape
 * @param window_strides
 * @param window_padding
 * @return spu::Value
 */
spu::Value MaxPoolScatter(
    SPUContext *ctx, const spu::Value &scatter_indices,
    const spu::Value &source, absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> base_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding);

}  // namespace spu::kernel::hlo
