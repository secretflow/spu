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

#include <cstdint>
#include <vector>

#include "absl/types/span.h"

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hlo {

using BatchedValueBinaryFn = std::function<std::vector<spu::Value>(
    absl::Span<spu::Value const> lhs, absl::Span<spu::Value const> rhs)>;

struct ReduceWindowConfig {
  Shape window_shape;
  Strides window_strides;
  Sizes window_dilations;
  absl::Span<const std::pair<int64_t, int64_t>> window_padding;
  Sizes base_dilations;
};

std::vector<spu::Value> ReduceWindow(SPUContext *ctx,
                                     absl::Span<const spu::Value> inputs,
                                     absl::Span<const spu::Value> init_values,
                                     const Shape &ret_shape,
                                     const ReduceWindowConfig &config,
                                     const BatchedValueBinaryFn &reducer,
                                     bool ignore_init_values = false);

std::vector<spu::Value> Reduce(SPUContext *ctx,
                               absl::Span<const spu::Value> inputs,
                               absl::Span<const spu::Value> init_values,
                               const Axes &dims_to_reduce,
                               const BatchedValueBinaryFn &reducer,
                               bool ignore_init_values = false);

std::pair<spu::Value, spu::Value> ArgMax(SPUContext *ctx,
                                         const spu::Value &input,
                                         const Shape &ret_shape,
                                         const ReduceWindowConfig &config);

/// ------------------- non-PPHLO APIs ------------------------------------
std::vector<spu::Value> TreeReduce(SPUContext *ctx,
                                   absl::Span<const spu::Value> inputs,
                                   int64_t axis,
                                   const BatchedValueBinaryFn &reducer);

}  // namespace spu::kernel::hlo
