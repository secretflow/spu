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

#include "libspu/core/context.h"
#include "libspu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

struct GatherConfig {
  absl::Span<const int64_t> sliceSizes;
  int64_t indexVectorDim;
  absl::Span<const int64_t> offsetDims;
  absl::Span<const int64_t> collapsedSliceDims;
  absl::Span<const int64_t> startIndexMap;
};

// This is ported from
// https://github.com/tensorflow/tensorflow/blob/bf4c6ad46dac1f7f69911e2bfc48e141a39b40af/tensorflow/compiler/xla/service/hlo_evaluator.cc#L1774
spu::Value Gather(SPUContext *ctx, const spu::Value &operand,
                  const spu::Value &start_indices, const GatherConfig &config,
                  absl::Span<const int64_t> result_shape);

spu::Value DynamicUpdateSlice(SPUContext *ctx, const spu::Value &operand,
                              const spu::Value &update,
                              absl::Span<const spu::Value> start_indices);

spu::Value DynamicSlice(SPUContext *ctx, const spu::Value &operand,
                        absl::Span<const int64_t> slice_size,
                        absl::Span<const spu::Value> start_indices);

/// ------------------- non-XLA APIs ------------------------------------
// @brief Update slice
spu::Value UpdateSlice(SPUContext *ctx, const spu::Value &in,
                       const spu::Value &update,
                       absl::Span<const int64_t> start_indices);

spu::Value FilterByMask(SPUContext *ctx, const spu::Value &operand,
                        absl::Span<const uint8_t> mask);

spu::Value LinearGather(SPUContext *ctx, const spu::Value &in,
                        absl::Span<const int64_t> indices);

void LinearScatterInPlace(SPUContext *ctx, spu::Value &in,
                          const spu::Value &update,
                          absl::Span<const int64_t> indices);

}  // namespace spu::kernel::hlo
