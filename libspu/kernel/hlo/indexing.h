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

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hlo {

spu::Value DynamicUpdateSlice(SPUContext *ctx, const spu::Value &operand,
                              const spu::Value &update,
                              absl::Span<const spu::Value> start_indices);

spu::Value DynamicSlice(SPUContext *ctx, const spu::Value &operand,
                        const Sizes &slice_size,
                        absl::Span<const spu::Value> start_indices);

/// ------------------- non-XLA APIs ------------------------------------
// @brief Update slice
spu::Value UpdateSlice(SPUContext *ctx, const spu::Value &in,
                       const spu::Value &update, const Index &start_indices);

spu::Value FilterByMask(SPUContext *ctx, const spu::Value &operand,
                        absl::Span<const uint8_t> mask);

spu::Value LinearGather(SPUContext *ctx, const spu::Value &in,
                        const Index &indices);

void LinearScatterInPlace(SPUContext *ctx, spu::Value &in,
                          const spu::Value &update, const Index &indices);

}  // namespace spu::kernel::hlo
