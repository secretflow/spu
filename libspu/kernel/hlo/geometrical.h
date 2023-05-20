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

#include "libspu/kernel/hlo/utils.h"
namespace spu::kernel::hlo {

spu::Value Transpose(SPUContext *ctx, const spu::Value &in,
                     absl::Span<const int64_t> permutation);

spu::Value Broadcast(SPUContext *ctx, const spu::Value &in,
                     absl::Span<const int64_t> to_shape,
                     absl::Span<const int64_t> in_dims);

spu::Value Reshape(SPUContext *ctx, const spu::Value &in,
                   absl::Span<const int64_t> to_shape);

spu::Value Concatenate(SPUContext *ctx, absl::Span<const spu::Value> operands,
                       int64_t axis);

spu::Value Slice(SPUContext *ctx, const spu::Value &in,
                 absl::Span<const int64_t> start, absl::Span<const int64_t> end,
                 absl::Span<const int64_t> strides);

spu::Value Pad(SPUContext *ctx, const spu::Value &in,
               const spu::Value &pad_value, absl::Span<const int64_t> edge_low,
               absl::Span<const int64_t> edge_high,
               absl::Span<const int64_t> inner);

spu::Value Reverse(SPUContext *ctx, const spu::Value &in,
                   absl::Span<const int64_t> dims);

}  // namespace spu::kernel::hlo
