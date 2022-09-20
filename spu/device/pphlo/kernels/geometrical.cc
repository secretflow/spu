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

#include "spu/device/pphlo/kernels/geometrical.h"

#include "spu/hal/concat.h"
#include "spu/hal/shape_ops.h"

namespace spu::device::pphlo::kernel {

hal::Value Transpose(HalContext *ctx, const hal::Value &in,
                     absl::Span<const int64_t> permutation) {
  return hal::transpose(ctx, in, permutation);
}

hal::Value Broadcast(HalContext *ctx, const hal::Value &in,
                     absl::Span<const int64_t> to_shape,
                     absl::Span<const int64_t> in_dims) {
  return hal::broadcast_to(ctx, in, to_shape, in_dims);
}

hal::Value Reshape(HalContext *ctx, const hal::Value &in,
                   absl::Span<const int64_t> to_shape) {
  return hal::reshape(ctx, in, to_shape);
}

hal::Value Concatenate(HalContext *ctx, absl::Span<const hal::Value> operands,
                       int64_t axis) {
  return hal::concatenate(ctx, operands, axis);
}

hal::Value Slice(HalContext *ctx, const hal::Value &in,
                 absl::Span<const int64_t> start, absl::Span<const int64_t> end,
                 absl::Span<const int64_t> strides) {
  return hal::slice(ctx, in, start, end, strides);
}

hal::Value Pad(HalContext *ctx, const hal::Value &in,
               const hal::Value &pad_value, absl::Span<const int64_t> edge_low,
               absl::Span<const int64_t> edge_high,
               absl::Span<const int64_t> inner) {
  return hal::pad(ctx, in, pad_value, edge_low, edge_high, inner);
}

hal::Value Reverse(HalContext *ctx, const hal::Value &in,
                   absl::Span<const int64_t> dims) {
  return hal::reverse(ctx, in, dims);
}

} // namespace spu::device::pphlo::kernel
