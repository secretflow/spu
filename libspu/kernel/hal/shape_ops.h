// Copyright 2021 Ant Group Co., Ltd.
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

#include "libspu/kernel/context.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hal {

/// the broadcast function
// @param in, the input
// @param to_shape, the target shape
Value broadcast_to(HalContext* ctx, const Value& in,
                   absl::Span<const int64_t> to_shape,
                   absl::Span<const int64_t> in_dims = {});

/// the reshape function
// @param in, the input
// @param to_shape, the target shape
Value reshape(HalContext* ctx, const Value& in,
              absl::Span<const int64_t> to_shape);

/// the slice function
// @param input, the param
// @param start_indices, the start indices
// @param end_indices, the end indices
// @param strides, the strides
Value slice(HalContext* ctx, const Value& input,
            absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> end_indices,
            absl::Span<const int64_t> strides);

/// the transpose function
// @param in, the param
Value transpose(HalContext* ctx, const Value& in,
                absl::Span<const int64_t> permutation = {});

//// the reverse function
// @param in, the param
// @param dimensions, dimensions to reverse
Value reverse(HalContext* ctx, const Value& in,
              absl::Span<const int64_t> dimensions);

/// Expand a scalar into to_shape.
/// Compare with broadcast, expand actually reallocates and assign memory
Value expand(HalContext* ctx, const Value& in,
             absl::Span<const int64_t> to_shape);

}  // namespace spu::kernel::hal
