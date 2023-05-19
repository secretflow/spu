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

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {

/// the broadcast function
// @param in, the input
// @param to_shape, the target shape
Value broadcast_to(SPUContext* ctx, const Value& in,
                   absl::Span<const int64_t> to_shape,
                   absl::Span<const int64_t> in_dims = {});

/// the reshape function
// @param in, the input
// @param to_shape, the target shape
Value reshape(SPUContext* ctx, const Value& in,
              absl::Span<const int64_t> to_shape);

/// the slice function
// @param input, the param
// @param start_indices, the start indices
// @param end_indices, the end indices
// @param strides, the strides
Value slice(SPUContext* ctx, const Value& input,
            absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> end_indices,
            absl::Span<const int64_t> strides);

/// This is a special slice for single element at indices
// @returns a array with empty shape (scalar)
Value slice_scalar_at(SPUContext* ctx, const Value& input,
                      absl::Span<const int64_t> indices);

// update a block of in with update, start_indices is postion at in
Value update_slice(SPUContext* ctx, const Value& in, const Value& update,
                   absl::Span<const int64_t> start_indices);

/// the transpose function
// @param in, the param
Value transpose(SPUContext* ctx, const Value& in,
                absl::Span<const int64_t> permutation = {});

//// the reverse function
// @param in, the param
// @param dimensions, dimensions to reverse
Value reverse(SPUContext* ctx, const Value& in,
              absl::Span<const int64_t> dimensions);

/// Expand a scalar into to_shape.
/// Compare with broadcast, expand actually reallocates and assign memory
Value expand(SPUContext* ctx, const Value& in,
             absl::Span<const int64_t> to_shape);

//// the pad function
// @param in, the param
// @param padding_value, to fill in the added padding
// @param edge_padding_low, the amount of padding added at the
//        low-end (next to index 0) of each dimension
// @param edge_padding_high, the amount of padding added at the high-end
//        (next to the highest index) of each dimension
// @param interior_padding, the amount of padding added between any two elements
//        in each dimension
Value pad(SPUContext* ctx, const Value& in, const Value& padding_value,
          absl::Span<const int64_t> edge_padding_low,
          absl::Span<const int64_t> edge_padding_high,
          absl::Span<const int64_t> interior_padding);

/// the concatenate function
// @param first, the first param
// @param second, the second param
// @param axis, the axis
Value concatenate(SPUContext* ctx, absl::Span<const Value> values,
                  const size_t& axis);

}  // namespace spu::kernel::hal
