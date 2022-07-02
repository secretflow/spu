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

#include "spu/hal/context.h"
#include "spu/hal/value.h"

namespace spu::hal {

/// FIXME: Remove these hacky templates once we get rid of xtensor here
template <size_t S>
struct element_t_s {
  std::array<std::byte, S> buf;
  // xtensor uses operator+ to compute type promotion rule of container element
  // So we provides a empty + to make it happy
  element_t_s operator+(const element_t_s& /*unused*/) { return *this; }
};

#define __CASE_SIZE(SIZE, ...)                       \
  case (SIZE): {                                     \
    using element_t = element_t_s<SIZE>;             \
    [[maybe_unused]] constexpr size_t _kSize = SIZE; \
    return __VA_ARGS__();                            \
  }

#define DISPATCH_ALL_ELSIZE(SIZE, ...)                         \
  [&] {                                                        \
    switch (SIZE) {                                            \
      __CASE_SIZE(4, __VA_ARGS__)                              \
      __CASE_SIZE(8, __VA_ARGS__)                              \
      __CASE_SIZE(16, __VA_ARGS__)                             \
      __CASE_SIZE(32, __VA_ARGS__)                             \
      default:                                                 \
        YASL_THROW("un-implemented for elment_size={}", SIZE); \
    }                                                          \
  }()

/// the broadcast function
// @param in, the input
// @param to_shape, the target shape
Value broadcast_to(HalContext* ctx, const Value& in,
                   const std::vector<int64_t>& to_shape,
                   const std::vector<size_t>& in_dims = {});

/// the concatenate function
// @param first, the first param
// @param second, the second param
// @param axis, the axis
Value concatenate(HalContext* ctx, absl::Span<const Value> values,
                  const size_t& axis);

/// the reshape function
// @param in, the input
// @param to_shape, the target shape
Value reshape(HalContext* ctx, const Value& in,
              const std::vector<int64_t>& to_shape);

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
                std::vector<int64_t> permutation = {});

//// the reverse function
// @param in, the param
// @param dimensions, dimensions to reverse
Value reverse(HalContext* ctx, const Value& in,
              const std::vector<size_t>& dimensions);

//// the pad function
// @param in, the param
// @param padding_value, to fill in the added padding
// @edge_padding_low, the amount of padding added at the low-end (next to index
// 0) of each dimension
// @edge_padding_high, the amount of padding added at the high-end (next to the
// highest index) of each dimension
// @interior_padding, the amount of padding added between any two elements in
// each dimension
Value pad(HalContext* ctx, const Value& in, const Value& padding_value,
          const std::vector<int64_t>& edge_padding_low,
          const std::vector<int64_t>& edge_padding_high,
          const std::vector<int64_t>& interior_padding);

}  // namespace spu::hal
