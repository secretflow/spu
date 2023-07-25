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

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type.h"

namespace spu::mpc {

void ring_print(const NdArrayRef& x, std::string_view name = "_");

NdArrayRef ring_rand(FieldType field, const Shape& shape);
NdArrayRef ring_rand(FieldType field, const Shape& shape, uint128_t prg_seed,
                     uint64_t* prg_counter);
NdArrayRef ring_rand_range(FieldType field, const Shape& shape, int32_t min,
                           int32_t max);

NdArrayRef ring_zeros(FieldType field, const Shape& shape);

NdArrayRef ring_ones(FieldType field, const Shape& shape);

NdArrayRef ring_randbit(FieldType field, const Shape& shape);

void ring_assign(NdArrayRef& x, const NdArrayRef& y);

// signed 2's complement negation.
NdArrayRef ring_neg(const NdArrayRef& x);
void ring_neg_(NdArrayRef& x);

NdArrayRef ring_add(const NdArrayRef& x, const NdArrayRef& y);
void ring_add_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef ring_sub(const NdArrayRef& x, const NdArrayRef& y);
void ring_sub_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef ring_mul(const NdArrayRef& x, const NdArrayRef& y);
void ring_mul_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef ring_mul(const NdArrayRef& x, uint128_t y);
void ring_mul_(NdArrayRef& x, uint128_t y);

NdArrayRef ring_mmul(const NdArrayRef& lhs, const NdArrayRef& rhs);
void ring_mmul_(NdArrayRef& out, const NdArrayRef& lhs, const NdArrayRef& rhs);

NdArrayRef ring_not(const NdArrayRef& x);
void ring_not_(NdArrayRef& x);

NdArrayRef ring_and(const NdArrayRef& x, const NdArrayRef& y);
void ring_and_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef ring_xor(const NdArrayRef& x, const NdArrayRef& y);
void ring_xor_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef ring_equal(const NdArrayRef& x, const NdArrayRef& y);
NdArrayRef ring_equal_(const NdArrayRef& x);

NdArrayRef ring_arshift(const NdArrayRef& x, size_t bits);
void ring_arshift_(NdArrayRef& x, size_t bits);

NdArrayRef ring_rshift(const NdArrayRef& x, size_t bits);
void ring_rshift_(NdArrayRef& x, size_t bits);

NdArrayRef ring_lshift(const NdArrayRef& x, size_t bits);
void ring_lshift_(NdArrayRef& x, size_t bits);

NdArrayRef ring_bitrev(const NdArrayRef& x, size_t start, size_t end);
void ring_bitrev_(NdArrayRef& x, size_t start, size_t end);

NdArrayRef ring_sum(absl::Span<NdArrayRef const> arrs);

bool ring_all_equal(const NdArrayRef& x, const NdArrayRef& y,
                    size_t abs_err = 0);

// Note: here we use uint8_t instead of bool because most of the time the casted
// boolean will participate in arithmetic computation in the future.
std::vector<uint8_t> ring_cast_boolean(const NdArrayRef& x);

// x & bits[low, high)
NdArrayRef ring_bitmask(const NdArrayRef& x, size_t low, size_t high);
void ring_bitmask_(NdArrayRef& x, size_t low, size_t high);

NdArrayRef ring_select(const std::vector<uint8_t>& c, const NdArrayRef& x,
                       const NdArrayRef& y);

// random additive splits.
std::vector<NdArrayRef> ring_rand_additive_splits(const NdArrayRef& arr,
                                                  size_t num_splits);
// random boolean splits.
std::vector<NdArrayRef> ring_rand_boolean_splits(const NdArrayRef& arr,
                                                 size_t num_splits);

template <typename T>
void ring_set_value(NdArrayRef& in, const T& value) {
  pforeach(0, in.numel(), [&](int64_t idx) { in.at<T>(idx) = value; });
}

}  // namespace spu::mpc
