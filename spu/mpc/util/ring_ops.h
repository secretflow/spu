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

#include "spu/core/array_ref.h"
#include "spu/core/type.h"

namespace spu::mpc {

void ring_print(const ArrayRef& x, std::string_view name = "_");

ArrayRef ring_rand(FieldType field, size_t size);
ArrayRef ring_rand(FieldType field, size_t size, uint128_t prg_seed,
                   uint64_t* prg_counter);
ArrayRef ring_rand_range(FieldType field, size_t size, int32_t min,
                         int32_t max);

ArrayRef ring_zeros(FieldType field, size_t size);

ArrayRef ring_ones(FieldType field, size_t size);

ArrayRef ring_randbit(FieldType field, size_t size);

void ring_assign(ArrayRef& lhs, const ArrayRef& rhs);

// signed 2's complement negation.
ArrayRef ring_neg(const ArrayRef& x);
void ring_neg_(ArrayRef& x);

ArrayRef ring_add(const ArrayRef& x, const ArrayRef& y);
void ring_add_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_sub(const ArrayRef& x, const ArrayRef& y);
void ring_sub_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_mul(const ArrayRef& x, const ArrayRef& y);
void ring_mul_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_mmul(const ArrayRef& x, const ArrayRef& y, size_t M, size_t N,
                   size_t K);

ArrayRef ring_not(const ArrayRef& x);
void ring_not_(ArrayRef& x);

ArrayRef ring_and(const ArrayRef& x, const ArrayRef& y);
void ring_and_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_xor(const ArrayRef& x, const ArrayRef& y);
void ring_xor_(ArrayRef& x, const ArrayRef& y);

ArrayRef ring_equal(const ArrayRef& x, const ArrayRef& y);
ArrayRef ring_equal_(const ArrayRef& x);

ArrayRef ring_arshift(const ArrayRef& x, size_t bits);
void ring_arshift_(ArrayRef& x, size_t bits);

ArrayRef ring_rshift(const ArrayRef& x, size_t bits);
void ring_rshift_(ArrayRef& x, size_t bits);

ArrayRef ring_lshift(const ArrayRef& x, size_t bits);
void ring_lshift_(ArrayRef& x, size_t bits);

ArrayRef ring_bitrev(const ArrayRef& x, size_t start, size_t end);
void ring_bitrev_(ArrayRef& x, size_t start, size_t end);

ArrayRef ring_sum(absl::Span<ArrayRef const> arrs);

bool ring_all_equal(const ArrayRef& x, const ArrayRef& y, size_t abs_err = 0);

std::vector<bool> ring_as_bool(const ArrayRef& x);

ArrayRef ring_select(const std::vector<uint8_t>& c, const ArrayRef& x,
                     const ArrayRef& y);

// random additive splits.
std::vector<ArrayRef> ring_rand_splits(const ArrayRef& arr, size_t num_splits);

}  // namespace spu::mpc
