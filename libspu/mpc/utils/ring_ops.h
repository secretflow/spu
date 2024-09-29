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

#include "libspu/core/memref.h"

namespace spu::mpc {

#define DEF_RVALUE_BINARY_RING_OP(op_name, commutative)                       \
  template <class X, class Y, bool COMMUTATIVE = commutative>                 \
  typename std::enable_if<                                                    \
      std::is_same_v<MemRef, std::remove_cv_t<std::remove_reference_t<X>>> && \
          std::is_same_v<MemRef,                                              \
                         std::remove_cv_t<std::remove_reference_t<Y>>>,       \
      MemRef>::type                                                           \
  op_name(X&& x, Y&& y) {                                                     \
    if constexpr (std::is_rvalue_reference_v<decltype(x)>) {                  \
      op_name##_(x, y);                                                       \
      if constexpr (std::is_rvalue_reference_v<decltype(y)>) {                \
        MemRef dummy = std::move(y);                                          \
      }                                                                       \
      return std::move(x);                                                    \
    } else if constexpr (std::is_rvalue_reference_v<decltype(y)> &&           \
                         COMMUTATIVE) {                                       \
      op_name##_(y, x);                                                       \
      return std::move(y);                                                    \
    } else {                                                                  \
      return op_name(static_cast<const MemRef&>(x),                           \
                     static_cast<const MemRef&>(y));                          \
    }                                                                         \
  }

void ring_print(const MemRef& x, std::string_view name = "_");

void ring_rand(MemRef& in);
void ring_rand(MemRef& in, uint128_t prg_seed, uint64_t* prg_counter);
void ring_rand_range(MemRef& in, int32_t min, int32_t max);

void ring_zeros(MemRef& in);

void ring_ones(MemRef& in);

void ring_randbit(MemRef& in);

void ring_assign(MemRef& x, const MemRef& y);

void ring_msb(MemRef& out, const MemRef& in);

// signed 2's complement negation.
MemRef ring_neg(const MemRef& x);
void ring_neg_(MemRef& x);

MemRef ring_add(const MemRef& x, const MemRef& y);
void ring_add_(MemRef& x, const MemRef& y);
DEF_RVALUE_BINARY_RING_OP(ring_add, true);

MemRef ring_sub(const MemRef& x, const MemRef& y);
void ring_sub_(MemRef& x, const MemRef& y);
DEF_RVALUE_BINARY_RING_OP(ring_sub, false);

MemRef ring_mul(const MemRef& x, const MemRef& y);
void ring_mul_(MemRef& x, const MemRef& y);
DEF_RVALUE_BINARY_RING_OP(ring_mul, true);

MemRef ring_mul(const MemRef& x, uint128_t y);
void ring_mul_(MemRef& x, uint128_t y);
MemRef ring_mul(MemRef&& x, uint128_t y);

MemRef ring_mmul(const MemRef& lhs, const MemRef& rhs);
void ring_mmul_(MemRef& out, const MemRef& lhs, const MemRef& rhs);

MemRef ring_not(const MemRef& x);
void ring_not_(MemRef& x);

MemRef ring_and(const MemRef& x, const MemRef& y);
void ring_and_(MemRef& x, const MemRef& y);
DEF_RVALUE_BINARY_RING_OP(ring_and, true);

MemRef ring_xor(const MemRef& x, const MemRef& y);
void ring_xor_(MemRef& x, const MemRef& y);
DEF_RVALUE_BINARY_RING_OP(ring_xor, true);

void ring_equal(MemRef& ret, const MemRef& x, const MemRef& y);

MemRef ring_arshift(const MemRef& x, const Sizes& bits);
void ring_arshift_(MemRef& x, const Sizes& bits);

MemRef ring_rshift(const MemRef& x, const Sizes& bits);
void ring_rshift_(MemRef& x, const Sizes& bits);

MemRef ring_lshift(const MemRef& x, const Sizes& bits);
void ring_lshift_(MemRef& x, const Sizes& bits);

MemRef ring_bitrev(const MemRef& x, size_t start, size_t end);
void ring_bitrev_(MemRef& x, size_t start, size_t end);

MemRef ring_sum(absl::Span<MemRef const> arrs);

bool ring_all_equal(const MemRef& x, const MemRef& y, size_t abs_err = 0);

bool ring_all_equal(const MemRef& x, int64_t val);

// Note: here we use uint8_t instead of bool because most of the time the casted
// boolean will participate in arithmetic computation in the future.
std::vector<uint8_t> ring_cast_boolean(const MemRef& x);

// x & bits[low, high)
MemRef ring_bitmask(const MemRef& x, size_t low, size_t high);
void ring_bitmask_(MemRef& x, size_t low, size_t high);

MemRef ring_select(const std::vector<uint8_t>& c, const MemRef& x,
                   const MemRef& y);

MemRef ring_select(const MemRef& c, const MemRef& x, const MemRef& y);

// random additive splits.
std::vector<MemRef> ring_rand_additive_splits(const MemRef& arr,
                                              size_t num_splits);
// random boolean splits.
std::vector<MemRef> ring_rand_boolean_splits(const MemRef& arr,
                                             size_t num_splits);

template <typename T>
void ring_set_value(MemRef& in, const T& value) {
  MemRefView<T> _in(in);
  pforeach(0, in.numel(), [&](int64_t idx) { _in[idx] = value; });
};

#undef DEF_RVALUE_BINARY_RING_OP

}  // namespace spu::mpc
