// Copyright 2024 Ant Group Co., Ltd.
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

#include "yacl/base/int128.h"

#include "libspu/core/type_util.h"

#define EIGEN_HAS_OPENMP

#include "Eigen/Core"

namespace spu::mpc {

inline uint8_t mul(uint8_t x, uint8_t y, uint8_t* z) {
  uint16_t hi = static_cast<uint16_t>(x) * static_cast<uint16_t>(y);
  auto lo = static_cast<uint8_t>(hi);
  if (z != nullptr) {
    *z = static_cast<uint8_t>(hi >> 8);
  }
  return lo;
}

inline uint32_t mul(uint32_t x, uint32_t y, uint32_t* z) {
  uint64_t hi = static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
  auto lo = static_cast<uint32_t>(hi);
  if (z != nullptr) {
    *z = static_cast<uint32_t>(hi >> 32);
  }
  return lo;
}

inline uint64_t mul(uint64_t x, uint64_t y, uint64_t* z) {
  uint128_t hi = static_cast<uint128_t>(x) * static_cast<uint128_t>(y);
  auto lo = static_cast<uint64_t>(hi);
  if (z != nullptr) {
    *z = static_cast<uint64_t>(hi >> 64);
  }
  return lo;
}

inline uint128_t mul(uint128_t x, uint128_t y, uint128_t* z) {
  uint64_t x_lo = x & 0xFFFFFFFFFFFFFFFF;
  uint64_t x_hi = x >> 64;
  uint64_t y_lo = y & 0xFFFFFFFFFFFFFFFF;
  uint64_t y_hi = y >> 64;

  uint128_t lo = static_cast<uint128_t>(x_lo) * y_lo;

  uint128_t xl_yh = static_cast<uint128_t>(x_lo) * y_hi;
  uint128_t xh_yl = static_cast<uint128_t>(x_hi) * y_lo;

  lo += xl_yh << 64;
  uint128_t hi = static_cast<uint128_t>(lo < (xl_yh << 64));

  lo += xh_yl << 64;
  hi += static_cast<uint128_t>(lo < (xh_yl << 64));
  hi += static_cast<uint128_t>(x_hi) * y_hi;

  hi += xl_yh >> 64;
  hi += xh_yl >> 64;
  if (z != nullptr) {
    *z = hi;
  }
  return lo;
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T mul_mod(T x, T y) {
  T c = 0;
  T e = mul(x, y, &c);
  T p = ScalarTypeToPrime<T>::prime;
  size_t mp_exp = ScalarTypeToPrime<T>::exp;
  T ret = (e & p) + ((e >> mp_exp) ^ (c << (sizeof(T) * 8 - mp_exp)));
  return (ret >= p) ? ret - p : ret;
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T add_mod(T x, T y) {
  T ret = x + y;
  T p = ScalarTypeToPrime<T>::prime;
  return (ret >= p) ? ret - p : ret;
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T add_inv(T x) {
  T p = ScalarTypeToPrime<T>::prime;
  return x ^ p;
}

// Extended Euclidean Algorithm
// ax + by =  gcd(a, b)
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
void extend_gcd(T a, T b, T& x, T& y) {
  if (b == 0) {
    x = 1;
    y = 0;
    return;
  }
  extend_gcd(b, static_cast<T>(a % b), y, x);
  T tmp = mul_mod(static_cast<T>(a / b), x);
  y = add_mod(y, add_inv(tmp));
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T mul_inv(T in) {
  T x;
  T y;
  T p = ScalarTypeToPrime<T>::prime;
  extend_gcd(p, in, x, y);
  return y;
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T mod_p(T in) {
  T p = ScalarTypeToPrime<T>::prime;
  size_t mp_exp = ScalarTypeToPrime<T>::exp;
  T i = (in & p) + (in >> mp_exp);
  return i >= p ? i - p : i;
}

// the following code references SEAL library
// https://github.com/microsoft/SEAL/blob/main/src/seal/util/uintarithsmallmod.cpp
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T exp_mod(T operand, T exponent) {
  // Fast cases
  if (exponent == 0) {
    // Result is supposed to be only one digit
    return 1;
  }

  if (exponent == 1) {
    return operand;
  }

  // Perform binary exponentiation.
  T power = operand;
  T product = 0;
  T intermediate = 1;

  // Initially: power = operand and intermediate = 1, product is irrelevant.
  while (true) {
    if (exponent & 1) {
      product = mul_mod(power, intermediate);
      std::swap(product, intermediate);
    }
    exponent >>= 1;
    if (exponent == 0) {
      break;
    }
    product = mul_mod(power, power);
    std::swap(product, power);
  }
  return intermediate;
}
}  // namespace spu::mpc