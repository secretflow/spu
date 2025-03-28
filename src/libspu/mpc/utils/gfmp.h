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
inline T mod_p(T in) {
  T p = ScalarTypeToPrime<T>::prime;
  size_t mp_exp = ScalarTypeToPrime<T>::exp;
  T i = (in & p) + (in >> mp_exp);
  return i >= p ? i - p : i;
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
  if (x == 0) {
    return 0;
  }
  T p = ScalarTypeToPrime<T>::prime;
  return x ^ p;
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T square_mod(T x) {
  return mul_mod(x, x);
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T pow_mod(T x, T exp) {
  T res = 1;
  while (exp) {
    if (exp & 1) {
      res *= x;
    }
    exp >>= 1;
    x = square_mod(x);
  }
  return mod_p(res);
}

// Define the sqrt(x) be the unique element in {1, ..., (p-1)/2}
// For p = 3 (mod 4), the square roots of x mod p are (-)x^{(p+1)/4}
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline T sqrt_mod(T x) {
  constexpr T prime = ScalarTypeToPrime<T>::prime;
  constexpr T MID_PR = ((prime - 1) >> 1) + 1;
  constexpr T sqrt_exp = (prime + 1) >> 2;
  T res = pow_mod(x, sqrt_exp);
  if (res >= 1 && res < MID_PR) {
    return res;
  } else {
    return add_inv(res);
  }
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

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline std::vector<T> lagrange_interpolation(const std::vector<T>& x,
                                             const std::vector<T>& y) {
  SPU_ENFORCE_EQ(x.size(), y.size());
  SPU_ENFORCE(!x.empty());

  std::vector<T> coeff(x.size(), 0);
  // Todo: optimize me
  for (size_t i = 0; i < x.size(); i++) {
    std::vector<T> tmp_coeff(x.size(), 0);
    tmp_coeff[0] = y[i];
    T prod = 1;
    for (size_t j = 0; j < x.size(); j++) {
      if (j != i) {
        prod = mul_mod(prod, add_mod(x[i], add_inv(x[j])));
        T precedent = 0;
        for (auto res_iter = tmp_coeff.begin(); res_iter < tmp_coeff.end();
             res_iter++) {
          T new_res = add_mod(mul_mod(*res_iter, add_inv(x[j])), precedent);
          precedent = *res_iter;
          *res_iter = new_res;
        }
      }
    }
    std::transform(coeff.begin(), coeff.end(), tmp_coeff.begin(), coeff.begin(),
                   [prod](T old, T add) {
                     return add_mod(old, mul_mod(add, mul_inv(prod)));
                   });
  }

  return coeff;
}

template <typename T>
class Gfmp {
 private:
  T data_{0};

 public:
  explicit Gfmp(T data) : data_(mod_p(data)) {}
  Gfmp(const Gfmp& other) = default;
  Gfmp() = default;
  ~Gfmp() = default;

  T data() const { return data_; }

  Gfmp operator+(const Gfmp& other) const {
    return Gfmp(add_mod(data_, other.data()));
  }

  Gfmp& operator+=(const Gfmp& other) {
    data_ = add_mod(data_, other.data());
    return *this;
  }

  Gfmp operator-(const Gfmp& other) const {
    return Gfmp(add_mod(data_, add_inv(other.data())));
  }

  Gfmp& operator-=(const Gfmp& other) {
    data_ = add_mod(data_, add_inv(other.data()));
    return *this;
  }

  Gfmp operator*(const Gfmp& other) const {
    return Gfmp(mul_mod(data_, other.data()));
  }

  Gfmp& operator*=(const Gfmp& other) {
    data_ = mul_mod(data_, other.data());
    return *this;
  }

  Gfmp operator/(const Gfmp& other) const {
    return Gfmp(mul_mod(data_, mul_inv(other.data())));
  }

  Gfmp& operator/=(const Gfmp& other) {
    data_ = mul_mod(data_, mul_inv(other.data()));
    return *this;
  }

  bool operator==(const Gfmp& other) const { return data_ == other.data(); }
  bool operator!=(const Gfmp& other) const { return data_ != other.data(); }

  friend std::ostream& operator<<(std::ostream& os, const Gfmp& value) {
    os << value.data_;
    return os;
  }
};

template <typename T>
using GfmpMatrix =
    Eigen::Matrix<Gfmp<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
GfmpMatrix<T> GenVandermondeMatrix(size_t rows, size_t cols) {
  GfmpMatrix<T> vander(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    Gfmp<T> prod(1);
    Gfmp<T> x(i + 1);
    for (size_t j = 0; j < cols; ++j) {
      prod = prod * x;
      vander(i, j) = prod;
    }
  }
  return vander;
}

template <typename T>
std::vector<T> GenReconstructVector(size_t n_shares) {
  std::vector<T> recon(n_shares);
  for (size_t i = 0; i < n_shares; ++i) {
    T prod = 1;
    for (size_t j = 0; j < n_shares; ++j) {
      if (i != j) {
        T xi = i + 1;
        T xj = j + 1;
        auto tmp = mul_mod(xj, mul_inv(add_mod(xj, add_inv(xi))));
        prod = mul_mod(prod, tmp);
      }
    }
    recon[i] = prod;
  }
  return recon;
}
}  // namespace spu::mpc