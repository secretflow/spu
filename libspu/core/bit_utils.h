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

#include <array>

#include "absl/numeric/bits.h"
#include "yacl/base/int128.h"

#include "libspu/core/platform_utils.h"

namespace spu {

inline constexpr int Log2Floor(uint64_t n) {
  return (n <= 1) ? 0 : (63 - absl::countl_zero(n));
}

inline constexpr int Log2Ceil(uint64_t n) {
  return (n <= 1) ? 0 : (64 - absl::countl_zero(n - 1));
}

// TODO: move to constexpr when yacl is ready.
template <typename T>
size_t BitWidth(const T& v) {
  if constexpr (sizeof(T) == 16) {
    auto [hi, lo] = yacl::DecomposeUInt128(v);
    if (hi != 0) {
      return absl::bit_width(hi) + 64;
    } else {
      return absl::bit_width(lo);
    }
  } else {
    return absl::bit_width(v);
  }
}

namespace detail {

uint64_t BitDeintlWithPdepext(uint64_t in, int64_t stride);
uint64_t BitIntlWithPdepext(uint64_t in, int64_t stride);

inline constexpr std::array<uint128_t, 6> kBitIntlSwapMasks = {{
    yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
    yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
    yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
    yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
    yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
    yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
}};

inline constexpr std::array<uint128_t, 6> kBitIntlKeepMasks = {{
    yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
    yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
    yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
    yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
    yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
    yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
}};

}  // namespace detail

// Bit de-interleave function.
//
// The reverse bit interleave method, put the even bits at lower half, and odd
// bits at upper half.
//
//   aXbYcZdW -> abcdXYZW
//
// stride represent the log shift of the interleaved distance.
//
//   01010101 -> 00001111     stride = 0
//   00110011 -> 00001111     stride = 1
//   00001111 -> 00001111     stride = 2
//
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
T BitDeintl(T in, int64_t stride, int64_t nbits = -1) {
  if (nbits == -1) {
    nbits = sizeof(T) * 8;
  }

  // TODO:
  // 1. handle nbits
  // 2. enable this when benchmark test passed.
  // if constexpr (std::is_same_v<T, uint64_t>) {
  //   return detail::BitDeintlWithPdepext(in, stride);
  // }

  // The general log(n) algorithm
  // algorithm:
  //      0101010101010101
  // swap  ^^  ^^  ^^  ^^
  //      0011001100110011
  // swap   ^^^^    ^^^^
  //      0000111100001111
  // swap     ^^^^^^^^
  //      0000000011111111
  T r = in;
  for (int64_t level = stride; level + 1 < Log2Ceil(nbits); level++) {
    const T K = static_cast<T>(detail::kBitIntlKeepMasks[level]);
    const T M = static_cast<T>(detail::kBitIntlSwapMasks[level]);
    int S = 1 << level;

    r = (r & K) ^ ((r >> S) & M) ^ ((r & M) << S);
  }
  return r;
}

/// Bit interleave function.
//
// Interleave bits of input, so the upper bits of input are in the even
// positions and lower bits in the odd. Also called Morton Number.
//
//   abcdXYZW -> aXbYcZdW
//
// stride represent the log shift of the interleaved distance.
//
//   00001111 -> 01010101     stride = 0
//   00001111 -> 00110011     stride = 1
//   00001111 -> 00001111     stride = 2
//
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
T BitIntl(T in, int64_t stride, int64_t nbits = -1) {
  if (nbits == -1) {
    nbits = sizeof(T) * 8;
  }

  // TODO: fast path for intrinsic.
  // 1. handle nbits
  // 2. enable this when benchmark test passed.
  // if constexpr (std::is_same_v<T, uint64_t>) {
  //  return detail::BitIntlWithPdepext(in, stride);
  // }

  // The general log(n) algorithm
  // algorithm:
  //      0000000011111111
  // swap     ^^^^^^^^
  //      0000111100001111
  // swap   ^^^^    ^^^^
  //      0011001100110011
  // swap  ^^  ^^  ^^  ^^
  //      0101010101010101
  T r = in;
  for (int64_t level = Log2Ceil(nbits) - 2; level >= stride; level--) {
    const T K = static_cast<T>(detail::kBitIntlKeepMasks[level]);
    const T M = static_cast<T>(detail::kBitIntlSwapMasks[level]);
    int S = 1 << level;

    r = (r & K) ^ ((r >> S) & M) ^ ((r & M) << S);
  }
  return r;
}

}  // namespace spu
