// Copyright 2023 Ant Group Co., Ltd.
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

#include "absl/types/span.h"
#include "yacl/base/dynamic_bitset.h"
#include "yacl/base/int128.h"

#include "libspu/core/prelude.h"
#include "libspu/core/xt_helper.h"

namespace spu::mpc::cheetah {

template <typename T>
inline T makeBitsMask(size_t nbits) {
  size_t max = sizeof(T) * 8;
  if (nbits == 0) {
    nbits = max;
  }
  SPU_ENFORCE(nbits <= max);
  T mask = static_cast<T>(-1);
  if (nbits < max) {
    mask = (static_cast<T>(1) << nbits) - 1;
  }
  return mask;
}

template <typename T>
inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
inline size_t ZipArray(absl::Span<const T> inp, size_t bit_width,
                       absl::Span<T> oup) {
  size_t width = sizeof(T) * 8;
  SPU_ENFORCE(bit_width > 0 && width >= bit_width);
  size_t shft = bit_width;
  size_t pack_load = width / shft;
  size_t numel = inp.size();
  size_t packed_sze = CeilDiv(numel, pack_load);
  SPU_ENFORCE(oup.size() >= packed_sze);

  const T mask = makeBitsMask<T>(bit_width);
  for (size_t i = 0; i < numel; i += pack_load) {
    size_t this_batch = std::min(pack_load, numel - i);
    T acc{0};
    for (size_t j = 0; j < this_batch; ++j) {
      acc = (acc << shft) | (inp[i + j] & mask);
    }
    oup[i / pack_load] = acc;
  }
  return packed_sze;
}

template <typename T>
inline size_t UnzipArray(absl::Span<const T> inp, size_t bit_width,
                         absl::Span<T> oup) {
  size_t width = sizeof(T) * 8;
  SPU_ENFORCE(bit_width > 0 && bit_width <= width);

  size_t shft = bit_width;
  size_t pack_load = width / shft;
  size_t packed_sze = inp.size();
  size_t n = oup.size();
  SPU_ENFORCE(n > 0 && n <= pack_load * packed_sze);

  const T mask = makeBitsMask<T>(bit_width);
  for (size_t i = 0; i < packed_sze; ++i) {
    size_t j0 = std::min(n, i * pack_load);
    size_t j1 = std::min(n, j0 + pack_load);
    size_t this_batch = j1 - j0;
    T package = inp[i];
    // NOTE (reversed order)
    for (size_t j = 0; j < this_batch; ++j) {
      oup[j1 - 1 - j] = package & mask;
      package >>= shft;
    }
  }

  return n;
}

template <typename T>
inline size_t PackU8Array(absl::Span<const uint8_t> u8array,
                          absl::Span<T> packed) {
  constexpr size_t elsze = sizeof(T);
  const size_t nbytes = u8array.size();
  const size_t numel = CeilDiv(nbytes, elsze);

  SPU_ENFORCE(packed.size() >= numel);

  for (size_t i = 0; i < nbytes; i += elsze) {
    size_t this_batch = std::min(nbytes - i, elsze);
    T acc{0};
    for (size_t j = 0; j < this_batch; ++j) {
      acc = (acc << 8) | u8array[i + j];
    }
    packed[i / elsze] = acc;
  }

  return numel;
}

template <typename T>
inline size_t UnpackU8Array(absl::Span<const T> input,
                            absl::Span<uint8_t> u8array) {
  using UT = typename std::make_unsigned<T>::type;
  constexpr size_t elsze = sizeof(T);
  const size_t numel = input.size();
  const size_t nbytes = u8array.size();
  SPU_ENFORCE(CeilDiv(nbytes, elsze) >= numel);

  constexpr T mask = (static_cast<T>(1) << 8) - 1;
  for (size_t i = 0; i < nbytes; i += elsze) {
    size_t this_batch = std::min(nbytes - i, elsze);
    UT acc = static_cast<UT>(input[i / elsze]);
    for (size_t j = 0; j < this_batch; ++j) {
      u8array[i + this_batch - 1 - j] = acc & mask;
      acc >>= 8;
    }
  }

  return nbytes;
}

inline uint8_t BoolToU8(absl::Span<const uint8_t> bits);

inline void U8ToBool(absl::Span<uint8_t> bits, uint8_t u8);

// Add by @wenfan
inline void VecU8toBitset(absl::Span<const uint8_t> bits,
                          yacl::dynamic_bitset<uint128_t>& bitset) {
  SPU_ENFORCE(bits.size() == bitset.size());
  uint64_t bits_num = bits.size();
  // low efficiency
  for (uint64_t i = 0; i < bits_num; ++i) {
    bitset[i] = (bool)bits[i];
  }
}

inline yacl::dynamic_bitset<uint128_t> VecU8toBitset(
    absl::Span<const uint8_t> bits) {
  yacl::dynamic_bitset<uint128_t> bitset(bits.size());
  VecU8toBitset(bits, bitset);
  return bitset;
}

inline void BitsettoVecU8(const yacl::dynamic_bitset<uint128_t>& bitset,
                          absl::Span<uint8_t> bits) {
  SPU_ENFORCE(bits.size() == bitset.size());
  uint64_t bits_num = bitset.size();
  // low efficiency
  for (uint64_t i = 0; i < bits_num; ++i) {
    bits[i] = bitset[i];
  }
}

inline std::vector<uint8_t> BitsettoVecU8(
    const yacl::dynamic_bitset<uint128_t>& bitset) {
  std::vector<uint8_t> bits(bitset.size());
  BitsettoVecU8(bitset, absl::MakeSpan(bits));
  return bits;
}

}  // namespace spu::mpc::cheetah
