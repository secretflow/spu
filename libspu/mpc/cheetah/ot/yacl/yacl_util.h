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
#include "yacl/base/aligned_vector.h"
#include "yacl/base/buffer.h"
#include "yacl/base/dynamic_bitset.h"
#include "yacl/base/int128.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::cheetah {

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

absl::Span<uint128_t> inline MakeSpan_Uint128(yacl::Buffer& buf) {
  return absl::MakeSpan(buf.data<uint128_t>(), buf.size() / sizeof(uint128_t));
}

}  // namespace spu::mpc::cheetah
