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

#include "libspu/core/bit_utils.h"

namespace spu::detail {

uint64_t BitDeintlWithPdepext(uint64_t in, int64_t stride) {
  constexpr std::array<uint64_t, 6> kMasks = {{
      0x5555555555555555,  // 01010101
      0x3333333333333333,  // 00110011
      0x0F0F0F0F0F0F0F0F,  // 00001111
      0x00FF00FF00FF00FF,  // ...
      0x0000FFFF0000FFFF,  // ...
      0x00000000FFFFFFFF,  // ...
  }};
  if (stride >= static_cast<int64_t>(kMasks.size())) {
    return in;
  }
  const uint64_t m = kMasks[stride];
  return pext_u64(in, m) ^ (pext_u64(in, ~m) << 32);
}

uint64_t BitIntlWithPdepext(uint64_t in, int64_t stride) {
  constexpr std::array<uint64_t, 6> kMasks = {{
      0x5555555555555555,  // 01010101
      0x3333333333333333,  // 00110011
      0x0F0F0F0F0F0F0F0F,  // 00001111
      0x00FF00FF00FF00FF,  // ...
      0x0000FFFF0000FFFF,  // ...
      0x00000000FFFFFFFF,  // ...
  }};
  if (stride >= static_cast<int64_t>(kMasks.size())) {
    return in;
  }
  const uint64_t m = kMasks[stride];
  return pdep_u64(in, m) ^ pdep_u64(in >> 32, ~m);
}

}  // namespace spu::detail
