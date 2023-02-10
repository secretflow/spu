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

#include "libspu/mpc/cheetah/ot/util.h"

#include <numeric>

#include "libspu/core/prelude.h"

namespace spu::mpc::cheetah {

uint8_t BoolToU8(absl::Span<const uint8_t> bits) {
  size_t len = bits.size();
  SPU_ENFORCE(len >= 1 && len <= 8);
  return std::accumulate(
      bits.data(), bits.data() + len,
      /*init*/ static_cast<uint8_t>(0),
      [](uint8_t init, uint8_t next) { return (init << 1) | (next & 1); });
}

void U8ToBool(absl::Span<uint8_t> bits, uint8_t u8) {
  size_t len = std::min(8UL, bits.size());
  SPU_ENFORCE(len >= 1);
  for (size_t i = 0; i < len; ++i) {
    bits[i] = (u8 & 1);
    u8 >>= 1;
  }
}

}  // namespace spu::mpc::cheetah
