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

#include "absl/numeric/bits.h"
#include "yacl/base/int128.h"

namespace spu {

inline constexpr int Log2Floor(uint64_t n) {
  return (n <= 1) ? 0 : (63 - absl::countl_zero(n));
}

inline constexpr int Log2Ceil(uint64_t n) {
  return (n <= 1) ? 0 : (64 - absl::countl_zero(n - 1));
}

// TODO: move to constexpr when yacl is ready.
// TODO: test me.
template <typename T>
size_t BitWidth(const T& v) {
  if constexpr (sizeof(T) == 16) {
    auto [hi, lo] = yacl::DecomposeUInt128(v);
    size_t hi_width = absl::bit_width(hi);
    if (hi_width != 0) {
      return hi_width + 64;
    } else {
      return absl::bit_width(lo);
    }
  } else {
    return absl::bit_width(v);
  }
}

}  // namespace spu
