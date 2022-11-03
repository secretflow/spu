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

namespace spu::mpc {

inline constexpr int log2Floor(uint64_t n) {
  return (n <= 1) ? 0 : (63 - absl::countl_zero(n));
}

inline constexpr int log2Ceil(uint64_t n) {
  return (n <= 1) ? 0 : (64 - absl::countl_zero(n - 1));
}

}  // namespace spu::mpc
