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

#include "absl/types/span.h"
#include "emp-tool/utils/block.h"
#include "yacl/base/int128.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::cheetah {

template <typename T>
inline emp::block ConvToBlock(T x) {
  return _mm_set_epi64x(0, static_cast<uint64_t>(x));
}

template <>
inline emp::block ConvToBlock(uint128_t x) {
  return emp::makeBlock(/*hi64*/ static_cast<uint64_t>(x >> 64),
                        /*lo64*/ static_cast<uint64_t>(x));
}

template <typename T>
inline T ConvFromBlock(const emp::block& x) {
  return static_cast<T>(_mm_extract_epi64(x, 0));
}

template <>
inline uint128_t ConvFromBlock(const emp::block& x) {
  return yacl::MakeUint128(/*hi64*/ _mm_extract_epi64(x, 1),
                           /*lo64*/ _mm_extract_epi64(x, 0));
}

}  // namespace spu::mpc::cheetah