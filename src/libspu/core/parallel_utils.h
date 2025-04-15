// Copyright 2022 Ant Group Co., Ltd.
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

#include <algorithm>
#include <cmath>

#include "yacl/utils/parallel.h"

namespace spu {

constexpr int64_t kMinTaskSize = 50000;

template <class F>
inline auto pforeach(int64_t begin, int64_t end, F&& f) -> std::enable_if_t<
    std::is_same_v<decltype(f(int64_t(), int64_t())), void>> {
  return yacl::parallel_for(begin, end, kMinTaskSize, f);
}

template <class F>
inline auto pforeach(int64_t begin, int64_t end, F&& f)
    -> std::enable_if_t<std::is_same_v<decltype(f(int64_t())), void>> {
  return yacl::parallel_for(begin, end, kMinTaskSize,
                            [&f](int64_t begin, int64_t end) {
                              for (int64_t idx = begin; idx < end; ++idx) {
                                f(idx);
                              }
                            });
}

}  // namespace spu
