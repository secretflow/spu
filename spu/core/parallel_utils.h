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

int getNumberOfProc();

inline int64_t computeTaskSize(int64_t numel) {
  auto grain_size = static_cast<int64_t>(
      std::ceil(static_cast<float>(numel) / getNumberOfProc()));
  return std::max(grain_size, kMinTaskSize);
}

template <class F>
inline void pfor(const int64_t begin, const int64_t end, F&& f) {
  const int64_t grain_size = computeTaskSize(end - begin);
  return yacl::parallel_for(begin, end, grain_size, f);
}

template <class F>
inline void pforeach(const int64_t begin, const int64_t end, F&& fn) {
  const int64_t grain_size = computeTaskSize(end - begin);
  yacl::parallel_for(begin, end, grain_size, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; idx++) {
      fn(idx);
    }
  });
}

#define PFOR(IDX, BEG, END, ...) \
  pforeach(BEG, END, [&](int64_t IDX) { __VA_ARGS__; });

}  // namespace spu
