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

#include <numeric>
#include <vector>

#include "yasl/base/exception.h"

#include "spu/core/type_util.h"

namespace spu {

int64_t calcNumel(absl::Span<const int64_t> shape);

std::vector<int64_t> deduceDotShape(absl::Span<const int64_t> lhs,
                                    absl::Span<const int64_t> rhs);

std::vector<int64_t> makeCompactStrides(absl::Span<const int64_t> shape);

// This function assumes row major
int64_t flattenIndex(absl::Span<const int64_t> indices,
                     absl::Span<const int64_t> shape);

void unflattenIndex(int64_t index, absl::Span<const int64_t> shape,
                    std::vector<int64_t> &unflattened);

std::vector<int64_t> unflattenIndex(int64_t index,
                                    absl::Span<const int64_t> shape);

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
bool bumpIndices(absl::Span<const T> shape, absl::Span<T> indices) {
  YASL_ENFORCE(shape.size() == indices.size());
  for (int64_t dimno = indices.size() - 1; dimno >= 0; --dimno) {
    T limit = shape[dimno];
    if (indices[dimno] + 1 < limit) {
      indices[dimno]++;
      // Whenever an index of a dimension is increased, it means that all
      // following dimensions have maxed out, so they must go to 0.
      std::fill(indices.begin() + dimno + 1, indices.end(), 0);
      return true;
    }
  }
  return false;
}

}  // namespace spu
