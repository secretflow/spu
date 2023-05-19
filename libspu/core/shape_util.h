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

#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/types/span.h"

#include "libspu/core/prelude.h"

namespace spu {

// This module assumes row major

int64_t calcNumel(absl::Span<const int64_t> shape);

bool isEmpty(absl::Span<const int64_t> shape);

std::vector<int64_t> makeCompactStrides(absl::Span<const int64_t> shape);

int64_t flattenIndex(absl::Span<const int64_t> index,
                     absl::Span<const int64_t> shape);

std::vector<int64_t> unflattenIndex(int64_t index,
                                    absl::Span<const int64_t> shape);

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
bool bumpIndices(absl::Span<const T> shape, absl::Span<T> indices) {
  SPU_ENFORCE(shape.size() == indices.size());
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

std::vector<int64_t> deduceDotShape(absl::Span<const int64_t> lhs,
                                    absl::Span<const int64_t> rhs);

inline size_t calcFlattenOffset(absl::Span<const int64_t> indices,
                                absl::Span<const int64_t> shape,
                                absl::Span<const int64_t> strides) {
  if (!shape.empty() && strides.empty()) {
    return calcFlattenOffset(indices, shape, makeCompactStrides(shape));
  }

  int64_t offset = 0;
  for (int64_t idx = indices.size() - 1; idx >= 0; --idx) {
    offset += indices[idx] * strides[idx];
  }
  return offset;
}

using ShapeView = absl::Span<int64_t const>;

class Shape : public std::vector<int64_t> {
  using Base = std::vector<int64_t>;

 public:
  Shape() = default;
  Shape(const Shape &other) = default;
  Shape &operator=(const Shape &other) = default;

  Shape(std::initializer_list<int64_t> list) : Base(list) {}
  Shape(iterator begin, iterator end) : Base(begin, end) {}
  explicit Shape(size_t size, int64_t element = 0) : Base(size, element) {}
  explicit Shape(const std::vector<int64_t> &other) : Base(other) {}
  explicit Shape(absl::Span<int64_t const> other)
      : Base(other.begin(), other.end()) {}

  int64_t ndim() const { return static_cast<int64_t>(size()); }

  int64_t dim(int64_t idx) const {
    SPU_ENFORCE(idx < ndim(), "invalid idx={}, ndim={}", idx, ndim());
    return at(idx);
  }

  int64_t numel() const {
    return calcNumel(absl::MakeConstSpan(data(), size()));
  }
};

}  // namespace spu
