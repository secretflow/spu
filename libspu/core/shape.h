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

#include <array>
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"

#include "libspu/core/prelude.h"

namespace spu {

class Shape : public std::vector<int64_t> {
 private:
  using Base = std::vector<int64_t>;

 public:
  using Base::Base;

  /*explicit*/ Shape(llvm::ArrayRef<int64_t> arr)
      : Base(arr.begin(), arr.end()) {}

  Shape(std::initializer_list<int64_t> list) : Base(list) {}

  template <size_t SZ>
  Shape(std::array<int64_t, SZ> arr) : Base(arr.begin(), arr.end()) {}

  int64_t ndim() const { return static_cast<int64_t>(size()); }

  int64_t dim(int64_t idx) const {
    SPU_ENFORCE(idx < ndim(), "invalid idx={}, ndim={}", idx, ndim());
    return at(idx);
  }

  int64_t numel() const {
    return std::accumulate(begin(), end(), static_cast<int64_t>(1),
                           std::multiplies<>());
  }

  bool isScalar() const { return empty(); }

  bool isTensor() const { return !isScalar(); }

  friend std::ostream &operator<<(std::ostream &out, const Shape &s) {
    out << fmt::format("{}", fmt::join(s, "x"));
    return out;
  }

 private:
  // Hide parent's empty method
  bool empty() const { return Base::empty(); }
};

inline auto format_as(const Shape &s) { return fmt::streamed(s); }

class Index : public std::vector<int64_t> {
 private:
  using Base = std::vector<int64_t>;

 public:
  using Base::Base;

  /*explicit*/ Index(llvm::ArrayRef<int64_t> arr)
      : Base(arr.begin(), arr.end()) {}

  /// Checks if an element `e` at kth axis of `this` object follows
  /// `0 <= e <= bounds[k]`.
  bool inBounds(const Shape &bounds) const;

  friend std::ostream &operator<<(std::ostream &out, const Index &s) {
    out << fmt::format("{}", fmt::join(s, "x"));
    return out;
  }
};

inline auto format_as(const Index &idx) { return fmt::streamed(idx); }

using Stride = int64_t;

class Strides : public std::vector<Stride> {
 private:
  using Base = std::vector<Stride>;

 public:
  using Base::Base;

  /*explicit*/ Strides(llvm::ArrayRef<Stride> arr)
      : Base(arr.begin(), arr.end()) {}

  Strides(std::initializer_list<int64_t> list) : Base(list) {}

  friend std::ostream &operator<<(std::ostream &out, const Strides &s) {
    out << fmt::format("{}", fmt::join(s, "x"));
    return out;
  }
};

inline auto format_as(const Strides &s) { return fmt::streamed(s); }

class Sizes : public std::vector<int64_t> {
 private:
  using Base = std::vector<int64_t>;

 public:
  using Base::Base;

  /*explicit*/ Sizes(llvm::ArrayRef<Stride> arr)
      : Base(arr.begin(), arr.end()) {}

  friend std::ostream &operator<<(std::ostream &out, const Sizes &s) {
    out << fmt::format("{}", fmt::join(s, "x"));
    return out;
  }
};

inline auto format_as(const Sizes &s) { return fmt::streamed(s); }

class Axes : public std::vector<int64_t> {
 private:
  using Base = std::vector<int64_t>;

 public:
  using Base::Base;

  /*explicit*/ Axes(llvm::ArrayRef<int64_t> arr)
      : Base(arr.begin(), arr.end()) {}

  friend std::ostream &operator<<(std::ostream &out, const Axes &s) {
    out << fmt::format("{}", fmt::join(s, "x"));
    return out;
  }
};

inline auto format_as(const Axes &axes) { return fmt::streamed(axes); }

Strides makeCompactStrides(const Shape &shape);

int64_t flattenIndex(const Index &index, const Shape &shape);

Index unflattenIndex(int64_t index, const Shape &shape);

inline bool bumpIndices(absl::Span<const int64_t> shape,
                        absl::Span<int64_t> indices) {
  SPU_ENFORCE(shape.size() == indices.size());
  for (int64_t dimno = indices.size() - 1; dimno >= 0; --dimno) {
    int64_t limit = shape[dimno];
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

inline size_t calcFlattenOffset(const Index &indices, const Shape &shape,
                                const Strides &strides) {
  if (shape.ndim() != 0 && strides.empty()) {
    return calcFlattenOffset(indices, shape, makeCompactStrides(shape));
  }

  int64_t offset = 0;
  for (int64_t idx = indices.size() - 1; idx >= 0; --idx) {
    offset += indices[idx] * strides[idx];
  }
  return offset;
}

}  // namespace spu
