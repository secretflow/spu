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

#include <memory>
#include <vector>

#include "yasl/base/buffer.h"

#include "spu/core/array_ref.h"
#include "spu/core/type.h"

namespace spu {
namespace detail {

// This function assumes row major
size_t calcFlattenOffset(absl::Span<const int64_t> indices,
                         absl::Span<const int64_t> shape,
                         absl::Span<const int64_t> strides = {});

}  // namespace detail

// N-dimensional array reference.
//
// About: 0-dimension processing.
// We use numpy/xtensor 0-dimension setting.
class NdArrayRef {
  std::shared_ptr<yasl::Buffer> buf_{nullptr};

  Type eltype_{};

  // the shape.
  std::vector<int64_t> shape_{};

  // the strides, in number of elements.
  std::vector<int64_t> strides_{};

  // start offset from the mem buffer.
  int64_t offset_{0};

 public:
  NdArrayRef() = default;

  // full constructor
  NdArrayRef(std::shared_ptr<yasl::Buffer> buf, Type eltype,
             std::vector<int64_t> shape, std::vector<int64_t> strides,
             int64_t offset);

  // constructor, view buf as a compact buffer with given shape.
  NdArrayRef(std::shared_ptr<yasl::Buffer> buf, Type eltype,
             std::vector<int64_t> shape);

  // constructor, create a new buffer of elements and ref to it.
  NdArrayRef(Type eltype, std::vector<int64_t> shape);

  // convenient constructor to accept shape/strides from xtensor.
  template <typename ShapeT, typename StridesT>
  NdArrayRef(std::shared_ptr<yasl::Buffer> buf, Type eltype, ShapeT&& shape,
             StridesT&& strides, int64_t offset)
      : NdArrayRef(std::move(buf), std::move(eltype),
                   {shape.begin(), shape.end()},
                   {strides.begin(), strides.end()}, offset) {}

  // copy and move constructable, using referential semantic.
  NdArrayRef(const NdArrayRef& other) = default;
  NdArrayRef(NdArrayRef&& other) = default;
  NdArrayRef& operator=(const NdArrayRef& other) = default;
  NdArrayRef& operator=(NdArrayRef&& other) = default;

  // Returns the number of dimension.
  size_t ndim() const { return shape_.size(); }

  // Return the size of the given dimension.
  size_t dim(size_t idx) const;

  // Return total number of elements.
  int64_t numel() const;

  // Return the element type.
  const Type& eltype() const { return eltype_; }

  // Return the element size.
  size_t elsize() const { return eltype_.size(); }

  std::vector<int64_t> const& strides() const { return strides_; }

  std::vector<int64_t> const& shape() const { return shape_; }

  int64_t offset() const { return offset_; }

  std::shared_ptr<yasl::Buffer> buf() const { return buf_; }

  bool isCompact() const;

  // create a compact clone.
  NdArrayRef clone() const;

  // View this array ref as another type.
  // @param force, true if ignore the type check, else only the same elsize type
  //               could be casted.
  NdArrayRef as(const Type& new_ty, bool force = false) const;

  // Get data pointer
  void* data() { return buf_->data<std::byte>() + offset_; }
  const void* data() const { return buf_->data<std::byte>() + offset_; }

  // Get element.
  template <typename T = std::byte>
  T& at(absl::Span<int64_t const> pos) {
    auto fi = detail::calcFlattenOffset(pos, shape_, strides_);
    return *reinterpret_cast<T*>(static_cast<std::byte*>(data()) +
                                 elsize() * fi);
  }

  template <typename T = std::byte>
  const T& at(absl::Span<int64_t const> pos) const {
    auto fi = detail::calcFlattenOffset(pos, shape_, strides_);
    return *reinterpret_cast<const T*>(static_cast<const std::byte*>(data()) +
                                       elsize() * fi);
  }
};

// Unflatten a 1d-array to an ndarray.
NdArrayRef unflatten(const ArrayRef& arr, std::vector<int64_t> shape);

// Flatten an nd-array to a 1d-array.
//
// note: it's not always to share the same underline buffer, sometimes a compact
// copy is required.
ArrayRef flatten(const NdArrayRef& ndarr);

std::ostream& operator<<(std::ostream& out, const NdArrayRef& v);

}  // namespace spu
