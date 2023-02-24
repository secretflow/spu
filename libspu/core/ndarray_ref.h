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

#include "absl/types/span.h"
#include "spdlog/spdlog.h"
#include "yacl/base/buffer.h"

#include "libspu/core/array_ref.h"
#include "libspu/core/shape_util.h"
#include "libspu/core/type.h"

// #define ITER_DEBUG

namespace spu {

// N-dimensional array reference.
//
// About: 0-dimension processing.
// We use numpy/xtensor 0-dimension setting.
class NdArrayRef {
  std::shared_ptr<yacl::Buffer> buf_{nullptr};

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
  NdArrayRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
             std::vector<int64_t> shape, std::vector<int64_t> strides,
             int64_t offset);

  // constructor, view buf as a compact buffer with given shape.
  NdArrayRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
             absl::Span<const int64_t> shape);

  // constructor, create a new buffer of elements and ref to it.
  NdArrayRef(const Type& eltype, absl::Span<const int64_t> shape);

  // convenient constructor to accept shape/strides from xtensor.
  template <typename ShapeT, typename StridesT>
  NdArrayRef(std::shared_ptr<yacl::Buffer> buf, Type eltype, ShapeT&& shape,
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

  std::shared_ptr<yacl::Buffer> buf() const { return buf_; }

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
    auto fi = calcFlattenOffset(pos, shape_, strides_);
    return *reinterpret_cast<T*>(static_cast<std::byte*>(data()) +
                                 elsize() * fi);
  }

  template <typename T = std::byte>
  const T& at(absl::Span<int64_t const> pos) const {
    auto fi = calcFlattenOffset(pos, shape_, strides_);
    return *reinterpret_cast<const T*>(static_cast<const std::byte*>(data()) +
                                       elsize() * fi);
  }

  struct Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::byte;
    using pointer = std::byte*;
    using reference = std::byte&;

    explicit Iterator(const NdArrayRef& array, std::vector<int64_t> coord,
                      bool invalid = false)
        : coord_(std::move(coord)),
          shape_(array.shape()),
          strides_(array.strides()),
          elsize_(array.elsize()),
          invalid_(invalid) {
      if (!invalid_) {
        ptr_ = const_cast<std::byte*>(&array.at(coord_));
      }
    }

    explicit Iterator(const NdArrayRef& array, absl::Span<const int64_t> coord,
                      bool invalid = false)
        : coord_(coord.begin(), coord.end()),
          shape_(array.shape()),
          strides_(array.strides()),
          elsize_(array.elsize()),
          invalid_(invalid) {
      if (!invalid_) {
        ptr_ = const_cast<std::byte*>(&array.at(coord_));
      }
    }

    reference operator*() const { return *ptr_; }
    pointer getRawPtr() { return ptr_; }
    Iterator& operator++();
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const Iterator& other) {
      return invalid_ == other.invalid_ && coord_ == other.coord_ &&
             shape_ == other.shape_ && strides_ == other.strides_;
    }

    bool operator!=(const Iterator& other) {
      return invalid_ != other.invalid_ || coord_ != other.coord_ ||
             shape_ != other.shape_ || strides_ != other.strides_;
    }

#ifdef ITER_DEBUG
    friend std::ostream& operator<<(std::ostream& s, const Iterator& iter) {
      auto fs = fmt::format("ptr = {} coord = {} shape={} strides={}",
                            (void*)iter.ptr_, fmt::join(iter.coord_, "x"),
                            fmt::join(iter.shape_, "x"),
                            fmt::join(iter.strides_, "x"));
      s << fs;
      return s;
    }

    bool validate() const { return !invalid_; }
#endif

   private:
    std::byte* ptr_ = nullptr;  // Exclude from equality check
    std::vector<int64_t> coord_;
    const std::vector<int64_t> shape_;
    const std::vector<int64_t> strides_;
    const int64_t elsize_;
    bool invalid_ = false;
  };

  Iterator begin() {
    return Iterator(*this, std::vector<int64_t>(shape().size(), 0));
  }

  Iterator end() {
    return Iterator(*this, std::vector<int64_t>(shape().size(), 0), true);
  }

  Iterator cbegin() const {
    return Iterator(*this, std::vector<int64_t>(shape().size(), 0));
  }

  Iterator cend() const {
    return Iterator(*this, std::vector<int64_t>(shape().size(), 0), true);
  }

  void copy_slice(const NdArrayRef& src, absl::Span<const int64_t> src_base,
                  absl::Span<const int64_t> dst_base, int64_t num_copy);

  /// the broadcast function
  /// Guarantee no copy
  NdArrayRef broadcast_to(absl::Span<const int64_t> to_shape,
                          absl::Span<const int64_t> in_dims) const;

  /// the reshape function
  /// No copy if can achieve through strides tricks
  NdArrayRef reshape(absl::Span<const int64_t> to_shape) const;

  /// the slice function
  /// Guarantee no copy
  NdArrayRef slice(absl::Span<const int64_t> start_indices,
                   absl::Span<const int64_t> end_indices,
                   absl::Span<const int64_t> slice_strides) const;

  /// the transpose function
  /// Guarantee no copy
  NdArrayRef transpose(absl::Span<const int64_t> permutation) const;

  /// the reverse function
  /// Guarantee no copy
  NdArrayRef reverse(absl::Span<const int64_t> dimensions) const;

  /// Expand a scalar into to_shape.
  /// Compare with broadcast, expand actually reallocates and assign memory
  NdArrayRef expand(absl::Span<const int64_t> to_shape) const;

  /// the concatenate function
  /// Always results a new NdArrayRef
  NdArrayRef concatenate(absl::Span<const NdArrayRef> others,
                         const size_t& axis) const;

  /// the pad function
  /// Always results a new NdArrayRef
  NdArrayRef pad(const NdArrayRef& padding_value,
                 absl::Span<const int64_t> edge_padding_low,
                 absl::Span<const int64_t> edge_padding_high,
                 absl::Span<const int64_t> interior_padding) const;

  /// Linear gather function
  /// Always results a new NdArrayRef
  NdArrayRef linear_gather(absl::Span<const int64_t> indices) const;

  /// Scatter new values into indices
  /// Update happens in-place
  /// must be 1D array and indicies
  NdArrayRef& linear_scatter(const NdArrayRef& new_values,
                             absl::Span<const int64_t> indices);
};

// Unflatten a 1d-array to an ndarray.
NdArrayRef unflatten(const ArrayRef& arr, absl::Span<const int64_t> shape);

// Flatten an nd-array to a 1d-array.
//
// note: it's not always to share the same underline buffer, sometimes a compact
// copy is required.
ArrayRef flatten(const NdArrayRef& ndarr);

std::ostream& operator<<(std::ostream& out, const NdArrayRef& v);

}  // namespace spu
