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

#include <functional>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"
#include "yacl/base/buffer.h"

#include "libspu/core/bit_utils.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/shape.h"
#include "libspu/core/type.h"
#include "libspu/core/vectorize.h"

namespace spu {

struct ValueProto {
  ValueMetaProto meta;
  std::vector<ValueChunkProto> chunks;
};

// N-dimensional array reference.
//
// About: 0-dimension processing.
// We use numpy/xMemRef 0-dimension setting.
class MemRef {
  std::shared_ptr<yacl::Buffer> buf_{nullptr};

  Type eltype_;

  // the shape.
  Shape shape_{};

  // the strides, in number of elements.
  Strides strides_{};

  // start offset from the mem buffer.
  int64_t offset_{0};

  // Indicate if storing complex numbers
  bool is_complex_{false};

  // Indicate this buffer can be indexing in a linear way
  bool use_fast_indexing_{false};
  Stride fast_indexing_stride_{0};

 public:
  MemRef() = default;

  // full constructor
  explicit MemRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
                  const Shape& shape, const Strides& strides, int64_t offset,
                  bool is_complex = false);

  // constructor, view buf as a compact buffer with given shape.
  explicit MemRef(std::shared_ptr<yacl::Buffer> buf, Type eltype,
                  const Shape& shape, bool is_complex = false);

  // constructor, create a new buffer of elements and ref to it.
  explicit MemRef(const Type& eltype, const Shape& shape,
                  bool is_complex = false);

  // copy and move constructable, using referential semantic.
  MemRef(const MemRef& other) = default;
  MemRef(MemRef&& other) = default;
  MemRef& operator=(const MemRef& other) = default;

#ifndef NDEBUG
  // GCC 11.4 with -O1 is not happy with default assign operator when using
  // std::reverse...
  MemRef& operator=(MemRef&& other) noexcept {
    if (this != &other) {
      std::swap(this->buf_, other.buf_);
      std::swap(this->eltype_, other.eltype_);
      std::swap(this->shape_, other.shape_);
      std::swap(this->strides_, other.strides_);
      std::swap(this->offset_, other.offset_);
      std::swap(this->use_fast_indexing_, other.use_fast_indexing_);
      std::swap(this->fast_indexing_stride_, other.fast_indexing_stride_);
    }
    return *this;
  }
#else
  MemRef& operator=(MemRef&& other) = default;
#endif

  bool operator==(const MemRef& other) const {
    return shape_ == other.shape_ && strides_ == other.strides_ &&
           offset_ == other.offset_ && buf_ == other.buf_;
  }

  bool operator!=(const MemRef& other) const { return !(*this == other); }

  bool isUninitialized() const { return buf_ == nullptr; }

  // Returns the number of dimension.
  size_t ndim() const { return shape_.size(); }

  // Return the size of the given dimension.
  size_t dim(size_t idx) const { return shape_.at(idx); }

  // Return total number of elements.
  int64_t numel() const { return shape_.numel(); }

  // Return the element type.
  const Type& eltype() const { return eltype_; }
  Type& eltype() { return eltype_; }

  // Return the element size.
  size_t elsize() const {
    return is_complex_ ? 2 * eltype_.size() : eltype_.size();
  }

  Strides const& strides() const { return strides_; }

  Shape const& shape() const { return shape_; }

  int64_t offset() const { return offset_; }

  std::shared_ptr<yacl::Buffer> buf() const { return buf_; }

  // create a compact clone.
  MemRef clone() const;

  Visibility vtype() const;
  bool isPublic() const { return vtype() == VIS_PUBLIC; }
  bool isSecret() const { return vtype() == VIS_SECRET; }
  bool isPrivate() const { return vtype() == VIS_PRIVATE; }

  bool isComplex() const { return is_complex_; }

  bool isCompact() const {
    if (numel() < 2) {
      return true;
    }
    return strides_ == makeCompactStrides(shape_);
  }

  bool isSplat() const {
    return std::all_of(strides_.begin(), strides_.end(),
                       [](Stride s) { return s == 0; });
  }

  // Test only
  bool canUseFastIndexing() const { return use_fast_indexing_; }
  const Stride& fastIndexingStride() const { return fast_indexing_stride_; }

  // View this array ref as another type.
  // @param force, true if ignore the type check, else only the same elsize type
  //               could be casted.
  MemRef as(const Type& new_ty, bool force = false) const;

  // Get element.
  template <typename T = std::byte>
  T& at(const Index& pos) {
    auto fi = calcFlattenOffset(pos, shape_, strides_);
    return *reinterpret_cast<T*>(static_cast<std::byte*>(data()) +
                                 (elsize() * fi));
  }

  template <typename T = std::byte>
  const T& at(const Index& pos) const {
    auto fi = calcFlattenOffset(pos, shape_, strides_);
    return *reinterpret_cast<const T*>(static_cast<const std::byte*>(data()) +
                                       (elsize() * fi));
  }

  // Get data pointer
  template <typename T = void>
  T* data() {
    return reinterpret_cast<T*>(buf_->data<std::byte>() + offset_);
  }

  template <typename T = void>
  const T* data() const {
    return reinterpret_cast<const T*>(buf_->data<std::byte>() + offset_);
  }

  template <typename T = std::byte>
  T& at(int64_t pos) {
    if (use_fast_indexing_) {
      return *reinterpret_cast<T*>(static_cast<std::byte*>(data()) +
                                   (elsize() * pos * fast_indexing_stride_));
    } else {
      return at<T>(unflattenIndex(pos, shape_));
    }
  }

  template <typename T = std::byte>
  const T& at(int64_t pos) const {
    if (use_fast_indexing_) {
      return *reinterpret_cast<const T*>(
          static_cast<const std::byte*>(data()) +
          (elsize() * pos * fast_indexing_stride_));
    } else {
      return at<T>(unflattenIndex(pos, shape_));
    }
  }

  // Serialize to protobuf.
  ValueProto toProto(size_t max_chunk_size) const;
  size_t chunksCount(size_t max_chunk_size) const;
  ValueMetaProto toMetaProto() const;

  // Deserialize from protobuf.
  static MemRef fromProto(const ValueProto& proto);

  struct Iterator {
   public:
    explicit Iterator(const MemRef& array)
        : shape_(array.shape_),
          strides_(array.strides_),
          elsize_(array.elsize()) {
      if (array.use_fast_indexing_) {
        auto flat_array = array.reshape({array.numel()});
#ifndef NDEBUG
        SPU_ENFORCE(array.data() == flat_array.data());
#endif
        shape_ = flat_array.shape_;
        strides_ = flat_array.strides_;
      }
    }

    explicit Iterator(const MemRef& array, const Index& index)
        : shape_(array.shape_),
          strides_(array.strides_),
          elsize_(array.elsize()),
          index_(index) {
      if (array.use_fast_indexing_) {
        auto flat_array = array.reshape({array.numel()});
#ifndef NDEBUG
        SPU_ENFORCE(array.data() == flat_array.data());
#endif
        shape_ = flat_array.shape_;
        strides_ = flat_array.strides_;
      }
      if (!index.inBounds(array.shape())) {
        index_.reset();
      } else {
        ptr_ = const_cast<std::byte*>(&array.at(*index_));
        if (array.use_fast_indexing_) {
          index_ = {flattenIndex(*index_, array.shape_)};
        }
      }
    }

    explicit Iterator(const MemRef& array, int64_t flat_idx)
        : Iterator(array, unflattenIndex(flat_idx, array.shape_)) {}

    std::byte& operator*() const {
#ifndef NDEBUG
      SPU_ENFORCE(index_, "Dereference an iterator that passes end.");
#endif
      return *ptr_;
    }
    std::byte* operator->() {
#ifndef NDEBUG
      SPU_ENFORCE(index_, "Dereference an iterator that passes end.");
#endif
      return ptr_;
    }

    Iterator& operator++();
    Iterator operator++(int);

    bool operator==(const Iterator& other) const {
      return this->shape_ == other.shape_ && this->strides_ == other.strides_ &&
             this->index_ == other.index_ && this->elsize_ == other.elsize_;
    }

    template <typename T>
    T& getScalarValue() const {
      return *reinterpret_cast<T*>(ptr_);
    }

    void* getRawPtr() const { return ptr_; }

    bool operator!=(const Iterator& other) const { return !(*this == other); }

#ifndef NDEBUG
    friend std::ostream& operator<<(std::ostream& s, const Iterator& iter) {
      if (iter.index_.has_value()) {
        s << fmt::format("index = {},", *iter.index_);
      } else {
        s << fmt::format("invalid index");
      }
      return s;
    }
    bool validate() const { return index_.has_value(); }
#endif
   private:
    Shape shape_;
    Strides strides_;
    size_t elsize_;
    // When optional has no value, this is an invalid iterator
    std::optional<Index> index_;
    std::byte* ptr_{nullptr};
  };

  Iterator begin() { return Iterator(*this, Index(this->shape_.size(), 0)); }
  Iterator end() { return Iterator(*this); }
  Iterator cbegin() const {
    return Iterator(*this, Index(this->shape_.size(), 0));
  }
  Iterator cend() const { return Iterator(*this); }

  void copy_slice(const MemRef& src, const Index& src_base,
                  const Index& dst_base, int64_t num_copy);

// The following APIs are not safe to directly use by downstream components.
// SPU_BUILD is only defined when building SPU itself.
#ifdef SPU_BUILD
 public:
#else
 private:
#endif
  /// the broadcast function
  /// Guarantee no copy
  MemRef broadcast_to(const Shape& to_shape, const Axes& in_dims) const;

  /// the reshape function
  /// No copy if can achieve through strides tricks
  MemRef reshape(const Shape& to_shape) const;

  /// the slice function
  /// Guarantee no copy
  MemRef slice(const Index& offsets, const Shape& sizes,
               const Strides& strides) const;

  /// This is a special slice for single element at indices
  MemRef slice_scalar_at(const Index& indices) const;

  /// the transpose function
  /// Guarantee no copy
  MemRef transpose(const Axes& permutation) const;

  /// the transpose function with reverse order.
  /// Guarantee no copy
  MemRef transpose() const;

  /// the reverse function
  /// Guarantee no copy
  MemRef reverse(const Axes& dimensions) const;

  /// Expand a scalar into to_shape.
  /// Compare with broadcast, expand actually reallocates and assign memory
  MemRef expand(const Shape& to_shape) const;

  /// the concatenate function
  /// Always results a new MemRef
  MemRef concatenate(absl::Span<const MemRef> others, int64_t axis) const;

  /// the pad function
  /// Always results a new MemRef
  MemRef pad(const MemRef& padding_value, const Sizes& edge_padding_low,
             const Sizes& edge_padding_high) const;

  /// Linear gather function
  /// Always results a new MemRef
  MemRef linear_gather(const Index& indices) const;

  /// Scatter new values into indices
  /// Update happens in-place
  /// must be 1D array and indices
  MemRef& linear_scatter(const MemRef& new_values, const Index& indices);

  void insert_slice(const MemRef& new_value, const Index& offsets,
                    const Strides& strides);

 private:
  void eliminate_zero_stride();
};

template <>
struct SimdTrait<MemRef> {
  using PackInfo = std::vector<Shape>;

  template <typename InputIt>
  static MemRef pack(InputIt first, InputIt last, PackInfo& pi) {
    SPU_ENFORCE(first != last);

    int64_t total_numel = 0;
    const Type ty = first->eltype();
    for (auto itr = first; itr != last; ++itr) {
      SPU_ENFORCE(itr->eltype().storage_type() == ty.storage_type(),
                  "storage type mismatch {} != {}", itr->eltype(), ty);
      total_numel += itr->numel();
    }
    MemRef result(first->eltype(), {total_numel});
    int64_t offset = 0;
    for (; first != last; ++first) {
      MemRef slice(result.buf(), ty, first->shape(),
                   makeCompactStrides(first->shape()), offset);
      Index start_index(first->ndim(), 0);
      slice.copy_slice(*first, start_index, start_index, first->numel());
      pi.push_back(first->shape());
      offset += first->numel() * ty.size();
    }
    return result;
  }

  template <typename OutputIt>
  static OutputIt unpack(const MemRef& v, OutputIt result, const PackInfo& pi) {
    int64_t total_num = 0;
    for (const auto& shape : pi) {
      total_num += shape.numel();
    }

    SPU_ENFORCE(v.numel() == total_num, "split number mismatch {} != {}",
                v.numel(), total_num);

    int64_t offset = 0;
    for (const auto& shape : pi) {
      *result++ =
          MemRef(v.buf(), v.eltype(), shape, makeCompactStrides(shape), offset);
      offset += shape.numel() * v.elsize();
    }

    return result;
  }
};

MemRef makeConstantArrayRef(const Type& eltype, const Shape& shape);

std::ostream& operator<<(std::ostream& out, const MemRef& v);
std::ostream& operator<<(std::ostream& out, const std::vector<MemRef>& v);

template <typename T>
class MemRefView {
 private:
  MemRef* arr_;
  size_t elsize_;

 public:
  // Note: we explicit discard const correctness due to the complexity.
  explicit MemRefView(const MemRef& arr)
      : arr_(const_cast<MemRef*>(&arr)), elsize_(sizeof(T)) {
    if (!arr.canUseFastIndexing()) {
      // When elsize does not match, flatten/unflatten computation is dangerous
      SPU_ENFORCE(elsize_ == arr_->elsize(), "T size = {}, arr elsize = {}",
                  elsize_, arr_->elsize());
    }
  }

  int64_t numel() const { return arr_->numel(); }

  T& operator[](size_t idx) {
    if (arr_->canUseFastIndexing()) {
      return *reinterpret_cast<T*>(
          arr_->data<std::byte>() +
          (elsize_ * idx * arr_->fastIndexingStride()));
    } else {
      const auto& indices = unflattenIndex(idx, arr_->shape());
      auto fi = calcFlattenOffset(indices, arr_->shape(), arr_->strides());
      return *reinterpret_cast<T*>(arr_->data<std::byte>() + (elsize_ * fi));
    }
  }

  T const& operator[](size_t idx) const {
    if (arr_->canUseFastIndexing()) {
      return *reinterpret_cast<T*>(
          arr_->data<std::byte>() +
          (elsize_ * idx * arr_->fastIndexingStride()));
    } else {
      const auto& indices = unflattenIndex(idx, arr_->shape());
      auto fi = calcFlattenOffset(indices, arr_->shape(), arr_->strides());
      return *reinterpret_cast<T*>(arr_->data<std::byte>() + (elsize_ * fi));
    }
  }
};

template <typename T>
size_t maxBitWidth(const MemRef& in) {
  auto numel = in.numel();
  if (numel == 0) {
    return sizeof(T) * 8;
  }

  if (std::all_of(in.strides().begin(), in.strides().end(),
                  [](int64_t s) { return s == 0; })) {
    return BitWidth(in.cbegin().getScalarValue<const T>());
  }

  MemRefView<T> _in(in);

  return yacl::parallel_reduce<size_t>(
      0, numel, kMinTaskSize,
      [&](int64_t begin, int64_t end) {
        size_t partial_max = 0;
        for (int64_t idx = begin; idx < end; ++idx) {
          partial_max = std::max<size_t>(partial_max, BitWidth(_in[idx]));
        }
        return partial_max;
      },
      [](size_t a, size_t b) { return std::max(a, b); });
}

}  // namespace spu

template <>
struct std::hash<spu::MemRef> {
  std::size_t operator()(const spu::MemRef& r) const noexcept {
    return std::hash<const void*>{}(r.data());
  }
};

namespace fmt {

template <>
struct formatter<spu::MemRef> : ostream_formatter {};

template <>
struct formatter<std::vector<spu::MemRef>> : ostream_formatter {};

}  // namespace fmt
