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

#include "spu/core/type.h"
#include "spu/core/vectorize.h"

namespace spu {
namespace detail {

void strided_copy(int64_t numel, int64_t elsize, void* dst, int64_t dstride,
                  void const* src, int64_t sstride);

}

// ArrayRef is a reference type which represent an strided array of objects.
class ArrayRef {
  std::shared_ptr<yasl::Buffer> buf_{nullptr};

  // element type.
  Type eltype_{};

  // number of elements.
  int64_t numel_{0};

  // element stride, in number of elements.
  int64_t stride_{0};

  // start offset from the mem buffer, in bytes.
  int64_t offset_{0};

 public:
  ArrayRef() = default;

  // full constructor
  ArrayRef(std::shared_ptr<yasl::Buffer> buf, Type eltype, int64_t numel,
           int64_t stride, int64_t offset);

  // create a new buffer of uninitialized elements and ref to it.
  ArrayRef(Type eltype, size_t numel);

  // Return total number of elements.
  int64_t numel() const { return numel_; }

  size_t elsize() const { return eltype_.size(); }

  int64_t stride() const { return stride_; }

  int64_t offset() const { return offset_; }

  const Type& eltype() const { return eltype_; }

  Type& eltype() { return eltype_; }

  // https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
  ArrayRef slice(int64_t start, int64_t stop, int64_t stride = 1) const;

  std::shared_ptr<yasl::Buffer> buf() const { return buf_; }

  // Create a new buffer if current underline buffer is not compact.
  // while compact means offset > 0 or stride != elsize.
  // Or return the underline buffer.
  std::shared_ptr<yasl::Buffer> getOrCreateCompactBuf() const;

  bool isCompact() const;

  ArrayRef clone() const;

  // View this array ref as another type.
  // @param force, true if ignore the type check.
  ArrayRef as(const Type& new_ty, bool force = false) const;

  // Test two array are bitwise equal
  bool operator==(const ArrayRef& other) const;

  // Get data pointer
  void* data() {
    return reinterpret_cast<void*>(buf_->data<std::byte>() + offset_);
  }
  void const* data() const {
    return reinterpret_cast<void const*>(buf_->data<std::byte>() + offset_);
  }

  // Get element.
  template <typename T = std::byte>
  T& at(int64_t pos) {
    return *reinterpret_cast<T*>(static_cast<std::byte*>(data()) +
                                 stride_ * pos * elsize());
  }
  template <typename T = std::byte>
  const T& at(int64_t pos) const {
    return *reinterpret_cast<const T*>(static_cast<const std::byte*>(data()) +
                                       stride_ * pos * elsize());
  }
};

std::ostream& operator<<(std::ostream& out, const ArrayRef& v);

template <>
struct SimdTrait<ArrayRef> {
  using PackInfo = std::vector<size_t>;

  template <typename InputIt>
  static ArrayRef pack(InputIt first, InputIt last, PackInfo& pi) {
    YASL_ENFORCE(first != last);

    size_t total_numel = 0;
    const Type ty = first->eltype();
    for (auto itr = first; itr != last; ++itr) {
      YASL_ENFORCE(itr->eltype() == ty, "type mismatch {} != {}", itr->eltype(),
                   ty);
      total_numel += itr->numel();
    }
    ArrayRef result(first->eltype(), total_numel);
    size_t res_idx = 0;
    for (; first != last; ++first) {
      detail::strided_copy(first->numel(), ty.size(), &result.at(res_idx),
                           result.stride(), &first->at(0), first->stride());
      pi.push_back(first->numel());
      res_idx += first->numel();
    }
    return result;
  }

  template <typename OutputIt>
  static OutputIt unpack(const ArrayRef& v, OutputIt result,
                         const PackInfo& pi) {
    const int64_t total_num =
        std::accumulate(pi.begin(), pi.end(), 0, std::plus<>());

    YASL_ENFORCE(v.numel() == total_num, "split number mismatch {} != {}",
                 v.numel(), total_num);

    int64_t offset = 0;
    for (const auto& sz : pi) {
      *result++ = ArrayRef(v.buf(), v.eltype(), sz, v.stride(), offset);
      offset += sz * v.elsize();
    }

    return result;
  }
};

ArrayRef makeConstantArrayRef(Type eltype, size_t numel);

// A strided array view type.
template <typename T>
class ArrayView {
  T* const data_;
  int64_t const stride_;
  int64_t const numel_;

 public:
  // TODO: we explicit discard const correctness due to the complexity.
  explicit ArrayView(const ArrayRef& arr)
      : data_(const_cast<T*>(&arr.at<T>(0))),
        stride_(arr.stride()),
        numel_(arr.numel()) {}

  int64_t numel() const { return numel_; }

  bool isCompact() const { return stride_ == 1; }

  ArrayRef clone() const {
    ArrayRef res(makePtType<T>(), numel_);
    detail::strided_copy(numel_, sizeof(T), res.data(), res.stride(), data_,
                         stride_);
    return res;
  }

  T* data() { return data_; }

  T& operator[](size_t idx) { return *(data_ + idx * stride_); }

  T const& operator[](size_t idx) const { return *(data_ + idx * stride_); }
};

// A vector like container which buffer could be stealed.
template <typename T>
class Array {
  T* const data_;
  int64_t const numel_;

 public:
  // TODO: we explicit discard const correctness due to the complexity.
  explicit Array(yasl::Buffer&& buf)
      : data_(buf.data()), numel_(buf.size() / sizeof(T)) {
    YASL_ENFORCE(buf.size() % sizeof(T) == 0);
    buf.release();
  }

  ~Array() { delete[] data_; }

  int64_t numel() const { return numel_; }

  T* data() { return data_; }

  T& operator[](size_t idx) { return *(data_ + idx); }

  T const& operator[](size_t idx) const { return *(data_ + idx); }
};

}  // namespace spu
