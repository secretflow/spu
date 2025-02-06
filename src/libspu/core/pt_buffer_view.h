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

#include <utility>

#include "spdlog/spdlog.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/shape.h"
#include "libspu/spu.h"  // PtType

namespace spu {
namespace detail {

template <typename, typename = void>
constexpr bool is_tensor_like_v = false;

// Detection idioms for size() and data().
template <typename T>
constexpr bool
    is_tensor_like_v<T, std::void_t<decltype(std::declval<T>().data()),
                                    decltype(std::declval<T>().shape()),
                                    decltype(std::declval<T>().strides())>> =
        true;

bool isCompact(const Strides& stride, const Shape& shape);

}  // namespace detail

// A view of a plaintext buffer.
struct PtBufferView {
  void* const ptr;               // Pointer to the underlying storage
  PtType const pt_type;          // Plaintext data type.
  Shape const shape;             // Shape of the tensor.
  Strides const strides;         // Strides in number of elements.
  bool const write_able{false};  // Whether this is a writable buffer
  bool const compacted{false};   // Whether this is a compacted buffer
  bool is_bitset{false};         // Bit data

  // We have to take a concrete buffer as a view.
  PtBufferView() = delete;

  // full constructor
  template <typename Pointer>
  explicit PtBufferView(Pointer ptr, PtType pt_type, Shape in_shape,
                        Strides in_strides, bool is_bitset = false)
      : ptr(const_cast<void*>(static_cast<const void*>(ptr))),
        pt_type(pt_type),
        shape(std::move(in_shape)),
        strides(std::move(in_strides)),
        write_able(!std::is_const_v<std::remove_pointer_t<Pointer>>),
        compacted(detail::isCompact(strides, shape)),
        is_bitset(is_bitset) {
    static_assert(std::is_pointer_v<Pointer>);
    if (is_bitset) {
      SPU_ENFORCE(pt_type == PT_I1 && compacted,
                  "Bitset must be I1 type with compacted data");
    }
  }

  // View c++ builtin scalar type as a buffer
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  /* implicit */ PtBufferView(T const& s)  // NOLINT
      : ptr(const_cast<void*>(static_cast<const void*>(&s))),
        pt_type(PtTypeToEnum<T>::value),
        shape(),
        strides(),
        compacted(true) {}

  explicit PtBufferView(bool const& s)
      : ptr(const_cast<void*>(static_cast<const void*>(&s))),
        pt_type(PT_I1),
        shape(),
        strides() {}

  template <typename T,
            std::enable_if_t<detail::is_container_like_v<T>, bool> = true>
  /* implicit */ PtBufferView(const T& c)  // NOLINT
      : ptr(const_cast<void*>(static_cast<const void*>(c.data()))),
        pt_type(PtTypeToEnum<typename T::value_type>::value),
        shape({static_cast<int64_t>(c.size())}),
        strides({1}),
        compacted(true) {}

  // View a tensor-like type (i.e. xt::xarray) as a buffer.
  template <typename T,
            std::enable_if_t<detail::is_tensor_like_v<T>, bool> = true>
  /* implicit */ PtBufferView(const T& t)  // NOLINT
      : ptr(const_cast<void*>(static_cast<const void*>(t.data()))),
        pt_type(PtTypeToEnum<typename T::value_type>::value),
        shape(t.shape().begin(), t.shape().end()),
        strides(t.strides().begin(), t.strides().end()),
        compacted(detail::isCompact(strides, shape)) {}

  template <typename T,
            std::enable_if_t<detail::is_tensor_like_v<T>, bool> = true>
  /* implicit */ PtBufferView(T& t)  // NOLINT
      : ptr(const_cast<void*>(static_cast<const void*>(t.data()))),
        pt_type(PtTypeToEnum<typename T::value_type>::value),
        shape(t.shape().begin(), t.shape().end()),
        strides(t.strides().begin(), t.strides().end()),
        write_able(true),
        compacted(detail::isCompact(strides, shape)) {}

  template <typename S = uint8_t>
  const S& get(const Index& indices) const {
    SPU_ENFORCE(!is_bitset);
    SPU_ENFORCE(PtTypeToEnum<S>::value == pt_type);
    auto fi = calcFlattenOffset(indices, shape, strides);
    const auto* addr =
        static_cast<const std::byte*>(ptr) + SizeOf(pt_type) * fi;
    return *reinterpret_cast<const S*>(addr);
  }

  template <typename S = uint8_t>
  const S& get(size_t idx) const {
    SPU_ENFORCE(!is_bitset);
    if (isCompact()) {
      const auto* addr =
          static_cast<const std::byte*>(ptr) + SizeOf(pt_type) * idx;
      return *reinterpret_cast<const S*>(addr);
    } else {
      const auto& indices = unflattenIndex(idx, shape);
      return get<S>(indices);
    }
  }

  template <typename S = uint8_t>
  void set(const Index& indices, S v) {
    SPU_ENFORCE(write_able);
    SPU_ENFORCE(PtTypeToEnum<S>::value == pt_type);
    SPU_ENFORCE(!is_bitset);
    auto fi = calcFlattenOffset(indices, shape, strides);
    auto* addr = static_cast<std::byte*>(ptr) + SizeOf(pt_type) * fi;
    *reinterpret_cast<S*>(addr) = v;
  }

  template <typename S = uint8_t>
  void set(size_t idx, S v) {
    SPU_ENFORCE(!is_bitset);
    if (isCompact()) {
      auto* addr = static_cast<std::byte*>(ptr) + SizeOf(pt_type) * idx;
      *reinterpret_cast<S*>(addr) = v;
    } else {
      const auto& indices = unflattenIndex(idx, shape);
      set<S>(indices, v);
    }
  }

  bool isCompact() const { return compacted; }

  bool isBitSet() const { return is_bitset; }

  bool getBit(size_t idx) const {
    SPU_ENFORCE(is_bitset);
    auto el_idx = idx / 8;
    auto bit_offset = idx % 8;

    uint8_t mask = (1 << bit_offset);
    uint8_t el = static_cast<uint8_t*>(ptr)[el_idx];

    return (mask & el) != 0;
  }
};

std::ostream& operator<<(std::ostream& out, PtBufferView v);

NdArrayRef convertToNdArray(PtBufferView bv);

}  // namespace spu
