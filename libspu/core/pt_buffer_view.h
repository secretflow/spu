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

#include "absl/types/span.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"

#include "libspu/spu.pb.h"  // PtType

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

}  // namespace detail

// A view of a plaintext buffer.
//
// Please do not direct use this class if possible.
struct PtBufferView {
  void const* const ptr;               // Pointer to the underlying storage
  PtType const pt_type;                // Plaintext data type.
  std::vector<int64_t> const shape;    // Shape of the tensor.
  std::vector<int64_t> const strides;  // Strides in byte.

  // We have to take a concrete buffer as a view.
  PtBufferView() = delete;

  // full constructor
  explicit PtBufferView(void const* ptr, PtType pt_type,
                        std::vector<int64_t> shape,
                        std::vector<int64_t> strides)
      : ptr(ptr),
        pt_type(pt_type),
        shape(std::move(shape)),
        strides(std::move(strides)) {}

  // View c++ builtin scalar type as a buffer
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  /* implicit */ PtBufferView(T const& s)  // NOLINT
      : ptr(static_cast<void const*>(&s)),
        pt_type(PtTypeToEnum<T>::value),
        shape(),
        strides() {}

  // FIXME(jint): make it work when T = bool
  template <typename T, typename std::enable_if_t<
                            detail::is_container_like_v<T>, bool> = true>
  /* implicit */ PtBufferView(const T& c)  // NOLINT
      : ptr(static_cast<void const*>(c.data())),
        pt_type(PtTypeToEnum<typename T::value_type>::value),
        shape({static_cast<int64_t>(c.size())}),
        strides({1}) {}

  // View a tensor-like type (i.e. xt::xarray) as a buffer.
  template <typename T,
            typename std::enable_if_t<detail::is_tensor_like_v<T>, bool> = true>
  /* implicit */ PtBufferView(const T& t)  // NOLINT
      : ptr(static_cast<void const*>(t.data())),
        pt_type(PtTypeToEnum<typename T::value_type>::value),
        shape(t.shape().begin(), t.shape().end()),
        strides({t.strides().begin(), t.strides().end()}) {}

  template <typename T = std::byte>
  const T* get(absl::Span<int64_t const> indices) const {
    auto fi = calcFlattenOffset(indices, shape, strides);
    const auto* addr =
        static_cast<const std::byte*>(ptr) + SizeOf(pt_type) * fi;
    return reinterpret_cast<const T*>(addr);
  }
};

std::ostream& operator<<(std::ostream& out, PtBufferView v);

// Make a ndarray from a plaintext buffer.
NdArrayRef convertToNdArray(PtBufferView bv);

}  // namespace spu
