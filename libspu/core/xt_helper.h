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

// TODO: we need a prelude for int128 before include xtensor
#include "yacl/base/int128.h"

#include "libspu/core/prelude.h"
#include "libspu/core/shape_util.h"
//

#include "xtensor/xadapt.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xio.hpp"

#include "libspu/core/array_ref.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type.h"

namespace spu {

// This module tries to unify all xt::adapt usage, to avoid potential abuse.

class ArrayRef;
class NdArrayRef;

template <typename T>
auto xt_adapt(const ArrayRef& aref) {
  SPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={} with size={}",
              aref.eltype(), sizeof(T));

  std::vector<int64_t> shape = {aref.numel()};
  std::vector<int64_t> strides = {aref.stride()};

  return xt::adapt(static_cast<const T*>(aref.data()),
                   static_cast<size_t>(aref.numel()), xt::no_ownership(), shape,
                   strides);
}

template <typename T>
auto xt_mutable_adapt(ArrayRef& aref) {
  SPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={} with size={}",
              aref.eltype(), sizeof(T));

  std::vector<int64_t> shape = {aref.numel()};
  std::vector<int64_t> strides = {aref.stride()};

  return xt::adapt(static_cast<T*>(aref.data()),
                   static_cast<size_t>(aref.numel()), xt::no_ownership(), shape,
                   strides);
}

// Make a array from xtensor expression.
template <typename E,
          typename T = typename std::remove_const<typename E::value_type>::type>
ArrayRef xt_to_array(const xt::xexpression<E>& e, const Type& eltype) {
  SPU_ENFORCE(sizeof(T) == sizeof(typename E::value_type));
  SPU_ENFORCE(sizeof(T) == eltype.size());

  auto&& ret = xt::eval(e.derived_cast());

  // TODO(jint) for matmul, we also pass as a 1D array, but the result is 2D
  SPU_ENFORCE(ret.shape().size() == 1);

  std::vector<std::size_t> shape = {ret.size()};
  ArrayRef aref(eltype, ret.size());

  xt::adapt(static_cast<T*>(aref.data()), ret.size(), xt::no_ownership(),
            shape) = ret;
  return aref;
}

template <typename T>
auto xt_mutable_adapt(NdArrayRef& aref) {
  SPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={} with size={}",
              aref.eltype(), sizeof(T));

  return xt::adapt(static_cast<T*>(aref.data()), aref.numel(),
                   xt::no_ownership(), aref.shape(), aref.strides());
}

template <typename T>
auto xt_adapt(const NdArrayRef& aref) {
  SPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={} with size={}",
              aref.eltype(), sizeof(T));

  return xt::adapt(static_cast<const T*>(aref.data()), aref.numel(),
                   xt::no_ownership(), aref.shape(), aref.strides());
}

// A view of a plaintext buffer.
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

  // View a xt::xarray as a buffer.
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  /* implicit */ PtBufferView(const xt::xarray<T>& xarr)  // NOLINT
      : ptr(static_cast<void const*>(xarr.data())),
        pt_type(PtTypeToEnum<T>::value),
        shape(xarr.shape().begin(), xarr.shape().end()),
        strides({xarr.strides().begin(), xarr.strides().end()}) {}

  // FIXME(jint): make it work when T = bool
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  /* implicit */ PtBufferView(const std::vector<T>& xarr)  // NOLINT
      : ptr(static_cast<void const*>(xarr.data())),
        pt_type(PtTypeToEnum<T>::value),
        shape({static_cast<int64_t>(xarr.size())}),
        strides({1}) {}
};

std::ostream& operator<<(std::ostream& out, const PtBufferView& v);

// Make a ndarray from a plaintext buffer.
// TODO: rename this function.
NdArrayRef xt_to_ndarray(const PtBufferView& bv);

// Make a NdArrayRef from an xt expression.
template <typename E,
          typename T = typename std::remove_const<typename E::value_type>::type,
          std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
NdArrayRef xt_to_ndarray(const xt::xexpression<E>& e) {
  auto&& ee = xt::eval(e.derived_cast());

  const Type eltype = makePtType<T>();
  auto arr = NdArrayRef(
      eltype, std::vector<int64_t>(ee.shape().begin(), ee.shape().end()));
  xt_mutable_adapt<T>(arr) = ee;

  return arr;
}

}  // namespace spu
