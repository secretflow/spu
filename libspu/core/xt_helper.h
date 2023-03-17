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

#include "xtensor/xadapt.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"

#include "libspu/core/array_ref.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/type_util.h"

namespace spu {

// This module tries to unify all xt::adapt usage, to avoid potential abuse.
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
