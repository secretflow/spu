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

#include <type_traits>

#include "type.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xio.hpp"

#include "libspu/core/memref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"

namespace spu {

template <typename T>
auto xt_mutable_adapt(MemRef& aref) {
  SPU_ENFORCE(aref.elsize() == sizeof(T), "adapt eltype={}({}) with size={}",
              aref.eltype(), aref.elsize(), sizeof(T));

  std::vector<int64_t> shape(aref.shape().begin(), aref.shape().end());
  std::vector<int64_t> stride(aref.strides().begin(), aref.strides().end());

  return xt::adapt(aref.data<T>(), aref.numel(), xt::no_ownership(), shape,
                   stride);
}

template <typename T>
auto xt_adapt(const MemRef& aref) {
  auto eltype = aref.eltype();
  auto elsize = aref.elsize();

  if constexpr (std::is_same_v<T, bool>) {
    // SemanticType must be SE_1 or has ONE valid bit
    SPU_ENFORCE(eltype.semantic_type() == SE_1 ||
                    eltype.as<BaseRingType>()->valid_bits() == 1,
                "adapt eltype={}({}) with size={}", eltype, elsize, sizeof(T));
  } else {
    SPU_ENFORCE(elsize == sizeof(T), "adapt eltype={}({}) with size={}", eltype,
                elsize, sizeof(T));
  }

  std::vector<int64_t> shape(aref.shape().begin(), aref.shape().end());
  std::vector<int64_t> stride(aref.strides().begin(), aref.strides().end());

  return xt::adapt(aref.data<const T>(), aref.numel(), xt::no_ownership(),
                   shape, stride);
}

}  // namespace spu

template <typename T>
struct fmt::is_range<xt::xarray<T>, char> : std::false_type {};
