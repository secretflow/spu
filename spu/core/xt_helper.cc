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

#include "spu/core/xt_helper.h"

#include <random>

namespace spu {
namespace {

template <typename T>
NdArrayRef make_ndarray_impl(PtBufferView bv) {
  // make a compact ndarray
  auto arr = NdArrayRef(makePtType(bv.pt_type), bv.shape);

  // assign to it.
  xt_mutable_adapt<T>(arr) =
      xt::adapt(static_cast<T const*>(bv.ptr), calcNumel(bv.shape),
                xt::no_ownership(), bv.shape, bv.strides);

  return arr;
}

}  // namespace

NdArrayRef xt_to_ndarray(PtBufferView bv) {
#define CASE(NAME, CTYPE, _)             \
  case NAME: {                           \
    return make_ndarray_impl<CTYPE>(bv); \
  }

  switch (bv.pt_type) {
    FOREACH_PT_TYPES(CASE)
    default:
      YASL_THROW("should not be here, pt_type={}", bv.pt_type);
  }

#undef CASE
}

std::ostream& operator<<(std::ostream& out, PtBufferView v) {
  out << fmt::format("PtBufferView<{},{}x{},{}>", v.ptr,
                     fmt::join(v.shape, "x"), v.pt_type,
                     fmt::join(v.strides, "x"));
  return out;
}

}  // namespace spu
