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

#include "spu/mpc/aby3/value.h"

#include "yacl/base/exception.h"

#include "spu/core/array_ref.h"
#include "spu/mpc/aby3/type.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::aby3 {

ArrayRef getShare(const ArrayRef& in, int64_t share_idx) {
  YACL_ENFORCE(share_idx == 0 || share_idx == 1);

  if (in.eltype().isa<AShrTy>()) {
    const auto field = in.eltype().as<AShrTy>()->field();
    const auto ty = makeType<RingTy>(field);
    return {in.buf(), ty, in.numel(), in.stride() * 2,
            in.offset() + share_idx * static_cast<int64_t>(ty.size())};
  } else if (in.eltype().isa<BShrTy>()) {
    const auto stype = in.eltype().as<BShrTy>()->getBacktype();
    const auto ty = makeType<PtTy>(stype);
    return {in.buf(), ty, in.numel(), in.stride() * 2,
            in.offset() + share_idx * static_cast<int64_t>(ty.size())};
  } else {
    YACL_THROW("unsupported type {}", in.eltype());
  }
}

ArrayRef getFirstShare(const ArrayRef& in) { return getShare(in, 0); }

ArrayRef getSecondShare(const ArrayRef& in) { return getShare(in, 1); }

ArrayRef makeAShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field,
                    int owner_rank) {
  const Type ty = makeType<AShrTy>(field, owner_rank);

  YACL_ENFORCE(s2.eltype().as<Ring2k>()->field() == field);
  YACL_ENFORCE(s1.eltype().as<Ring2k>()->field() == field);
  YACL_ENFORCE(s1.numel() == s2.numel(), "got s1={}, s2={}", s1.numel(),
               s2.numel());
  YACL_ENFORCE(ty.size() == 2 * s1.elsize());

  ArrayRef res(ty, s1.numel());

  if (res.numel() != 0) {
    auto res_s1 = getFirstShare(res);
    auto res_s2 = getSecondShare(res);

    ring_assign(res_s1, s1);
    ring_assign(res_s2, s2);
  }

  return res;
}

ArrayRef makeBShare(const ArrayRef& s1, const ArrayRef& s2, size_t nbits) {
  const auto pt_type = s1.eltype().as<PtTy>()->pt_type();
  YACL_ENFORCE(pt_type == s2.eltype().as<PtTy>()->pt_type());
  YACL_ENFORCE(s1.elsize() >= 8 * nbits);
  const Type ty = makeType<BShrTy>(pt_type, nbits);
  ArrayRef res(ty, s1.numel());

  DISPATCH_INT_PT_TYPES(pt_type, "makeBShare", [&]() {
    auto _x1 = ArrayView<ScalarT>(getFirstShare(res));
    auto _x2 = ArrayView<ScalarT>(getSecondShare(res));
    auto _s1 = ArrayView<ScalarT>(s1);
    auto _s2 = ArrayView<ScalarT>(s2);

    for (int64_t idx = 0; idx < s1.numel(); ++idx) {
      _x1[idx] = _s1[idx];
      _x2[idx] = _s2[idx];
    }
  });

  return res;
}

PtType calcBShareBacktype(size_t nbits) {
  if (nbits <= 8) {
    return PT_U8;
  }
  if (nbits <= 16) {
    return PT_U16;
  }
  if (nbits <= 32) {
    return PT_U32;
  }
  if (nbits <= 64) {
    return PT_U64;
  }
  if (nbits <= 128) {
    return PT_U128;
  }
  YACL_THROW("invalid number of bits={}", nbits);
}

}  // namespace spu::mpc::aby3
