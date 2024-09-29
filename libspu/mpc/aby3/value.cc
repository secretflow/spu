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

#include "libspu/mpc/aby3/value.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {

MemRef getShare(const MemRef& in, int64_t share_idx) {
  SPU_ENFORCE(share_idx == 0 || share_idx == 1);

  auto new_strides = in.strides();
  std::transform(new_strides.cbegin(), new_strides.cend(), new_strides.begin(),
                 [](int64_t s) { return 2 * s; });

  const auto& in_ty = in.eltype();
  if (in_ty.isa<BoolShareTy>() || in_ty.isa<ArithShareTy>() ||
      in_ty.isa<OramShareTy>() || in_ty.isa<PermShareTy>()) {
    auto out_ty = makeType<RingTy>(in_ty.semantic_type(),
                                   SizeOf(in_ty.storage_type()) * 8);

    return MemRef(in.buf(), out_ty, in.shape(), new_strides,
                  in.offset() + share_idx * out_ty.size());
  } else {
    SPU_THROW("unsupported type {}", in_ty);
  }
}

MemRef getFirstShare(const MemRef& in) { return getShare(in, 0); }

MemRef getSecondShare(const MemRef& in) { return getShare(in, 1); }

MemRef makeArithShare(const MemRef& s1, const MemRef& s2,
                      SemanticType seman_type, size_t valid_bits) {
  const Type ty = makeType<ArithShareTy>(seman_type, valid_bits);

  SPU_ENFORCE(s1.eltype().semantic_type() == seman_type);
  SPU_ENFORCE(s2.eltype().semantic_type() == seman_type);
  SPU_ENFORCE(s1.shape() == s2.shape(), "got s1={}, s2={}", s1, s2);

  MemRef res(ty, s1.shape());

  if (res.numel() != 0) {
    auto res_s1 = getFirstShare(res);
    auto res_s2 = getSecondShare(res);

    ring_assign(res_s1, s1);
    ring_assign(res_s2, s2);
  }

  return res;
}

SemanticType calcBShareSemanticType(size_t nbits) {
  SPU_ENFORCE(nbits <= 128, "unsupported semantic type for {} bits", nbits);
  if (nbits > 64) return SE_I128;
  if (nbits > 32) return SE_I64;
  if (nbits > 16) return SE_I32;
  if (nbits > 8) return SE_I16;
  if (nbits > 1) return SE_I8;
  return SE_1;
}

}  // namespace spu::mpc::aby3
