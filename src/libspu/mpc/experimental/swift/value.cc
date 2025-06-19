// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/experimental/swift/value.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/experimental/swift/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::swift {

NdArrayRef getShare(const NdArrayRef& in, int64_t share_idx) {
  SPU_ENFORCE(share_idx == 0 || share_idx == 1 || share_idx == 2,
              "expect share_idx = 1 or 2 or 3, got={}", share_idx);

  auto new_strides = in.strides();
  std::transform(new_strides.cbegin(), new_strides.cend(), new_strides.begin(),
                 [](int64_t s) { return 3 * s; });

  if (in.eltype().isa<AShrTy>()) {
    const auto field = in.eltype().as<AShrTy>()->field();
    const auto ty = makeType<RingTy>(field);

    return NdArrayRef(
        in.buf(), ty, in.shape(), new_strides,
        in.offset() + share_idx * static_cast<int64_t>(ty.size()));
  } else if (in.eltype().isa<BShrTy>()) {
    const auto field = in.eltype().as<BShrTy>()->field();
    const auto ty = makeType<RingTy>(field);

    return NdArrayRef(
        in.buf(), ty, in.shape(), new_strides,
        in.offset() + share_idx * static_cast<int64_t>(ty.size()));
  } else if (in.eltype().isa<PShrTy>()) {
    const auto field = in.eltype().as<PShrTy>()->field();
    const auto ty = makeType<RingTy>(field);

    return NdArrayRef(
        in.buf(), ty, in.shape(), new_strides,
        in.offset() + share_idx * static_cast<int64_t>(ty.size()));
  } else {
    SPU_THROW("unsupported type {}", in.eltype());
  }
}

NdArrayRef getFirstShare(const NdArrayRef& in) { return getShare(in, 0); }

NdArrayRef getSecondShare(const NdArrayRef& in) { return getShare(in, 1); }

NdArrayRef getThirdShare(const NdArrayRef& in) { return getShare(in, 2); }

NdArrayRef makeAShare(const NdArrayRef& s1, const NdArrayRef& s2,
                      const NdArrayRef& s3, FieldType field) {
  const Type ty = makeType<AShrTy>(field);

  SPU_ENFORCE(s1.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s2.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s3.eltype().as<Ring2k>()->field() == field);
  SPU_ENFORCE(s1.shape() == s2.shape(), "got s1={}, s2={}", s1, s2);
  SPU_ENFORCE(s1.shape() == s3.shape(), "got s1={}, s3={}", s1, s3);
  SPU_ENFORCE(ty.size() == 3 * s1.elsize());

  NdArrayRef res(ty, s1.shape());

  if (res.numel() != 0) {
    auto res_s1 = getFirstShare(res);
    auto res_s2 = getSecondShare(res);
    auto res_s3 = getThirdShare(res);

    ring_assign(res_s1, s1);
    ring_assign(res_s2, s2);
    ring_assign(res_s3, s3);
  }

  return res;
}

}  // namespace spu::mpc::swift
