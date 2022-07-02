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

#include "spu/mpc/aby3/type.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::aby3 {
namespace {

ArrayRef makeShare(const ArrayRef& s1, const ArrayRef& s2, Type ty) {
  const auto field = ty.as<Ring2k>()->field();

  YASL_ENFORCE(s2.eltype().as<Ring2k>()->field() == field);
  YASL_ENFORCE(s1.eltype().as<Ring2k>()->field() == field);
  YASL_ENFORCE(s1.numel() == s2.numel(), "got s1={}, s2={}", s1.numel(),
               s2.numel());
  YASL_ENFORCE(ty.size() == 2 * s1.elsize());

  ArrayRef res(ty, s1.numel());

  auto res_s1 = getFirstShare(res);
  auto res_s2 = getSecondShare(res);

  ring_assign(res_s1, s1);
  ring_assign(res_s2, s2);
  return res;
}

}  // namespace

ArrayRef getFirstShare(const ArrayRef& in) {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto ty = makeType<RingTy>(field);

  YASL_ENFORCE(in.stride() != 0);
  return {in.buf(), ty, in.numel(), in.stride() * 2, in.offset()};
}

ArrayRef getSecondShare(const ArrayRef& in) {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto ty = makeType<RingTy>(field);

  YASL_ENFORCE(in.stride() != 0);
  return {in.buf(), ty, in.numel(), in.stride() * 2,
          in.offset() + static_cast<int64_t>(ty.size())};
}

ArrayRef makeAShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field) {
  return makeShare(s1, s2, makeType<AShrTy>(field));
}

ArrayRef makeBShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field,
                    size_t nbits) {
  return makeShare(s1, s2, makeType<BShrTy>(field, nbits));
}

}  // namespace spu::mpc::aby3
