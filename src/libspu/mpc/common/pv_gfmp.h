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

#pragma once

#include "libspu/core/object.h"
#include "libspu/core/type.h"

namespace spu::mpc {

class PubGfmpTy : public TypeImpl<PubGfmpTy, GfmpTy, Public> {
  using Base = TypeImpl<PubGfmpTy, GfmpTy, Public>;

 public:
  using Base::Base;
  explicit PubGfmpTy(FieldType field) {
    field_ = field;
    mersenne_prime_exp_ = GetMersennePrimeExp(field);
    prime_ = (static_cast<uint128_t>(1) << mersenne_prime_exp_) - 1;
  }

  static std::string_view getStaticId() { return "PubGfmp"; }
};

class PrivGfmpTy : public TypeImpl<PrivGfmpTy, GfmpTy, Private> {
  using Base = TypeImpl<PrivGfmpTy, GfmpTy, Private>;

 public:
  using Base::Base;
  explicit PrivGfmpTy(FieldType field, int64_t owner) {
    field_ = field;
    owner_ = owner;
    mersenne_prime_exp_ = GetMersennePrimeExp(field);
    prime_ = (static_cast<uint128_t>(1) << mersenne_prime_exp_) - 1;
  }

  static std::string_view getStaticId() { return "PrivGfmp"; }

  std::string toString() const override;
  void fromString(std::string_view str) override;

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<PrivGfmpTy const*>(other);
    SPU_ENFORCE(derived_other);

    return field_ == derived_other->field_ && owner_ == derived_other->owner();
  }
};

void regPVGfmpTypes();

void regPVGfmpKernels(Object* obj);

}  // namespace spu::mpc
