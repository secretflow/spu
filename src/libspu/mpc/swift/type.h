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

#include "libspu/core/type.h"

namespace spu::mpc::swift {

class AShrTy : public TypeImpl<AShrTy, RingTy, Secret, AShare> {
  using Base = TypeImpl<AShrTy, RingTy, Secret, AShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "swift.AShr"; }
  explicit AShrTy(FieldType field) { field_ = field; }

  size_t size() const override {
    if (field_ == FT_INVALID) {
      return 0;
    }
    return SizeOf(GetStorageType(field_)) * 3;
  }
};

class BShrTy : public TypeImpl<BShrTy, RingTy, Secret, BShare> {
  using Base = TypeImpl<BShrTy, RingTy, Secret, BShare>;

  static constexpr size_t kDefaultNumBits = std::numeric_limits<size_t>::max();

 public:
  using Base::Base;
  explicit BShrTy(FieldType field, size_t nbits = kDefaultNumBits) {
    field_ = field;
    nbits_ = nbits == kDefaultNumBits ? SizeOf(field) * 8 : nbits;
    SPU_ENFORCE(nbits_ <= SizeOf(field) * 8);
  }

  static std::string_view getStaticId() { return "swift.BShr"; }

  size_t size() const override {
    if (field_ == FT_INVALID) {
      return 0;
    }
    return SizeOf(GetStorageType(field_)) * 3;
  }

  void fromString(std::string_view detail) override;
  std::string toString() const override;

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<BShrTy const*>(other);
    SPU_ENFORCE(derived_other);
    return field_ == derived_other->field_ && nbits_ == derived_other->nbits();
  }
};

class PShrTy : public TypeImpl<PShrTy, RingTy, Secret, PShare> {
  using Base = TypeImpl<PShrTy, RingTy, Secret, PShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "swift.PShr"; }
  PShrTy() { field_ = FieldType::FM64; }
};

void registerTypes();

}  // namespace spu::mpc::swift
