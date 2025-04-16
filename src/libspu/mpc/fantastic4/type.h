// Copyright 2025 Ant Group Co., Ltd.
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

#include "magic_enum.hpp"

#include "libspu/core/type.h"

namespace spu::mpc::fantastic4 {

class AShrTy : public TypeImpl<AShrTy, RingTy, Secret, AShare> {
  using Base = TypeImpl<AShrTy, RingTy, Secret, AShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "fantastic4.AShr"; }

  explicit AShrTy(FieldType field) { field_ = field; }

  // 3-out-of-4 shares
  size_t size() const override { return SizeOf(GetStorageType(field_)) * 3; }
};

class BShrTy : public TypeImpl<BShrTy, TypeObject, Secret, BShare> {
  using Base = TypeImpl<BShrTy, TypeObject, Secret, BShare>;
  PtType back_type_ = PT_INVALID;

 public:
  using Base::Base;
  explicit BShrTy(PtType back_type, size_t nbits) {
    SPU_ENFORCE(SizeOf(back_type) * 8 >= nbits,
                "backtype={} has not enough bits={}", back_type, nbits);
    back_type_ = back_type;
    nbits_ = nbits;
  }

  PtType getBacktype() const { return back_type_; }

  static std::string_view getStaticId() { return "fantastic4.BShr"; }

  void fromString(std::string_view detail) override {
    auto comma = detail.find_first_of(',');
    auto back_type_str = detail.substr(0, comma);
    auto nbits_str = detail.substr(comma + 1);
    auto back_type = magic_enum::enum_cast<PtType>(back_type_str);
    SPU_ENFORCE(back_type.has_value(), "parse failed from={}", detail);
    back_type_ = back_type.value();
    nbits_ = std::stoul(std::string(nbits_str));
  }

  std::string toString() const override {
    return fmt::format("{},{}", magic_enum::enum_name(back_type_), nbits_);
  }

  // 3-out-of-4 shares
  size_t size() const override { return SizeOf(back_type_) * 3; }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<BShrTy const*>(other);
    SPU_ENFORCE(derived_other);
    return getBacktype() == derived_other->getBacktype() &&
           nbits() == derived_other->nbits();
  }
};

void registerTypes();

}  // namespace spu::mpc::fantastic4