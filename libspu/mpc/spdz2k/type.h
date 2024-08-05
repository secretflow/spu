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

#include "libspu/core/type.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::spdz2k {

class AShrTy : public TypeImpl<AShrTy, RingTy, Secret, AShare> {
  using Base = TypeImpl<AShrTy, RingTy, Secret, AShare>;

  bool has_mac_ = false;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "spdz2k.AShr"; }

  explicit AShrTy(FieldType field) { field_ = field; }

  explicit AShrTy(FieldType field, bool has_mac) {
    field_ = field;
    has_mac_ = has_mac;
  }

  bool hasMac() const { return has_mac_; }

  size_t size() const override { return SizeOf(GetStorageType(field_)) * 2; }
};

class BShrTy : public TypeImpl<BShrTy, RingTy, Secret, BShare> {
  using Base = TypeImpl<BShrTy, RingTy, Secret, BShare>;
  PtType back_type_ = PT_INVALID;
  size_t k_ = 0;

 public:
  using Base::Base;

  explicit BShrTy(PtType back_type, size_t nbits, FieldType field) {
    SPU_ENFORCE(SizeOf(back_type) * 8 >= nbits,
                "backtype={} has not enough bits={}", back_type, nbits);
    back_type_ = back_type;
    nbits_ = nbits;
    max_bits_ = SizeOf(field) * 8;
    field_ = field;
    k_ = SizeOf(field) * 8 / 2;
  }

  PtType getBacktype() const { return back_type_; }

  static std::string_view getStaticId() { return "spdz2k.BShr"; }

  void fromString(std::string_view detail) override {
    auto comma = detail.find_first_of(',');
    auto last_comma = detail.find_last_of(',');
    auto back_type_str = detail.substr(0, comma);
    auto nbits_str = detail.substr(comma + 1, last_comma);
    SPU_ENFORCE(PtType_Parse(std::string(back_type_str), &back_type_),
                "parse failed from={}", back_type_str);
    nbits_ = std::stoul(std::string(nbits_str));
    auto field_str = detail.substr(last_comma + 1);
    SPU_ENFORCE(FieldType_Parse(std::string(field_str), &field_),
                "parse failed from={}", field_str);
  };

  std::string toString() const override {
    return fmt::format("{},{},{}", PtType_Name(back_type_), nbits_, field_);
  }

  size_t nbits() const { return nbits_; }

  size_t k() const { return k_; }

  size_t size() const override {
    return SizeOf(GetStorageType(field_)) * 2 * k_;
  }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<BShrTy const*>(other);
    SPU_ENFORCE(derived_other);
    return getBacktype() == derived_other->getBacktype() &&
           nbits() == derived_other->nbits() &&
           field() == derived_other->field();
  }
};

void registerTypes();

}  // namespace spu::mpc::spdz2k
