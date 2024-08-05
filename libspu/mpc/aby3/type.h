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

#include "absl/strings/str_split.h"

#include "libspu/core/type.h"

namespace spu::mpc::aby3 {

class AShrTy : public TypeImpl<AShrTy, RingTy, Secret, AShare> {
  using Base = TypeImpl<AShrTy, RingTy, Secret, AShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "aby3.AShr"; }

  explicit AShrTy(FieldType field) { field_ = field; }

  size_t size() const override { return SizeOf(GetStorageType(field_)) * 2; }
};

// class Z2k {
//   int64_t k_;

// public:
//  virtual ~Z2k() = default;

//  void setK(int64_t k) { k_ = k; }
//  int64_t getK() const { return k_; }
//};

// class Z2Packed {
//   size_t nbits_;

// public:
//  virtual ~Z2Packed() = default;
//  size_t setNbits() const { return nbits_; }
//  void getNbits(size_t nbits) { nbits_ = nbits; }
//};

class BShrTy : public TypeImpl<BShrTy, TypeObject, Secret, BShare> {
  using Base = TypeImpl<BShrTy, TypeObject, Secret, BShare>;
  PtType back_type_ = PT_INVALID;
  FieldType mapping_field_ = FT_INVALID;

 public:
  using Base::Base;
  explicit BShrTy(PtType back_type, size_t nbits, FieldType ftype) {
    SPU_ENFORCE(SizeOf(back_type) * 8 >= nbits,
                "backtype={} has not enough bits={}", back_type, nbits);
    back_type_ = back_type;
    mapping_field_ = ftype;
    nbits_ = nbits;
    max_bits_ = SizeOf(mapping_field_) * 8;
  }

  PtType getBacktype() const { return back_type_; }

  FieldType getMappingField() const { return mapping_field_; }

  static std::string_view getStaticId() { return "aby3.BShr"; }

  void fromString(std::string_view detail) override {
    std::vector<std::string> parts = absl::StrSplit(detail, ',');
    SPU_ENFORCE(PtType_Parse(parts[0], &back_type_), "parse failed from={}",
                detail);
    nbits_ = std::stoul(std::string(parts[1]));
    SPU_ENFORCE(FieldType_Parse(parts[2], &mapping_field_),
                "parse failed from={}", detail);
  }

  std::string toString() const override {
    return fmt::format("{},{},{}", PtType_Name(back_type_), nbits_,
                       FieldType_Name(mapping_field_));
  }

  size_t size() const override { return SizeOf(back_type_) * 2; }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<BShrTy const*>(other);
    SPU_ENFORCE(derived_other);
    return getBacktype() == derived_other->getBacktype() &&
           nbits() == derived_other->nbits();
  }
};

void registerTypes();

}  // namespace spu::mpc::aby3
