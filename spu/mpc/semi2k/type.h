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

#include "spu/core/type.h"

namespace spu::mpc::semi2k {

class AShrTy : public TypeImpl<AShrTy, RingTy, Secret, AShare> {
  using Base = TypeImpl<AShrTy, RingTy, Secret, AShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "semi2k.AShr"; }

  explicit AShrTy(FieldType field) { field_ = field; }
};

class BShrTy : public TypeImpl<BShrTy, RingTy, Secret, BShare> {
  using Base = TypeImpl<BShrTy, RingTy, Secret, BShare>;

  static constexpr size_t kDefaultNumBits = std::numeric_limits<size_t>::max();

 public:
  using Base::Base;
  explicit BShrTy(FieldType field, size_t nbits = kDefaultNumBits) {
    field_ = field;
    nbits_ = nbits == kDefaultNumBits ? SizeOf(field) * 8 : nbits;
    YASL_ENFORCE(nbits_ <= SizeOf(field) * 8);
  }

  static std::string_view getStaticId() { return "semi2k.BShr"; }

  void fromString(std::string_view detail) override {
    auto comma = detail.find_first_of(',');
    auto field_str = detail.substr(0, comma);
    auto nbits_str = detail.substr(comma + 1);
    YASL_ENFORCE(FieldType_Parse(std::string(field_str), &field_),
                 "parse failed from={}", detail);
    nbits_ = std::stoul(std::string(nbits_str));
  };

  std::string toString() const override {
    return fmt::format("{},{}", FieldType_Name(field()), nbits_);
  }
};

void registerTypes();

}  // namespace spu::mpc::semi2k
