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

#include "libspu/core/object.h"
#include "libspu/core/type.h"

namespace spu::mpc {

class Pub2kTy : public TypeImpl<Pub2kTy, RingTy, Public> {
  using Base = TypeImpl<Pub2kTy, RingTy, Public>;

 public:
  using Base::Base;
  explicit Pub2kTy(FieldType field) { field_ = field; }

  static std::string_view getStaticId() { return "Pub2k"; }
};

// Note: currently, private type is not exposed to spu.proto, which means:
// - it's transparent to compiler.
// - it's optional to the whole system.
class Priv2kTy : public TypeImpl<Priv2kTy, TypeObject, Ring2k, Private> {
  using Base = TypeImpl<Priv2kTy, TypeObject, Ring2k, Private>;

 public:
  using Base::Base;

  explicit Priv2kTy(FieldType field, int64_t owner) {
    field_ = field;
    owner_ = owner;
  }

  static std::string_view getStaticId() { return "Priv2k"; }

  size_t size() const override {
    // Note: for non-owner party, use 0-strided buffer to store it, the eltype
    // size is not changed.
    return SizeOf(GetStorageType(field_));
  }

  std::string toString() const override {
    return fmt::format("{},{}", FieldType_Name(field()), owner_);
  }

  void fromString(std::string_view str) override {
    auto comma = str.find_first_of(',');
    auto field_str = str.substr(0, comma);
    auto owner_str = str.substr(comma + 1);
    SPU_ENFORCE(FieldType_Parse(std::string(field_str), &field_),
                "parse failed from={}", str);
    owner_ = std::stoll(std::string(owner_str));
  }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<Priv2kTy const*>(other);
    SPU_ENFORCE(derived_other);

    return field_ == derived_other->field_ && owner_ == derived_other->owner();
  }
};

// Z2k related states.
class Z2kState : public State {
  FieldType field_;

 public:
  static constexpr char kBindName[] = "Z2kState";

  explicit Z2kState(FieldType field) : field_(field) {}

  FieldType getDefaultField() const { return field_; }

  std::unique_ptr<State> fork() override {
    return std::make_unique<Z2kState>(field_);
  }

  bool hasLowCostFork() const override { return true; }
};

void regPV2kTypes();

void regPV2kKernels(Object* obj);

}  // namespace spu::mpc
