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

  void inferTypeInfo() {
    valid_bits_ = SizeOf(semantic_type_) * 8;
    storage_type_ = GetStorageType(valid_bits_);
  }

 public:
  using Base::Base;
  explicit Pub2kTy(SemanticType type) {
    semantic_type_ = type;
    inferTypeInfo();
  }

  static std::string_view getStaticId() { return "Pub2k"; }

  void fromString(std::string_view detail) override {
    SemanticType_Parse(std::string(detail), &semantic_type_);
    inferTypeInfo();
  };

  std::string toString() const override {
    return fmt::format("{}", semantic_type_);
  }
};

// Note: currently, private type is not exposed to spu.proto, which means:
// - it's transparent to compiler.
// - it's optional to the whole system.
class Priv2kTy : public TypeImpl<Priv2kTy, RingTy, Private> {
  using Base = TypeImpl<Priv2kTy, RingTy, Private>;

  void inferTypeInfo() {
    valid_bits_ = SizeOf(semantic_type_) * 8;
    storage_type_ = GetStorageType(valid_bits_);
  }

 public:
  using Base::Base;

  explicit Priv2kTy(SemanticType type, int64_t owner) {
    semantic_type_ = type;
    owner_ = owner;
    inferTypeInfo();
  }

  explicit Priv2kTy(SemanticType set, StorageType sst, int64_t owner) {
    semantic_type_ = set;
    owner_ = owner;
    storage_type_ = sst;
    valid_bits_ = SizeOf(storage_type_) * 8;
  }

  static std::string_view getStaticId() { return "Priv2k"; }

  std::string toString() const override {
    return fmt::format("{},{}", semantic_type_, owner_);
  }

  void fromString(std::string_view str) override {
    std::vector<std::string> tokens = absl::StrSplit(str, ',');
    SemanticType_Parse(tokens[0], &semantic_type_);
    owner_ = std::stoll(tokens[1]);
    inferTypeInfo();
  }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<Priv2kTy const*>(other);
    SPU_ENFORCE(derived_other);

    return semantic_type_ == derived_other->semantic_type_ &&
           owner_ == derived_other->owner_;
  }
};

// Z2k related states.
class Z2kState : public State {
  size_t field_;

 public:
  static constexpr const char* kBindName() { return "Z2kState"; }

  explicit Z2kState(size_t field) : field_(field) {}

  size_t getDefaultField() const { return field_; }

  void setDefaultField(size_t field) { field_ = field; }

  std::unique_ptr<State> fork() override {
    return std::make_unique<Z2kState>(field_);
  }

  bool hasLowCostFork() const override { return true; }
};

void regPV2kTypes();

void regPV2kKernels(Object* obj);

}  // namespace spu::mpc
