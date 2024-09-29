// Copyright 2023 Ant Group Co., Ltd.
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

namespace spu::mpc::securenn {

class ArithShareTy : public TypeImpl<ArithShareTy, RingTy, Secret, ArithShare> {
  using Base = TypeImpl<ArithShareTy, RingTy, Secret, ArithShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "securenn.ArithShare"; }
  explicit ArithShareTy(SemanticType semantic_type, size_t width) {
    semantic_type_ = semantic_type;
    valid_bits_ = width;
    storage_type_ = GetStorageType(valid_bits_);
  }
};

class BoolShareTy : public TypeImpl<BoolShareTy, RingTy, Secret, BoolShare> {
  using Base = TypeImpl<BoolShareTy, RingTy, Secret, BoolShare>;

 public:
  using Base::Base;
  // Optimize Me: we can use storage type according to width
  explicit BoolShareTy(SemanticType semantic_type, StorageType storage_type,
                       size_t width) {
    semantic_type_ = semantic_type;
    valid_bits_ = width;
    storage_type_ = storage_type;
    packed_ = true;
  }

  static std::string_view getStaticId() { return "securenn.BoolShare"; }

  void fromString(std::string_view detail) override {
    std::vector<std::string> tokens = absl::StrSplit(detail, ',');
    SemanticType_Parse(tokens[0], &semantic_type_);
    StorageType_Parse(tokens[1], &storage_type_);
    valid_bits_ = std::stoul(tokens[2]);
  };

  std::string toString() const override {
    return fmt::format("{},{},{}", semantic_type_, storage_type_, valid_bits_);
  }

  bool equals(TypeObject const* other) const override {
    auto const* derived_other = dynamic_cast<BoolShareTy const*>(other);
    SPU_ENFORCE(derived_other);
    return valid_bits_ == derived_other->valid_bits_ &&
           semantic_type_ == derived_other->semantic_type_ &&
           storage_type_ == derived_other->storage_type_;
    ;
  }
};

void registerTypes();

}  // namespace spu::mpc::securenn
