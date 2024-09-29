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

namespace spu::mpc::aby3 {

class ArithShareTy : public TypeImpl<ArithShareTy, RingTy, Secret, ArithShare> {
  using Base = TypeImpl<ArithShareTy, RingTy, Secret, ArithShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "aby3.AShr"; }

  explicit ArithShareTy(SemanticType semantic_type, size_t width) {
    semantic_type_ = semantic_type;
    valid_bits_ = width;
    storage_type_ = GetStorageType(valid_bits_);
  }

  size_t size() const override { return SizeOf(storage_type_) * 2; }
};

class OramShareTy : public TypeImpl<OramShareTy, RingTy, Secret, OramShare> {
  using Base = TypeImpl<OramShareTy, RingTy, Secret, OramShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "aby3.OShr"; }

  explicit OramShareTy(SemanticType semantic_type, StorageType storage_type) {
    semantic_type_ = semantic_type;
    valid_bits_ = SizeOf(storage_type) * 8;
    storage_type_ = storage_type;
  }

  size_t size() const override { return SizeOf(storage_type_) * 2; }
};

class OramPubShareTy
    : public TypeImpl<OramPubShareTy, RingTy, Secret, OramPubShare> {
  using Base = TypeImpl<OramPubShareTy, RingTy, Secret, OramPubShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "aby3.OPShr"; }

  explicit OramPubShareTy(SemanticType semantic_type,
                          StorageType storage_type) {
    semantic_type_ = semantic_type;
    valid_bits_ = SizeOf(storage_type) * 8;
    storage_type_ = storage_type;
  }

  // two shares in oram rep share of two different values
  size_t size() const override { return SizeOf(storage_type_); }
};

class BoolShareTy : public TypeImpl<BoolShareTy, RingTy, Secret, BoolShare> {
  using Base = TypeImpl<BoolShareTy, RingTy, Secret, BoolShare>;

 public:
  using Base::Base;

  static std::string_view getStaticId() { return "aby3.BShr"; }

  explicit BoolShareTy(SemanticType semantic_type, StorageType storage_type,
                       size_t width) {
    semantic_type_ = semantic_type;
    valid_bits_ = width;
    storage_type_ = storage_type;
    packed_ = true;
  }

  size_t size() const override { return SizeOf(storage_type_) * 2; }
};

// Permutation share
class PermShareTy : public TypeImpl<PermShareTy, RingTy, Secret, PermShare> {
  using Base = TypeImpl<PermShareTy, RingTy, Secret, PermShare>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "aby3.PShr"; }

  explicit PermShareTy() {
    semantic_type_ = SE_I64;
    valid_bits_ = 64;
    storage_type_ = GetStorageType(valid_bits_);
  }

  size_t size() const override { return SizeOf(storage_type_) * 2; }
};

void registerTypes();

}  // namespace spu::mpc::aby3
