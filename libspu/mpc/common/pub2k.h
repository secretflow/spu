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

void regPub2kTypes();

void regPub2kKernels(Object* obj);

}  // namespace spu::mpc
