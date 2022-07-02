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

#include <utility>

#include "yasl/link/context.h"

#include "spu/core/type_util.h"
#include "spu/mpc/beaver/beaver.h"

namespace spu::mpc {

// HE-based beaver protocol implementation.
class BeaverHE : public Beaver {
 protected:
  struct Impl;
  std::shared_ptr<Impl> impl_;  // private implementation

 public:
  BeaverHE(std::shared_ptr<yasl::link::Context> lctx);

  Beaver::Triple Mul(FieldType field, size_t size) override;

  Beaver::Triple Dot(FieldType field, size_t M, size_t N, size_t K) override;

  // Not supported by BeaverHE
  Beaver::Triple And(FieldType field, size_t size) override;

  // Not supported by BeaverHE
  Beaver::Pair Trunc(FieldType field, size_t size, size_t bits) override;

  // Not supported by BeaverHE
  ArrayRef RandBit(FieldType field, size_t size) override;
};

}  // namespace spu::mpc
