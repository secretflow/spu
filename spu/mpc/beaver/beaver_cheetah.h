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

#include "yasl/link/context.h"

#include "spu/crypto/ot/silent/primitives.h"
#include "spu/mpc/beaver/beaver.h"
#include "spu/mpc/beaver/beaver_he.h"

namespace spu::mpc {

// Cheetah beaver implementation.
class BeaverCheetah : public Beaver {
 protected:
  std::shared_ptr<yasl::link::Context> lctx_;

  std::shared_ptr<spu::CheetahPrimitives> cheetah_ot_primitives_{nullptr};

  std::shared_ptr<BeaverHE> cheetah_he_primitives_{nullptr};

 public:
  BeaverCheetah(std::shared_ptr<yasl::link::Context> lctx);

  void set_primitives(
      std::shared_ptr<spu::CheetahPrimitives> cheetah_primitives);

  Beaver::Triple Mul(FieldType field, size_t size) override;

  Beaver::Triple And(FieldType field, size_t size) override;

  Beaver::Triple Dot(FieldType field, size_t M, size_t N, size_t K) override;

  bool SupportTrunc() override { return false; }
  Beaver::Pair Trunc(FieldType field, size_t size, size_t bits) override;

  bool SupportRandBit() override { return false; }
  ArrayRef RandBit(FieldType field, size_t size) override;
};

}  // namespace spu::mpc
