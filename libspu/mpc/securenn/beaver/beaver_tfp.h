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

#include <memory>

#include "yacl/link/context.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/securenn/beaver/beaver_interface.h"

namespace spu::mpc::securenn {

// Trusted First Party beaver implementation.
//
// Warn: The first party acts TrustedParty directly, it is NOT SAFE and SHOULD
// NOT BE used in production.
//
// Check security implications before moving on.
class BeaverTfpUnsafe final : public Beaver {
 private:
  // Only for rank0 party.
  std::vector<PrgSeed> seeds_;

  std::shared_ptr<yacl::link::Context> lctx_;

  PrgSeed seed_;

  PrgCounter counter_;

 public:
  explicit BeaverTfpUnsafe(std::shared_ptr<yacl::link::Context> lctx);

  Triple Mul(FieldType field, size_t size) override;

  Triple And(FieldType field, size_t size) override;

  Triple Dot(FieldType field, size_t M, size_t N, size_t K) override;

  Pair Trunc(FieldType field, size_t size, size_t bits) override;

  Triple TruncPr(FieldType field, size_t size, size_t bits) override;

  ArrayRef RandBit(FieldType field, size_t size) override;

  std::unique_ptr<Beaver> Spawn() override;
};

}  // namespace spu::mpc::securenn
