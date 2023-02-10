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

#include "libspu/mpc/spdz2k/beaver/beaver_tfp.h"
#include "libspu/mpc/spdz2k/beaver/trusted_party.h"

namespace spu::mpc::spdz2k {

// Trusted First Party beaver implementation.
//
// Warn: The first party acts TrustedParty directly, it is NOT SAFE and SHOULD
// NOT BE used in production.
//
// Check security implications before moving on.
class BeaverTfpUnsafe final {
 protected:
  // Only for rank0 party.
  TrustedParty tp_;

  std::shared_ptr<yacl::link::Context> lctx_;

  PrgSeed seed_;

  PrgCounter counter_;

  uint128_t global_key_;

 public:
  using Triple = std::tuple<ArrayRef, ArrayRef, ArrayRef>;
  using Pair = std::pair<ArrayRef, ArrayRef>;
  using Pair_Pair = std::pair<Pair, Pair>;
  using Triple_Pair = std::pair<Triple, Triple>;

 public:
  explicit BeaverTfpUnsafe(std::shared_ptr<yacl::link::Context> lctx);

  std::shared_ptr<yacl::link::Context> GetLink() const { return lctx_; }

  uint128_t GetSpdzKey(FieldType field, size_t s);

  Pair AuthCoinTossing(FieldType field, size_t size, size_t s);

  Triple_Pair AuthMul(FieldType field, size_t size);

  Triple_Pair AuthDot(FieldType field, size_t M, size_t N, size_t K);

  Pair_Pair AuthTrunc(FieldType field, size_t size, size_t bits);
};

}  // namespace spu::mpc::spdz2k
