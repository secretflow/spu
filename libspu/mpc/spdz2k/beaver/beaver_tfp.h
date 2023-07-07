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

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/spdz2k/beaver/beaver_interface.h"
#include "libspu/mpc/spdz2k/beaver/trusted_party.h"

namespace spu::mpc::spdz2k {

// Trusted First Party beaver implementation.
//
// Warn: The first party acts TrustedParty directly, it is NOT SAFE and SHOULD
// NOT BE used in production.
//
// Check security implications before moving on.
class BeaverTfpUnsafe final : public Beaver {
 protected:
  // Only for rank0 party.
  TrustedParty tp_;

  std::unique_ptr<Communicator> comm_;

  PrgSeed seed_;

  PrgCounter counter_;

  uint128_t global_key_;

  // spzd key
  uint128_t spdz_key_;

  // security parameters
  static constexpr int kappa_ = 128;

 public:
  using Triple = std::tuple<ArrayRef, ArrayRef, ArrayRef>;
  using Pair = std::pair<ArrayRef, ArrayRef>;
  using Pair_Pair = std::pair<Pair, Pair>;
  using Triple_Pair = std::pair<Triple, Triple>;

 public:
  explicit BeaverTfpUnsafe(std::shared_ptr<yacl::link::Context> lctx);

  uint128_t InitSpdzKey(FieldType field, size_t s) override;

  ArrayRef AuthArrayRef(const ArrayRef& value, FieldType field, size_t k,
                        size_t s) override;

  Pair AuthCoinTossing(FieldType field, size_t size, size_t k,
                       size_t s) override;

  Triple_Pair AuthMul(FieldType field, size_t size, size_t k,
                      size_t s) override;

  Triple_Pair AuthDot(FieldType field, size_t M, size_t N, size_t K, size_t k,
                      size_t s) override;

  Triple_Pair AuthAnd(FieldType field, size_t size, size_t s) override;

  Pair_Pair AuthTrunc(FieldType field, size_t size, size_t bits, size_t k,
                      size_t s) override;

  Pair AuthRandBit(FieldType field, size_t size, size_t k, size_t s) override;

  // Check the opened value only
  bool BatchMacCheck(const ArrayRef& open_value, const ArrayRef& mac, size_t k,
                     size_t s);

  // Open the low k_bits of value only
  std::pair<ArrayRef, ArrayRef> BatchOpen(const ArrayRef& value,
                                          const ArrayRef& mac, size_t k,
                                          size_t s);

  // public coin, used in malicious model, all party generate new seed, then
  // get exactly the same random variable.
  ArrayRef genPublCoin(FieldType field, size_t numel);
};

}  // namespace spu::mpc::spdz2k
