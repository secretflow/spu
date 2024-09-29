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

#include "absl/types/span.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/link/context.h"

#include "libspu/core/memref.h"
#include "libspu/core/object.h"

namespace spu::mpc {

// The Pseudo-Random-Generator state.
//
// PRG could be used to generate random variable, for
// 1. public, all party get exactly the same random variable.
// 2. private, each party get different local random variable.
// 3. correlated, for instance zero sharing.
class PrgState : public State {
  // public seed, known to all parties.
  uint128_t pub_seed_ = 0;
  uint64_t pub_counter_ = 0;

  // private seed, known to self only.
  uint128_t priv_seed_ = 0;
  uint64_t priv_counter_ = 0;

  // Pseudorandom Secret Sharing seeds.
  uint128_t self_seed_ = 0;
  uint128_t next_seed_ = 0;
  uint64_t r0_counter_ = 0;  // cnt for self_seed
  uint64_t r1_counter_ = 0;  // cnt for next_seed

 public:
  static constexpr const char* kBindName() { return "PrgState"; }
  static constexpr auto kAesType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;

  PrgState();
  explicit PrgState(const std::shared_ptr<yacl::link::Context>& lctx);

  bool hasLowCostFork() const override { return true; }

  std::unique_ptr<State> fork() override;

  void fillPriv(void* in, size_t nbytes);
  void fillPubl(void* in, size_t nbytes);

  // Generate a random pair (r0, r1), where
  //   r1 = next_party.r0
  //
  // This correlation could be used to construct zero shares.
  void fillPrssPair(void* r0, void* r1, size_t nbytes);
};

}  // namespace spu::mpc
