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
#include "yacl/link/link.h"

#include "libspu/core/array_ref.h"
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
  uint128_t next_seed_ = 0;
  uint128_t self_seed_ = 0;
  uint64_t prss_counter_ = 0;

 public:
  static constexpr char kBindName[] = "PrgState";
  static constexpr auto kAesType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;

  PrgState();
  explicit PrgState(const std::shared_ptr<yacl::link::Context>& lctx);

  bool hasLowCostFork() const override { return true; }

  std::unique_ptr<State> fork() override;

  ArrayRef genPriv(FieldType field, size_t numel);

  ArrayRef genPubl(FieldType field, size_t numel);

  // Generate a random pair (r0, r1), where
  //   r1 = next_party.r0
  //
  // This correlation could be used to construct zero shares.
  //
  // Note: ignore_first, ignore_second is for perf improvement.
  std::pair<ArrayRef, ArrayRef> genPrssPair(FieldType field, size_t size,
                                            bool ignore_first = false,
                                            bool ignore_second = false);

  template <typename T>
  void fillPubl(absl::Span<T> r) {
    pub_counter_ =
        yacl::crypto::FillPRand(kAesType, pub_seed_, 0, pub_counter_, r);
  }

  template <typename T>
  void fillPriv(absl::Span<T> r) {
    priv_counter_ =
        yacl::crypto::FillPRand(kAesType, priv_seed_, 0, priv_counter_, r);
  }

  template <typename T>
  void fillPrssPair(absl::Span<T> r0, absl::Span<T> r1,
                    bool ignore_first = false, bool ignore_second = false) {
    uint64_t new_counter = prss_counter_;
    if (!ignore_first) {
      new_counter =
          yacl::crypto::FillPRand(kAesType, self_seed_, 0, prss_counter_, r0);
    }
    if (!ignore_second) {
      new_counter =
          yacl::crypto::FillPRand(kAesType, next_seed_, 0, prss_counter_, r1);
    }

    if (new_counter == prss_counter_) {
      // both part ignored, dummy run to update counter...
      new_counter = yacl::crypto::DummyUpdateRandomCount(prss_counter_, r0);
    }
    prss_counter_ = new_counter;
  }
};

}  // namespace spu::mpc
