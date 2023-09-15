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

#include "libspu/core/ndarray_ref.h"
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

  NdArrayRef genPriv(FieldType field, const Shape& shape);

  NdArrayRef genPubl(FieldType field, const Shape& shape);

  // Generate a random pair (r0, r1), where
  //   r1 = next_party.r0
  //
  // This correlation could be used to construct zero shares.
  //
  // Note: ignore_first, ignore_second is for perf improvement.
  enum class GenPrssCtrl { Both, First, Second, None };
  std::pair<NdArrayRef, NdArrayRef> genPrssPair(FieldType field,
                                                const Shape& shape,
                                                GenPrssCtrl ctrl);

  template <typename T>
  void fillPrssPair(T* r0, T* r1, size_t numel, GenPrssCtrl ctrl) {
    switch (ctrl) {
      case GenPrssCtrl::None: {
        // Nothing to generate, pure dummy
        prss_counter_ = yacl::crypto::DummyUpdateRandomCount(prss_counter_,
                                                             numel * sizeof(T));
        return;
      }
      case GenPrssCtrl::First: {
        prss_counter_ = yacl::crypto::FillPRand(
            kAesType, self_seed_, 0, prss_counter_, absl::MakeSpan(r0, numel));
        return;
      }
      case GenPrssCtrl::Second: {
        prss_counter_ = yacl::crypto::FillPRand(
            kAesType, next_seed_, 0, prss_counter_, absl::MakeSpan(r1, numel));
        return;
      }
      case GenPrssCtrl::Both: {
        auto counter0 = yacl::crypto::FillPRand(
            kAesType, self_seed_, 0, prss_counter_, absl::MakeSpan(r0, numel));
        auto counter1 = yacl::crypto::FillPRand(
            kAesType, next_seed_, 0, prss_counter_, absl::MakeSpan(r1, numel));
        SPU_ENFORCE(counter0 == counter1);
        prss_counter_ = counter0;
        return;
      }
    }
  }

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
};

}  // namespace spu::mpc
