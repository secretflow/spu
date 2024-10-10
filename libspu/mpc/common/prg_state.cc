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

#include "libspu/mpc/common/prg_state.h"

#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/link/algorithm/allgather.h"
#include "yacl/utils/serialize.h"

#include "libspu/mpc/utils/permute.h"

namespace spu::mpc {

PrgState::PrgState() {
  pub_seed_ = 0;

  priv_seed_ = yacl::crypto::SecureRandSeed();

  self_seed_ = 0;
  next_seed_ = 0;
}

PrgState::PrgState(const std::shared_ptr<yacl::link::Context>& lctx) {
  // synchronize public state.
  {
    uint128_t self_pk = yacl::crypto::SecureRandSeed();

    const auto all_buf = yacl::link::AllGather(
        lctx, yacl::SerializeUint128(self_pk), "Random::PK");

    pub_seed_ = 0;
    for (const auto& buf : all_buf) {
      uint128_t seed = yacl::DeserializeUint128(buf);
      pub_seed_ += seed;
    }
  }

  // init private state.
  priv_seed_ = yacl::crypto::SecureRandSeed();

  // init PRSS state.
  {
    self_seed_ = yacl::crypto::SecureRandSeed();

    constexpr char kCommTag[] = "Random:PRSS";

    // send seed to prev party, receive seed from next party
    lctx->SendAsync(lctx->PrevRank(), yacl::SerializeUint128(self_seed_),
                    kCommTag);
    next_seed_ =
        yacl::DeserializeUint128(lctx->Recv(lctx->NextRank(), kCommTag));
  }
}

std::unique_ptr<State> PrgState::fork() {
  //
  auto new_prg = std::make_unique<PrgState>();

  fillPubl(&new_prg->pub_seed_, sizeof(int128_t));

  new_prg->priv_seed_ = yacl::crypto::SecureRandSeed();

  fillPrssPair(&new_prg->self_seed_, &new_prg->next_seed_, sizeof(int128_t));

  return new_prg;
}

void PrgState::fillPrssPair(void* r0, void* r1, size_t nbytes) {
  if (nbytes == 0) {
    return;
  }

  if (r0 != nullptr) {
    r0_counter_ = yacl::crypto::FillPRand(
        kAesType, self_seed_, 0, r0_counter_,
        absl::MakeSpan(static_cast<std::byte*>(r0), nbytes));
  }

  if (r1 != nullptr) {
    r1_counter_ = yacl::crypto::FillPRand(
        kAesType, next_seed_, 0, r1_counter_,
        absl::MakeSpan(static_cast<std::byte*>(r1), nbytes));
  }
}

void PrgState::fillPriv(void* in, size_t nbytes) {
  priv_counter_ = yacl::crypto::FillPRand(
      kAesType, priv_seed_, 0, priv_counter_,
      absl::MakeSpan(static_cast<std::byte*>(in), nbytes));
}

void PrgState::fillPubl(void* in, size_t nbytes) {
  pub_counter_ = yacl::crypto::FillPRand(
      kAesType, pub_seed_, 0, pub_counter_,
      absl::MakeSpan(static_cast<std::byte*>(in), nbytes));
}

Index PrgState::genPrivPerm(size_t numel) {
  return genRandomPerm(numel, priv_seed_, &priv_counter_);
}

std::pair<Index, Index> PrgState::genPrssPermPair(size_t numel) {
  std::pair<Index, Index> res;
  res.first = genRandomPerm(numel, self_seed_, &r0_counter_);
  res.second = genRandomPerm(numel, next_seed_, &r1_counter_);
  return res;
}

}  // namespace spu::mpc
