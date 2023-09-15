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

#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/serialize.h"

namespace spu::mpc {

PrgState::PrgState() {
  pub_seed_ = 0;

  priv_seed_ = yacl::crypto::RandSeed();

  self_seed_ = 0;
  next_seed_ = 0;
}

PrgState::PrgState(const std::shared_ptr<yacl::link::Context>& lctx) {
  // synchronize public state.
  {
    uint128_t self_pk = yacl::crypto::RandSeed();

    const auto all_buf = yacl::link::AllGather(
        lctx, yacl::SerializeUint128(self_pk), "Random::PK");

    pub_seed_ = 0;
    for (const auto& buf : all_buf) {
      uint128_t seed = yacl::DeserializeUint128(buf);
      pub_seed_ += seed;
    }
  }

  // init private state.
  priv_seed_ = yacl::crypto::RandSeed();

  // init PRSS state.
  {
    self_seed_ = yacl::crypto::RandSeed();

    constexpr char kCommTag[] = "Random:PRSS";

    // send seed to next party, receive seed from prev party
    lctx->SendAsync(lctx->PrevRank(), yacl::SerializeUint128(self_seed_),
                    kCommTag);
    next_seed_ =
        yacl::DeserializeUint128(lctx->Recv(lctx->NextRank(), kCommTag));
  }
}

std::unique_ptr<State> PrgState::fork() {
  //
  auto new_prg = std::make_unique<PrgState>();

  fillPubl(absl::MakeSpan(&new_prg->pub_seed_, 1));

  new_prg->priv_seed_ = yacl::crypto::RandSeed();

  fillPrssPair(&new_prg->self_seed_, &new_prg->next_seed_, 1,
               PrgState::GenPrssCtrl::Both);

  return new_prg;
}

std::pair<NdArrayRef, NdArrayRef> PrgState::genPrssPair(FieldType field,
                                                        const Shape& shape,
                                                        GenPrssCtrl ctrl) {
  const Type ty = makeType<RingTy>(field);

  NdArrayRef r_self(ty, shape);
  NdArrayRef r_next(ty, shape);

  fillPrssPair(r_self.data<std::uint8_t>(), r_next.data<std::uint8_t>(),
               shape.numel() * ty.size(), ctrl);

  return std::make_pair(r_self, r_next);
}

NdArrayRef PrgState::genPriv(FieldType field, const Shape& shape) {
  NdArrayRef res(makeType<RingTy>(field), shape);
  priv_counter_ = yacl::crypto::FillPRand(
      kAesType, priv_seed_, 0, priv_counter_,
      absl::MakeSpan(res.data<char>(), res.buf()->size()));

  return res;
}

NdArrayRef PrgState::genPubl(FieldType field, const Shape& shape) {
  NdArrayRef res(makeType<RingTy>(field), shape);
  pub_counter_ = yacl::crypto::FillPRand(
      kAesType, pub_seed_, 0, pub_counter_,
      absl::MakeSpan(res.data<char>(), res.buf()->size()));

  return res;
}

}  // namespace spu::mpc
