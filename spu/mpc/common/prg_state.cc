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

#include "spu/mpc/common/prg_state.h"

#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/serialize.h"

namespace spu::mpc {

PrgState::PrgState() {
  pub_seed_ = 0;
  pub_counter_ = 0;

  priv_seed_ = yacl::RandSeed();
  priv_counter_ = 0;

  self_seed_ = 0;
  next_seed_ = 0;
  prss_counter_ = 0;
}

PrgState::PrgState(std::shared_ptr<yacl::link::Context> lctx) {
  // synchronize public state.
  {
    uint128_t self_pk = yacl::RandSeed();

    const auto all_buf = yacl::link::AllGather(
        lctx, yacl::SerializeUint128(self_pk), "Random::PK");

    pub_seed_ = 0;
    for (const auto& buf : all_buf) {
      uint128_t seed = yacl::DeserializeUint128(buf);
      pub_seed_ += seed;
    }

    pub_counter_ = 0;
  }

  // init private state.
  {
    priv_seed_ = yacl::RandSeed();
    priv_counter_ = 0;
  }

  // init PRSS state.
  {
    self_seed_ = yacl::RandSeed();

    constexpr char kCommTag[] = "Random:PRSS";

    // send seed to next party, receive seed from prev party
    lctx->SendAsync(lctx->PrevRank(), yacl::SerializeUint128(self_seed_),
                    kCommTag);
    next_seed_ =
        yacl::DeserializeUint128(lctx->Recv(lctx->NextRank(), kCommTag));

    prss_counter_ = 0;
  }
}

std::pair<ArrayRef, ArrayRef> PrgState::genPrssPair(FieldType field,
                                                    size_t size,
                                                    bool ignore_first,
                                                    bool ignore_second) {
  const Type ty = makeType<RingTy>(field);

  ArrayRef r_self(ty, size);
  ArrayRef r_next(ty, size);

  uint64_t new_counter = prss_counter_;
  if (!ignore_first) {
    new_counter =
        yacl::FillPseudoRandom(kAesType, self_seed_, 0, prss_counter_,
                               absl::MakeSpan(static_cast<char*>(r_self.data()),
                                              r_self.buf()->size()));
  }
  if (!ignore_second) {
    new_counter =
        yacl::FillPseudoRandom(kAesType, next_seed_, 0, prss_counter_,
                               absl::MakeSpan(static_cast<char*>(r_next.data()),
                                              r_next.buf()->size()));
  }

  if (new_counter == prss_counter_) {
    // both part ignored, dummy run to update counter...
    new_counter = yacl::DummyUpdateRandomCount(
        prss_counter_, absl::MakeSpan(static_cast<char*>(r_next.data()),
                                      r_next.buf()->size()));
  }

  prss_counter_ = new_counter;
  return std::make_pair(r_self, r_next);
}

ArrayRef PrgState::genPriv(FieldType field, size_t numel) {
  ArrayRef res(makeType<RingTy>(field), numel);
  priv_counter_ = yacl::FillPseudoRandom(
      kAesType, priv_seed_, 0, priv_counter_,
      absl::MakeSpan(static_cast<char*>(res.data()), res.buf()->size()));

  return res;
}

ArrayRef PrgState::genPubl(FieldType field, size_t numel) {
  ArrayRef res(makeType<RingTy>(field), numel);
  pub_counter_ = yacl::FillPseudoRandom(
      kAesType, pub_seed_, 0, pub_counter_,
      absl::MakeSpan(static_cast<char*>(res.data()), res.buf()->size()));

  return res;
}

}  // namespace spu::mpc
