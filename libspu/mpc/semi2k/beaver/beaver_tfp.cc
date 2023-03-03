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

#include "libspu/mpc/semi2k/beaver/beaver_tfp.h"

#include <random>
#include <utility>

#include "yacl/crypto/utils/rand.h"
#include "yacl/link/link.h"
#include "yacl/utils/serialize.h"

#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/semi2k/beaver/trusted_party.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

BeaverTfpUnsafe::BeaverTfpUnsafe(std::shared_ptr<yacl::link::Context> lctx)
    : lctx_(std::move(std::move(lctx))),
      seed_(yacl::crypto::RandSeed(true)),
      counter_(0) {
  auto buf = yacl::SerializeUint128(seed_);
  std::vector<yacl::Buffer> all_bufs =
      yacl::link::Gather(lctx_, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (lctx_->Rank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < lctx_->WorldSize(); ++rank) {
      PrgSeed seed = yacl::DeserializeUint128(all_bufs[rank]);
      seeds_.push_back(seed);
    }
  }
}

BeaverTfpUnsafe::Triple BeaverTfpUnsafe::Mul(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    auto adjust = TrustedParty::adjustMul(descs, seeds_);
    ring_add_(c, adjust);
  }

  return {a, b, c};
}

BeaverTfpUnsafe::Triple BeaverTfpUnsafe::Dot(FieldType field, size_t m,
                                             size_t n, size_t k) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, m * k, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, k * n, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, m * n, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    auto adjust = TrustedParty::adjustDot(descs, seeds_, m, n, k);
    ring_add_(c, adjust);
  }

  return {a, b, c};
}

BeaverTfpUnsafe::Triple BeaverTfpUnsafe::And(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    auto adjust = TrustedParty::adjustAnd(descs, seeds_);
    ring_xor_(c, adjust);
  }

  return {a, b, c};
}

BeaverTfpUnsafe::Pair BeaverTfpUnsafe::Trunc(FieldType field, size_t size,
                                             size_t bits) {
  std::vector<PrgArrayDesc> descs(2);
  auto a = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  if (lctx_->Rank() == 0) {
    auto adjust = TrustedParty::adjustTrunc(descs, seeds_, bits);
    ring_add_(b, adjust);
  }
  return {a, b};
}

BeaverTfpUnsafe::Triple BeaverTfpUnsafe::TruncPr(FieldType field, size_t size,
                                                 size_t bits) {
  std::vector<PrgArrayDesc> descs(3);

  auto r = prgCreateArray(field, size, seed_, &counter_, descs.data());
  auto rc = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto rb = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    auto adjusts = TrustedParty::adjustTruncPr(descs, seeds_, bits);
    ring_add_(rc, std::get<0>(adjusts));
    ring_add_(rb, std::get<1>(adjusts));
  }

  return {r, rc, rb};
}

ArrayRef BeaverTfpUnsafe::RandBit(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(1);
  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);

  if (lctx_->Rank() == 0) {
    auto adjust = TrustedParty::adjustRandBit(descs, seeds_);
    ring_add_(a, adjust);
  }

  return a;
}

std::unique_ptr<Beaver> BeaverTfpUnsafe::Spawn() {
  return std::make_unique<BeaverTfpUnsafe>(lctx_->Spawn());
}

}  // namespace spu::mpc::semi2k
