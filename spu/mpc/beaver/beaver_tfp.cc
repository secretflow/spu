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

#include "spu/mpc/beaver/beaver_tfp.h"

#include <random>

#include "yasl/link/link.h"
#include "yasl/utils/serialize.h"

#include "spu/mpc/beaver/prg_tensor.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {
namespace {

uint128_t GetHardwareRandom128() {
  std::random_device rd;
  // call random_device four times, make sure uint128 is random in 2^128 set.
  uint64_t lhs = static_cast<uint64_t>(rd()) << 32 | rd();
  uint64_t rhs = static_cast<uint64_t>(rd()) << 32 | rd();
  return yasl::MakeUint128(lhs, rhs);
}

}  // namespace

BeaverTfpUnsafe::BeaverTfpUnsafe(std::shared_ptr<yasl::link::Context> lctx)
    : lctx_(lctx), seed_(GetHardwareRandom128()), counter_(0) {
  auto buf = yasl::SerializeUint128(seed_);
  std::vector<yasl::Buffer> all_bufs =
      yasl::link::Gather(lctx_, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (lctx_->Rank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < lctx_->WorldSize(); ++rank) {
      PrgSeed seed = yasl::DeserializeUint128(all_bufs[rank]);
      tp_.setSeed(rank, lctx_->WorldSize(), seed);
    }
  }
}

Beaver::Triple BeaverTfpUnsafe::Mul(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustMul(descs);
  }

  return {a, b, c};
}

Beaver::Triple BeaverTfpUnsafe::Dot(FieldType field, size_t M, size_t N,
                                    size_t K) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, M * K, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, K * N, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, M * N, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustDot(descs, M, N, K);
  }

  return {a, b, c};
}

Beaver::Triple BeaverTfpUnsafe::And(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustAnd(descs);
  }

  return {a, b, c};
}

Beaver::Pair BeaverTfpUnsafe::Trunc(FieldType field, size_t size, size_t bits) {
  std::vector<PrgArrayDesc> descs(2);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == 0) {
    b = tp_.adjustTrunc(descs, bits);
  }

  return {a, b};
}

ArrayRef BeaverTfpUnsafe::RandBit(FieldType field, size_t size) {
  PrgArrayDesc desc{};
  auto a = prgCreateArray(field, size, seed_, &counter_, &desc);

  if (lctx_->Rank() == 0) {
    a = tp_.adjustRandBit(desc);
  }

  return a;
}

}  // namespace spu::mpc
