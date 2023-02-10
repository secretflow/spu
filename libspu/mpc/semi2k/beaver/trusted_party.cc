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

#include "libspu/mpc/semi2k/beaver/trusted_party.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {
namespace {

enum class RecOp : uint8_t {
  ADD = 0,
  XOR = 1,
};

// reconstruct P0's data &
std::pair<std::vector<ArrayRef>, std::vector<ArrayRef>> reconstruct(
    RecOp op, absl::Span<const PrgSeed> seeds,
    absl::Span<const PrgArrayDesc> descs) {
  std::vector<ArrayRef> r0(descs.size());
  std::vector<ArrayRef> rs(descs.size());

  for (size_t rank = 0; rank < seeds.size(); rank++) {
    for (size_t idx = 0; idx < descs.size(); idx++) {
      auto t = prgReplayArray(seeds[rank], descs[idx]);

      if (rank == 0) {
        r0[idx] = t;
        rs[idx] = t.clone();
      } else {
        if (op == RecOp::ADD) {
          ring_add_(rs[idx], t);
        } else if (op == RecOp::XOR) {
          ring_xor_(rs[idx], t);
        } else {
          SPU_ENFORCE("not supported reconstruct op");
        }
      }
    }
  }

  return {r0, rs};
}

void checkDescs(absl::Span<const PrgArrayDesc> descs) {
  for (size_t idx = 1; idx < descs.size(); idx++) {
    SPU_ENFORCE(descs[0].field == descs[idx].field);
    SPU_ENFORCE(descs[0].numel == descs[idx].numel);
  }
}

}  // namespace

void TrustedParty::setSeed(size_t rank, size_t world_size,
                           const PrgSeed& seed) {
  SPU_ENFORCE(rank < world_size, "rank={} should be smaller then world_size={}",
              rank, world_size);

  std::unique_lock lock(mutex_);

  if (seeds_.empty()) {
    seeds_.resize(world_size);
    seeds_[rank] = seed;
  } else {
    SPU_ENFORCE(world_size == seeds_.size(),
                "parties claim different world_size, prev={}, cur={}",
                seeds_.size(), world_size);

    SPU_ENFORCE(!seeds_[rank].has_value() ||
                seeds_[rank].value() == seed);  // NOLINT: checked

    seeds_[rank] = seed;
  }
}

std::vector<PrgSeed> TrustedParty::getSeeds() const {
  std::shared_lock lock(mutex_);

  std::vector<PrgSeed> seeds(seeds_.size());

  for (size_t rank = 0; rank < seeds_.size(); rank++) {
    SPU_ENFORCE(seeds_[rank].has_value(), "seed for rank={} not set", rank);
    seeds[rank] = seeds_[rank].value();  // NOLINT: checked
  }

  return seeds;
}

ArrayRef TrustedParty::adjustMul(absl::Span<const PrgArrayDesc> descs) const {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[2] += rs[0] * rs[1] - rs[2];
  ring_add_(r0[2], ring_sub(ring_mul(rs[0], rs[1]), rs[2]));
  return r0[2];
}

ArrayRef TrustedParty::adjustDot(absl::Span<const PrgArrayDesc> descs, size_t m,
                                 size_t n, size_t k) const {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  SPU_ENFORCE(descs[0].numel == m * k);
  SPU_ENFORCE(descs[1].numel == k * n);
  SPU_ENFORCE(descs[2].numel == m * n);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[2] += rs[0] dot rs[1] - rs[2];
  ring_add_(r0[2], ring_sub(ring_mmul(rs[0], rs[1], m, n, k), rs[2]));
  return r0[2];
}

ArrayRef TrustedParty::adjustAnd(absl::Span<const PrgArrayDesc> descs) const {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto [r0, rs] = reconstruct(RecOp::XOR, getSeeds(), descs);
  // r0[2] ^= (rs[0] & rs[1]) ^ rs[2];
  ring_xor_(r0[2], ring_xor(ring_and(rs[0], rs[1]), rs[2]));
  return r0[2];
}

ArrayRef TrustedParty::adjustTrunc(absl::Span<const PrgArrayDesc> descs,
                                   size_t bits) const {
  SPU_ENFORCE_EQ(descs.size(), 2U);
  checkDescs(descs);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[1] += (rs[0] >> bits) - rs[1];
  ring_add_(r0[1], ring_sub(ring_arshift(rs[0], bits), rs[1]));
  return r0[1];
}

std::pair<ArrayRef, ArrayRef> TrustedParty::adjustTruncPr(
    absl::Span<const PrgArrayDesc> descs, size_t bits) const {
  // descs[0] is r, descs[1] adjust to r[k-2, bits], descs[2] adjust to r[k-1]
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);

  // r0[1] += ((rs[0] << 1) >> (bits + 1)) - rs[1];
  ring_add_(r0[1],
            ring_sub(ring_rshift(ring_lshift(rs[0], 1), bits + 1), rs[1]));

  // r0[2] += (rs[0] >> (k - 1)) - rs[2];
  const size_t k = SizeOf(descs[0].field) * 8;
  ring_add_(r0[2], ring_sub(ring_rshift(rs[0], k - 1), rs[2]));

  return {r0[1], r0[2]};
}

ArrayRef TrustedParty::adjustRandBit(const PrgArrayDesc& desc) const {
  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), absl::MakeSpan(&desc, 1));
  SPU_ENFORCE(r0.size() == 1 && rs.size() == 1);

  // r0[0] += bitrev - rs[0];
  ring_add_(r0[0], ring_sub(ring_randbit(desc.field, desc.numel), rs[0]));
  return r0[0];
}

}  // namespace spu::mpc::semi2k
