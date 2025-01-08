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

#include "libspu/mpc/spdz2k/beaver/trusted_party.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {
namespace {

enum class RecOp : uint8_t {
  ADD = 0,
  XOR = 1,
};

// reconstruct P0's data &
std::pair<std::vector<NdArrayRef>, std::vector<NdArrayRef>> reconstruct(
    RecOp op, absl::Span<const PrgSeed> seeds,
    absl::Span<const PrgArrayDesc> descs) {
  std::vector<NdArrayRef> r0(descs.size());
  std::vector<NdArrayRef> rs(descs.size());

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
    SPU_ENFORCE(descs[0].shape == descs[idx].shape);
  }
}

}  // namespace

void TrustedParty::setSeed(size_t rank, size_t world_size,
                           const PrgSeed& seed) {
  SPU_ENFORCE(rank < world_size, "rank={} should be smaller then world_size={}",
              rank, world_size);

  std::unique_lock lock(seeds_mutex_);

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
  std::unique_lock lock(seeds_mutex_);

  std::vector<PrgSeed> seeds;

  for (size_t rank = 0; rank < seeds_.size(); rank++) {
    SPU_ENFORCE(seeds_[rank].has_value(), "seed for rank={} not set", rank);
    seeds.push_back(seeds_[rank].value());  // NOLINT: checked
  }

  return seeds;
}

NdArrayRef TrustedParty::adjustSpdzKey(const PrgArrayDesc& desc) const {
  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), absl::MakeSpan(&desc, 1));
  SPU_ENFORCE(r0.size() == 1 && rs.size() == 1);

  return rs[0];
}

std::vector<NdArrayRef> TrustedParty::adjustAuthCoinTossing(
    const PrgArrayDesc& desc, const PrgArrayDesc& mac_desc,
    uint128_t global_key, size_t k, size_t s) const {
  SPU_ENFORCE(s <= SizeOf(desc.field) * 8);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), absl::MakeSpan(&desc, 1));
  SPU_ENFORCE(r0.size() == 1 && rs.size() == 1);
  auto r = ring_bitmask(ring_rand(desc.field, desc.shape), 0, k);
  ring_add_(r0[0], ring_sub(r, rs[0]));

  auto [mac_r0, mac_rs] =
      reconstruct(RecOp::ADD, getSeeds(), absl::MakeSpan(&mac_desc, 1));
  SPU_ENFORCE(mac_r0.size() == 1 && mac_rs.size() == 1);

  // mac_r0[0] += r * global_key - mac_rs[0];
  auto mac = ring_mul(r, global_key);
  ring_add_(mac_r0[0], ring_sub(mac, mac_rs[0]));

  return {r0[0], mac_r0[0]};
}

std::vector<NdArrayRef> TrustedParty::adjustAuthRandBit(
    const PrgArrayDesc& desc, const PrgArrayDesc& mac_desc,
    uint128_t global_key, size_t /*s*/) const {
  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), absl::MakeSpan(&desc, 1));
  SPU_ENFORCE(r0.size() == 1 && rs.size() == 1);

  // r0[0] += r - rs[0];
  auto r = ring_bitmask(ring_rand(desc.field, desc.shape), 0, 1);
  ring_add_(r0[0], ring_sub(r, rs[0]));

  auto [mac_r0, mac_rs] =
      reconstruct(RecOp::ADD, getSeeds(), absl::MakeSpan(&mac_desc, 1));
  SPU_ENFORCE(mac_r0.size() == 1 && mac_rs.size() == 1);

  // mac_r0[0] += r * global_key - mac_rs[0];
  auto mac = ring_mul(r, global_key);
  ring_add_(mac_r0[0], ring_sub(mac, mac_rs[0]));

  return {r0[0], mac_r0[0]};
}

std::vector<NdArrayRef> TrustedParty::adjustAuthMul(
    absl::Span<const PrgArrayDesc> descs,
    absl::Span<const PrgArrayDesc> mac_descs, uint128_t global_key) const {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  SPU_ENFORCE_EQ(mac_descs.size(), 3U);
  checkDescs(mac_descs);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[2] += rs[0] * rs[1] - rs[2];
  ring_add_(r0[2], ring_sub(ring_mul(rs[0], rs[1]), rs[2]));

  auto [mac_r0, mac_rs] = reconstruct(RecOp::ADD, getSeeds(), mac_descs);
  // mac_r0[0] += rs[0] * global_key - mac_rs[0];
  auto amac = ring_mul(rs[0], global_key);
  ring_add_(mac_r0[0], ring_sub(amac, mac_rs[0]));

  // mac_r0[1] += rs[1] * global_key - mac_rs[1];
  auto bmac = ring_mul(rs[1], global_key);
  ring_add_(mac_r0[1], ring_sub(bmac, mac_rs[1]));

  // mac_r0[2] += rs[0] * rs[1] * global_key - mac_rs[2];
  auto c = ring_mul(rs[0], rs[1]);
  auto cmac = ring_mul(c, global_key);
  ring_add_(mac_r0[2], ring_sub(cmac, mac_rs[2]));
  return {r0[2], mac_r0[0], mac_r0[1], mac_r0[2]};
}

std::vector<NdArrayRef> TrustedParty::adjustAuthDot(
    absl::Span<const PrgArrayDesc> descs,
    absl::Span<const PrgArrayDesc> mac_descs, int64_t m, int64_t n, int64_t k,
    uint128_t global_key) const {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  SPU_ENFORCE(descs[0].shape == (std::vector<int64_t>{m, k}));
  SPU_ENFORCE(descs[1].shape == (std::vector<int64_t>{k, n}));
  SPU_ENFORCE(descs[2].shape == (std::vector<int64_t>{m, n}));

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[2] += rs[0] dot rs[1] - rs[2];
  ring_add_(r0[2], ring_sub(ring_mmul(rs[0], rs[1]), rs[2]));

  auto [mac_r0, mac_rs] = reconstruct(RecOp::ADD, getSeeds(), mac_descs);
  // mac_r0[0] += rs[0] * global_key - mac_rs[0];
  auto amac = ring_mul(rs[0], global_key);
  ring_add_(mac_r0[0], ring_sub(amac, mac_rs[0]));

  // mac_r0[1] += rs[1] * global_key - mac_rs[1];
  auto bmac = ring_mul(rs[1], global_key);
  ring_add_(mac_r0[1], ring_sub(bmac, mac_rs[1]));

  // mac_r0[2] += rs[0] dot rs[1] * global_key - mac_rs[2];
  auto c = ring_mmul(rs[0], rs[1]);
  auto cmac = ring_mul(c, global_key);
  ring_add_(mac_r0[2], ring_sub(cmac, mac_rs[2]));
  return {r0[2], mac_r0[0], mac_r0[1], mac_r0[2]};
}

std::vector<NdArrayRef> TrustedParty::adjustAuthAnd(
    absl::Span<const PrgArrayDesc> descs,
    absl::Span<const PrgArrayDesc> mac_descs, uint128_t global_key) const {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  SPU_ENFORCE_EQ(mac_descs.size(), 3U);
  checkDescs(mac_descs);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[2] += rs[0] * rs[1] - rs[2];
  ring_add_(r0[2], ring_sub(ring_mul(rs[0], rs[1]), rs[2]));

  auto [mac_r0, mac_rs] = reconstruct(RecOp::ADD, getSeeds(), mac_descs);
  // mac_r0[0] += rs[0] * global_key - mac_rs[0];
  auto amac = ring_mul(rs[0], global_key);
  ring_add_(mac_r0[0], ring_sub(amac, mac_rs[0]));

  // mac_r0[1] += rs[1] * global_key - mac_rs[1];
  auto bmac = ring_mul(rs[1], global_key);
  ring_add_(mac_r0[1], ring_sub(bmac, mac_rs[1]));

  // mac_r0[2] += rs[0] * rs[1] * global_key - mac_rs[2];
  auto c = ring_mul(rs[0], rs[1]);
  auto cmac = ring_mul(c, global_key);
  ring_add_(mac_r0[2], ring_sub(cmac, mac_rs[2]));
  return {r0[2], mac_r0[0], mac_r0[1], mac_r0[2]};
}

std::vector<NdArrayRef> TrustedParty::adjustAuthTrunc(
    absl::Span<const PrgArrayDesc> descs,
    absl::Span<const PrgArrayDesc> mac_descs, size_t bits, uint128_t global_key,
    size_t k, size_t /*s*/) const {
  SPU_ENFORCE_EQ(descs.size(), 2U);
  checkDescs(descs);
  const auto field = descs[0].field;

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[0] += (rs[0] & ((1 << k) - 1)) - rs[0];
  auto t_rs = rs[0].clone();
  ring_bitmask_(rs[0], 0, k);
  ring_add_(r0[0], ring_sub(rs[0], t_rs));

  // r0[1] += (rs[0] >> bits) - rs[1];
  const int64_t bit_len = SizeOf(field) * 8;
  auto tr_rs0 =
      ring_arshift(ring_lshift(rs[0], {static_cast<int64_t>(bit_len - k)}),
                   {static_cast<int64_t>(bit_len - k + bits)});
  ring_bitmask_(tr_rs0, 0, k);
  ring_add_(r0[1], ring_sub(tr_rs0, rs[1]));

  auto [mac_r0, mac_rs] = reconstruct(RecOp::ADD, getSeeds(), mac_descs);
  // mac_r0[0] += rs[0] * global_key - mac_rs[0];
  auto mac = ring_mul(rs[0], global_key);
  ring_add_(mac_r0[0], ring_sub(mac, mac_rs[0]));

  // mac_r0[1] += (rs[0] >> bits) * global_key - mac_rs[1];
  auto tr_mac = ring_mul(tr_rs0, global_key);
  ring_add_(mac_r0[1], ring_sub(tr_mac, mac_rs[1]));
  return {r0[0], r0[1], mac_r0[0], mac_r0[1]};
}

}  // namespace spu::mpc::spdz2k
