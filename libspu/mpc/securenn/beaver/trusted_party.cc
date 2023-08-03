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

#include "libspu/mpc/securenn/beaver/trusted_party.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::securenn {
namespace {

enum class RecOp : uint8_t {
  ADD = 0,
  XOR = 1,
};

std::vector<ArrayRef> reconstruct(RecOp op, absl::Span<const PrgSeed> seeds,
                                  absl::Span<const PrgArrayDesc> descs) {
  std::vector<ArrayRef> rs(descs.size());

  for (size_t rank = 0; rank < seeds.size(); rank++) {
    for (size_t idx = 0; idx < descs.size(); idx++) {
      // FIXME: TTP adjuster server and client MUST have same endianness.
      auto t = prgReplayArray(seeds[rank], descs[idx]);

      if (rank == 0) {
        rs[idx] = t;
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

  return rs;
}

void checkDescs(absl::Span<const PrgArrayDesc> descs) {
  for (size_t idx = 1; idx < descs.size(); idx++) {
    SPU_ENFORCE(descs[0].field == descs[idx].field);
    SPU_ENFORCE(descs[0].numel == descs[idx].numel);
  }
}

}  // namespace

ArrayRef TrustedParty::adjustMul(Descs descs, Seeds seeds) {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto rs = reconstruct(RecOp::ADD, seeds, descs);
  // adjust = rs[0] * rs[1] - rs[2];
  return ring_sub(ring_mul(rs[0], rs[1]), rs[2]);
}

ArrayRef TrustedParty::adjustDot(Descs descs, Seeds seeds, size_t m, size_t n,
                                 size_t k) {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  SPU_ENFORCE_EQ(descs[0].numel, m * k);
  SPU_ENFORCE_EQ(descs[1].numel, k * n);
  SPU_ENFORCE_EQ(descs[2].numel, m * n);

  auto rs = reconstruct(RecOp::ADD, seeds, descs);
  // adjust = rs[0] dot rs[1] - rs[2];
  return ring_sub(ring_mmul(rs[0], rs[1], m, n, k), rs[2]);
}

ArrayRef TrustedParty::adjustAnd(Descs descs, Seeds seeds) {
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto rs = reconstruct(RecOp::XOR, seeds, descs);
  // adjust = (rs[0] & rs[1]) ^ rs[2];
  return ring_xor(ring_and(rs[0], rs[1]), rs[2]);
}

ArrayRef TrustedParty::adjustTrunc(Descs descs, Seeds seeds, size_t bits) {
  SPU_ENFORCE_EQ(descs.size(), 2U);
  checkDescs(descs);

  auto rs = reconstruct(RecOp::ADD, seeds, descs);
  // adjust = (rs[0] >> bits) - rs[1];
  return ring_sub(ring_arshift(rs[0], bits), rs[1]);
}

std::pair<ArrayRef, ArrayRef> TrustedParty::adjustTruncPr(Descs descs,
                                                          Seeds seeds,
                                                          size_t bits) {
  // descs[0] is r, descs[1] adjust to r[k-2, bits], descs[2] adjust to r[k-1]
  SPU_ENFORCE_EQ(descs.size(), 3U);
  checkDescs(descs);

  auto rs = reconstruct(RecOp::ADD, seeds, descs);

  // adjust1 = ((rs[0] << 1) >> (bits + 1)) - rs[1];
  auto adjust1 = ring_sub(ring_rshift(ring_lshift(rs[0], 1), bits + 1), rs[1]);

  // adjust2 = (rs[0] >> (k - 1)) - rs[2];
  const size_t k = SizeOf(descs[0].field) * 8;
  auto adjust2 = ring_sub(ring_rshift(rs[0], k - 1), rs[2]);

  return {adjust1, adjust2};
}

ArrayRef TrustedParty::adjustRandBit(Descs descs, Seeds seeds) {
  SPU_ENFORCE_EQ(descs.size(), 1U);
  auto rs = reconstruct(RecOp::ADD, seeds, descs);

  // adjust = bitrev - rs[0];
  return ring_sub(ring_randbit(descs[0].field, descs[0].numel), rs[0]);
}

}  // namespace spu::mpc::securenn
