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

#include "libspu/mpc/semi2k/beaver/beaver_impl/trusted_party/trusted_party.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {
namespace {

enum class ReduceOp : uint8_t {
  ADD = 0,
  XOR = 1,
  MUL = 2,
};

enum class RecOp : uint8_t {
  ADD = 0,
  XOR = 1,
};

std::vector<NdArrayRef> reduce(ReduceOp op,
                               absl::Span<TrustedParty::Operand> ops) {
  std::vector<NdArrayRef> rs(ops.size());

  const auto world_size = ops[0].seeds.size();

  for (size_t rank = 0; rank < world_size; rank++) {
    for (size_t idx = 0; idx < ops.size(); idx++) {
      // FIXME: TTP adjuster server and client MUST have same endianness.
      NdArrayRef t;
      if (rank < world_size - 1) {
        t = prgReplayArray(ops[idx].seeds[rank], ops[idx].desc);
      } else {
        t = prgReplayArrayMutable(ops[idx].seeds[rank], ops[idx].desc);
      }

      if (rank == 0) {
        rs[idx] = t;
      } else {
        if (op == ReduceOp::ADD) {
          if (ops[idx].desc.eltype == ElementType::kGfmp) {
            // TODO: generalize the reduction
            gfmp_add_mod_(rs[idx], t);
          } else {
            ring_add_(rs[idx], t);
          }
        } else if (op == ReduceOp::XOR) {
          // gfmp has no xor inplementation
          ring_xor_(rs[idx], t);
        } else if (op == ReduceOp::MUL) {
          if (ops[idx].desc.eltype == ElementType::kGfmp) {
            // TODO: generalize the reduction
            gfmp_mul_mod_(rs[idx], t);
          } else {
            ring_mul_(rs[idx], t);
          }
        } else {
          SPU_THROW("not supported reduction op");
        }
      }
    }
  }

  return rs;
}

std::vector<NdArrayRef> reconstruct(RecOp op,
                                    absl::Span<TrustedParty::Operand> ops) {
  return reduce(ReduceOp(op), ops);
}

void checkOperands(absl::Span<const TrustedParty::Operand> ops,
                   bool skip_shape = false, bool allow_transpose = false) {
  for (size_t idx = 1; idx < ops.size(); idx++) {
    SPU_ENFORCE(skip_shape || ops[0].desc.shape == ops[idx].desc.shape);
    SPU_ENFORCE(allow_transpose || ops[0].transpose == false);
    SPU_ENFORCE(ops[0].desc.eltype == ops[idx].desc.eltype);
    SPU_ENFORCE(ops[0].desc.field == ops[idx].desc.field);
    SPU_ENFORCE(ops[0].seeds.size() == ops[idx].seeds.size(), "{} <> {}",
                ops[0].seeds.size(), ops[idx].seeds.size());
  }
}

}  // namespace

// TODO: gfmp support more operations
NdArrayRef TrustedParty::adjustMul(absl::Span<Operand> ops) {
  SPU_ENFORCE_EQ(ops.size(), 3U);
  checkOperands(ops);

  auto rs = reconstruct(RecOp::ADD, ops);
  // adjust = rs[0] * rs[1] - rs[2];
  if (ops[0].desc.eltype == ElementType::kGfmp) {
    return gfmp_sub_mod(gfmp_mul_mod(rs[0], rs[1]), rs[2]);
  } else {
    return ring_sub(ring_mul(rs[0], rs[1]), rs[2]);
  }
}

// ops are [a_or_b, c]
// P0 generate a, c0
// P1 generate b, c1
// The adjustment is ab - (c0 + c1),
// which only needs to be sent to adjust party, e.g. P0.
// P0 with adjust is ab - c1 = ab - (c0 + c1) + c0
// Therefore,
// P0 holds: a, ab - c1
// P1 holds: b, c1
NdArrayRef TrustedParty::adjustMulPriv(absl::Span<Operand> ops) {
  SPU_ENFORCE_EQ(ops.size(), 2U);
  checkOperands(ops);

  auto ab = reduce(ReduceOp::MUL, ops.subspan(0, 1))[0];
  auto c = reconstruct(RecOp::ADD, ops.subspan(1, 1))[0];
  // adjust = ab - c;
  if (ops[0].desc.eltype == ElementType::kGfmp) {
    return gfmp_sub_mod(ab, c);
  } else {
    return ring_sub(ab, c);
  }
}

NdArrayRef TrustedParty::adjustSquare(absl::Span<Operand> ops) {
  SPU_ENFORCE_EQ(ops.size(), 2U);

  auto rs = reconstruct(RecOp::ADD, ops);
  // adjust = rs[0] * rs[0] - rs[1];

  return ring_sub(ring_mul(rs[0], rs[0]), rs[1]);
}

NdArrayRef TrustedParty::adjustDot(absl::Span<Operand> ops) {
  SPU_ENFORCE_EQ(ops.size(), 3U);
  checkOperands(ops, true, true);
  SPU_ENFORCE(ops[2].transpose == false);
  auto rs = reconstruct(RecOp::ADD, ops);

  if (ops[0].transpose) {
    rs[0] = rs[0].transpose();
  }
  if (ops[1].transpose) {
    rs[1] = rs[1].transpose();
  }

  // adjust = rs[0] dot rs[1] - rs[2];
  return ring_sub(ring_mmul(rs[0], rs[1]), rs[2]);
}

NdArrayRef TrustedParty::adjustAnd(absl::Span<Operand> ops) {
  SPU_ENFORCE_EQ(ops.size(), 3U);
  checkOperands(ops);

  auto rs = reconstruct(RecOp::XOR, ops);
  // adjust = (rs[0] & rs[1]) ^ rs[2];
  return ring_xor(ring_and(rs[0], rs[1]), rs[2]);
}

NdArrayRef TrustedParty::adjustTrunc(absl::Span<Operand> ops, size_t bits) {
  SPU_ENFORCE_EQ(ops.size(), 2U);
  checkOperands(ops);

  auto rs = reconstruct(RecOp::ADD, ops);
  // adjust = (rs[0] >> bits) - rs[1];
  return ring_sub(ring_arshift(rs[0], {static_cast<int64_t>(bits)}), rs[1]);
}

std::pair<NdArrayRef, NdArrayRef> TrustedParty::adjustTruncPr(
    absl::Span<Operand> ops, size_t bits) {
  // descs[0] is r, descs[1] adjust to r[k-2, bits], descs[2] adjust to r[k-1]
  SPU_ENFORCE_EQ(ops.size(), 3U);
  checkOperands(ops);

  auto rs = reconstruct(RecOp::ADD, ops);

  // adjust1 = ((rs[0] << 1) >> (bits + 1)) - rs[1];
  auto adjust1 = ring_sub(
      ring_rshift(ring_lshift(rs[0], {1}), {static_cast<int64_t>(bits + 1)}),
      rs[1]);

  // adjust2 = (rs[0] >> (k - 1)) - rs[2];
  const size_t k = SizeOf(ops[0].desc.field) * 8;
  auto adjust2 =
      ring_sub(ring_rshift(rs[0], {static_cast<int64_t>(k - 1)}), rs[2]);

  return {adjust1, adjust2};
}

NdArrayRef TrustedParty::adjustRandBit(absl::Span<Operand> ops) {
  SPU_ENFORCE_EQ(ops.size(), 1U);
  auto rs = reconstruct(RecOp::ADD, ops);

  // adjust = bitrev - rs[0];
  return ring_sub(ring_randbit(ops[0].desc.field, ops[0].desc.shape), rs[0]);
}

NdArrayRef TrustedParty::adjustEqz(absl::Span<Operand> ops) {
  SPU_ENFORCE_EQ(ops.size(), 2U);
  checkOperands(ops);
  auto rs_a = reconstruct(RecOp::ADD, ops.subspan(0, 1));
  auto rs_b = reconstruct(RecOp::XOR, ops.subspan(1, 2));
  // adjust = rs[0] ^ rs[1];
  return ring_xor(rs_a[0], rs_b[0]);
}

NdArrayRef TrustedParty::adjustPerm(absl::Span<Operand> ops,
                                    absl::Span<const int64_t> perm_vec) {
  SPU_ENFORCE_EQ(ops.size(), 2U);
  auto rs = reconstruct(RecOp::ADD, ops);

  return ring_sub(applyInvPerm(rs[0], perm_vec), rs[1]);
}

}  // namespace spu::mpc::semi2k
