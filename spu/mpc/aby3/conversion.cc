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

#include "spu/mpc/aby3/conversion.h"

#include "spu/core/profile.h"
#include "spu/mpc/aby3/ot.h"
#include "spu/mpc/aby3/type.h"
#include "spu/mpc/aby3/value.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/interfaces.h"
#include "spu/mpc/util/circuits.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::aby3 {

// Referrence:
// ABY3: A Mixed Protocol Framework for Machine Learning
// P16 5.3 Share Conversions, Bit Decomposition
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2 + log(nbits) from 1 rotate and 1 ppa.
//
// See:
// https://github.com/tf-encrypted/tf-encrypted/blob/master/tf_encrypted/protocol/aby3/aby3.py#L2889
ArrayRef A2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_END_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  // Let
  //   X = [(x0, x1), (x1, x2), (x2, x0)] as input.
  //   Z = (z0, z1, z2) as boolean zero share.
  //
  // Construct
  //   M = [((x0+x1)^z0, z1) (z1, z2), (z2, (x0+x1)^z0)]
  //   N = [(0, 0), (0, x2), (x2, 0)]
  // Then
  //   Y = PPA(M, N) as the output.
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  auto [r_fst, r_snd] = prg_state->genPrssPair(field, numel);
  auto m1 = ring_xor(r_fst, r_snd);
  auto m2 = ring_zeros(field, numel);
  auto n1 = ring_zeros(field, numel);
  auto n2 = ring_zeros(field, numel);

  // Shr(x) = [in1, in2, 0]
  // Shr(y) = [0, 0, in3]
  if (comm->getRank() == 0) {
    m1 = ring_xor(m1, ring_add(x1, x2));
  } else if (comm->getRank() == 1) {
    n2 = x2;
  } else if (comm->getRank() == 2) {
    n1 = x1;
  }

  m2 = comm->rotate(m1, kBindName);  // comm => 1, k
  const auto n = makeBShare(n1, n2, field);
  const auto m = makeBShare(m1, m2, field);

  return add_bb(ctx->caller(), m, n);  // comm => log(k) + 1, 2k(logk) + k
}

// Referrence:
// IV.E Boolean to Arithmetic Sharing (B2A), extended to 3pc settings.
// https://encrypto.de/papers/DSZ15.pdf
//
// Latency: 4 + log(nbits) - 3 rotate + 1 send/rec + 1 ppa.
// TODO(junfeng): Optimize anount of comm.
ArrayRef B2A::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto numel = x.numel();

  auto ra1 = prg_state->genPriv(field, numel);
  auto ra2 = comm->rotate(ra1, kBindName);  // comm => 1, k

  auto [z0, z1] = prg_state->genPrssPair(field, numel);
  auto rb1 = ring_xor(z0, z1);
  if (comm->getRank() == 1) {
    ring_xor_(rb1, ring_add(ra1, ra2));
  }
  auto rb2 = comm->rotate(rb1, kBindName);  // comm => 1, k

  // comm => log(k) + 1, 2k(logk) + k
  auto x_plus_r = add_bb(ctx->caller(), x, makeBShare(rb1, rb2, field));

  const auto& x_plus_r_1 = getFirstShare(x_plus_r);
  const auto& x_plus_r_2 = getSecondShare(x_plus_r);

  auto y1 = ring_neg(ra1);

  // we only record the maximum communication, we need to manually add comm
  const auto kcomm = y1.elsize() * y1.numel();
  comm->addCommStatsManually(1, kcomm);  // comm => 1, k

  if (comm->getRank() == 0) {
    auto tmp = comm->recv(2, y1.eltype(), kBindName);
    y1 = ring_xor(ring_xor(x_plus_r_1, x_plus_r_2), tmp);
  } else if (comm->getRank() == 2) {
    comm->sendAsync(0, x_plus_r_1, kBindName);
  }

  auto y2 = comm->rotate(y1, kBindName);  // comm => 1, k

  return makeAShare(y1, y2, field);
}

// Referrence:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2.
ArrayRef B2AByOT::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  // Aby3 paper algorithm reference.
  //
  // P1 & P3 locally samples c1.
  // P2 & P3 locally samples c3.
  //
  // P3 (the OT sender) defines two messages.
  //   m[i] := (i^b1^b3)−c1−c3 for i in {0, 1}
  // P2 (the receiver) defines his input to be b2 in order to learn the message
  //   c2 = m[b2] = (b2^b1^b3)−c1−c3 = b − c1 − c3.
  // P1 (the helper) also knows b2 and therefore the three party OT can be used.
  //
  // However, to make this a valid 2-out-of-3 secret sharing, P1 needs to learn
  // c2.
  //
  // Current implementation
  // - P2 could send c2 resulting in 2 rounds and 4k bits of communication.
  //
  // TODO:
  // - Alternatively, the three-party OT procedure can be repeated (in parallel)
  // with again party 3 playing the sender with inputs m0,mi so that party 1
  // (the receiver) with input bit b2 learns the message c2 (not m[b2]) in the
  // first round, totaling 6k bits and 1 round.

  const auto& x_fst = getFirstShare(in);
  const auto& x_snd = getSecondShare(in);

  const size_t nbits = in.eltype().as<BShare>()->nbits();
  YASL_ENFORCE(nbits <= SizeOf(field) * 8, "invalid nbits={}", nbits);

  if (nbits == 0) {
    // special case, it's known to be zero.
    return makeAShare(ring_zeros(field, numel), ring_zeros(field, numel),
                      field);
  }

  Ot3::RoleRanks roles = {
      2,  // P3(rank=2) plays as sender
      1,  // P2(rank=1) plays as receiver
      0,  // P1(rank=0) plays as helper
  };
  Ot3 ot(field, in.numel() * nbits, roles, comm, prg_state);
  comm->addCommStatsManually(
      1,
      2 * SizeOf(field) * in.numel() * nbits);  // 1, 2 * k * k

  auto buildChoices = [&](const ArrayRef& x) {
    std::vector<uint8_t> choices(numel * nbits);

    const auto& ones = ring_ones(field, numel);
    for (size_t i = 0; i < nbits; i++) {
      auto x_i = ring_as_bool(ring_and(ring_rshift(x, i), ones));
      for (size_t j = 0; j < static_cast<size_t>(numel); j++) {
        choices[i * numel + j] = static_cast<uint8_t>(x_i[j]);
      }
    }
    return choices;
  };

  auto bitCompose = [&](ArrayRef flatten, size_t numel) {
    YASL_ENFORCE(flatten.numel() % numel == 0);

    ArrayRef res = ring_zeros(field, numel);
    // flatten is a random bit in mod 2k space.
    const size_t nbits = flatten.numel() / numel;
    for (size_t i = 0; i < nbits; i++) {
      auto in_i = flatten.slice(i * numel, (i + 1) * numel);
      ring_add_(res, ring_lshift(in_i, i));
    }

    return res;
  };

  // send c2 cost
  comm->addCommStatsManually(1, SizeOf(field) * in.numel());  // 2, k
  const auto& [r_fst, r_snd] = prg_state->genPrssPair(field, numel * nbits);
  switch (comm->getRank()) {
    case 0: {
      // the helper
      const auto& c1 = r_fst;
      const auto& b2 = x_snd;

      auto choices = buildChoices(b2);
      ot.help(choices);

      auto c2_ = comm->recv(1, x_fst.eltype(), "c2_");
      auto c1_ = bitCompose(c1, numel);
      return makeAShare(c1_, c2_, field);
    }
    case 1: {
      // the receiver
      const auto& c3 = r_snd;
      const auto& b2 = x_fst;

      auto choices = buildChoices(b2);
      ArrayRef c2 = ot.recv(choices);

      auto c2_ = bitCompose(c2, numel);
      auto c3_ = bitCompose(c3, numel);
      comm->sendAsync(0, c2_, "c2_");
      return makeAShare(c2_, c3_, field);
    }
    case 2: {
      // the sender.
      const auto& c3 = r_fst;
      const auto& c1 = r_snd;

      // let v0 = 0 ^ x3 ^ x1
      //     v1 = 1 ^ x3 ^ x1
      auto v0 = ring_xor(x_fst, x_snd);
      auto v1 = ring_not(v0);

      // let m[i] := (i^b1^b3)−c1−c3 for i in {0, 1}
      auto m0 = ring_neg(ring_add(c1, c3));
      auto m1 = m0.clone();
      const auto& ones = ring_ones(field, numel);
      for (size_t i = 0; i < nbits; i++) {
        const auto start = i * numel;
        const auto stop = start + numel;

        auto m0_i = m0.slice(start, stop);
        auto m1_i = m1.slice(start, stop);

        ring_add_(m0_i, ring_and(ring_rshift(v0, i), ones));
        ring_add_(m1_i, ring_and(ring_rshift(v1, i), ones));
      }

      //
      ot.send(m0, m1);

      auto c3_ = bitCompose(c3, numel);
      auto c1_ = bitCompose(c1, numel);
      return makeAShare(c3_, c1_, field);
    }
    default:
      YASL_THROW("expected party=3, got={}", comm->getRank());
  }
}

ArrayRef AddBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  CircuitBasicBlock<ArrayRef> cbb;
  {
    cbb.num_bits = SizeOf(field) * 8;
    cbb._xor = [&](ArrayRef const& x, ArrayRef const& y) -> ArrayRef {
      return xor_bb(ctx->caller(), x, y);
    };
    cbb._and = [&](ArrayRef const& x, ArrayRef const& y) -> ArrayRef {
      return and_bb(ctx->caller(), x, y);
    };
    cbb.lshift = [&](ArrayRef const& x, size_t bits) -> ArrayRef {
      return lshift_b(ctx->caller(), x, bits);
    };
    cbb.rshift = [&](ArrayRef const& x, size_t bits) -> ArrayRef {
      return rshift_b(ctx->caller(), x, bits);
    };
  }

  return KoggleStoneAdder<ArrayRef>(lhs, rhs, cbb);
}

}  // namespace spu::mpc::aby3
