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

#include <functional>

#include "spu/core/parallel_utils.h"
#include "spu/core/trace.h"
#include "spu/mpc/aby3/ot.h"
#include "spu/mpc/aby3/type.h"
#include "spu/mpc/aby3/value.h"
#include "spu/mpc/common/abprotocol.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
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
ArrayRef A2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();

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
  const PtType out_btype = calcBShareBacktype(SizeOf(field) * 8);
  const auto out_ty = makeType<BShrTy>(out_btype, SizeOf(out_btype) * 8);
  ArrayRef m(out_ty, in.numel());
  ArrayRef n(out_ty, in.numel());

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    const auto _in = ArrayView<std::array<ring2k_t, 2>>(in);

    DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using BShrT = ScalarT;

      std::vector<BShrT> r0(in.numel());
      std::vector<BShrT> r1(in.numel());
      prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

      pforeach(0, in.numel(), [&](int64_t idx) {
        r0[idx] ^= r1[idx];
        if (comm->getRank() == 0) {
          r0[idx] ^= _in[idx][0] + _in[idx][1];
        }
      });

      r1 = comm->rotate<BShrT>(r0, "a2b");  // comm => 1, k

      auto _m = ArrayView<std::array<BShrT, 2>>(m);
      auto _n = ArrayView<std::array<BShrT, 2>>(n);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _m[idx][0] = r0[idx];
        _m[idx][1] = r1[idx];

        if (comm->getRank() == 0) {
          _n[idx][0] = 0;
          _n[idx][1] = 0;
        } else if (comm->getRank() == 1) {
          _n[idx][0] = 0;
          _n[idx][1] = _in[idx][1];
        } else if (comm->getRank() == 2) {
          _n[idx][0] = _in[idx][0];
          _n[idx][1] = 0;
        }
      });
    });
  });

  return add_bb(ctx->caller(), m, n);  // comm => log(k) + 1, 2k(logk) + k
}

ArrayRef B2ASelector::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();

  // PPA: latency=3+log(k), comm = 2*k*log(k) +3k
  // OT:  latency=2, comm=K*K
  if (in_nbits <= 8) {
    return B2AByOT().proc(ctx, in);
  } else {
    return B2AByPPA().proc(ctx, in);
  }
}

// Referrence:
// 5.3 Share Conversions
// https://eprint.iacr.org/2018/403.pdf
//
// In the semi-honest setting, this can be further optimized by having party 2
// provide (−x2−x3) as private input and compute
//   [x1]B = [x]B + [-x2-x3]B
// using a parallel prefix adder. Regardless, x1 is revealed to parties
// 1,3 and the final sharing is defined as
//   [x]A := (x1, x2, x3)
// Overall, the conversion requires 1 + log k rounds and k + k log k gates.
//
// TODO: convert to single share, will reduce number of rotate.
ArrayRef B2AByPPA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = ctx->caller()->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();

  YACL_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);
  const auto out_ty = makeType<AShrTy>(field);
  ArrayRef out(out_ty, in.numel());
  if (in_nbits == 0) {
    // special case, it's known to be zero.
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using AShrT = ring2k_t;
      auto _out = ArrayView<std::array<AShrT, 2>>(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = 0;
      });
    });
    return out;
  }

  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using BShrT = ScalarT;
    auto _in = ArrayView<std::array<BShrT, 2>>(in);
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using AShrT = ring2k_t;

      // first expand b share to a share length.
      const auto expanded_ty = makeType<BShrTy>(
          calcBShareBacktype(SizeOf(field) * 8), SizeOf(field) * 8);
      ArrayRef x(expanded_ty, in.numel());
      auto _x = ArrayView<std::array<AShrT, 2>>(x);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _x[idx][0] = _in[idx][0];
        _x[idx][1] = _in[idx][1];
      });

      // P1 & P2 local samples ra, note P0's ra is not used.
      std::vector<AShrT> ra0(in.numel());
      std::vector<AShrT> ra1(in.numel());
      std::vector<AShrT> rb0(in.numel());
      std::vector<AShrT> rb1(in.numel());

      prg_state->fillPrssPair(absl::MakeSpan(ra0), absl::MakeSpan(ra1));
      prg_state->fillPrssPair(absl::MakeSpan(rb0), absl::MakeSpan(rb1));

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto zb = rb0[idx] ^ rb1[idx];
        if (comm->getRank() == 1) {
          rb0[idx] = zb ^ (ra0[idx] + ra1[idx]);
        } else {
          rb0[idx] = zb;
        }
      });
      rb1 = comm->rotate<AShrT>(rb0, "b2a.rand");  // comm => 1, k

      // compute [x+r]B
      ArrayRef r(expanded_ty, in.numel());
      auto _r = ArrayView<std::array<AShrT, 2>>(r);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _r[idx][0] = rb0[idx];
        _r[idx][1] = rb1[idx];
      });

      // comm => log(k) + 1, 2k(logk) + k
      auto x_plus_r = add_bb(ctx->caller(), x, r);
      auto _x_plus_r = ArrayView<std::array<AShrT, 2>>(x_plus_r);

      // reveal
      std::vector<AShrT> x_plus_r_2(in.numel());
      if (comm->getRank() == 0) {
        x_plus_r_2 = comm->recv<AShrT>(2, "reveal.x_plus_r.to.P0");
      } else if (comm->getRank() == 2) {
        std::vector<AShrT> x_plus_r_0(in.numel());
        pforeach(0, in.numel(),
                 [&](int64_t idx) { x_plus_r_0[idx] = _x_plus_r[idx][0]; });
        comm->sendAsync<AShrT>(0, x_plus_r_0, "reveal.x_plus_r.to.P0");
      }

      // P0 hold x+r, P1 & P2 hold -r, reuse ra0 and ra1 as output
      pforeach(0, in.numel(), [&](int64_t idx) {
        if (comm->getRank() == 0) {
          ra0[idx] = _x_plus_r[idx][0] ^ _x_plus_r[idx][1] ^ x_plus_r_2[idx];
        } else {
          ra0[idx] = -ra0[idx];
        }
      });

      ra1 = comm->rotate<AShrT>(ra0, "b2a.rotate");

      auto _out = ArrayView<std::array<AShrT, 2>>(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = ra0[idx];
        _out[idx][1] = ra1[idx];
      });
    });
  });
  return out;
}

template <typename T>
static std::vector<bool> bitDecompose(ArrayView<T> in, size_t nbits) {
  // decompose each bit of an array of element.
  std::vector<bool> dep(in.numel() * nbits);
  pforeach(0, in.numel(), [&](int64_t idx) {
    for (size_t bit = 0; bit < nbits; bit++) {
      size_t flat_idx = idx * nbits + bit;
      dep[flat_idx] = static_cast<bool>((in[idx] >> bit) & 0x1);
    }
  });
  return dep;
}

template <typename T>
static std::vector<T> bitCompose(absl::Span<T const> in, size_t nbits) {
  YACL_ENFORCE(in.size() % nbits == 0);
  std::vector<T> out(in.size() / nbits, 0);
  pforeach(0, out.size(), [&](int64_t idx) {
    for (size_t bit = 0; bit < nbits; bit++) {
      size_t flat_idx = idx * nbits + bit;
      out[idx] += in[flat_idx] << bit;
    }
  });
  return out;
}

// Referrence:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2.
//
// Aby3 paper algorithm reference.
//
// P1 & P3 locally samples c1.
// P2 & P3 locally samples c3.
//
// P3 (the OT sender) defines two messages.
//   m{i} := (i^b1^b3)−c1−c3 for i in {0, 1}
// P2 (the receiver) defines his input to be b2 in order to learn the message
//   c2 = m{b2} = (b2^b1^b3)−c1−c3 = b − c1 − c3.
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
ArrayRef B2AByOT::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = ctx->caller()->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();

  YACL_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);

  ArrayRef out(makeType<AShrTy>(field), in.numel());
  if (in_nbits == 0) {
    // special case, it's known to be zero.
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using AShrT = ring2k_t;
      auto _out = ArrayView<std::array<AShrT, 2>>(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = 0;
      });
    });
    return out;
  }

  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  // P0 as the helper/dealer, helps to prepare correlated randomness.
  // P1, P2 as the receiver and sender of OT.
  size_t pivot;
  prg_state->fillPubl(absl::MakeSpan(&pivot, 1));
  size_t P0 = pivot % 3;
  size_t P1 = (pivot + 1) % 3;
  size_t P2 = (pivot + 2) % 3;

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using BShrT = ScalarT;

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using AShrT = ring2k_t;

      auto _in = ArrayView<std::array<BShrT, 2>>(in);
      auto _out = ArrayView<std::array<AShrT, 2>>(out);

      const size_t total_nbits = in.numel() * in_nbits;
      std::vector<AShrT> r0(total_nbits);
      std::vector<AShrT> r1(total_nbits);
      prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

      if (comm->getRank() == P0) {
        // the helper
        auto b2 = bitDecompose(ArrayView<BShrT>(getShare(in, 1)), in_nbits);

        // gen masks with helper.
        std::vector<AShrT> m0(total_nbits);
        std::vector<AShrT> m1(total_nbits);
        prg_state->fillPrssPair(absl::MakeSpan(m0), {}, false, true);
        prg_state->fillPrssPair(absl::MakeSpan(m1), {}, false, true);

        // build selected mask
        YACL_ENFORCE(b2.size() == m0.size() && b2.size() == m1.size());
        pforeach(0, total_nbits,
                 [&](int64_t idx) { m0[idx] = !b2[idx] ? m0[idx] : m1[idx]; });

        // send selected masked to receiver.
        comm->sendAsync<AShrT>(P1, m0, "mc");

        auto c1 = bitCompose<AShrT>(r0, in_nbits);
        auto c2 = comm->recv<AShrT>(P1, "c2");

        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx][0] = c1[idx];
          _out[idx][1] = c2[idx];
        });
      } else if (comm->getRank() == P1) {
        // the receiver
        prg_state->fillPrssPair(absl::MakeSpan(r0), {}, false, false);
        prg_state->fillPrssPair(absl::MakeSpan(r0), {}, false, false);

        auto b2 = bitDecompose(ArrayView<BShrT>(getShare(in, 0)), in_nbits);

        // ot.recv
        auto mc = comm->recv<AShrT>(P0, "mc");
        auto m0 = comm->recv<AShrT>(P2, "m0");
        auto m1 = comm->recv<AShrT>(P2, "m1");

        // rebuild c2 = (b1^b2^b3)-c1-c3
        pforeach(0, total_nbits, [&](int64_t idx) {
          mc[idx] = !b2[idx] ? m0[idx] ^ mc[idx] : m1[idx] ^ mc[idx];
        });
        auto c2 = bitCompose<AShrT>(mc, in_nbits);
        comm->sendAsync<AShrT>(P0, c2, "c2");
        auto c3 = bitCompose<AShrT>(r1, in_nbits);

        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx][0] = c2[idx];
          _out[idx][1] = c3[idx];
        });
      } else if (comm->getRank() == P2) {
        // the sender.
        auto c3 = bitCompose<AShrT>(r0, in_nbits);
        auto c1 = bitCompose<AShrT>(r1, in_nbits);

        // c3 = r0, c1 = r1
        // let mi := (i^b1^b3)−c1−c3 for i in {0, 1}
        // reuse r's memory for m
        pforeach(0, in.numel(), [&](int64_t idx) {
          auto xx = _in[idx][0] ^ _in[idx][1];
          for (size_t bit = 0; bit < in_nbits; bit++) {
            size_t flat_idx = idx * in_nbits + bit;
            AShrT t = r0[flat_idx] + r1[flat_idx];
            r0[flat_idx] = ((xx >> bit) & 0x1) - t;
            r1[flat_idx] = ((~xx >> bit) & 0x1) - t;
          }
        });

        // gen masks with helper.
        std::vector<AShrT> m0(total_nbits);
        std::vector<AShrT> m1(total_nbits);
        prg_state->fillPrssPair({}, absl::MakeSpan(m0), true, false);
        prg_state->fillPrssPair({}, absl::MakeSpan(m1), true, false);
        pforeach(0, total_nbits, [&](int64_t idx) {
          m0[idx] ^= r0[idx];
          m1[idx] ^= r1[idx];
        });

        comm->sendAsync<AShrT>(P1, m0, "m0");
        comm->sendAsync<AShrT>(P1, m1, "m1");

        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx][0] = c3[idx];
          _out[idx][1] = c1[idx];
        });
      } else {
        YACL_THROW("expected party=3, got={}", comm->getRank());
      }
    });
  });

  return out;
}

ArrayRef AddBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  YACL_ENFORCE(lhs_ty->nbits() == rhs_ty->nbits());
  const size_t nbits = lhs_ty->nbits();

  auto* obj = ctx->caller();
  auto cbb = makeABProtBasicBlock(obj);
  // sklansky has more local computation which leads to lower performance.
  // return sklansky<ArrayRef>(cbb, lhs, rhs, nbits);
  return kogge_stone<ArrayRef>(cbb, lhs, rhs, nbits);
}

}  // namespace spu::mpc::aby3
