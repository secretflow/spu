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

#include "libspu/mpc/aby3/conversion.h"

#include <functional>
#include <iostream>

#include "yacl/utils/platform_utils.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

// Reference:
// ABY3: A Mixed Protocol Framework for Machine Learning
// P16 5.3 Share Conversions, Bit Decomposition
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2 + log(nbits) from 1 rotate and 1 ppa.
NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

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
  NdArrayRef m(out_ty, in.shape());
  NdArrayRef n(out_ty, in.shape());

  auto numel = in.numel();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using ashr_t = std::array<ring2k_t, 2>;
    NdArrayView<ashr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 2>;

      std::vector<bshr_el_t> r0(in.numel());
      std::vector<bshr_el_t> r1(in.numel());
      prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                              PrgState::GenPrssCtrl::Both);

      pforeach(0, numel, [&](int64_t idx) {
        r0[idx] ^= r1[idx];
        if (comm->getRank() == 0) {
          const auto& v = _in[idx];
          r0[idx] ^= v[0] + v[1];
        }
      });

      r1 = comm->rotate<bshr_el_t>(r0, "a2b");  // comm => 1, k

      NdArrayView<bshr_t> _m(m);
      NdArrayView<bshr_t> _n(n);

      pforeach(0, numel, [&](int64_t idx) {
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

  return wrap_add_bb(ctx->sctx(), m, n);  // comm => log(k) + 1, 2k(logk) + k
}

NdArrayRef B2ASelector::proc(KernelEvalContext* ctx,
                             const NdArrayRef& in) const {
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

// Reference:
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
NdArrayRef B2AByPPA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();

  SPU_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);
  const auto out_ty = makeType<AShrTy>(field);
  NdArrayRef out(out_ty, in.shape());

  auto numel = in.numel();

  if (in_nbits == 0) {
    // special case, it's known to be zero.
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      NdArrayView<std::array<ring2k_t, 2>> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = 0;
      });
    });
    return out;
  }

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using bshr_t = std::array<ScalarT, 2>;
    NdArrayView<bshr_t> _in(in);

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using ashr_el_t = ring2k_t;
      using ashr_t = std::array<ashr_el_t, 2>;

      // first expand b share to a share length.
      const auto expanded_ty = makeType<BShrTy>(
          calcBShareBacktype(SizeOf(field) * 8), SizeOf(field) * 8);
      NdArrayRef x(expanded_ty, in.shape());
      NdArrayView<ashr_t> _x(x);

      pforeach(0, numel, [&](int64_t idx) {
        const auto& v = _in[idx];
        _x[idx][0] = v[0];
        _x[idx][1] = v[1];
      });

      // P1 & P2 local samples ra, note P0's ra is not used.
      std::vector<ashr_el_t> ra0(numel);
      std::vector<ashr_el_t> ra1(numel);
      std::vector<ashr_el_t> rb0(numel);
      std::vector<ashr_el_t> rb1(numel);

      prg_state->fillPrssPair(ra0.data(), ra1.data(), ra0.size(),
                              PrgState::GenPrssCtrl::Both);
      prg_state->fillPrssPair(rb0.data(), rb1.data(), rb0.size(),
                              PrgState::GenPrssCtrl::Both);

      pforeach(0, numel, [&](int64_t idx) {
        const auto zb = rb0[idx] ^ rb1[idx];
        if (comm->getRank() == 1) {
          rb0[idx] = zb ^ (ra0[idx] + ra1[idx]);
        } else {
          rb0[idx] = zb;
        }
      });
      rb1 = comm->rotate<ashr_el_t>(rb0, "b2a.rand");  // comm => 1, k

      // compute [x+r]B
      NdArrayRef r(expanded_ty, in.shape());
      NdArrayView<ashr_t> _r(r);
      pforeach(0, numel, [&](int64_t idx) {
        _r[idx][0] = rb0[idx];
        _r[idx][1] = rb1[idx];
      });

      // comm => log(k) + 1, 2k(logk) + k
      auto x_plus_r = wrap_add_bb(ctx->sctx(), x, r);
      NdArrayView<ashr_t> _x_plus_r(x_plus_r);

      // reveal
      std::vector<ashr_el_t> x_plus_r_2(numel);
      if (comm->getRank() == 0) {
        x_plus_r_2 = comm->recv<ashr_el_t>(2, "reveal.x_plus_r.to.P0");
      } else if (comm->getRank() == 2) {
        std::vector<ashr_el_t> x_plus_r_0(numel);
        pforeach(0, numel,
                 [&](int64_t idx) { x_plus_r_0[idx] = _x_plus_r[idx][0]; });
        comm->sendAsync<ashr_el_t>(0, x_plus_r_0, "reveal.x_plus_r.to.P0");
      }

      // P0 hold x+r, P1 & P2 hold -r, reuse ra0 and ra1 as output
      auto self_rank = comm->getRank();
      pforeach(0, numel, [&](int64_t idx) {
        if (self_rank == 0) {
          const auto& x_r_v = _x_plus_r[idx];
          ra0[idx] = x_r_v[0] ^ x_r_v[1] ^ x_plus_r_2[idx];
        } else {
          ra0[idx] = -ra0[idx];
        }
      });

      ra1 = comm->rotate<ashr_el_t>(ra0, "b2a.rotate");

      NdArrayView<ashr_t> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = ra0[idx];
        _out[idx][1] = ra1[idx];
      });
    });
  });
  return out;
}

template <typename T>
static std::vector<bool> bitDecompose(const NdArrayRef& in, size_t nbits) {
  auto numel = in.numel();
  // decompose each bit of an array of element.
  // FIXME: this is not thread-safe.
  std::vector<bool> dep(numel * nbits);

  NdArrayView<T> _in(in);

  pforeach(0, numel, [&](int64_t idx) {
    const auto& v = _in[idx];
    for (size_t bit = 0; bit < nbits; bit++) {
      size_t flat_idx = idx * nbits + bit;
      dep[flat_idx] = static_cast<bool>((v >> bit) & 0x1);
    }
  });
  return dep;
}

template <typename T>
static std::vector<T> bitCompose(absl::Span<T const> in, size_t nbits) {
  SPU_ENFORCE(in.size() % nbits == 0);
  std::vector<T> out(in.size() / nbits, 0);
  pforeach(0, out.size(), [&](int64_t idx) {
    for (size_t bit = 0; bit < nbits; bit++) {
      size_t flat_idx = idx * nbits + bit;
      out[idx] += in[flat_idx] << bit;
    }
  });
  return out;
}

// Reference:
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
NdArrayRef B2AByOT::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();

  SPU_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);

  NdArrayRef out(makeType<AShrTy>(field), in.shape());
  auto numel = in.numel();

  if (in_nbits == 0) {
    // special case, it's known to be zero.
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      NdArrayView<std::array<ring2k_t, 2>> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = 0;
      });
    });
    return out;
  }

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  // P0 as the helper/dealer, helps to prepare correlated randomness.
  // P1, P2 as the receiver and sender of OT.
  size_t pivot;
  prg_state->fillPubl(absl::MakeSpan(&pivot, 1));
  size_t P0 = pivot % 3;
  size_t P1 = (pivot + 1) % 3;
  size_t P2 = (pivot + 2) % 3;

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using bshr_el_t = ScalarT;
    using bshr_t = std::array<bshr_el_t, 2>;
    NdArrayView<bshr_t> _in(in);

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using ashr_el_t = ring2k_t;
      using ashr_t = std::array<ashr_el_t, 2>;

      NdArrayView<ashr_t> _out(out);

      const size_t total_nbits = numel * in_nbits;
      std::vector<ashr_el_t> r0(total_nbits);
      std::vector<ashr_el_t> r1(total_nbits);
      prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                              PrgState::GenPrssCtrl::Both);

      if (comm->getRank() == P0) {
        // the helper
        auto b2 = bitDecompose<bshr_el_t>(getShare(in, 1), in_nbits);

        // gen masks with helper.
        std::vector<ashr_el_t> m0(total_nbits);
        std::vector<ashr_el_t> m1(total_nbits);
        prg_state->fillPrssPair<ashr_el_t>(m0.data(), nullptr, m0.size(),
                                           PrgState::GenPrssCtrl::First);
        prg_state->fillPrssPair<ashr_el_t>(m1.data(), nullptr, m1.size(),
                                           PrgState::GenPrssCtrl::First);

        // build selected mask
        SPU_ENFORCE(b2.size() == m0.size() && b2.size() == m1.size());
        pforeach(0, total_nbits,
                 [&](int64_t idx) { m0[idx] = !b2[idx] ? m0[idx] : m1[idx]; });

        // send selected masked to receiver.
        comm->sendAsync<ashr_el_t>(P1, m0, "mc");

        auto c1 = bitCompose<ashr_el_t>(r0, in_nbits);
        auto c2 = comm->recv<ashr_el_t>(P1, "c2");

        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = c1[idx];
          _out[idx][1] = c2[idx];
        });
      } else if (comm->getRank() == P1) {
        // the receiver
        prg_state->fillPrssPair<ashr_el_t>(nullptr, nullptr, r0.size(),
                                           PrgState::GenPrssCtrl::None);
        prg_state->fillPrssPair<ashr_el_t>(nullptr, nullptr, r0.size(),
                                           PrgState::GenPrssCtrl::None);

        auto b2 = bitDecompose<bshr_el_t>(getShare(in, 0), in_nbits);

        // ot.recv
        auto mc = comm->recv<ashr_el_t>(P0, "mc");
        auto m0 = comm->recv<ashr_el_t>(P2, "m0");
        auto m1 = comm->recv<ashr_el_t>(P2, "m1");

        // rebuild c2 = (b1^b2^b3)-c1-c3
        pforeach(0, total_nbits, [&](int64_t idx) {
          mc[idx] = !b2[idx] ? m0[idx] ^ mc[idx] : m1[idx] ^ mc[idx];
        });
        auto c2 = bitCompose<ashr_el_t>(mc, in_nbits);
        comm->sendAsync<ashr_el_t>(P0, c2, "c2");
        auto c3 = bitCompose<ashr_el_t>(r1, in_nbits);

        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = c2[idx];
          _out[idx][1] = c3[idx];
        });
      } else if (comm->getRank() == P2) {
        // the sender.
        auto c3 = bitCompose<ashr_el_t>(r0, in_nbits);
        auto c1 = bitCompose<ashr_el_t>(r1, in_nbits);

        // c3 = r0, c1 = r1
        // let mi := (i^b1^b3)−c1−c3 for i in {0, 1}
        // reuse r's memory for m
        pforeach(0, numel, [&](int64_t idx) {
          const auto x = _in[idx];
          auto xx = x[0] ^ x[1];
          for (size_t bit = 0; bit < in_nbits; bit++) {
            size_t flat_idx = idx * in_nbits + bit;
            ashr_el_t t = r0[flat_idx] + r1[flat_idx];
            r0[flat_idx] = ((xx >> bit) & 0x1) - t;
            r1[flat_idx] = ((~xx >> bit) & 0x1) - t;
          }
        });

        // gen masks with helper.
        std::vector<ashr_el_t> m0(total_nbits);
        std::vector<ashr_el_t> m1(total_nbits);
        prg_state->fillPrssPair<ashr_el_t>(nullptr, m0.data(), m0.size(),
                                           PrgState::GenPrssCtrl::Second);
        prg_state->fillPrssPair<ashr_el_t>(nullptr, m1.data(), m1.size(),
                                           PrgState::GenPrssCtrl::Second);
        pforeach(0, total_nbits, [&](int64_t idx) {
          m0[idx] ^= r0[idx];
          m1[idx] ^= r1[idx];
        });

        comm->sendAsync<ashr_el_t>(P1, m0, "m0");
        comm->sendAsync<ashr_el_t>(P1, m1, "m1");

        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = c3[idx];
          _out[idx][1] = c1[idx];
        });
      } else {
        SPU_THROW("expected party=3, got={}", comm->getRank());
      }
    });
  });

  return out;
}

// TODO: Accelerate bit scatter.
// split even and odd bits. e.g.
//   xAyBzCwD -> (xyzw, ABCD)
[[maybe_unused]] std::pair<NdArrayRef, NdArrayRef> bit_split(
    const NdArrayRef& in) {
  constexpr std::array<uint128_t, 6> kSwapMasks = {{
      yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
      yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
      yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
      yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
      yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
      yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
  }};
  constexpr std::array<uint128_t, 6> kKeepMasks = {{
      yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
      yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
      yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
      yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
      yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
      yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
  }};

  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();
  SPU_ENFORCE(in_nbits != 0 && in_nbits % 2 == 0, "in_nbits={}", in_nbits);
  const size_t out_nbits = in_nbits / 2;
  const auto out_backtype = calcBShareBacktype(out_nbits);
  const auto out_type = makeType<BShrTy>(out_backtype, out_nbits);

  NdArrayRef lo(out_type, in.shape());
  NdArrayRef hi(out_type, in.shape());

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_backtype, "_", [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayView<out_shr_t> _lo(lo);
      NdArrayView<out_shr_t> _hi(hi);

      if constexpr (sizeof(out_el_t) <= 8) {
        pforeach(0, in.numel(), [&](int64_t idx) {
          constexpr uint64_t S = 0x5555555555555555;  // 01010101
          const out_el_t M = (out_el_t(1) << (in_nbits / 2)) - 1;

          const auto& r = _in[idx];

          _lo[idx][0] = yacl::pext_u64(r[0], S) & M;
          _hi[idx][0] = yacl::pext_u64(r[0], ~S) & M;
          _lo[idx][1] = yacl::pext_u64(r[1], S) & M;
          _hi[idx][1] = yacl::pext_u64(r[1], ~S) & M;
        });
      } else {
        pforeach(0, in.numel(), [&](int64_t idx) {
          auto r = _in[idx];
          // algorithm:
          //      0101010101010101
          // swap  ^^  ^^  ^^  ^^
          //      0011001100110011
          // swap   ^^^^    ^^^^
          //      0000111100001111
          // swap     ^^^^^^^^
          //      0000000011111111
          for (int k = 0; k + 1 < Log2Ceil(in_nbits); k++) {
            auto keep = static_cast<in_el_t>(kKeepMasks[k]);
            auto move = static_cast<in_el_t>(kSwapMasks[k]);
            int shift = 1 << k;

            r[0] = (r[0] & keep) ^ ((r[0] >> shift) & move) ^
                   ((r[0] & move) << shift);
            r[1] = (r[1] & keep) ^ ((r[1] >> shift) & move) ^
                   ((r[1] & move) << shift);
          }
          in_el_t mask = (in_el_t(1) << (in_nbits / 2)) - 1;
          _lo[idx][0] = static_cast<out_el_t>(r[0]) & mask;
          _hi[idx][0] = static_cast<out_el_t>(r[0] >> (in_nbits / 2)) & mask;
          _lo[idx][1] = static_cast<out_el_t>(r[1]) & mask;
          _hi[idx][1] = static_cast<out_el_t>(r[1] >> (in_nbits / 2)) & mask;
        });
      }
    });
  });

  return std::make_pair(hi, lo);
}

NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  // First construct 2 boolean shares.
  // Let
  //   X = [(x0, x1), (x1, x2), (x2, x0)] as input.
  //   Z = (z0, z1, z2) as boolean zero share.
  //
  // Construct M, N as boolean shares,
  //   M = [((x0+x1)^z0, z1), (z1, z2), (z2, (x0+x1)^z0)]
  //   N = [(0,          0),  (0,  x2), (x2, 0         )]
  //
  // That
  //  M + N = (x0+x1)^z0^z1^z2 + x2
  //        = x0 + x1 + x2 = X
  const Type bshr_type =
      makeType<BShrTy>(GetStorageType(field), SizeOf(field) * 8);
  NdArrayRef m(bshr_type, in.shape());
  NdArrayRef n(bshr_type, in.shape());
  DISPATCH_ALL_FIELDS(field, "aby3.msb.split", [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    NdArrayView<shr_t> _in(in);
    NdArrayView<shr_t> _m(m);
    NdArrayView<shr_t> _n(n);

    std::vector<el_t> r0(numel);
    std::vector<el_t> r1(numel);
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);

    pforeach(0, numel, [&](int64_t idx) {
      r0[idx] = r0[idx] ^ r1[idx];
      if (comm->getRank() == 0) {
        const auto& v = _in[idx];
        r0[idx] ^= (v[0] + v[1]);
      }
    });

    // Now, we hold ((x0+x1)^z0, z1, z2) which is stored in r0.

    // 1. rotate k bits
    r1 = comm->rotate<el_t>(r0, "m");                   // send r0 to the previous party, get r1 from the next party.

    pforeach(0, numel, [&](int64_t idx) {
      const auto& v = _in[idx];
      _m[idx][0] = r0[idx];
      _m[idx][1] = r1[idx];
      _n[idx][0] = comm->getRank() == 2 ? v[0] : 0;
      _n[idx][1] = comm->getRank() == 1 ? v[1] : 0;
    });
  });

  // Compute the k-1'th carry bit.
  size_t nbits = SizeOf(field) * 8 - 1;
  auto* sctx = ctx->sctx();

  const Shape shape = {in.numel()};
  auto wrap_m = WrapValue(m);
  auto wrap_n = WrapValue(n);
  {
    // 2. 2k + 16 * 2 bits
    auto carry = carry_a2b(sctx, wrap_m, wrap_n, nbits);

    // Compute the k'th bit.
    //   (m^n)[k] ^ carry
    auto msb = xor_bb(sctx, rshift_b(sctx, xor_bb(sctx, wrap_m, wrap_n), nbits),
                      carry);

    return UnwrapValue(msb);
  }
}

// Reference:
// New Primitives for Actively-Secure MPC over Rings with Applications to
// Private Machine Learning
// P8 IV.D protocol eqz
// https://eprint.iacr.org/2019/599.pdf
//
// Improved Primitives for MPC over Mixed Arithmetic-Binary Circuits
// https://eprint.iacr.org/2020/338.pdf
//
// P0 as the helper/dealer, samples r, deals [r]a and [r]b.
// P1 and P2 get new share [a]
//   P1: [a] = x2 + x3
//   P2: [a] = x1
// reveal c = [a]+[r]a
// check [a] == 0  <=> c == r
// c == r <=> ~c ^ rb  to be bit wise all 1
// then eqz(a) = bit_wise_and(~c ^ rb)
NdArrayRef eqz(KernelEvalContext* ctx, const NdArrayRef& in) {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  const auto field = in.eltype().as<AShrTy>()->field();
  const PtType in_bshr_btype = calcBShareBacktype(SizeOf(field) * 8);
  const auto numel = in.numel();

  NdArrayRef out(makeType<BShrTy>(calcBShareBacktype(8), 8), in.shape());

  size_t pivot;
  prg_state->fillPubl(absl::MakeSpan(&pivot, 1));
  size_t P0 = pivot % 3;
  size_t P1 = (pivot + 1) % 3;
  size_t P2 = (pivot + 2) % 3;

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 2>;
    DISPATCH_UINT_PT_TYPES(in_bshr_btype, "_", [&]() {
      using bshr_el_t = ScalarT;
      std::vector<bshr_el_t> zero_flag_3pc_0(numel);
      std::vector<bshr_el_t> zero_flag_3pc_1(numel);

      // algorithm begins
      if (comm->getRank() == P0) {
        std::vector<ashr_el_t> r(numel);
        prg_state->fillPriv(absl::MakeSpan(r));

        std::vector<ashr_el_t> r_arith_0(numel);
        prg_state->fillPrssPair<ashr_el_t>({}, r_arith_0.data(), numel,
                                           PrgState::GenPrssCtrl::Second);
        std::vector<bshr_el_t> r_bool_0(numel);
        prg_state->fillPrssPair<bshr_el_t>({}, r_bool_0.data(), numel,
                                           PrgState::GenPrssCtrl::Second);

        std::vector<ashr_el_t> r_arith_1(numel);
        pforeach(0, numel, [&](int64_t idx) {
          r_arith_1[idx] = r[idx] - r_arith_0[idx];
        });
        comm->sendAsync<ashr_el_t>(P2, r_arith_1, "r_arith");

        std::vector<bshr_el_t> r_bool_1(numel);
        pforeach(0, numel,
                 [&](int64_t idx) { r_bool_1[idx] = r[idx] ^ r_bool_0[idx]; });
        comm->sendAsync<bshr_el_t>(P2, r_bool_1, "r_bool");

        // back to 3 pc
        // P0 zero_flag = (rb1, rz)
        pforeach(0, numel,
                 [&](int64_t idx) { zero_flag_3pc_0[idx] = r_bool_1[idx]; });

        prg_state->fillPrssPair<bshr_el_t>({}, zero_flag_3pc_1.data(), numel,
                                           PrgState::GenPrssCtrl::Second);

      } else {
        std::vector<ashr_el_t> a_s(numel);
        NdArrayView<ashr_t> _in(in);
        std::vector<ashr_el_t> r_arith(numel);
        std::vector<bshr_el_t> r_bool(numel);

        if (comm->getRank() == P1) {
          pforeach(0, numel,
                   [&](int64_t idx) { a_s[idx] = _in[idx][0] + _in[idx][1]; });

          prg_state->fillPrssPair<ashr_el_t>(r_arith.data(), {}, numel,
                                             PrgState::GenPrssCtrl::First);
          prg_state->fillPrssPair<bshr_el_t>(r_bool.data(), {}, numel,
                                             PrgState::GenPrssCtrl::First);
        } else {
          pforeach(0, numel, [&](int64_t idx) { a_s[idx] = _in[idx][1]; });
          prg_state->fillPrssPair<ashr_el_t>({}, {}, numel,
                                             PrgState::GenPrssCtrl::None);
          prg_state->fillPrssPair<bshr_el_t>({}, {}, numel,
                                             PrgState::GenPrssCtrl::None);
          r_arith = comm->recv<ashr_el_t>(P0, "r_arith");
          r_bool = comm->recv<bshr_el_t>(P0, "r_bool");
        }

        // c in secret share
        std::vector<ashr_el_t> c_s(numel);
        pforeach(0, numel,
                 [&](int64_t idx) { c_s[idx] = r_arith[idx] + a_s[idx]; });

        std::vector<bshr_el_t> zero_flag_2pc(numel);
        if (comm->getRank() == P1) {
          auto c_p = comm->recv<ashr_el_t>(P2, "c_s");

          // reveal c
          pforeach(0, numel,
                   [&](int64_t idx) { c_p[idx] = c_p[idx] + c_s[idx]; });
          // P1 zero_flag = (rz, not(c_p xor [r]b0)^ rz)
          std::vector<bshr_el_t> r_z(numel);
          prg_state->fillPrssPair<bshr_el_t>(r_z.data(), {}, numel,
                                             PrgState::GenPrssCtrl::First);
          pforeach(0, numel, [&](int64_t idx) {
            zero_flag_2pc[idx] = ~(c_p[idx] ^ r_bool[idx]) ^ r_z[idx];
          });

          comm->sendAsync<bshr_el_t>(P2, zero_flag_2pc, "flag_split");

          pforeach(0, numel, [&](int64_t idx) {
            zero_flag_3pc_0[idx] = r_z[idx];
            zero_flag_3pc_1[idx] = zero_flag_2pc[idx];
          });
        } else {
          comm->sendAsync<ashr_el_t>(P1, c_s, "c_s");
          // P1 zero_flag = (not(c_p xor [r]b0)^ rz, rb1)
          pforeach(0, numel,
                   [&](int64_t idx) { zero_flag_3pc_1[idx] = r_bool[idx]; });
          prg_state->fillPrssPair<bshr_el_t>({}, {}, numel,
                                             PrgState::GenPrssCtrl::None);

          auto flag_split = comm->recv<bshr_el_t>(P1, "flag_split");
          pforeach(0, numel, [&](int64_t idx) {
            zero_flag_3pc_0[idx] = flag_split[idx];
          });
        }
      }

      // Reference:
      // Improved Primitives for Secure Multiparty Integer Computation
      // P10 4.1 k-ary
      // https://link.springer.com/chapter/10.1007/978-3-642-15317-4_13
      //
      // if a == 0, zero_flag supposed to be all 1
      // do log k round bit wise and
      // in each round, bit wise split zero_flag in half
      // compute  and(left_half, right_half)
      auto cur_bytes = SizeOf(field) * numel;
      auto cur_bits = cur_bytes * 8;
      auto cur_numel = (unsigned long)numel;
      std::vector<std::byte> round_res_0(cur_bytes);
      std::memcpy(round_res_0.data(), zero_flag_3pc_0.data(), cur_bytes);
      std::vector<std::byte> round_res_1(cur_bytes);
      std::memcpy(round_res_1.data(), zero_flag_3pc_1.data(), cur_bytes);
      while (cur_bits != cur_numel) {
        // byte num per element
        auto byte_num_el = cur_bytes == cur_numel ? 1 : (cur_bytes / numel);
        // byte num of left/right_bits
        auto half_num_bytes =
            cur_bytes == cur_numel ? cur_numel : (cur_bytes / 2);

        // break into left_bits and right_bits
        std::vector<std::vector<std::byte>> left_bits(
            2, std::vector<std::byte>(half_num_bytes));
        std::vector<std::vector<std::byte>> right_bits(
            2, std::vector<std::byte>(half_num_bytes));

        // cur_bits <= 8, use rshift to split in half
        if (cur_bytes == cur_numel) {
          pforeach(0, numel, [&](int64_t idx) {
            left_bits[0][idx] =
                round_res_0[idx] >> (cur_bits / (cur_numel * 2));
            left_bits[1][idx] =
                round_res_1[idx] >> (cur_bits / (cur_numel * 2));
            right_bits[0][idx] = round_res_0[idx];
            right_bits[1][idx] = round_res_1[idx];
          });
          // cur_bits > 8
        } else {
          pforeach(0, numel, [&](int64_t idx) {
            auto cur_byte_idx = idx * byte_num_el;
            for (size_t i = 0; i < (byte_num_el / 2); i++) {
              left_bits[0][cur_byte_idx / 2 + i] =
                  round_res_0[cur_byte_idx + i];
              left_bits[1][cur_byte_idx / 2 + i] =
                  round_res_1[cur_byte_idx + i];
            }
            for (size_t i = 0; i < (byte_num_el / 2); i++) {
              right_bits[0][cur_byte_idx / 2 + i] =
                  round_res_0[cur_byte_idx + byte_num_el / 2 + i];
              right_bits[1][cur_byte_idx / 2 + i] =
                  round_res_1[cur_byte_idx + byte_num_el / 2 + i];
            }
          });
        }

        // compute and(left_half, right_half)
        std::vector<std::byte> r0(half_num_bytes);
        std::vector<std::byte> r1(half_num_bytes);
        prg_state->fillPrssPair<std::byte>(r0.data(), r1.data(), half_num_bytes,
                                           PrgState::GenPrssCtrl::Both);

        // z1 = (x1 & y1) ^ (x1 & y2) ^ (x2 & y1) ^ (r0 ^ r1);
        pforeach(0, half_num_bytes, [&](int64_t idx) {
          r0[idx] = (left_bits[0][idx] & right_bits[0][idx]) ^
                    (left_bits[0][idx] & right_bits[1][idx]) ^
                    (left_bits[1][idx] & right_bits[0][idx]) ^
                    (r0[idx] ^ r1[idx]);
        });

        auto temp = comm->rotate<std::byte>(r0, "andbb");
        r1.assign(temp.begin(), temp.end());

        cur_bytes = cur_bytes == cur_numel ? cur_numel : (cur_bytes / 2);
        cur_bits /= 2;
        round_res_0.assign(r0.begin(), r0.end());
        round_res_1.assign(r1.begin(), r1.end());
      }

      NdArrayView<std::array<std::byte, 2>> _out(out);

      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = round_res_0[idx];
        _out[idx][1] = round_res_1[idx];
      });
    });
  });

  return out;
}

NdArrayRef EqualAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  NdArrayRef out(makeType<AShrTy>(field), lhs.shape());

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using shr_t = std::array<ring2k_t, 2>;
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] - _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] - _rhs[idx][1];
    });
  });

  return eqz(ctx, out);
}

NdArrayRef EqualAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  NdArrayRef out(makeType<AShrTy>(field), lhs.shape());

  auto rank = comm->getRank();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      if (rank == 0) _out[idx][1] -= _rhs[idx];
      if (rank == 1) _out[idx][0] -= _rhs[idx];
    });
    return out;
  });

  return eqz(ctx, out);
}

void CommonTypeV::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

  const auto* lhs_v = lhs.as<Priv2kTy>();
  const auto* rhs_v = rhs.as<Priv2kTy>();

  ctx->setOutput(makeType<AShrTy>(std::max(lhs_v->field(), rhs_v->field())));
}

#define EQ_USE_PRG_STATE 0

// Xor gate for ASS.
NdArrayRef AssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;
        NdArrayView<out_shr_t> _out(out);

        // online.
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          out_shr_t& o = _out[idx];
          o[0] = l[0] ^ r[0];
        });
        return out;
      });
    });
  });
}

// Xor gate for RSS.
NdArrayRef RssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) {

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;
        NdArrayView<out_shr_t> _out(out);

        // online.
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          out_shr_t& o = _out[idx];
          o[0] = l[0] ^ r[0];
          o[1] = l[1] ^ r[1];
        });
        return out;
      });
    });
  });
}

// Xor gate for MSS.
NdArrayRef MssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) {

  const auto* lhs_ty = lhs.eltype().as<BShrTyMss>();
  const auto* rhs_ty = rhs.eltype().as<BShrTyMss>();

  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTyMss>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 3>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 3>;
        NdArrayView<out_shr_t> _out(out);

        // online.
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          out_shr_t& o = _out[idx];
          o[0] = l[0] ^ r[0];
          o[1] = l[1] ^ r[1];
          o[2] = l[2] ^ r[2];
        });
        return out;
      });
    });
  });
}

// And gate for RSS which outputs ASS result (no comunication).
NdArrayRef RssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) {
  #ifdef EQ_USE_PRG_STATE
  auto* prg_state = ctx->getState<PrgState>();
  #endif

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;
        NdArrayView<out_shr_t> _out(out);

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> r0(lhs.numel(), 0);
        std::vector<out_el_t> r1(lhs.numel(), 0);
        #ifdef EQ_USE_PRG_STATE
        prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                                PrgState::GenPrssCtrl::Both);
        #endif

        // online.
        // dxy = dx & dy = (dx0 & dy0) ^ (dx0 & dy1) ^ (dx1 & dy0);
        // r0 is dxy0, r1 is dxy1.
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          out_shr_t& o = _out[idx];
          o[0] = (l[0] & r[0]) ^ (l[0] & r[1]) ^ (l[1] & r[0]) ^
                    (r0[idx] ^ r1[idx]);
        });
        return out;
      });
    });
  });
}

// And gate for MSS which outputs RSS result (no comunication).
NdArrayRef MssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) {
  #ifdef EQ_USE_PRG_STATE
  auto* prg_state = ctx->getState<PrgState>();
  #endif
  auto* comm = ctx->getState<Communicator>();

  const auto* lhs_ty = lhs.eltype().as<BShrTyMss>();
  const auto* rhs_ty = rhs.eltype().as<BShrTyMss>();

  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 3>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> r0(lhs.numel(), 0);
        std::vector<out_el_t> r1(lhs.numel(), 0);
        #ifdef EQ_USE_PRG_STATE
        prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                                PrgState::GenPrssCtrl::Both);
        #endif

        // offline.
        // dxy = dx & dy = (dx0 & dy0) ^ (dx0 & dy1) ^ (dx1 & dy0);
        // r0 is dxy0, r1 is dxy1.
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          r0[idx] = (l[1] & r[1]) ^ (l[1] & r[2]) ^ (l[2] & r[1]) ^
                    (r0[idx] ^ r1[idx]);
        });

        r1 = comm->rotate<out_el_t>(r0, "MssAndBB, offline");  // comm => 1, k

        // online, compute [out] locally.
        NdArrayView<out_shr_t> _out(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];

          out_shr_t& o = _out[idx];
          // z = x & y = (Dx ^ dx) & (Dy ^ dy) = Dx & Dy ^ Dx & dy ^ dx & Dy ^ dxy
          // o[0] = ((comm->getRank() == 0) * (l[0] & r[0])) ^ (l[0] & r[1]) ^ (l[1] & r[0]) ^ r0[idx];   // r0 is dxy0
          // o[1] = ((comm->getRank() == 2) * (l[0] & r[0])) ^ (l[0] & r[2]) ^ (l[2] & r[0]) ^ r1[idx];   // r1 is dxy1
          o[0] = ((l[0] & r[0])) ^ (l[0] & r[1]) ^ (l[1] & r[0]) ^ r0[idx];   // r0 is dxy0
          o[1] = ((l[0] & r[0])) ^ (l[0] & r[2]) ^ (l[2] & r[0]) ^ r1[idx];   // r1 is dxy1
        });
        return out;
      });
    });
  });
}

// And gate for MSS which outputs ASS result (no comunication).
NdArrayRef MssAnd3NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                       const NdArrayRef& op2, const NdArrayRef& op3) {

  // const auto* op1_ty = op1.eltype().as<BShrTyMss>();
  // const auto* op2_ty = op2.eltype().as<BShrTyMss>();
  // const auto* op3_ty = op3.eltype().as<BShrTyMss>();

  // const size_t out_nbits = std::min(std::min(op1_ty->nbits(), op2_ty->nbits()), op3_ty->nbits());
  // const PtType out_btype = calcBShareBacktype(out_nbits);

  // return DISPATCH_UINT_PT_TYPES(op1_ty->getBacktype(), "_", [&]() {
  //   using op1_el_t = ScalarT;
  //   using op1_shr_t = std::array<op1_el_t, 3>;
  //   NdArrayView<op1_shr_t> _op1(op1);

  // return DISPATCH_UINT_PT_TYPES(op2_ty->getBacktype(), "_", [&]() {
  //   using op2_el_t = ScalarT;
  //   using op2_shr_t = std::array<op2_el_t, 3>;
  //   NdArrayView<op2_shr_t> _op2(op2);

  // return DISPATCH_UINT_PT_TYPES(op3_ty->getBacktype(), "_", [&]() {
  //   using op3_el_t = ScalarT;
  //   using op3_shr_t = std::array<op3_el_t, 3>;
  //   NdArrayView<op3_shr_t> _op3(op3);

  // return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
  //   using out_el_t = ScalarT;
  //   // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
  //   using out_shr_t = std::array<out_el_t, 2>;        // use MSS container to hold ass result

    auto lo_res = MssAnd2NoComm(ctx, op1, op2);
    auto hi_res = ResharingMss2Rss(ctx, op3);
    auto out = RssAnd2NoComm(ctx, lo_res, hi_res);
    
    return out;
  // });
  // });
  // });
  // });
}

// And gate for MSS which outputs ASS result (no comunication).
NdArrayRef MssAnd4NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                       const NdArrayRef& op2, const NdArrayRef& op3, const NdArrayRef& op4) {

  // const auto* op1_ty = op1.eltype().as<BShrTyMss>();
  // const auto* op2_ty = op2.eltype().as<BShrTyMss>();
  // const auto* op3_ty = op3.eltype().as<BShrTyMss>();
  // const auto* op4_ty = op4.eltype().as<BShrTyMss>();

  // const size_t out_nbits = std::min(std::min(op1_ty->nbits(), op2_ty->nbits()), std::min(op3_ty->nbits(), op4_ty->nbits()));
  // const PtType out_btype = calcBShareBacktype(out_nbits);

  // return DISPATCH_UINT_PT_TYPES(op1_ty->getBacktype(), "_", [&]() {
  //   using op1_el_t = ScalarT;
  //   using op1_shr_t = std::array<op1_el_t, 3>;
  //   NdArrayView<op1_shr_t> _op1(op1);

  // return DISPATCH_UINT_PT_TYPES(op2_ty->getBacktype(), "_", [&]() {
  //   using op2_el_t = ScalarT;
  //   using op2_shr_t = std::array<op2_el_t, 3>;
  //   NdArrayView<op2_shr_t> _op2(op2);

  // return DISPATCH_UINT_PT_TYPES(op3_ty->getBacktype(), "_", [&]() {
  //   using op3_el_t = ScalarT;
  //   using op3_shr_t = std::array<op3_el_t, 3>;
  //   NdArrayView<op3_shr_t> _op3(op3);

  // return DISPATCH_UINT_PT_TYPES(op4_ty->getBacktype(), "_", [&]() {
  //   using op4_el_t = ScalarT;
  //   using op4_shr_t = std::array<op4_el_t, 3>;
  //   NdArrayView<op4_shr_t> _op4(op4);

  // return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
  //   using out_el_t = ScalarT;
  //   // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
  //   using out_shr_t = std::array<out_el_t, 2>;        // use MSS container to hold ass result

    auto lo_res = MssAnd2NoComm(ctx, op1, op2);
    auto hi_res = MssAnd2NoComm(ctx, op3, op4);
    auto out = RssAnd2NoComm(ctx, lo_res, hi_res);
    
    return out;
  // });
  // });
  // });
  // });
  // });
}

// Resharing protocol from RSS to MSS.
NdArrayRef ResharingRss2Mss(KernelEvalContext* ctx, const NdArrayRef& in) {
  #ifdef EQ_USE_PRG_STATE
  auto* prg_state = ctx->getState<PrgState>();
  #endif
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTyMss>(out_btype, out_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
      using in_el_t = ScalarT;
      using in_shr_t = std::array<in_el_t, 2>;
      NdArrayView<in_shr_t> _in(in);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 3>;
        NdArrayView<out_shr_t> _out(out);

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> r0(in.numel(), 0);
        std::vector<out_el_t> r1(in.numel(), 0);
        #ifdef EQ_USE_PRG_STATE
        prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                                PrgState::GenPrssCtrl::Both);
        #endif

        // online.
        pforeach(0, in.numel(), [&](int64_t idx) {
          in_shr_t& i = _in[idx];
          out_shr_t& o = _out[idx];
          o[1] = r0[idx];
          o[2] = r1[idx];
          r0[idx] = i[0] ^ r0[idx];
        });

        r0 = comm->rotateR<out_el_t>(r0, "Resharing RSS to MSS, online");  // comm => 1, k

        pforeach(0, in.numel(), [&](int64_t idx) {
          in_shr_t& i = _in[idx];
          out_shr_t& o = _out[idx];

          o[0] = i[0] ^ i[1] ^ o[1] ^ o[2] ^ r0[idx];
        });
        return out;
      });
    });
}

// Resharing protocol from ASS to RSS.
// using RSS container to hold ASS.
NdArrayRef ResharingAss2Rss(KernelEvalContext* ctx, const NdArrayRef& in) {
  #ifdef EQ_USE_PRG_STATE
  auto* prg_state = ctx->getState<PrgState>();
  #endif
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
      using in_el_t = ScalarT;
      using in_shr_t = std::array<in_el_t, 2>;
      NdArrayView<in_shr_t> _in(in);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;
        NdArrayView<out_shr_t> _out(out);

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> r0(in.numel(), 0);
        std::vector<out_el_t> r1(in.numel(), 0);
        #ifdef EQ_USE_PRG_STATE
        prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                                PrgState::GenPrssCtrl::Both);
        #endif

        // online.
        pforeach(0, in.numel(), [&](int64_t idx) {
          in_shr_t& i = _in[idx];
          out_shr_t& o = _out[idx];
          o[0] = i[0] ^ r0[idx] ^ r1[idx];
          r0[idx] = i[0] ^ r0[idx] ^ r1[idx];
        });

        // TODO: not safe. should add a mask to r1.
        r0 = comm->rotate<out_el_t>(r0, "Resharing ASS to RSS, online");  // comm => 1, k

        pforeach(0, in.numel(), [&](int64_t idx) {
          out_shr_t& o = _out[idx];

          o[1] = r0[idx];
        });
        return out;
      });
    });
}

// Resharing protocol from ASS to MSS.
// using RSS container to hold ASS.
NdArrayRef ResharingAss2Mss(KernelEvalContext* ctx, const NdArrayRef& in) {
  #ifdef EQ_USE_PRG_STATE
  auto* prg_state = ctx->getState<PrgState>();
  #endif
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTyMss>(out_btype, out_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
      using in_el_t = ScalarT;
      using in_shr_t = std::array<in_el_t, 2>;
      NdArrayView<in_shr_t> _in(in);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 3>;
        NdArrayView<out_shr_t> _out(out);

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> r0(in.numel());
        std::vector<out_el_t> r1(in.numel());
        #ifdef EQ_USE_PRG_STATE
        prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                                PrgState::GenPrssCtrl::Both);
        #endif

        // online.
        pforeach(0, in.numel(), [&](int64_t idx) {
          in_shr_t& i = _in[idx];
          out_shr_t& o = _out[idx];
          o[1] = r0[idx];
          o[2] = r1[idx];
          r0[idx] = i[0] ^ r0[idx];
          r1[idx] = i[0];
        });

        // TODO: not safe. should add a mask to r1.
        r0 = comm->rotateR<out_el_t>(r0, "Resharing ASS to MSS, online, message 1");  // comm => 1, k
        r1 = comm->rotate<out_el_t>(r1, "Resharing ASS to MSS, online, message 2");  // comm => 1, k

        pforeach(0, in.numel(), [&](int64_t idx) {
          in_shr_t& i = _in[idx];
          out_shr_t& o = _out[idx];

          o[0] = i[0] ^ o[1] ^ o[2] ^ r0[idx] ^ r1[idx];
        });
        return out;
      });
    });
}

// Resharing protocol from MSS to RSS.
NdArrayRef ResharingMss2Rss(KernelEvalContext* ctx, const NdArrayRef& in) {

  const auto* in_ty = in.eltype().as<BShrTyMss>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
      using in_el_t = ScalarT;
      using in_shr_t = std::array<in_el_t, 3>;
      NdArrayView<in_shr_t> _in(in);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;
        NdArrayView<out_shr_t> _out(out);

        // online.
        pforeach(0, in.numel(), [&](int64_t idx) {
          in_shr_t& i = _in[idx];
          out_shr_t& o = _out[idx];
          o[0] = i[0] ^ i[1];
          o[1] = i[0] ^ i[2];
        });

        return out;
      });
    });
}

// Resharing protocol from RSS to ASS.
NdArrayRef ResharingRss2Ass(KernelEvalContext* ctx, const NdArrayRef& in) {

  const auto* in_ty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());

    return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
      using in_el_t = ScalarT;
      using in_shr_t = std::array<in_el_t, 2>;
      NdArrayView<in_shr_t> _in(in);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;
        NdArrayView<out_shr_t> _out(out);

        // online.
        pforeach(0, in.numel(), [&](int64_t idx) {
          in_shr_t& i = _in[idx];
          out_shr_t& o = _out[idx];
          o[0] = i[0];
          o[1] = 0;
        });

        return out;
      });
    });
}

uint64_t lshift(uint64_t x, size_t shift) {
  return x << shift;
}

uint64_t select(uint64_t x, uint64_t mask, uint64_t offset, size_t idx) {
  return (x & (mask << (idx * offset))) << ((3 - idx) * offset);
}

NdArrayRef MsbA2BMultiFanIn(KernelEvalContext* ctx, const NdArrayRef& in) {
  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->getState<Communicator>();
  #ifdef EQ_USE_PRG_STATE
  auto* prg_state = ctx->getState<PrgState>();
  #endif

  #define EQ_U64(x) static_cast<uint64_t>(x)

  // First construct 2 boolean shares.
  // Let
  //   X = [(x0, x1), (x1, x2), (x2, x0)] as input.
  //   Z = (z0, z1, z2) as boolean zero share.
  //
  // Construct edabitsB = [(ebb0, ebb1), (ebb1, ebb2), (ebb2, ebb0)] as boolean shares,
  //   edabitsA = [(eba0, eba1), (eba1, eba2), (eba2, eba0)] as arithmetic shares,
  //   where edabitsA = edabitsB.
  //
  // Open mask = x - edabitsA.
  //
  // That
  //  mask + edabitsB = x0 + x1 + x2 = X
  const Type rss_ashr_type =
      makeType<AShrTy>(field);
  const Type rss_bshr_type =
      makeType<BShrTy>(GetStorageType(field), SizeOf(field) * 8);
  const Type mss_bshr_type =
      makeType<BShrTyMss>(GetStorageType(field), SizeOf(field) * 8);
  // NdArrayRef edabitsA(rss_ashr_type, in.shape());
  // NdArrayRef edabitsB(mss_bshr_type, in.shape());

  NdArrayRef m(mss_bshr_type, in.shape());
  NdArrayRef n(mss_bshr_type, in.shape());

  NdArrayRef p(mss_bshr_type, in.shape());
  NdArrayRef g(mss_bshr_type, in.shape());
  NdArrayRef out(rss_bshr_type, in.shape());
  return DISPATCH_ALL_FIELDS(field, "aby3.msb.split", [&]() {
    using el_t = ring2k_t;
    using rss_shr_t = std::array<el_t, 2>;
    using mss_shr_t = std::array<el_t, 3>;

    NdArrayView<rss_shr_t> _in(in);           // rss
    
    // NdArrayView<rss_shr_t> _eba(edabitsA);    
    // NdArrayView<mss_shr_t> _ebb(edabitsB);    

    NdArrayView<mss_shr_t> _m(m);
    NdArrayView<mss_shr_t> _n(n);

    NdArrayView<rss_shr_t> _out(out);

    /**
     * 1. Convert RSS-shared x into MSS-shared m (Dm, RSS(dm)) and n (Dn, RSS(dn)).
    */
    // generate (compressed) correlated randomness: ((dm0, dm1), (dm1, dn2), (dn2, dm0)). 
    std::vector<el_t> r0(numel, 0);
    std::vector<el_t> r1(numel, 0);
    #ifdef EQ_USE_PRG_STATE
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);
    #endif

    // copy the correlated randomness into m and n
    pforeach(0, numel, [&](int64_t idx) {
      if (comm->getRank() == 0) 
      {
        // Wait for x2 ^ dn2 from P1.
        _m[idx][1] = r0[idx];                               // dm0
        _m[idx][2] = r1[idx];                               // dm1
        r1[idx] ^= r0[idx] ^ (_in[idx][0] + _in[idx][1]);     
        _m[idx][0] = r1[idx];                               // Dm = (x0 + x1) ^ dm0 ^ dm1

        _n[idx][1] = 0;
        _n[idx][2] = 0;
      } 
      else if (comm->getRank() == 1) 
      {
        // Wait for Dm from P0.
        _m[idx][1] = r0[idx];                               // dm1
        _n[idx][2] = r1[idx];                               // dn2
        r1[idx] ^= _in[idx][1];                             // dn2 ^ x2
        _n[idx][0] = r1[idx];                               // Dn = x2 ^ dn2

        _m[idx][2] = 0;
        _n[idx][1] = 0;
      }
      else
      {
        // Wait for Dm from P0.
        _n[idx][1] = r0[idx];                               // dn2
        _m[idx][2] = r1[idx];                               // dm0
        _n[idx][0] = _in[idx][0] ^ r0[idx];                 // Dn = x2 ^ dn2

        _m[idx][1] = 0;
        _n[idx][2] = 0;
      }
    });

    // rotate k bits
    // r0 = comm->rotate<el_t>(r1, "MsbA2B, special resharing from ASS to MSS.");   // send r1 to the previous party, get r0 from the next party.
    r0 = comm->bcast<el_t>(r1, 0, "MsbA2B, special resharing from ASS to MSS, broadcast Dm");
    if (comm->getRank() == 0) 
    {
      r0 = comm->recv<el_t>(1, "MsbA2B, special resharing from ASS to MSS, get dn2");
    }
    else if (comm->getRank() == 1) 
    {
      comm->sendAsync<el_t>(0, r1, "MsbA2B, special resharing from ASS to MSS, send dn2");
    }

    // compute external value Dm, Dn
    pforeach(0, numel, [&](int64_t idx) {
      if (comm->getRank() == 0) 
      {
        _n[idx][0] = r0[idx];                               // Dn = x2 + dn2
      } 
      else if (comm->getRank() == 1) 
      {
        _m[idx][0] = r0[idx];                              // Dm = (x0 + x1) ^ dm0 ^ dm1
      }
      else
      {
        _m[idx][0] = r0[idx];                            
      }
    });

    if (comm->getRank() == 0) std::cout << "PPA: m " << _m[0][0] << " " << _m[1][0] << std::endl;
    if (comm->getRank() == 0) std::cout << "PPA: n " << _n[0][0] << " " << _n[1][0] << std::endl;
    // std::cout << "PPA: mask " << mask[0] << " " << mask[1] << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: compute mask." << std::endl;

    // 4. generate signal p and g.
    NdArrayView<mss_shr_t> _p(p);
    NdArrayView<mss_shr_t> _g(g);

    auto sig_g_rss = MssAnd2NoComm(ctx, m, n);
    auto sig_g_mss = ResharingAss2Mss(ctx, ResharingRss2Ass(ctx, sig_g_rss));
    NdArrayView<mss_shr_t> _g_mss(sig_g_mss);
    if (comm->getRank() == 0) std::cout << "PPA: sig_g_mss " << _g_mss[0][0] << " " << _g_mss[1][0] << std::endl;
    pforeach(0, numel, [&](int64_t idx) {
      _p[idx][0] = _m[idx][0] ^ _n[idx][0];
      _p[idx][1] = _m[idx][1] ^ _n[idx][1];
      _p[idx][2] = _m[idx][2] ^ _n[idx][2];
      _g[idx][0] = _g_mss[idx][0];
      _g[idx][1] = _g_mss[idx][1];
      _g[idx][2] = _g_mss[idx][2];
    });
    // assert(_p[0][1] == 0 && _p[1][1] == 0 && _p[2][1] == 0 && _p[0][2] == 0 && _p[1][2] == 0 && _p[2][2] == 0);
    // assert(_g[0][1] == 0 && _g[1][1] == 0 && _g[2][1] == 0 && _g[0][2] == 0 && _g[1][2] == 0 && _g[2][2] == 0);

    // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;

    // 5. PPA.
    // we dont use the carryout circuit from aby 2.0. By limitting p's msb to be 1 and g's msb to be 0,
    // we could build a simpler carryout circuit.
    size_t nbits = SizeOf(field) * 8 - 1;
    size_t k = nbits;
    
    pforeach(0, numel, [&](int64_t idx) {
      _out[idx][0] = (_p[idx][0] ^ _p[idx][1]) >> nbits;
      _out[idx][1] = (_p[idx][0] ^ _p[idx][2]) >> nbits;
      _p[idx][0] = EQ_U64((EQ_U64(1) << nbits) | EQ_U64(_p[idx][0]));    
      _p[idx][1] = EQ_U64(((EQ_U64(1) << nbits) - 1) & EQ_U64(_p[idx][1]));
      _p[idx][2] = EQ_U64(((EQ_U64(1) << nbits) - 1) & EQ_U64(_p[idx][2]));
      _g[idx][0] = EQ_U64(((EQ_U64(1) << nbits) - 1) & EQ_U64(_g[idx][0]));
      _g[idx][1] = EQ_U64(((EQ_U64(1) << nbits) - 1) & EQ_U64(_g[idx][1]));
      _g[idx][2] = EQ_U64(((EQ_U64(1) << nbits) - 1) & EQ_U64(_g[idx][2]));
    });

    // if (comm->getRank() == 0) std::cout << "PPA: msb(m) ^ msb(n) " << _out[0][0] << " " << _out[1][0] << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;

    // uint64_t ONLY. do not use el_t as it involves lshift error.
    std::array<uint64_t, 3> bit_mask = {
      static_cast<uint64_t>(0x1111111111111111),
      static_cast<uint64_t>(0x0008000800080008),
      static_cast<uint64_t>(0x0000000000008000)
    };
    std::array<uint64_t, 3> bit_offset = {
      1, 4, 16 
    };

    // if (comm->getRank() == 0) std::cout << "PPA: initialize." << std::endl;

    size_t lev = 0;
    while (k > 1) 
    {
      if (comm->getRank() == 0) std::cout << "PPA: " << k << " bits, level " << lev << std::endl;
      NdArrayRef pops[4];
      NdArrayRef gops[4];

      uint64_t debug_offset = static_cast<uint64_t>(1) << ((1 << (2 * lev + 2)) - 1);
      // if (comm->getRank() == 0) std::cout << debug_offset << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal_p." << _p[0][0] << " " << _p[1][0] << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal_g." << _g[0][0] << " " << _g[1][0] << std::endl;

      // for the i-th bit in a four-bits element vector, select the i-the bit and rotate it to the msb.
      // for example: when i = 1, sample (1)000 (0)000 from 10(1)1 11(0)1. 
      for (int i = 0; i < 4; i++) {
        pops[i] = NdArrayRef(mss_bshr_type, in.shape());
        NdArrayView<mss_shr_t> _pops(pops[i]);
        gops[i] = NdArrayRef(mss_bshr_type, in.shape());
        NdArrayView<mss_shr_t> _gops(gops[i]);

        // taking the i-th bit of p and g.
        // 1 << (4 ** （lev + 1) - 1) = 1 << ((1 << (2 * lev + 2)) - 1)
        pforeach(0, numel, [&](int64_t idx) {
          _pops[idx][0] = select(EQ_U64(_p[idx][0]), EQ_U64(bit_mask[lev]), EQ_U64(bit_offset[lev]), i);
          _pops[idx][1] = select(EQ_U64(_p[idx][1]), EQ_U64(bit_mask[lev]), EQ_U64(bit_offset[lev]), i);
          _pops[idx][2] = select(EQ_U64(_p[idx][2]), EQ_U64(bit_mask[lev]), EQ_U64(bit_offset[lev]), i);
          _gops[idx][0] = select(EQ_U64(_g[idx][0]), EQ_U64(bit_mask[lev]), EQ_U64(bit_offset[lev]), i);
          _gops[idx][1] = select(EQ_U64(_g[idx][1]), EQ_U64(bit_mask[lev]), EQ_U64(bit_offset[lev]), i);
          _gops[idx][2] = select(EQ_U64(_g[idx][2]), EQ_U64(bit_mask[lev]), EQ_U64(bit_offset[lev]), i);

          // assert(_pops[idx][0] % debug_offset == 0 && _pops[idx][1] % debug_offset == 0 && _pops[idx][2] % debug_offset == 0);
          // assert(_gops[idx][0] % debug_offset == 0 && _gops[idx][1] % debug_offset == 0 && _gops[idx][2] % debug_offset == 0);
        });
      }

      // if (comm->getRank() == 0) std::cout << "PPA: select bits in p and g. " << std::endl;

      auto p_res = MssAnd4NoComm(ctx, pops[0], pops[1], pops[2], pops[3]);
      auto p_final = ResharingAss2Mss(ctx, p_res);
      auto g_res_3 = ResharingRss2Ass(ctx, ResharingMss2Rss(ctx, gops[3]));
      auto g_res_2 = ResharingRss2Ass(ctx, MssAnd2NoComm(ctx, gops[2], pops[3]));
      auto g_res_1 = MssAnd3NoComm(ctx, gops[1], pops[3], pops[2]);
      auto g_res_0 = MssAnd4NoComm(ctx, gops[0], pops[3], pops[2], pops[1]);
      auto g_combined = AssXor2(ctx, AssXor2(ctx, g_res_0, g_res_1), AssXor2(ctx, g_res_2, g_res_3));
      // online communication
      auto g_final = ResharingAss2Mss(ctx, g_combined);

      // if (comm->getRank() == 0) std::cout << "PPA: protocol flow. " << std::endl;

      NdArrayView<mss_shr_t> _pf(p_final);
      NdArrayView<mss_shr_t> _gf(g_final);
      pforeach(0, numel, [&](int64_t idx) {
        _p[idx][0] = _pf[idx][0];
        _p[idx][1] = _pf[idx][1];
        _p[idx][2] = _pf[idx][2];
        _g[idx][0] = _gf[idx][0];
        _g[idx][1] = _gf[idx][1];
        _g[idx][2] = _gf[idx][2];
      });

      // assert(_p[0][1] == 0 && _p[1][1] == 0 && _p[2][1] == 0 && _p[0][2] == 0 && _p[1][2] == 0 && _p[2][2] == 0);
      // assert(_g[0][1] == 0 && _g[1][1] == 0 && _g[2][1] == 0 && _g[0][2] == 0 && _g[1][2] == 0 && _g[2][2] == 0);
      // if (comm->getRank() == 0) std::cout << "PPA: p_final " << _pf[0][0] << " " << _pf[1][0] << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: g_final " << _gf[0][0] << " " << _gf[1][0] << std::endl;

      // if (comm->getRank() == 0) std::cout << "PPA: epilogue. " << std::endl;

      lev += 1;
      k = k / 4;
    }

    auto g_rss = ResharingMss2Rss(ctx, g);
    NdArrayView<rss_shr_t> _g_rss(g_rss);
    pforeach(0, numel, [&](int64_t idx) {

      _out[idx][0] ^= (static_cast<uint64_t>(_g_rss[idx][0]) >> nbits);
      _out[idx][1] ^= (static_cast<uint64_t>(_g_rss[idx][1]) >> nbits);
    });
    // if (comm->getRank() == 0) std::cout << "PPA: carry " << _g_rss[0][0] << " " << _g_rss[1][0] << std::endl;

    return out;
  });  
}

}  // namespace spu::mpc::aby3
