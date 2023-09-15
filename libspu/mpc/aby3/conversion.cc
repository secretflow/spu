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

#include "yacl/utils/platform_utils.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"

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

    r1 = comm->rotate<el_t>(r0, "m");

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
    auto carry = carry_a2b(sctx, wrap_m, wrap_n, nbits);

    // Compute the k'th bit.
    //   (m^n)[k] ^ carry
    auto msb = xor_bb(sctx, rshift_b(sctx, xor_bb(sctx, wrap_m, wrap_n), nbits),
                      carry);

    return UnwrapValue(msb);
  }
}

}  // namespace spu::mpc::aby3
