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

#include "libspu/mpc/albo/conversion.h"

#include <functional>
#include <iostream>
#include <utility>
#include <atomic>

#include "yacl/utils/platform_utils.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/albo/type.h"
#include "libspu/mpc/albo/value.h"
#include "libspu/mpc/albo/mss_utils.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

#define EQ_PACK_SINGLE_BIT

#define EQ_TEST_PPA

namespace spu::mpc::albo {

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
  #ifdef EQ_TEST_PPA
  return PPATest(ctx, in);
  #else
    #ifdef EQ_TEST_AND_TREE
    #else
    return A2BMultiFanIn(ctx, in);
    #endif
  #endif
}

NdArrayRef B2ASelector::proc(KernelEvalContext* ctx,
                             const NdArrayRef& in) const {
  return B2AMultiFanIn(ctx, in);
}

// Alkaid's B2A. RSS input, RSS output.
// Let P0, P1 sample rb1, P1, P2 sample rb2, ra2, P0, P2 sample ra0.
// P1 computes ra1 = rb0 xor rb1 - ra2 and sends it to P0.
// Now, (0, (0, rb1, rb2)) and (0, (0, ra1, ra2)) come to MRSS.
// We invoke a PPA to compute z = x + r where r = rb0 xor rb1.
// Then, we reveal z to P0 and P2, who compute m = z + ra0 and send it to P1.
// We get arithmetic MRSS (m, (ra0, ra1, ra2)) as m = z + ra0 = x + ra1 + ra2 + ra0.
// Online: log2(k) + 1 rounds.
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
    DISPATCH_ALL_FIELDS(field, [&]() {
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

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using bshr_t = std::array<ScalarT, 2>;
    NdArrayView<bshr_t> _in(in);

    DISPATCH_ALL_FIELDS(field, [&]() {
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
// Alkaid paper algorithm reference.
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
    DISPATCH_ALL_FIELDS(field, [&]() {
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

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using bshr_el_t = ScalarT;
    using bshr_t = std::array<bshr_el_t, 2>;
    NdArrayView<bshr_t> _in(in);

    DISPATCH_ALL_FIELDS(field, [&]() {
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

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_backtype, [&]() {
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

[[maybe_unused]] std::pair<NdArrayRef, NdArrayRef> bit_split_mss(
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

  const auto* in_ty = in.eltype().as<BShrTyMss>();
  const size_t in_nbits = in_ty->nbits();
  SPU_ENFORCE(in_nbits != 0, "in_nbits={}", in_nbits);
  const size_t out_nbits = in_nbits / 2 + in_nbits % 2;
  const auto out_backtype = calcBShareBacktype(out_nbits);
  const auto out_type = makeType<BShrTyMss>(out_backtype, out_nbits);

  NdArrayRef lo(out_type, in.shape());
  NdArrayRef hi(out_type, in.shape());

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 3>;
    NdArrayView<in_shr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_backtype, [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 3>;

      NdArrayView<out_shr_t> _lo(lo);
      NdArrayView<out_shr_t> _hi(hi);

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
            int shift = 1ull << k;

            r[0] = (r[0] & keep) ^ ((r[0] >> shift) & move) ^
                   ((r[0] & move) << shift);
            r[1] = (r[1] & keep) ^ ((r[1] >> shift) & move) ^
                   ((r[1] & move) << shift);
            r[2] = (r[2] & keep) ^ ((r[2] >> shift) & move) ^
                   ((r[2] & move) << shift);

            // assert(r[1] == 0 && r[2] == 0);
          }
          in_el_t mask = (in_el_t(1) << (in_nbits / 2)) - 1;
          _lo[idx][0] = static_cast<out_el_t>(r[0]) & mask;
          _hi[idx][0] = static_cast<out_el_t>(r[0] >> (in_nbits / 2)) & mask;
          _lo[idx][1] = static_cast<out_el_t>(r[1]) & mask;
          _hi[idx][1] = static_cast<out_el_t>(r[1] >> (in_nbits / 2)) & mask;
          _lo[idx][2] = static_cast<out_el_t>(r[2]) & mask;
          _hi[idx][2] = static_cast<out_el_t>(r[2] >> (in_nbits / 2)) & mask;

          // assert(_lo[idx][1] == 0 && _lo[idx][2] == 0);
          // assert(_hi[idx][1] == 0 && _hi[idx][2] == 0);
        });
    });
  });

  return std::make_pair(hi, lo);
}

NdArrayRef bit_interleave_ass(
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
  SPU_ENFORCE(in_nbits != 0, "in_nbits={}", in_nbits);
  const size_t out_nbits = in_nbits;
  const auto out_backtype = calcBShareBacktype(out_nbits);
  const auto out_type = makeType<BShrTy>(out_backtype, out_nbits);

  NdArrayRef out(out_type, in.shape());

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_backtype, [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayView<out_shr_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = static_cast<out_el_t>(_in[idx][0]);
        // algorithm:
        //      0101010101010101
        // swap  ^^  ^^  ^^  ^^
        //      0011001100110011
        // swap   ^^^^    ^^^^
        //      0000111100001111
        // swap     ^^^^^^^^
        //      0000000011111111
        for (int k = Log2Ceil(in_nbits) - 2; k >= 0; k--) {
          auto keep = static_cast<in_el_t>(kKeepMasks[k]);
          auto move = static_cast<in_el_t>(kSwapMasks[k]);
          int shift = 1ull << k;
            _out[idx][0] = (_out[idx][0] & keep) ^ ((_out[idx][0] >> shift) & move) ^
                    ((_out[idx][0] & move) << shift);
        }
      });
    });
  });
  return out;
}

NdArrayRef bit_interleave_mss(
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

  const auto* in_ty = in.eltype().as<BShrTyMss>();
  const size_t in_nbits = in_ty->nbits();
  SPU_ENFORCE(in_nbits != 0 && in_nbits % 2 == 0, "in_nbits={}", in_nbits);
  const size_t out_nbits = in_nbits;
  const auto out_backtype = calcBShareBacktype(out_nbits);
  const auto out_type = makeType<BShrTyMss>(out_backtype, out_nbits);

  NdArrayRef out(out_type, in.shape());

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 3>;
    NdArrayView<in_shr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_backtype, [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 3>;

      NdArrayView<out_shr_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = static_cast<out_el_t>(_in[idx][0]);
        _out[idx][1] = static_cast<out_el_t>(_in[idx][1]);
        _out[idx][2] = static_cast<out_el_t>(_in[idx][2]);
        // algorithm:
        //      0101010101010101
        // swap  ^^  ^^  ^^  ^^
        //      0011001100110011
        // swap   ^^^^    ^^^^
        //      0000111100001111
        // swap     ^^^^^^^^
        //      0000000011111111
        for (int k = Log2Ceil(in_nbits) - 2; k >= 0; k--) {
          auto keep = static_cast<in_el_t>(kKeepMasks[k]);
          auto move = static_cast<in_el_t>(kSwapMasks[k]);
          int shift = 1ull << k;
            _out[idx][0] = (_out[idx][0] & keep) ^ ((_out[idx][0] >> shift) & move) ^
                    ((_out[idx][0] & move) << shift);
            _out[idx][1] = (_out[idx][1] & keep) ^ ((_out[idx][1] >> shift) & move) ^
                    ((_out[idx][1] & move) << shift);
            _out[idx][2] = (_out[idx][2] & keep) ^ ((_out[idx][2] >> shift) & move) ^
                    ((_out[idx][2] & move) << shift);

            // assert(r[1] == 0 && r[2] == 0);
        }
      });
    });
  });
  return out;
}

NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  // size_t numel = in.numel();
  // size_t elsize = in.elsize();

  // NdArrayRef res(makeType<BShrTy>(calcBShareBacktype(1), 1), in.shape());

  // for (size_t p = 0; p < 3; p++)
  // {
  //   size_t offset = p * numel / 3;
  //   size_t op_numel = p == 2 ? numel - offset : numel / 3;
  //   NdArrayRef op(in.eltype(), in.shape());
  //   auto src_ptr = (uint8_t *) in.cbegin().getRawPtr() + offset * elsize;
  //   auto res_ptr = (uint8_t *) res.cbegin().getRawPtr() + offset;
  //   auto* op_ptr = static_cast<std::byte*>(op.data());
  //   std::memcpy(op_ptr, src_ptr, elsize * op_numel);
  //   auto tmp = MsbA2BMultiFanIn(ctx, op, p);
  //   auto* tmp_ptr = static_cast<std::byte*>(tmp.data());
  //   std::memcpy(res_ptr, tmp_ptr, op_numel);
  // }

  return MsbA2BMultiFanIn(ctx, in);
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

  size_t pivot;
  prg_state->fillPubl(absl::MakeSpan(&pivot, 1));
  size_t P0 = pivot % 3;
  size_t P1 = (pivot + 1) % 3;
  size_t P2 = (pivot + 2) % 3;

  NdArrayRef out(makeType<BShrTy>(calcBShareBacktype(1), 1), in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 2>;
    DISPATCH_UINT_PT_TYPES(in_bshr_btype, [&]() {
      using bshr_el_t = ScalarT;
      NdArrayRef zero_flag(makeType<BShrTy>(in_bshr_btype, SizeOf(field) * 8), in.shape());
      NdArrayView<std::array<bshr_el_t, 2>> _zf(zero_flag);

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
        std::vector<bshr_el_t> temp(numel);
        prg_state->fillPrssPair<bshr_el_t>({}, temp.data(), numel,
                                           PrgState::GenPrssCtrl::Second);

        pforeach(0, numel,
                 [&](int64_t idx) { _zf[idx][0] = r_bool_1[idx], _zf[idx][1] = temp[idx];});
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
            _zf[idx][0] = r_z[idx];
            _zf[idx][1] = zero_flag_2pc[idx];
          });
        } else {
          comm->sendAsync<ashr_el_t>(P1, c_s, "c_s");
          // P1 zero_flag = (not(c_p xor [r]b0)^ rz, rb1)
          pforeach(0, numel,
                   [&](int64_t idx) { _zf[idx][1] = r_bool[idx]; });

          auto flag_split = comm->recv<bshr_el_t>(P1, "flag_split");
          pforeach(0, numel, [&](int64_t idx) {
            _zf[idx][0] = flag_split[idx];
          });
        }
      }

      // Alkaid.
      zero_flag = ResharingRss2Mss(ctx, zero_flag);      
      auto cur_bits = SizeOf(field) * 8;
      while (cur_bits > 1)
      {
        NdArrayRef op[4];
        std::tie(op[0], op[2]) = bit_split_mss(zero_flag);
        std::tie(op[0], op[1]) = bit_split_mss(op[0]);
        std::tie(op[2], op[3]) = bit_split_mss(op[2]);
        cur_bits /= 4;
        if (cur_bits > 1) zero_flag = ResharingAss2Mss(ctx, MssAnd4NoComm(ctx, op[0], op[1], op[2], op[3]));
        else out = ResharingAss2Rss(ctx, MssAnd4NoComm(ctx, op[0], op[1], op[2], op[3]));
      }
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

  DISPATCH_ALL_FIELDS(field, [&]() {
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

  DISPATCH_ALL_FIELDS(field, [&]() {
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

  ctx->pushOutput(makeType<AShrTy>(std::max(lhs_v->field(), rhs_v->field())));
}

template<typename NativeCppType, size_t share_number=2>
NdArrayRef lshift(const NdArrayRef& in, size_t shift) {
  NdArrayRef out = in.clone();

  using el_t = NativeCppType;
  using shr_t = std::array<el_t, share_number>;
  NdArrayView<shr_t> _out(out);

  pforeach(0, in.numel(), [&](int64_t idx) {
    for (size_t i = 0; i < share_number; i++) {
      _out[idx][i] <<= shift;
    }
  });
  return out;
}


uint64_t select(uint64_t x, uint64_t mask, uint64_t offset, size_t idx) {
  return (x & (mask << (idx * offset))) << ((3 - idx) * offset);
}

// Select substring of x corresponding to mask and lshift it stride bits.
uint64_t SelectAndRotate(uint64_t x, uint64_t mask, uint64_t stride) {
  return (x & mask) << stride;
}

// given lo and hi, output hi | lo.
NdArrayRef pack_2_bitvec_ass(const NdArrayRef& lo, const NdArrayRef& hi) {
  const auto* lo_ty = lo.eltype().as<BShrTy>();
  const auto* hi_ty = hi.eltype().as<BShrTy>();

  // assert(lo_ty->nbits() == hi_ty->nbits());
  const size_t out_nbits = lo_ty->nbits() + hi_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lo.shape());

  return DISPATCH_UINT_PT_TYPES(hi_ty->getBacktype(), [&]() {
    using hi_el_t = ScalarT;
    using hi_shr_t = std::array<hi_el_t, 2>;
    NdArrayView<hi_shr_t> _hi(hi);

    return DISPATCH_UINT_PT_TYPES(lo_ty->getBacktype(), [&]() {
      using lo_el_t = ScalarT;
      using lo_shr_t = std::array<lo_el_t, 2>;
      NdArrayView<lo_shr_t> _lo(lo);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;
        NdArrayView<out_shr_t> _out(out);

        pforeach(0, lo.numel(), [&](int64_t idx) {
          const auto& l = _lo[idx];
          const auto& h = _hi[idx];
          out_shr_t& o = _out[idx];
          o[0] = l[0] | (static_cast<out_el_t>(h[0]) << lo_ty->nbits());
        });
        return out;
      });
    });
  });
}

// NdArrayRef pack_2_bitvec_rss(const NdArrayRef& lo, const NdArrayRef& hi) {
//   const auto* lo_ty = lo.eltype().as<BShrTy>();
//   const auto* hi_ty = hi.eltype().as<BShrTy>();

//   assert(lo_ty->nbits() == hi_ty->nbits());
//   const size_t out_nbits = lo_ty->nbits() + hi_ty->nbits();
//   const PtType out_btype = calcBShareBacktype(out_nbits);
//   NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lo.shape());

//   return DISPATCH_UINT_PT_TYPES(hi_ty->getBacktype(), [&]() {
//     using hi_el_t = ScalarT;
//     using hi_shr_t = std::array<hi_el_t, 2>;
//     NdArrayView<hi_shr_t> _hi(hi);

//     return DISPATCH_UINT_PT_TYPES(lo_ty->getBacktype(), [&]() {
//       using lo_el_t = ScalarT;
//       using lo_shr_t = std::array<lo_el_t, 2>;
//       NdArrayView<lo_shr_t> _lo(lo);

//       return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
//         using out_el_t = ScalarT;
//         using out_shr_t = std::array<out_el_t, 2>;
//         NdArrayView<out_shr_t> _out(out);

//         pforeach(0, lo.numel(), [&](int64_t idx) {
//           const auto& l = _lo[idx];
//           const auto& h = _hi[idx];
//           out_shr_t& o = _out[idx];
//           o[0] = l[0] | (static_cast<out_el_t>(h[0]) << lo_ty->nbits());
//           o[1] = l[1] | (static_cast<out_el_t>(h[1]) << lo_ty->nbits());
//         });
//         return out;
//       });
//     });
//   });
// }

// NdArrayRef pack_2_bitvec_mss(const NdArrayRef& lo, const NdArrayRef& hi) {
//   const auto* lo_ty = lo.eltype().as<BShrTyMss>();
//   const auto* hi_ty = hi.eltype().as<BShrTyMss>();

//   assert(lo_ty->nbits() == hi_ty->nbits());
//   const size_t out_nbits = lo_ty->nbits() + hi_ty->nbits();
//   const PtType out_btype = calcBShareBacktype(out_nbits);
//   NdArrayRef out(makeType<BShrTyMss>(out_btype, out_nbits), lo.shape());

//   return DISPATCH_UINT_PT_TYPES(hi_ty->getBacktype(), [&]() {
//     using hi_el_t = ScalarT;
//     using hi_shr_t = std::array<hi_el_t, 3>;
//     NdArrayView<hi_shr_t> _hi(hi);

//     return DISPATCH_UINT_PT_TYPES(lo_ty->getBacktype(), [&]() {
//       using lo_el_t = ScalarT;
//       using lo_shr_t = std::array<lo_el_t, 3>;
//       NdArrayView<lo_shr_t> _lo(lo);

//       return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
//         using out_el_t = ScalarT;
//         using out_shr_t = std::array<out_el_t, 3>;
//         NdArrayView<out_shr_t> _out(out);

//         pforeach(0, lo.numel(), [&](int64_t idx) {
//           const auto& l = _lo[idx];
//           const auto& h = _hi[idx];
//           out_shr_t& o = _out[idx];
//           o[0] = l[0] | (static_cast<out_el_t>(h[0]) << lo_ty->nbits());
//           o[1] = l[1] | (static_cast<out_el_t>(h[1]) << lo_ty->nbits());
//           o[2] = l[2] | (static_cast<out_el_t>(h[2]) << lo_ty->nbits());
//         });
//         return out;
//       });
//     });
//   });
// }

// std::pair<NdArrayRef, NdArrayRef> unpack_2_bitvec_ass(const NdArrayRef& in) {
//   const auto* in_ty = in.eltype().as<BShrTy>();
//   assert(in_ty->nbits() != 0 && in_ty->nbits() % 2 == 0);

//   const size_t lo_nbits = in_ty->nbits() / 2;
//   const size_t hi_nbits = in_ty->nbits() - lo_nbits;
//   const PtType lo_btype = calcBShareBacktype(lo_nbits);
//   const PtType hi_btype = calcBShareBacktype(hi_nbits);
//   NdArrayRef lo(makeType<BShrTy>(lo_btype, lo_nbits), in.shape());
//   NdArrayRef hi(makeType<BShrTy>(hi_btype, hi_nbits), in.shape());

//   return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
//     using in_el_t = ScalarT;
//     using in_shr_t = std::array<in_el_t, 2>;
//     NdArrayView<in_shr_t> _in(in);

//     return DISPATCH_UINT_PT_TYPES(lo_btype, [&]() {
//       using lo_el_t = ScalarT;
//       using lo_shr_t = std::array<lo_el_t, 2>;
//       NdArrayView<lo_shr_t> _lo(lo);

//       return DISPATCH_UINT_PT_TYPES(hi_btype, [&]() {
//         using hi_el_t = ScalarT;
//         using hi_shr_t = std::array<hi_el_t, 2>;
//         NdArrayView<hi_shr_t> _hi(hi);

//         pforeach(0, in.numel(), [&](int64_t idx) {
//           const auto& i = _in[idx];
//           lo_shr_t& l = _lo[idx];
//           hi_shr_t& h = _hi[idx];
//           l[0] = i[0] & ((1 << lo_nbits) - 1);
//           h[0] = (i[0] >> lo_nbits) & ((1 << hi_nbits) - 1);
//         });
//         return std::make_pair(hi, lo);
//       });
//     });
//   });
// }

std::pair<NdArrayRef, NdArrayRef> unpack_2_bitvec_rss(const NdArrayRef& in) {
  const auto* in_ty = in.eltype().as<BShrTy>();
  assert(in_ty->nbits() != 0 && in_ty->nbits() % 2 == 0);

  const size_t lo_nbits = in_ty->nbits() / 2;
  const size_t hi_nbits = in_ty->nbits() - lo_nbits;
  const PtType lo_btype = calcBShareBacktype(lo_nbits);
  const PtType hi_btype = calcBShareBacktype(hi_nbits);
  NdArrayRef lo(makeType<BShrTy>(lo_btype, lo_nbits), in.shape());
  NdArrayRef hi(makeType<BShrTy>(hi_btype, hi_nbits), in.shape());

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(lo_btype, [&]() {
      using lo_el_t = ScalarT;
      using lo_shr_t = std::array<lo_el_t, 2>;
      NdArrayView<lo_shr_t> _lo(lo);

      return DISPATCH_UINT_PT_TYPES(hi_btype, [&]() {
        using hi_el_t = ScalarT;
        using hi_shr_t = std::array<hi_el_t, 2>;
        NdArrayView<hi_shr_t> _hi(hi);

        pforeach(0, in.numel(), [&](int64_t idx) {
          const auto& i = _in[idx];
          lo_shr_t& l = _lo[idx];
          hi_shr_t& h = _hi[idx];
          l[0] = i[0] & ((1 << lo_nbits) - 1);
          l[1] = i[1] & ((1 << lo_nbits) - 1);
          h[0] = (i[0] >> lo_nbits) & ((1 << hi_nbits) - 1);
          h[1] = (i[1] >> lo_nbits) & ((1 << hi_nbits) - 1);
        });
        return std::make_pair(hi, lo);
      });
    });
  });
}

std::pair<NdArrayRef, NdArrayRef> unpack_2_bitvec_mss(const NdArrayRef& in) {
  const auto* in_ty = in.eltype().as<BShrTyMss>();
  assert(in_ty->nbits() != 0 && in_ty->nbits() % 2 == 0);

  const size_t lo_nbits = in_ty->nbits() / 2;
  const size_t hi_nbits = in_ty->nbits() - lo_nbits;
  const PtType lo_btype = calcBShareBacktype(lo_nbits);
  const PtType hi_btype = calcBShareBacktype(hi_nbits);
  NdArrayRef lo(makeType<BShrTyMss>(lo_btype, lo_nbits), in.shape());
  NdArrayRef hi(makeType<BShrTyMss>(hi_btype, hi_nbits), in.shape());

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 3>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(lo_btype, [&]() {
      using lo_el_t = ScalarT;
      using lo_shr_t = std::array<lo_el_t, 3>;
      NdArrayView<lo_shr_t> _lo(lo);

      return DISPATCH_UINT_PT_TYPES(hi_btype, [&]() {
        using hi_el_t = ScalarT;
        using hi_shr_t = std::array<hi_el_t, 3>;
        NdArrayView<hi_shr_t> _hi(hi);

        pforeach(0, in.numel(), [&](int64_t idx) {
          const auto& i = _in[idx];
          lo_shr_t& l = _lo[idx];
          hi_shr_t& h = _hi[idx];
          l[0] = i[0] & ((1 << lo_nbits) - 1);
          l[1] = i[1] & ((1 << lo_nbits) - 1);
          l[2] = i[2] & ((1 << lo_nbits) - 1);
          h[0] = (i[0] >> lo_nbits) & ((1 << hi_nbits) - 1);
          h[1] = (i[1] >> lo_nbits) & ((1 << hi_nbits) - 1);
          h[2] = (i[2] >> lo_nbits) & ((1 << hi_nbits) - 1);
        });
        return std::make_pair(hi, lo);
      });
    });
  });
}

template<typename NativeCppType, size_t share_number=2>
std::array<NdArrayRef, 4> sklanky_split(const NdArrayRef& signal, size_t layer) {

  static std::array<uint64_t, 3> pattern = {
    0x1111111111111111ull,    // layer 1
    0x8888888888888888ull,    // layer 2
    0x8000800080008000ull     // layer 3
  };
  static std::array<uint64_t, 3> block_len = {
    4, 16, 64
  };
  static std::array<uint64_t, 3> start_bit = {
    0, 3, 15
  };  // start_bit + 1 actually

  std::array<NdArrayRef, 4> out;
  for (auto& o : out) {
    o = signal.clone();
  }

  using el_t = NativeCppType;
  using shr_t = std::array<el_t, share_number>;
  NdArrayView<shr_t> _s(signal);

  // (block_len, o0_start_bit) should be o[0].
  // repeat 64 / block_len times.
  // (block_len, o0_start_bit + block_len / 4 * j) should be o[j].
  // for o[j] in step layer,
  // (block_len, start_bit + block_len / 4 * j) should be filled with o[j].
  pforeach(0, signal.numel(), [&](int64_t idx) {
    for (size_t j = 0; j < 4; j++) 
    {
      if (j < 3)
      {
        uint64_t mask = pattern[layer] << j * block_len[layer] / 4;
        uint64_t fill_width = block_len[layer] - start_bit[layer] - 1 - j * block_len[layer] / 4;
        NdArrayView<shr_t> _o(out[j]);
        for (size_t i = 0; i < share_number; i++) {
          el_t temp = (_o[idx][i] & mask);
          for (size_t k = 0; k < fill_width; k++) {
            temp |= (temp << 1);
          }
          _o[idx][i] = temp & (~mask);
        }
      }      
    }
  });
  return out;
}

NdArrayRef MsbA2BMultiFanIn(KernelEvalContext* ctx, const NdArrayRef& in, size_t start_rank) {
  const auto field = in.eltype().as<AShrTyMss>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  const size_t start_rank_next = (start_rank + 1) % 3;

  const Type rss_ashr_type =
      makeType<AShrTy>(field);
  const Type rss_bshr_type =
      makeType<BShrTy>(GetStorageType(field), SizeOf(field) * 8);
  const Type rss_bshr_type_u8 =
      makeType<BShrTy>(PtType::PT_U8, 1);
  const Type mss_bshr_type =
      makeType<BShrTyMss>(GetStorageType(field), SizeOf(field) * 8);
  const Type mss_bshr_type_u8 =
      makeType<BShrTyMss>(PtType::PT_U8, 1);

  NdArrayRef m(mss_bshr_type, in.shape());
  NdArrayRef n(mss_bshr_type, in.shape());
  NdArrayRef p(mss_bshr_type, in.shape());
  NdArrayRef g(mss_bshr_type, in.shape());
  NdArrayRef out(mss_bshr_type, in.shape());
  auto in_rss = ResharingMss2RssAri(ctx, in);

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using rss_shr_t = std::array<el_t, 2>;
    using mss_shr_t = std::array<el_t, 3>;

    NdArrayView<rss_shr_t> _in(in_rss);           // rss
    NdArrayView<mss_shr_t> _m(m);
    NdArrayView<mss_shr_t> _n(n);
    NdArrayView<typename std::array<uint8_t, 3>> _out(out);

    /**
     * 1. Convert RSS-shared x into MSS-shared m (Dm, RSS(dm)) and n (Dn, RSS(dn)).
    */
    // generate (compressed) correlated randomness: ((dm0, dm1), (dm1, dn2), (dn2, dm0)). 
    std::vector<el_t> r0(numel, 0);
    std::vector<el_t> r1(numel, 0);
    
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);
    #if !defined(EQ_USE_OFFLINE) || !defined(EQ_USE_PRG_STATE)
    std::fill(r0.begin(), r0.end(), 0);
    std::fill(r1.begin(), r1.end(), 0);
    #endif

    // copy the correlated randomness into m and n
    pforeach(0, numel, [&](int64_t idx) {
      if (comm->getRank() == start_rank) 
      {
        // Wait for x2 ^ dn2 from P1.
        _m[idx][1] = r0[idx];                               // dm0
        _m[idx][2] = r1[idx];                               // dm1
        r1[idx] ^= r0[idx] ^ (_in[idx][0] + _in[idx][1]);     
        _m[idx][0] = r1[idx];                               // Dm = (x0 + x1) ^ dm0 ^ dm1

        _n[idx][1] = 0;
        _n[idx][2] = 0;
      } 
      else if (comm->getRank() == start_rank_next) 
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
    r0 = comm->bcast<el_t>(r1, start_rank, "MsbA2B, special resharing from ASS to MSS, broadcast Dm");
    if (comm->getRank() == start_rank) 
    {
      r0 = comm->recv<el_t>(start_rank_next, "MsbA2B, special resharing from ASS to MSS, get dn2");
      // TODO: The atomized cost of there is k. However, since P0 sends two elements, the SPU
      // logs the transmission of 2k for P0. We manually reduced the logging of k elements. 
      // Nevertheless, this implementation does not adhere to the principle of thread safety.
      // comm->addCommStatsManually(0, -sizeof(el_t) * numel);   // bcast
      // const std::atomic<size_t> & lctx_sent_bytes = comm->lctx().get()->GetStats().get()->sent_bytes;
      // const_cast<std::atomic<size_t> &>(lctx_sent_bytes) -= sizeof(el_t) * numel;
    }
    else if (comm->getRank() == start_rank_next) 
    {
      comm->sendAsync<el_t>(start_rank, r1, "MsbA2B, special resharing from ASS to MSS, send dn2");
    }

    // compute external value Dm, Dn
    pforeach(0, numel, [&](int64_t idx) {
      if (comm->getRank() == start_rank) 
      {
        _n[idx][0] = r0[idx];                               // Dn = x2 + dn2
      } 
      else if (comm->getRank() == start_rank_next) 
      {
        _m[idx][0] = r0[idx];                              // Dm = (x0 + x1) ^ dm0 ^ dm1
      }
      else
      {
        _m[idx][0] = r0[idx];                            
      }
    });

    // 4. generate signal p and g.
    g = MssAnd2NoComm(ctx, ResharingRss2Mss(ctx, sig_g_rss));
    p = MssXor2(ctx, m, n);
    NdArrayView<mss_shr_t> _p(p);
    NdArrayView<mss_shr_t> _g(g);

    // 5. PPA.
    // we dont use the carryout circuit from aby 2.0. By limitting p's msb to be 1 and g's msb to be 0,
    // we could build a simpler carryout circuit.
    size_t nbits = SizeOf(field) * 8 - 1;
    size_t k = nbits;
    
    pforeach(0, numel, [&](int64_t idx) {
      _out[idx][0]  = (_p[idx][0]) >> nbits;
      _out[idx][1]  = (_p[idx][1]) >> nbits;
      _out[idx][2]  = (_p[idx][2]) >> nbits;
      _p[idx][0]    = ( 1ull << nbits     ) | _p[idx][0];    
      _p[idx][1]    = ((1ull << nbits) - 1) & _p[idx][1];
      _p[idx][2]    = ((1ull << nbits) - 1) & _p[idx][2];
      _g[idx][0]    = ((1ull << nbits) - 1) & _g[idx][0];
      _g[idx][1]    = ((1ull << nbits) - 1) & _g[idx][1];
      _g[idx][2]    = ((1ull << nbits) - 1) & _g[idx][2];
    });

    while (k > 1) 
    {
      NdArrayRef pops[4];
      NdArrayRef gops[4];

      auto [g_hi, g_lo] = bit_split_mss(g);
      std::tie(gops[3], gops[1]) = bit_split_mss(g_hi);
      std::tie(gops[2], gops[0]) = bit_split_mss(g_lo);
      auto [p_hi, p_lo] = bit_split_mss(p);
      std::tie(pops[3], pops[1]) = bit_split_mss(p_hi);
      std::tie(pops[2], pops[0]) = bit_split_mss(p_lo);

      auto p_res      = MssAnd4NoComm(ctx, pops[0], pops[1], pops[2], pops[3]);
      auto g_res_3    = ResharingRss2Ass(ctx, ResharingMss2Rss(ctx, gops[3]));
      auto g_res_2    = ResharingRss2Ass(ctx, MssAnd2NoComm(ctx, gops[2], pops[3]));
      auto g_res_1    = MssAnd3NoComm(ctx, gops[1], pops[3], pops[2]);
      auto g_res_0    = MssAnd4NoComm(ctx, gops[0], pops[3], pops[2], pops[1]);
      auto g_combined = AssXor2(ctx, AssXor2(ctx, g_res_0, g_res_1), AssXor2(ctx, g_res_2, g_res_3));

      // online communication
      k /= 4;
      if (k > 1)
      {
        auto pg = pack_2_bitvec_ass(p_res, g_combined);
        pg = ResharingAss2Mss(ctx, pg);
        std::tie(g, p) = unpack_2_bitvec_mss(pg);
        // p = ResharingAss2Mss(ctx, p_res);
        // g = ResharingAss2Mss(ctx, g_combined);
      } else {
        #ifndef EQ_PACK_SINGLE_BIT
        g = ResharingAss2Mss(ctx, g_combined);
        #else
        // pack 8 element's bit into 1 uint8_t
        size_t packed_numel = numel / 8 + ((numel && 0b111) > 0);
        Shape packed_shape = {1, static_cast<int64_t>(packed_numel)};
        NdArrayRef packed_c(makeType<BShrTy>(PtType::PT_U8, 8), packed_shape);
        NdArrayView<std::array<uint8_t, 2>> _c(g_combined);
        NdArrayView<std::array<uint8_t, 2>> _pc(packed_c);
        // if (comm->getRank() == 0) std::cout << "MSB: c." << (int)_c[0][0] << " " << (int)_c[1][0] << " " << (int)_c[2][0] << " " << (int)_c[3][0] << std::endl;
        // if (comm->getRank() == 0) std::cout << "MSB: c." << (int)_c[4][0] << " " << (int)_c[5][0] << " " << (int)_c[6][0] << " " << (int)_c[7][0] << std::endl;
        pforeach(0, packed_numel, [&](int64_t idx) {
          size_t loc = idx * 8;
          uint8_t& op_pc = _pc[idx][0];
          op_pc = 0;                      // NdArrayRef's buffer is not empty. We should clear it manually.
          uint8_t op_c;
          for (size_t i = 0; i < 8; i++)
          {
            if (loc + i < static_cast<size_t>(numel)) op_c = _c[loc + i][0] & 1;
            else op_c = 0;
            op_pc ^= op_c << (7 - i);
          }
        });
        // if (comm->getRank() == 0) std::cout << "MSB: packed c." << (int)_pc[0][0] << std::endl;
        auto packed_c_rss = ResharingAss2Mss(ctx, packed_c);
        NdArrayView<std::array<uint8_t, 3>> _pcr(packed_c_rss);
        pforeach(0, packed_numel, [&](int64_t idx) {
          size_t loc = idx * 8;
          uint8_t op_pcr0 = _pcr[idx][0];
          uint8_t op_pcr1 = _pcr[idx][1];
          uint8_t op_pcr2 = _pcr[idx][2];
          for (size_t i = 0; i < 8; i++)
          {
            if (loc + i >= static_cast<size_t>(numel)) break;
            _c[loc + i][0] = (op_pcr0 >> (7 - i)) & 1;
            _c[loc + i][1] = (op_pcr1 >> (7 - i)) & 1;
            _c[loc + i][2] = (op_pcr2 >> (7 - i)) & 1;
          }
        });
        g = g_combined;
        #endif
      }
    }

    pforeach(0, numel, [&](size_t idx) {
      _out[idx][0] ^= (static_cast<uint8_t>(_g[idx][0]));
      _out[idx][1] ^= (static_cast<uint8_t>(_g[idx][1]));
      _out[idx][2] ^= (static_cast<uint8_t>(_g[idx][2]));
    });
    // if (comm->getRank() == 0) std::cout << "MSB: out." << (int)_out[0][0] << " " << (int)_out[1][0] << " " << (int)_out[2][0] << std::endl;

    return out;
  });  
}

/**
 * A 4 fan-in 4 outputs protocol for black cell in PPA.
 */
std::pair<NdArrayRef, NdArrayRef> PGCell_4FanIn4Out(KernelEvalContext* ctx, 
  const NdArrayRef& p0, const NdArrayRef& p1, const NdArrayRef& p2, const NdArrayRef& p3,
  const NdArrayRef& g0, const NdArrayRef& g1, const NdArrayRef& g2, const NdArrayRef& g3) 

// std::array<NdArrayRef, 8> PGCell_4FanIn4Out(KernelEvalContext* ctx, 
//   const NdArrayRef& p, const NdArrayRef& g, const size_t nbits, const size_t mask, const size_t stride) 
{
  /**
   *  p3    p2    p1    p0
   *  g3    g2    g1    g0
   * --------------------------------
   *  g'3   g'2   g'1   g'0
   *  p'3   p'2   p'1   p'0
   * where 
   * p'3 = (p0 & p1) & (p2 & p3)
   * p'2 = (p0 & p1) & p2
   * p'1 = (p0 & p1)
   * p'0 = p0
   * g'3 = g3 ^ g2 & p3 ^ g1 & (p2 & p3) ^ (g0 & p1) & (p2 & p3)
   * g'2 = g2 ^ g1 & p2 ^ (g0 & p1) & p2
   * g'1 = g1 ^ (g0 & p1)
   * g'0 = g0.
   * 
   * All the AND gates is concluded here:
   * AND2 in MSS:
   *  p01_rss = p0 & p1, p23_rss = p2 & p3, g0p1_rss = g0 & p1
   * AND2 in RSS:
   *  p0123_ass = p01_rss & p23_rss, p012_ass = p01_rss & p2_rss
   *  g2p3_ass = g2_rss & p3_rss, g1p23_ass = g1_rss & p23_rss, g0p123_ass = g0p1_rss & p23_rss
   *  g1p2_ass = g1_rss & p2_rss, g0p12_ass = g0p1_rss & p2_rss
   *  
   * All the Resharing steps is here:
   *  p3 -> p3_rss, p2 -> p2_rss, g2 -> g2_rss, g1 -> g1_rss              (down)
   *  p01_rss -> p01_mss, p012_ass -> p012_mss, p0123_ass -> p0123_mss    (up)
   *  gr3_ass -> gr3_mss, gr2_ass -> gr2_mss, gr1_rss -> gr1_mss          (up)
   */

  auto* comm = ctx->getState<Communicator>();

  auto p3_rss = ResharingMss2Rss(ctx, p3);
  auto p2_rss = ResharingMss2Rss(ctx, p2);
  auto g2_rss = ResharingMss2Rss(ctx, g2);
  auto g1_rss = ResharingMss2Rss(ctx, g1);

  auto p01_rss = MssAnd2NoComm(ctx, p0, p1);
  auto p23_rss = MssAnd2NoComm(ctx, p2, p3);
  auto g0p1_rss = MssAnd2NoComm(ctx, g0, p1);

  auto p0123_ass = RssAnd2NoComm(ctx, p01_rss, p23_rss);
  auto p012_ass = RssAnd2NoComm(ctx, p01_rss, p2_rss);
  auto g2p3_ass = RssAnd2NoComm(ctx, g2_rss, p3_rss);
  auto g1p23_ass = RssAnd2NoComm(ctx, g1_rss, p23_rss);
  auto g0p123_ass = RssAnd2NoComm(ctx, g0p1_rss, p23_rss);
  auto g1p2_ass = RssAnd2NoComm(ctx, g1_rss, p2_rss);
  auto g0p12_ass = RssAnd2NoComm(ctx, g0p1_rss, p2_rss);
  
  // gr3 = g3 ^ gr3_ass
  auto gr3_ass = AssXor2(ctx, g2p3_ass, AssXor2(ctx, g1p23_ass, g0p123_ass));
  auto gr2_ass = AssXor2(ctx, g1p2_ass, g0p12_ass);
  auto gr1_ass = ResharingRss2Ass(ctx, g0p1_rss);
  auto gr0_ass = ResharingRss2Ass(ctx, ResharingMss2Rss(ctx, g0));

  auto p3_ass = p0123_ass;
  auto p2_ass = p012_ass;
  auto p1_ass = ResharingRss2Ass(ctx, p01_rss);
  auto p0_ass = ResharingRss2Ass(ctx, ResharingMss2Rss(ctx, p0));
  auto g3_ass = AssXor2(ctx, 
      gr3_ass, ResharingRss2Ass(ctx, ResharingMss2Rss(ctx, g3)));
  auto g2_ass = AssXor2(ctx,
      gr2_ass, ResharingRss2Ass(ctx, g2_rss));
  auto g1_ass = AssXor2(ctx,
      gr1_ass, ResharingRss2Ass(ctx, g1_rss));
  auto g0_ass = gr0_ass;

  // 3 3, 2 2, 1 1, 0 0 -> 3 3 1 1, 2 2 0 0 -> 3 1 3 1, 2 0 2 0 -> 3 1 3 1 2 0 2 0 -> 3 2 1 0 3 2 1 0 
  auto g_packed_ass = bit_interleave_ass(pack_2_bitvec_ass( 
      bit_interleave_ass(pack_2_bitvec_ass(g0_ass, g2_ass)),
      bit_interleave_ass(pack_2_bitvec_ass(g1_ass, g3_ass))));
  auto p_packed_ass = bit_interleave_ass(pack_2_bitvec_ass(
      bit_interleave_ass(pack_2_bitvec_ass(p0_ass, p2_ass)),
      bit_interleave_ass(pack_2_bitvec_ass(p1_ass, p3_ass))));

  std::pair<NdArrayRef, NdArrayRef> result;
  result.first = ResharingAss2Mss(ctx, g_packed_ass);
  result.second = ResharingAss2Mss(ctx, p_packed_ass);
  // comm->addCommStatsManually(-1, 0);
  // const std::atomic<size_t> & lctx_sent_actions = comm->lctx().get()->GetStats().get()->sent_actions;
  // const_cast<std::atomic<size_t> &>(lctx_sent_actions) -= 1;

  return result;
}

/**
 * A 4 fan-in 1 output protocol for black cell in PPA.
 */
std::pair<NdArrayRef, NdArrayRef> PGCell_4FanIn1Out(KernelEvalContext* ctx, 
  const NdArrayRef& p0, const NdArrayRef& p1, const NdArrayRef& p2, const NdArrayRef& p3,
  const NdArrayRef& g0, const NdArrayRef& g1, const NdArrayRef& g2, const NdArrayRef& g3,
  bool output_p = true) 

// std::array<NdArrayRef, 8> PGCell_4FanIn4Out(KernelEvalContext* ctx, 
//   const NdArrayRef& p, const NdArrayRef& g, const size_t nbits, const size_t mask, const size_t stride) 
{
  /**
   *  p3    p2    p1    p0
   *  g3    g2    g1    g0
   * --------------------------------
   *  g'3   
   *  p'3   
   * where 
   * p'3 = (p0 & p1) & (p2 & p3)
   * g'3 = g3 ^ g2 & p3 ^ g1 & (p2 & p3) ^ (g0 & p1) & (p2 & p3)
   * 
   * All the AND gates is concluded here:
   * AND2 in MSS:
   *  p01_rss = p0 & p1, p23_rss = p2 & p3, g0p1_rss = g0 & p1
   * AND2 in RSS:
   *  p0123_ass = p01_rss & p23_rss, p012_ass = p01_rss & p2_rss
   *  g2p3_ass = g2_rss & p3_rss, g1p23_ass = g1_rss & p23_rss, g0p123_ass = g0p1_rss & p23_rss
   *  g1p2_ass = g1_rss & p2_rss, g0p12_ass = g0p1_rss & p2_rss
   *  
   * All the Resharing steps is here:
   *  p3 -> p3_rss, p2 -> p2_rss, g2 -> g2_rss, g1 -> g1_rss              (down)
   *  p01_rss -> p01_mss, p012_ass -> p012_mss, p0123_ass -> p0123_mss    (up)
   *  gr3_ass -> gr3_mss, gr2_ass -> gr2_mss, gr1_rss -> gr1_mss          (up)
   */

  auto* comm = ctx->getState<Communicator>();

  auto p3_rss = ResharingMss2Rss(ctx, p3);
  auto g2_rss = ResharingMss2Rss(ctx, g2);
  auto g1_rss = ResharingMss2Rss(ctx, g1);

  auto p01_rss = MssAnd2NoComm(ctx, p0, p1);
  auto p23_rss = MssAnd2NoComm(ctx, p2, p3);
  auto g0p1_rss = MssAnd2NoComm(ctx, g0, p1);

  auto p0123_ass = RssAnd2NoComm(ctx, p01_rss, p23_rss);
  auto g2p3_ass = RssAnd2NoComm(ctx, g2_rss, p3_rss);
  auto g1p23_ass = RssAnd2NoComm(ctx, g1_rss, p23_rss);
  auto g0p123_ass = RssAnd2NoComm(ctx, g0p1_rss, p23_rss);

  auto g3_ass = ResharingRss2Ass(ctx, ResharingMss2Rss(ctx, g3));
  
  // gr3 = g3 ^ gr3_ass
  auto gr3_ass = AssXor2(ctx, AssXor2(ctx, g3_ass, g2p3_ass), AssXor2(ctx, g1p23_ass, g0p123_ass));
  auto pr3_ass = p0123_ass;

  auto gr3_mss = ResharingAss2Mss(ctx, gr3_ass);
  NdArrayRef pr3_mss = gr3_mss;
  if (output_p) {
    pr3_mss = ResharingAss2Mss(ctx, pr3_ass);
    comm->addCommStatsManually(-1, 0);
    const std::atomic<size_t> & lctx_sent_actions = comm->lctx().get()->GetStats().get()->sent_actions;
    const_cast<std::atomic<size_t> &>(lctx_sent_actions) -= 1;
  }

  std::pair<NdArrayRef, NdArrayRef> result;
  result.first = gr3_mss;
  result.second = pr3_mss;
  return result;
}

NdArrayRef PPAFromABY2(KernelEvalContext* ctx, const NdArrayRef& x, const NdArrayRef& y)
{
  const auto numel = x.numel();
  const auto* in_ty = x.eltype().as<BShrTyMss>();
  const size_t in_nbits = in_ty->nbits();
  const auto in_shape = x.shape();

  SPU_ENFORCE(in_nbits == y.eltype().as<BShrTyMss>()->nbits(), "invalid nbits={}", in_nbits);
  SPU_ENFORCE(x.numel() == y.numel(), "invalid numel x.numel()={}, y.numel()={}", x.numel(), y.numel());

  const Type rss_bshr_type = 
      makeType<BShrTy>(calcBShareBacktype(in_nbits), in_nbits);
  const Type mss_bshr_type =
      makeType<BShrTyMss>(calcBShareBacktype(in_nbits), in_nbits);
    
  NdArrayRef p(mss_bshr_type, in_shape);
  NdArrayRef g(mss_bshr_type, in_shape);
  NdArrayRef out(mss_bshr_type, in_shape);

  return DISPATCH_UINT_PT_TYPES(calcBShareBacktype(in_nbits), [&]() {
    using bshr_el_t = ScalarT;
    using mss_shr_t = std::array<bshr_el_t, 3>;
    // using rss_shr_t = std::array<bshr_el_t, 2>;

    NdArrayView<mss_shr_t> _m(x);
    NdArrayView<mss_shr_t> _n(y);
    NdArrayView<mss_shr_t> _p(p);
    NdArrayView<mss_shr_t> _g(g);
    NdArrayView<mss_shr_t> _out(out);


    // 1. Compute signal g and p.
    auto sig_g_rss = MssAnd2NoComm(ctx, x, y);
    auto sig_g_mss = ResharingRss2Mss(ctx, sig_g_rss);
    NdArrayView<mss_shr_t> _g_mss(sig_g_mss);
    // if (comm->getRank() == 0) std::cout << "PPA: sig_g_mss " << _g_mss[0][0] << " " << _g_mss[1][0] << std::endl;
    pforeach(0, numel, [&](int64_t idx) {
      _p[idx][0] = _m[idx][0] ^ _n[idx][0];
      _p[idx][1] = _m[idx][1] ^ _n[idx][1];
      _p[idx][2] = _m[idx][2] ^ _n[idx][2];
      _g[idx][0] = _g_mss[idx][0];
      _g[idx][1] = _g_mss[idx][1];
      _g[idx][2] = _g_mss[idx][2];
    });

    // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;

    // 2. PPA.
    // we dont use the carryout circuit from aby 2.0. By limitting p's msb to be 1 and g's msb to be 0,
    // we could build a simpler carryout circuit.    
    pforeach(0, numel, [&](int64_t idx) {
      _out[idx][0] = _p[idx][0];
      _out[idx][1] = _p[idx][1];
      _out[idx][2] = _p[idx][2];
    });

    // Construnction from aby 2.0. See https://eprint.iacr.org/2020/1225
    // Level 0. Use 4 fan-in and 4 outputs cell. 
    // p3, p2, p1, p0 -> p3 & p2 & p1 & p0, p2 & p1 & p0, p1 & p0, p0
    // g works in the same way.
    {
      NdArrayRef pops[4];
      NdArrayRef gops[4];

      auto [g_hi, g_lo] = bit_split_mss(g);
      std::tie(gops[3], gops[1]) = bit_split_mss(g_hi);
      std::tie(gops[2], gops[0]) = bit_split_mss(g_lo);
      auto [p_hi, p_lo] = bit_split_mss(p);
      std::tie(pops[3], pops[1]) = bit_split_mss(p_hi);
      std::tie(pops[2], pops[0]) = bit_split_mss(p_lo);

      std::tie(g, p) = PGCell_4FanIn4Out(ctx, pops[0], pops[1], pops[2], pops[3], gops[0], gops[1], gops[2], gops[3]);

      // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;
    }

    const Type mss_bshr_type_16 =
      makeType<BShrTyMss>(PtType::PT_U16, 16);

    // Level 1. Use 4 fan-in and 1 output cell. 
    // p3, p2, p1, p0 -> p3 & p2 & p1 & p0
    // g works in the same way.
    { 
      NdArrayRef pops[4];
      NdArrayRef gops[4];
      NdArrayRef p_sel, g_sel;

      std::tie(p_sel, std::ignore) = bit_split_mss(p);
      std::tie(p_sel, std::ignore) = bit_split_mss(p_sel);
      std::tie(g_sel, std::ignore) = bit_split_mss(g);
      std::tie(g_sel, std::ignore) = bit_split_mss(g_sel);
      NdArrayView<std::array<uint16_t, 3>> _p_sel(p_sel);
      NdArrayView<std::array<uint16_t, 3>> _g_sel(g_sel);

      for (int i = 0; i < 4; i++) 
      {
        pops[i] = lshift<uint16_t, 3>(p_sel, 3 - i);
        gops[i] = lshift<uint16_t, 3>(g_sel, 3 - i);
      }

      std::tie(gops[0], pops[0]) = PGCell_4FanIn1Out(ctx, pops[0], pops[1], pops[2], pops[3], gops[0], gops[1], gops[2], gops[3]);
      pops[1] = NdArrayRef(mss_bshr_type, in_shape);
      gops[1] = NdArrayRef(mss_bshr_type, in_shape);

      NdArrayView<std::array<uint16_t, 3>> _pops0(pops[0]);
      NdArrayView<std::array<uint16_t, 3>> _gops0(gops[0]);
      NdArrayView<std::array<uint64_t, 3>> _pops1(pops[1]);
      NdArrayView<std::array<uint64_t, 3>> _gops1(gops[1]);
      pforeach(0, numel, [&](int64_t idx) {
        _pops1[idx][0] = static_cast<uint64_t>(_pops0[idx][0]) << 48;
        _pops1[idx][1] = static_cast<uint64_t>(_pops0[idx][1]) << 48;
        _pops1[idx][2] = static_cast<uint64_t>(_pops0[idx][2]) << 48;
        _gops1[idx][0] = static_cast<uint64_t>(_gops0[idx][0]) << 48;
        _gops1[idx][1] = static_cast<uint64_t>(_gops0[idx][1]) << 48;
        _gops1[idx][2] = static_cast<uint64_t>(_gops0[idx][2]) << 48;
      });

      pops[1] = bit_interleave_mss(pops[1]);
      pops[1] = bit_interleave_mss(pops[1]);
      gops[1] = bit_interleave_mss(gops[1]);
      gops[1] = bit_interleave_mss(gops[1]);

      NdArrayView<std::array<uint64_t, 3>> _pops(pops[1]);
      NdArrayView<std::array<uint64_t, 3>> _gops(gops[1]);
      pforeach(0, numel, [&](int64_t idx) {
        _g[idx][0] = (_g[idx][0] & 0x7777777777777777) ^ _gops[idx][0];
        _g[idx][1] = (_g[idx][1] & 0x7777777777777777) ^ _gops[idx][1];
        _g[idx][2] = (_g[idx][2] & 0x7777777777777777) ^ _gops[idx][2];
        _p[idx][0] = (_p[idx][0] & 0x7777777777777777) ^ _pops[idx][0];
        _p[idx][1] = (_p[idx][1] & 0x7777777777777777) ^ _pops[idx][1];
        _p[idx][2] = (_p[idx][2] & 0x7777777777777777) ^ _pops[idx][2];
      });

      // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;
    }

    // Level 2. Use 4 fan-in and 1 output cell. 
    // p3, p2, p1, p0 -> p3 & p2 & p1 & p0
    // g works in the same way.
    {
      NdArrayRef pops[4];
      NdArrayRef gops[4];
      NdArrayRef p_sel, g_sel;

      std::tie(p_sel, std::ignore) = bit_split_mss(p);
      std::tie(p_sel, std::ignore) = bit_split_mss(p_sel);
      std::tie(g_sel, std::ignore) = bit_split_mss(g);
      std::tie(g_sel, std::ignore) = bit_split_mss(g_sel);
      NdArrayView<std::array<uint16_t, 3>> _p_sel(p_sel);
      NdArrayView<std::array<uint16_t, 3>> _g_sel(g_sel);

      for (int i = 0; i < 4; i++) {
        pops[i] = lshift<uint16_t, 3>(p_sel, 4 * (3 - i));
        gops[i] = lshift<uint16_t, 3>(g_sel, 4 * (3 - i));

        // pops[i] = NdArrayRef(mss_bshr_type_16, in_shape);
        // gops[i] = NdArrayRef(mss_bshr_type_16, in_shape);
        // NdArrayView<std::array<uint16_t, 3>> _pops(pops[i]);
        // NdArrayView<std::array<uint16_t, 3>> _gops(gops[i]);

        // pforeach(0, numel, [&](int64_t idx) {
        //   _pops[idx][0] = _p_sel[idx][0] << 4 * (3 - i);
        //   _gops[idx][0] = _g_sel[idx][0] << 4 * (3 - i);
        //   _pops[idx][1] = _p_sel[idx][1] << 4 * (3 - i);
        //   _gops[idx][1] = _g_sel[idx][1] << 4 * (3 - i);
        //   _pops[idx][2] = _p_sel[idx][2] << 4 * (3 - i);
        //   _gops[idx][2] = _g_sel[idx][2] << 4 * (3 - i);
        // });
      }

      std::tie(gops[0], std::ignore) = PGCell_4FanIn1Out(ctx, 
          pops[0], pops[1], pops[2], pops[3], 
          gops[0], gops[1], gops[2], gops[3], false);
      // pops[1] = NdArrayRef(mss_bshr_type, in_shape);
      gops[1] = NdArrayRef(mss_bshr_type, in_shape);

      // NdArrayView<std::array<uint16_t, 3>> _pops0(pops[0]);
      NdArrayView<std::array<uint16_t, 3>> _gops0(gops[0]);
      // NdArrayView<std::array<uint64_t, 3>> _pops1(pops[1]);
      NdArrayView<std::array<uint64_t, 3>> _gops1(gops[1]);
      pforeach(0, numel, [&](int64_t idx) {
        // _pops1[idx][0] = static_cast<uint64_t>(_pops0[idx][0]) << 48;
        // _pops1[idx][1] = static_cast<uint64_t>(_pops0[idx][1]) << 48;
        // _pops1[idx][2] = static_cast<uint64_t>(_pops0[idx][2]) << 48;
        _gops1[idx][0] = static_cast<uint64_t>(_gops0[idx][0]) << 48;
        _gops1[idx][1] = static_cast<uint64_t>(_gops0[idx][1]) << 48;
        _gops1[idx][2] = static_cast<uint64_t>(_gops0[idx][2]) << 48;
      });

      // pops[1] = bit_interleave_mss(pops[1]);
      // pops[1] = bit_interleave_mss(pops[1]);
      gops[1] = bit_interleave_mss(gops[1]);
      gops[1] = bit_interleave_mss(gops[1]);

      // NdArrayView<std::array<uint64_t, 3>> _pops(pops[1]);
      NdArrayView<std::array<uint64_t, 3>> _gops(gops[1]);
      pforeach(0, numel, [&](int64_t idx) {
        _g[idx][0] = (_g[idx][0] & 0x7777777777777777) ^ _gops[idx][0];
        _g[idx][1] = (_g[idx][1] & 0x7777777777777777) ^ _gops[idx][1];
        _g[idx][2] = (_g[idx][2] & 0x7777777777777777) ^ _gops[idx][2];
        // _p[idx][0] = (_p[idx][0] & 0x7777777777777777) ^ _pops[idx][0];
        // _p[idx][1] = (_p[idx][1] & 0x7777777777777777) ^ _pops[idx][1];
        // _p[idx][2] = (_p[idx][2] & 0x7777777777777777) ^ _pops[idx][2];
      });

      // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;
    }

    // Level 3. Use 2 fan-in and 1 output cell. 
    // p3, p2, p1, p0 -> p3 & p2 & p1 & p0
    // g works in the same way.
    {
      NdArrayRef pops = NdArrayRef(mss_bshr_type, in_shape);
      NdArrayRef gops0 = NdArrayRef(mss_bshr_type, in_shape);
      NdArrayRef gops1 = NdArrayRef(mss_bshr_type, in_shape);
      NdArrayView<mss_shr_t> _pops(pops);
      NdArrayView<mss_shr_t> _gops0(gops0);
      NdArrayView<mss_shr_t> _gops1(gops1);

      pforeach(0, numel, [&](int64_t idx) {
        _gops0[idx][0] =  SelectAndRotate(_g[idx][0], 0x8888888888888888ull, 1) ^ \
                            SelectAndRotate(_g[idx][0], 0x8888888888888888ull, 2) ^ \
                              SelectAndRotate(_g[idx][0], 0x8888888888888888ull, 3); 
        _gops0[idx][1] =  SelectAndRotate(_g[idx][1], 0x8888888888888888ull, 1) ^ \
                            SelectAndRotate(_g[idx][1], 0x8888888888888888ull, 2) ^ \
                              SelectAndRotate(_g[idx][1], 0x8888888888888888ull, 3); 
        _gops0[idx][2] =  SelectAndRotate(_g[idx][2], 0x8888888888888888ull, 1) ^ \
                            SelectAndRotate(_g[idx][2], 0x8888888888888888ull, 2) ^ \
                              SelectAndRotate(_g[idx][2], 0x8888888888888888ull, 3); 
        _gops1[idx][0] =  _g[idx][0];
        _gops1[idx][1] =  _g[idx][1];
        _gops1[idx][2] =  _g[idx][2];
        _pops[idx][0]  =  SelectAndRotate(_p[idx][0], 0x7777777777777777ull, 0);
        _pops[idx][1]  =  SelectAndRotate(_p[idx][1], 0x7777777777777777ull, 0);
        _pops[idx][2]  =  SelectAndRotate(_p[idx][2], 0x7777777777777777ull, 0);
      });

      auto c = MssXor2(ctx, gops1, ResharingRss2Mss(ctx, MssAnd2NoComm(ctx, gops0, pops)));
      NdArrayView<mss_shr_t> _c(c);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] ^= _c[idx][0] << 1;
        _out[idx][1] ^= _c[idx][1] << 1;
        _out[idx][2] ^= _c[idx][2] << 1;
      });
      return out;
    }
  });
}

NdArrayRef PPASklanky(KernelEvalContext* ctx, const NdArrayRef& x, const NdArrayRef& y)
{
  const auto numel = x.numel();
  const auto* in_ty = x.eltype().as<BShrTyMss>();
  const size_t in_nbits = in_ty->nbits();
  const auto in_shape = x.shape();

  SPU_ENFORCE(in_nbits == y.eltype().as<BShrTyMss>()->nbits(), "invalid nbits={}", in_nbits);
  SPU_ENFORCE(x.numel() == y.numel(), "invalid numel x.numel()={}, y.numel()={}", x.numel(), y.numel());

  const Type rss_bshr_type = 
      makeType<BShrTy>(calcBShareBacktype(in_nbits), in_nbits);
  const Type mss_bshr_type =
      makeType<BShrTyMss>(calcBShareBacktype(in_nbits), in_nbits);
    
  NdArrayRef p(mss_bshr_type, in_shape);
  NdArrayRef g(mss_bshr_type, in_shape);
  NdArrayRef out(mss_bshr_type, in_shape);

  return DISPATCH_UINT_PT_TYPES(calcBShareBacktype(in_nbits), [&]() {
    using bshr_el_t = ScalarT;
    using mss_shr_t = std::array<bshr_el_t, 3>;

    NdArrayView<mss_shr_t> _m(x);
    NdArrayView<mss_shr_t> _n(y);
    NdArrayView<mss_shr_t> _p(p);
    NdArrayView<mss_shr_t> _g(g);
    NdArrayView<mss_shr_t> _out(out);


    // 1. Compute signal g and p.
    auto sig_g_rss = MssAnd2NoComm(ctx, x, y);
    auto sig_g_mss = ResharingRss2Mss(ctx, sig_g_rss);
    NdArrayView<mss_shr_t> _g_mss(sig_g_mss);
    // if (comm->getRank() == 0) std::cout << "PPA: sig_g_mss " << _g_mss[0][0] << " " << _g_mss[1][0] << std::endl;
    pforeach(0, numel, [&](int64_t idx) {
      _p[idx][0] = _m[idx][0] ^ _n[idx][0];
      _p[idx][1] = _m[idx][1] ^ _n[idx][1];
      _p[idx][2] = _m[idx][2] ^ _n[idx][2];
      _g[idx][0] = _g_mss[idx][0];
      _g[idx][1] = _g_mss[idx][1];
      _g[idx][2] = _g_mss[idx][2];
    });

    // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
    // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;

    // 2. PPA.
    // we dont use the carryout circuit from aby 2.0. By limitting p's msb to be 1 and g's msb to be 0,
    // we could build a simpler carryout circuit.    
    pforeach(0, numel, [&](int64_t idx) {
      _out[idx][0] = _p[idx][0];
      _out[idx][1] = _p[idx][1];
      _out[idx][2] = _p[idx][2];
    });

    // Construnction from aby 2.0. See https://eprint.iacr.org/2020/1225
    // Level 0. Use 4 fan-in and 4 outputs cell. 
    // p3, p2, p1, p0 -> p3 & p2 & p1 & p0, p2 & p1 & p0, p1 & p0, p0
    // g works in the same way.
    {
      auto gops = sklanky_split<bshr_el_t>(g, 0);
      auto pops = sklanky_split<bshr_el_t>(p, 0);

      std::tie(g, p) = PGCell_4FanIn1Out(ctx, pops[0], pops[1], pops[2], pops[3], gops[0], gops[1], gops[2], gops[3]);

      // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;
    }

    // Level 1. Use 4 fan-in and 1 output cell. 
    // p3, p2, p1, p0 -> p3 & p2 & p1 & p0
    // g works in the same way.
    { 
      auto gops = sklanky_split<bshr_el_t>(g, 1);
      auto pops = sklanky_split<bshr_el_t>(p, 1);

      std::tie(g, p) = PGCell_4FanIn1Out(ctx, pops[0], pops[1], pops[2], pops[3], gops[0], gops[1], gops[2], gops[3]);

      // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;
    }

    // Level 2. Use 4 fan-in and 1 output cell. 
    // p3, p2, p1, p0 -> p3 & p2 & p1 & p0
    // g works in the same way.
    {
      auto gops = sklanky_split<bshr_el_t>(g, 2);
      auto pops = sklanky_split<bshr_el_t>(p, 2);

      std::tie(g, p) = PGCell_4FanIn1Out(ctx, pops[0], pops[1], pops[2], pops[3], gops[0], gops[1], gops[2], gops[3], false);

      // if (comm->getRank() == 0) std::cout << "PPA: generate signal p and signal g." << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal p." << _p[0][0] << " " << _p[1][0] << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal g." << _g[0][0] << " " << _g[1][0] << std::endl;
    }

    NdArrayView<mss_shr_t> _c(g);
    pforeach(0, numel, [&](int64_t idx) {
      _out[idx][0] ^= _c[idx][0] << 1;
      _out[idx][1] ^= _c[idx][1] << 1;
      _out[idx][2] ^= _c[idx][2] << 1;
    });
    return out;
  });
}

NdArrayRef A2BMultiFanIn(KernelEvalContext* ctx, const NdArrayRef& in) {
  const auto field = in.eltype().as<AShrTyMss>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
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

  NdArrayRef m(mss_bshr_type, in.shape());
  NdArrayRef n(mss_bshr_type, in.shape());
  // NdArrayRef p(mss_bshr_type, in.shape());
  // NdArrayRef g(mss_bshr_type, in.shape());
  // NdArrayRef c(mss_bshr_type, in.shape());
  NdArrayRef out(mss_bshr_type, in.shape());
  auto in_rss = ResharingMss2RssAri(ctx, in);

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using rss_shr_t = std::array<el_t, 2>;
    using mss_shr_t = std::array<el_t, 3>;

    NdArrayView<rss_shr_t> _in(in_rss);           // rss
    NdArrayView<mss_shr_t> _m(m);
    NdArrayView<mss_shr_t> _n(n);
    NdArrayView<mss_shr_t> _out(out);

    /**
     * 1. Convert RSS-shared x into MSS-shared m (Dm, RSS(dm)) and n (Dn, RSS(dn)).
    */
    // generate (compressed) correlated randomness: ((dm0, dm1), (dm1, dn2), (dn2, dm0)). 
    std::vector<el_t> r0(numel, 0);
    std::vector<el_t> r1(numel, 0);
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);
    #if !defined(EQ_USE_OFFLINE) || !defined(EQ_USE_PRG_STATE)
    std::fill(r0.begin(), r0.end(), 0);
    std::fill(r1.begin(), r1.end(), 0);
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
    r0 = comm->bcast<el_t>(r1, 0, "MsbA2B, special resharing from ASS to MSS, broadcast Dm");
    if (comm->getRank() == 0) 
    {
      r0 = comm->recv<el_t>(1, "MsbA2B, special resharing from ASS to MSS, get dn2");
      const std::atomic<size_t> & lctx_sent_bytes = comm->lctx().get()->GetStats().get()->sent_bytes;
      const_cast<std::atomic<size_t> &>(lctx_sent_bytes) -= sizeof(el_t) * numel;
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

    return PPASklanky(ctx, m, n);
  });  
}

// Alkaid's B2A. RSS input, RSS output.
// Let P0, P1 sample rb1, ra1, P0, P2 sample rb0, P1, P2 sample r2.
// P0 computes ra0 = rb0 xor rb1 - ra1 and sends it to P2.
// Now, (0, (rb0, rb1, 0)) and (0, (ra0, ra1, 0)) come to MRSS(-r).
// We invoke a PPA to compute z = x + r where r = -(rb0 xor rb1).
// Notice that a ResharingAss2Mss is invoked while computing signal g.
// Then, we reveal z to P1 and P2. P1 and P2 compute and send zp = z + r2 to P0.
// (zp, (-ra0, -ra1, r2)) is what we want.
// Online: log2(k) + 1 rounds.
NdArrayRef B2AMultiFanIn(KernelEvalContext* ctx, const NdArrayRef& in) {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTyMss>();
  const size_t in_nbits = in_ty->nbits();

  SPU_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);
  const Type rss_ashr_type = 
      makeType<AShrTy>(field);
  const Type mss_ashr_type =
      makeType<AShrTyMss>(field);
  const int field_bit_width = SizeOf(field) * 8;
  const Type rss_bshr_type = 
      makeType<BShrTy>(calcBShareBacktype(field_bit_width), field_bit_width);
  const Type mss_bshr_type =
      makeType<BShrTyMss>(calcBShareBacktype(field_bit_width), field_bit_width);

  NdArrayRef dabit_a(mss_ashr_type, in.shape());
  NdArrayRef dabit_b(mss_bshr_type, in.shape());
  NdArrayRef p(mss_bshr_type, in.shape());
  NdArrayRef g(mss_bshr_type, in.shape());
  // NdArrayRef z(rss_bshr_type, in.shape());
  NdArrayRef out(mss_ashr_type, in.shape());

  auto numel = in.numel();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  if (in_nbits == 0) {
    // special case, it's known to be zero.
    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<std::array<ring2k_t, 3>> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = 0;
        _out[idx][2] = 0;
      });
    });
    return out;
  }

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;

    return DISPATCH_UINT_PT_TYPES(calcBShareBacktype(field_bit_width), [&]() {
      // TODO: Insecure when convert a 8-bit boolean sharing to 64-bit ring.
      using bshr_el_t = ScalarT;
      using mss_shr_t = std::array<bshr_el_t, 3>;
      using rss_shr_t = std::array<bshr_el_t, 2>;

      auto in_mss = in;

      NdArrayView<mss_shr_t> _dabit_a(dabit_a);
      NdArrayView<mss_shr_t> _dabit_b(dabit_b);
      NdArrayView<std::array<ashr_el_t, 3>> _out(out);

      /**
       * 1. P0 generate dabits.
       */
      if (comm->getRank() == 0) {
        std::vector<bshr_el_t> r_bool_0(numel, 0);
        std::vector<ashr_el_t> r_arith_1(numel, 0);
        std::vector<bshr_el_t> r_bool_1(numel, 0);

        // sample rb1, ra1 with p1
        prg_state->fillPrssPair<ashr_el_t>({}, r_arith_1.data(), numel,
                                          PrgState::GenPrssCtrl::Second);
        prg_state->fillPrssPair<bshr_el_t>({}, r_bool_1.data(), numel,
                                          PrgState::GenPrssCtrl::Second);
        // sample rb0 with p2
        prg_state->fillPrssPair<bshr_el_t>(r_bool_0.data(), {}, numel,
                                          PrgState::GenPrssCtrl::First);
        #if !defined(EQ_USE_OFFLINE) || !defined(EQ_USE_PRG_STATE)
        std::fill(r_bool_0.begin(), r_bool_0.end(), 0);
        std::fill(r_arith_1.begin(), r_arith_1.end(), 0);
        std::fill(r_bool_1.begin(), r_bool_1.end(), 0);
        #endif

        // send rb0 = (ra0 + ra1) ^ rb1 to P2
        std::vector<ashr_el_t> r_arith_0(numel, 0);
        pforeach(0, numel, [&](int64_t idx) {
          r_arith_0[idx] = (r_bool_0[idx] ^ r_bool_1[idx]) - r_arith_1[idx];
          _dabit_a[idx][0] = 0;
          _dabit_a[idx][1] = r_arith_0[idx];
          _dabit_a[idx][2] = r_arith_1[idx];
          _dabit_b[idx][0] = 0;
          _dabit_b[idx][1] = r_bool_0[idx];
          _dabit_b[idx][2] = r_bool_1[idx];
          _out[idx][0] = 0;
          _out[idx][1] = -r_arith_0[idx];
          _out[idx][2] = -r_arith_1[idx];
        });
        comm->sendAsync<ashr_el_t>(2, r_arith_0, "r_arith");
      } else {
        // std::vector<ashr_el_t> a_s(numel);
        std::vector<ashr_el_t> r_arith(numel, 0);
        std::vector<bshr_el_t> r_bool(numel, 0);

        if (comm->getRank() == 1) {
          // sample rb1, ra1 with p0
          prg_state->fillPrssPair<ashr_el_t>(r_arith.data(), {}, numel,
                                            PrgState::GenPrssCtrl::First);
          prg_state->fillPrssPair<bshr_el_t>(r_bool.data(), {}, numel,
                                            PrgState::GenPrssCtrl::First);
          #if !defined(EQ_USE_OFFLINE) || !defined(EQ_USE_PRG_STATE)
          std::fill(r_arith.begin(), r_arith.end(), 0);
          std::fill(r_bool.begin(), r_bool.end(), 0);
          #endif
          pforeach(0, numel, [&](int64_t idx) {
            _dabit_a[idx][0] = 0;
            _dabit_a[idx][1] = r_arith[idx];
            _dabit_a[idx][2] = 0;
            _dabit_b[idx][0] = 0;
            _dabit_b[idx][1] = r_bool[idx];
            _dabit_b[idx][2] = 0;
            _out[idx][0] = 0;
            _out[idx][1] = -r_arith[idx];
            _out[idx][2] = 0;
          });
        } else {
          prg_state->fillPrssPair<bshr_el_t>({}, r_bool.data(), numel,
                                            PrgState::GenPrssCtrl::Second);
          #if !defined(EQ_USE_OFFLINE) || !defined(EQ_USE_PRG_STATE)
          std::fill(r_arith.begin(), r_arith.end(), 0);
          std::fill(r_bool.begin(), r_bool.end(), 0);
          #endif
          r_arith = comm->recv<ashr_el_t>(0, "r_arith");   

          pforeach(0, numel, [&](int64_t idx) {
            _dabit_a[idx][0] = 0;
            _dabit_a[idx][1] = 0;
            _dabit_a[idx][2] = r_arith[idx];
            _dabit_b[idx][0] = 0;
            _dabit_b[idx][1] = 0;
            _dabit_b[idx][2] = r_bool[idx];
            _out[idx][0] = 0;
            _out[idx][1] = 0;
            _out[idx][2] = -r_arith[idx];
          }); 
        }
      }

      auto ppa_result = ResharingMss2Rss(ctx, PPASklanky(ctx, in, dabit_b));

      // if (comm->getRank() == 0) std::cout << "PPA: generate signal c." << std::endl;
      // if (comm->getRank() == 0) std::cout << "PPA: signal c." << _c[0][0] << " " << _c[1][0] << std::endl;
      {
        NdArrayView<rss_shr_t> _z(ppa_result);

        // open z = x + (-r) to P1 and P2.
        // P0 sends z0 to P1, P1 sends z1 to P2.
        std::vector<bshr_el_t> zp(numel, 0);
        if (comm->getRank() == 0) 
        {
          pforeach(0, numel, [&](int64_t idx) {
            zp[idx] = _z[idx][0];
          });
          comm->sendAsync<bshr_el_t>(1, zp, "z0");
          zp = comm->recv<bshr_el_t>(1, "delta_z");
          pforeach(0, numel, [&](int64_t idx) {
            _out[idx][0] = zp[idx];
          });
        }
        else if (comm->getRank() == 1) 
        {
          pforeach(0, numel, [&](int64_t idx) {
            zp[idx] = _z[idx][0];
          });
          comm->sendAsync<bshr_el_t>(2, zp, "z1");
          zp = comm->recv<bshr_el_t>(0, "z0");
          std::vector<ashr_el_t> r2(numel, 0);
          prg_state->fillPrssPair<ashr_el_t>({}, r2.data(), numel,
            PrgState::GenPrssCtrl::Second);
          pforeach(0, numel, [&](int64_t idx) {
            zp[idx] = (zp[idx] ^ _z[idx][0] ^ _z[idx][1]) + r2[idx];
            _out[idx][0] = zp[idx];
            _out[idx][2] = r2[idx];
          });
          comm->sendAsync<bshr_el_t>(0, zp, "delta_z");
        }
        else if (comm->getRank() == 2) 
        {
          zp = comm->recv<bshr_el_t>(1, "z1");
          std::vector<ashr_el_t> r2(numel, 0);
          prg_state->fillPrssPair<ashr_el_t>(r2.data(), {}, numel,
            PrgState::GenPrssCtrl::First);
          pforeach(0, numel, [&](int64_t idx) {
            zp[idx] = (zp[idx] ^ _z[idx][0] ^ _z[idx][1]) + r2[idx];
            _out[idx][0] = zp[idx];
            _out[idx][1] = r2[idx];
          });
        }
      }
      return out;
    });  
  });
}

NdArrayRef PPATest(KernelEvalContext* ctx, const NdArrayRef& in) {
  const auto field = in.eltype().as<AShrTyMss>()->field();
  const auto numel = in.numel();

  const Type mss_bshr_type =
      makeType<BShrTyMss>(GetStorageType(field), SizeOf(field) * 8);

  NdArrayRef m(mss_bshr_type, in.shape());
  NdArrayRef n(mss_bshr_type, in.shape());

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using mss_shr_t = std::array<el_t, 3>;

    NdArrayView<mss_shr_t> _in(in);          
    NdArrayView<mss_shr_t> _m(m);
    NdArrayView<mss_shr_t> _n(n);

    // only for test
    pforeach(0, numel, [&](int64_t idx) {
      _m[idx][0] = _in[idx][0];
      _m[idx][1] = _in[idx][1];
      _m[idx][2] = _in[idx][2];
      _n[idx][0] = _in[idx][0];
      _n[idx][1] = _in[idx][1];
      _n[idx][2] = _in[idx][2];
    });

    return PPASklanky(ctx, m, n);
  });  
}

}  // namespace spu::mpc::albo
