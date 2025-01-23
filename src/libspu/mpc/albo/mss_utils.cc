#include "libspu/mpc/albo/mss_utils.h"

#include <atomic>
#include <functional>
#include <iostream>
#include <utility>

#include "yacl/utils/platform_utils.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/albo/type.h"
#include "libspu/mpc/albo/value.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/offline_recorder.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::albo {
// Xor gate for ASS.
NdArrayRef AssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                   const NdArrayRef& rhs) {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
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

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
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
NdArrayRef MrssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                    const NdArrayRef& rhs) {
  const auto* lhs_ty = lhs.eltype().as<BShrTyMrss>();
  const auto* rhs_ty = rhs.eltype().as<BShrTyMrss>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTyMrss>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 3>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
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
  auto* prg_state = ctx->getState<PrgState>();

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;
        NdArrayView<out_shr_t> _out(out);

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> r0(lhs.numel(), 0);
        std::vector<out_el_t> r1(lhs.numel(), 0);

        prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                                PrgState::GenPrssCtrl::Both);
#ifndef ALKAID_USE_PRG_STATE
        std::fill(r0.begin(), r0.end(), 0);
        std::fill(r1.begin(), r1.end(), 0);
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
NdArrayRef MrssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                          const NdArrayRef& rhs) {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  const auto* lhs_ty = lhs.eltype().as<BShrTyMrss>();
  const auto* rhs_ty = rhs.eltype().as<BShrTyMrss>();

  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.shape());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 3>;
    NdArrayView<rhs_shr_t> _rhs(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;
      NdArrayView<lhs_shr_t> _lhs(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
        using out_el_t = ScalarT;
        // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
        using out_shr_t = std::array<out_el_t, 2>;

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> r0(lhs.numel(), 0);
        std::vector<out_el_t> r1(lhs.numel(), 0);
        prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                                PrgState::GenPrssCtrl::Both);

        // offline.
        if (comm->getRank() == 0)
          OfflineRecorder::RecordMult(lhs.numel(),
                                      lhs.numel() * ((out_nbits + 7) / 8));
#if !defined(ALKAID_USE_PRG_STATE) || !defined(ALKAID_USE_OFFLINE)
        std::fill(r0.begin(), r0.end(), 0);
        std::fill(r1.begin(), r1.end(), 0);
        comm->addCommStatsManually(0, 0);  // deal with unused-variable warning.
#endif
#ifdef ALKAID_USE_OFFLINE
        // dxy = dx & dy = (dx0 & dy0) ^ (dx0 & dy1) ^ (dx1 & dy0);
        // r0 is dxy0, r1 is dxy1.
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          r0[idx] = (l[1] & r[1]) ^ (l[1] & r[2]) ^ (l[2] & r[1]) ^
                    (r0[idx] ^ r1[idx]);
        });

        r1 = comm->rotate<out_el_t>(r0, "MrssAndBB, offline");  // comm => 1, k
// comm->addCommStatsManually(-1, -r0.size() * sizeof(out_el_t));
#endif

        // online, compute [out] locally.
        NdArrayView<out_shr_t> _out(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];

          out_shr_t& o = _out[idx];
          // z = x & y = (Dx ^ dx) & (Dy ^ dy) = Dx & Dy ^ Dx & dy ^ dx & Dy ^
          // dxy o[0] = ((comm->getRank() == 0) * (l[0] & r[0])) ^ (l[0] & r[1])
          // ^ (l[1] & r[0]) ^ r0[idx];   // r0 is dxy0 o[1] = ((comm->getRank()
          // == 2) * (l[0] & r[0])) ^ (l[0] & r[2]) ^ (l[2] & r[0]) ^ r1[idx];
          // // r1 is dxy1
          o[0] = ((l[0] & r[0])) ^ (l[0] & r[1]) ^ (l[1] & r[0]) ^
                 r0[idx];  // r0 is dxy0
          o[1] = ((l[0] & r[0])) ^ (l[0] & r[2]) ^ (l[2] & r[0]) ^
                 r1[idx];  // r1 is dxy1
        });
        return out;
      });
    });
  });
}

// And gate for MSS which outputs ASS result (no comunication).
NdArrayRef MrssAnd3NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                          const NdArrayRef& op2, const NdArrayRef& op3) {
  auto lo_res = MrssAnd2NoComm(ctx, op1, op2);
  auto hi_res = ResharingMrss2Rss(ctx, op3);
  auto out = RssAnd2NoComm(ctx, lo_res, hi_res);

  return out;
}

// And gate for MSS which outputs ASS result (no comunication).
NdArrayRef MrssAnd4NoComm(KernelEvalContext* ctx, const NdArrayRef& op1,
                          const NdArrayRef& op2, const NdArrayRef& op3,
                          const NdArrayRef& op4) {
  auto lo_res = MrssAnd2NoComm(ctx, op1, op2);
  auto hi_res = MrssAnd2NoComm(ctx, op3, op4);
  auto out = RssAnd2NoComm(ctx, lo_res, hi_res);

  return out;
}

// Resharing protocol from RSS to MSS.
NdArrayRef ResharingRss2Mrss(KernelEvalContext* ctx, const NdArrayRef& in) {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTyMrss>(out_btype, out_nbits), in.shape());

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using out_el_t = ScalarT;
      // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
      using out_shr_t = std::array<out_el_t, 3>;
      NdArrayView<out_shr_t> _out(out);

      // correlated randomness for RSS based multiplication.
      std::vector<out_el_t> r0(in.numel(), 0);
      std::vector<out_el_t> r1(in.numel(), 0);
      prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                              PrgState::GenPrssCtrl::Both);
#if !defined(ALKAID_USE_OFFLINE) || !defined(ALKAID_USE_PRG_STATE)
      std::fill(r0.begin(), r0.end(), 0);
      std::fill(r1.begin(), r1.end(), 0);
#endif

      // online.
      pforeach(0, in.numel(), [&](int64_t idx) {
        in_shr_t& i = _in[idx];
        out_shr_t& o = _out[idx];
        o[1] = r0[idx];
        o[2] = r1[idx];
        r0[idx] = i[0] ^ r0[idx];
      });

      r0 = comm->rotate2Next<out_el_t>(
          r0, "Resharing RSS to MSS, online");  // comm => 1, k

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
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using out_el_t = ScalarT;
      // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
      using out_shr_t = std::array<out_el_t, 2>;
      NdArrayView<out_shr_t> _out(out);

      // correlated randomness for RSS based multiplication.
      std::vector<out_el_t> r0(in.numel(), 0);
      std::vector<out_el_t> r1(in.numel(), 0);
      prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                              PrgState::GenPrssCtrl::Both);
#if !defined(ALKAID_USE_OFFLINE) || !defined(ALKAID_USE_PRG_STATE)
      std::fill(r0.begin(), r0.end(), 0);
      std::fill(r1.begin(), r1.end(), 0);
#endif

      // online.
      pforeach(0, in.numel(), [&](int64_t idx) {
        in_shr_t& i = _in[idx];
        out_shr_t& o = _out[idx];
        o[0] = i[0] ^ r0[idx] ^ r1[idx];
        r0[idx] = i[0] ^ r0[idx] ^ r1[idx];
      });

      // TODO: not safe. should add a mask to r1.
      r0 = comm->rotate<out_el_t>(
          r0, "Resharing ASS to RSS, online");  // comm => 1, k

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
NdArrayRef ResharingAss2Mrss(KernelEvalContext* ctx, const NdArrayRef& in) {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTyMrss>(out_btype, out_nbits), in.shape());

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using out_el_t = ScalarT;
      // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
      using out_shr_t = std::array<out_el_t, 3>;
      NdArrayView<out_shr_t> _out(out);

      // correlated randomness for RSS based multiplication.
      std::vector<out_el_t> r0(in.numel());
      std::vector<out_el_t> r1(in.numel());
      prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                              PrgState::GenPrssCtrl::Both);
#if !defined(ALKAID_USE_OFFLINE) || !defined(ALKAID_USE_PRG_STATE)
      std::fill(r0.begin(), r0.end(), 0);
      std::fill(r1.begin(), r1.end(), 0);
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
      // r0 = comm->rotate2Next<out_el_t>(r0, "Resharing ASS to MSS, online,
      // message 1");  // comm => 1, k r1 = comm->rotate<out_el_t>(r1,
      // "Resharing ASS to MSS, online, message 2");  // comm => 1, k
      comm->sendAsync<out_el_t>(
          comm->nextRank(), r0,
          "Resharing ASS to MSS, online, message 1");  // comm => 1, k
      comm->sendAsync<out_el_t>(
          comm->prevRank(), r1,
          "Resharing ASS to MSS, online, message 2");  // comm => 1, k
      r0 = comm->recv<out_el_t>(
          comm->prevRank(),
          "Resharing ASS to MSS, online, message 1");  // comm => 1, k
      r1 = comm->recv<out_el_t>(
          comm->nextRank(),
          "Resharing ASS to MSS, online, message 2");  // comm => 1, k
      comm->addCommStatsManually(1, 0);
      const std::atomic<size_t>& lctx_sent_actions =
          comm->lctx().get()->GetStats().get()->sent_actions;
      const std::atomic<size_t>& lctx_recv_actions =
          comm->lctx().get()->GetStats().get()->recv_actions;
      const_cast<std::atomic<size_t>&>(lctx_sent_actions) -= 1;
      const_cast<std::atomic<size_t>&>(lctx_recv_actions) -= 1;

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
NdArrayRef ResharingMrss2Rss(KernelEvalContext* ctx, const NdArrayRef& in) {
  const auto* in_ty = in.eltype().as<BShrTyMrss>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 3>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
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

        // assert(i[1] == 0 && i[2] == 0);
      });

      return out;
    });
  });
}

NdArrayRef ResharingMrss2RssAri(KernelEvalContext* ctx, const NdArrayRef& in) {
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();

  const auto field = in.eltype().as<AShrTyMrss>()->field();
  NdArrayRef out(makeType<AShrTy>(field), in.shape());

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using in_el_t = ring2k_t;
    using in_shr_t = std::array<in_el_t, 3>;
    NdArrayView<in_shr_t> _in(in);

    using out_el_t = ring2k_t;
    // mss(x) = (Dx, dx0, dx1), x = Dx ^ dx0 ^ dx1
    using out_shr_t = std::array<out_el_t, 2>;
    NdArrayView<out_shr_t> _out(out);

    // online.
    pforeach(0, in.numel(), [&](int64_t idx) {
      in_shr_t& i = _in[idx];
      out_shr_t& o = _out[idx];
      o[0] = (rank == 1) * i[0] - i[1];
      o[1] = (rank == 0) * i[0] - i[2];

      // assert(i[1] == 0 && i[2] == 0);
    });

    return out;
  });
}

// Resharing protocol from RSS to ASS.
NdArrayRef ResharingRss2Ass(KernelEvalContext* ctx, const NdArrayRef& in) {
  const auto* in_ty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_ty->nbits();
  const PtType out_btype = calcBShareBacktype(out_nbits);
  NdArrayRef out(makeType<BShrTy>(out_btype, out_nbits), in.shape());

  return DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    return DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
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

std::pair<NdArrayRef, NdArrayRef> unpack_2_bitvec_mss(const NdArrayRef& in) {
  const auto* in_ty = in.eltype().as<BShrTyMrss>();
  assert(in_ty->nbits() != 0 && in_ty->nbits() % 2 == 0);

  const size_t lo_nbits = in_ty->nbits() / 2;
  const size_t hi_nbits = in_ty->nbits() - lo_nbits;
  const PtType lo_btype = calcBShareBacktype(lo_nbits);
  const PtType hi_btype = calcBShareBacktype(hi_nbits);
  NdArrayRef lo(makeType<BShrTyMrss>(lo_btype, lo_nbits), in.shape());
  NdArrayRef hi(makeType<BShrTyMrss>(hi_btype, hi_nbits), in.shape());

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

void AddRounds(KernelEvalContext* ctx, size_t rounds, bool reduce_spu,
               bool reduce_yacl) {
  auto* comm = ctx->getState<Communicator>();
  if (reduce_yacl) {
    const std::atomic<size_t>& lctx_sent_actions =
        comm->lctx().get()->GetStats().get()->sent_actions;
    const_cast<std::atomic<size_t>&>(lctx_sent_actions) += rounds;
  }

  if (reduce_spu) {
    comm->addCommStatsManually(rounds, 0);
  }
}

void SubRounds(KernelEvalContext* ctx, size_t rounds, bool reduce_spu,
               bool reduce_yacl) {
  auto* comm = ctx->getState<Communicator>();
  if (reduce_yacl) {
    const std::atomic<size_t>& lctx_sent_actions =
        comm->lctx().get()->GetStats().get()->sent_actions;
    const_cast<std::atomic<size_t>&>(lctx_sent_actions) -= rounds;
  }

  if (reduce_spu) {
    comm->addCommStatsManually(-rounds, 0);
  }
}
}  // namespace spu::mpc::albo