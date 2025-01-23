#include "libspu/mpc/alkaid/core/bshare.h"

#include <atomic>
#include <functional>
#include <iostream>
#include <utility>

#include "yacl/utils/platform_utils.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/alkaid/core/type_helper.h"
#include "libspu/mpc/alkaid/type.h"
#include "libspu/mpc/alkaid/value.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/offline_recorder.h"
#include "libspu/mpc/utils/ring_ops.h"

#define P0_COUT \
  if (comm->getRank() == 0) std::cout

namespace spu::mpc::alkaid::core {
// Xor gate for ASS.
NdArrayRef AssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                   const NdArrayRef& rhs) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* lhs_shty = lhs.eltype().as<BShrTy>();
  const auto* rhs_shty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_shty->nbits(), rhs_shty->nbits());
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTy);
  const auto* out_shty = out_ty.as<BShrTy>();
  NdArrayRef out(out_ty, lhs.shape());

  DISPATCH_UINT_PT_TYPES(rhs_shty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;

    DISPATCH_UINT_PT_TYPES(lhs_shty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        NdArrayView<lhs_shr_t> _lhs(lhs);
        NdArrayView<rhs_shr_t> _rhs(rhs);
        NdArrayView<out_shr_t> _out(out);

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          auto& o = _out[idx];
          o[0] = l[0] ^ r[0];
        });
      });
    });
  });

  return out;
}

// Xor gate for RSS.
NdArrayRef RssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                   const NdArrayRef& rhs) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* lhs_shty = lhs.eltype().as<BShrTy>();
  const auto* rhs_shty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_shty->nbits(), rhs_shty->nbits());
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTy);
  const auto* out_shty = out_ty.as<BShrTy>();
  NdArrayRef out(out_ty, lhs.shape());

  DISPATCH_UINT_PT_TYPES(rhs_shty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;

    DISPATCH_UINT_PT_TYPES(lhs_shty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        NdArrayView<rhs_shr_t> _rhs(rhs);
        NdArrayView<lhs_shr_t> _lhs(lhs);
        NdArrayView<out_shr_t> _out(out);

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          out_shr_t& o = _out[idx];
          o[0] = l[0] ^ r[0];
          o[1] = l[1] ^ r[1];
        });
      });
    });
  });

  return out;
}

// Xor gate for MSS.
NdArrayRef MrssXor2(KernelEvalContext* ctx, const NdArrayRef& lhs,
                    const NdArrayRef& rhs) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* lhs_shty = lhs.eltype().as<BShrTyMrss>();
  const auto* rhs_shty = rhs.eltype().as<BShrTyMrss>();

  const size_t out_nbits = std::max(lhs_shty->nbits(), rhs_shty->nbits());
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTyMrss);
  const auto* out_shty = out_ty.as<BShrTyMrss>();
  NdArrayRef out(out_ty, lhs.shape());

  DISPATCH_UINT_PT_TYPES(rhs_shty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 3>;

    DISPATCH_UINT_PT_TYPES(lhs_shty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;

      DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 3>;

        NdArrayView<rhs_shr_t> _rhs(rhs);
        NdArrayView<lhs_shr_t> _lhs(lhs);
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
      });
    });
  });

  return out;
}

// And gate for RSS which outputs ASS result (no comunication).
NdArrayRef RssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* lhs_shty = lhs.eltype().as<BShrTy>();
  const auto* rhs_shty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::min(lhs_shty->nbits(), rhs_shty->nbits());
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTy);
  const auto* out_shty = out_ty.as<BShrTy>();
  NdArrayRef out(out_ty, lhs.shape());

  DISPATCH_UINT_PT_TYPES(rhs_shty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 2>;

    DISPATCH_UINT_PT_TYPES(lhs_shty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 2>;

      DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        NdArrayView<rhs_shr_t> _rhs(rhs);
        NdArrayView<lhs_shr_t> _lhs(lhs);
        NdArrayView<out_shr_t> _out(out);

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> tmp0(lhs.numel(), 0);
        std::vector<out_el_t> tmp1(lhs.numel(), 0);

        prg_state->fillPrssPair(tmp0.data(), tmp1.data(), tmp0.size(),
                                PrgState::GenPrssCtrl::Both);
#ifndef ALKAID_USE_PRG_STATE
        std::fill(tmp0.begin(), tmp0.end(), 0);
        std::fill(tmp1.begin(), tmp1.end(), 0);
#endif

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          out_shr_t& o = _out[idx];
          o[0] = (l[0] & r[0]) ^ (l[0] & r[1]) ^ (l[1] & r[0]) ^
                 (tmp0[idx] ^ tmp1[idx]);
        });
      });
    });
  });

  return out;
}

// And gate for MSS which outputs RSS result (no comunication).
NdArrayRef MrssAnd2NoComm(KernelEvalContext* ctx, const NdArrayRef& lhs,
                          const NdArrayRef& rhs) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* lhs_shty = lhs.eltype().as<BShrTyMrss>();
  const auto* rhs_shty = rhs.eltype().as<BShrTyMrss>();

  const size_t out_nbits = std::min(lhs_shty->nbits(), rhs_shty->nbits());
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTy);
  const auto* out_shty = out_ty.as<BShrTy>();
  NdArrayRef out(out_ty, lhs.shape());

  DISPATCH_UINT_PT_TYPES(rhs_shty->getBacktype(), [&]() {
    using rhs_el_t = ScalarT;
    using rhs_shr_t = std::array<rhs_el_t, 3>;

    DISPATCH_UINT_PT_TYPES(lhs_shty->getBacktype(), [&]() {
      using lhs_el_t = ScalarT;
      using lhs_shr_t = std::array<lhs_el_t, 3>;

      DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
        using out_el_t = ScalarT;
        using out_shr_t = std::array<out_el_t, 2>;

        NdArrayView<rhs_shr_t> _rhs(rhs);
        NdArrayView<lhs_shr_t> _lhs(lhs);
        NdArrayView<out_shr_t> _out(out);

        // correlated randomness for RSS based multiplication.
        std::vector<out_el_t> tmp0(lhs.numel(), 0);
        std::vector<out_el_t> tmp1(lhs.numel(), 0);
        prg_state->fillPrssPair(tmp0.data(), tmp1.data(), tmp0.size(),
                                PrgState::GenPrssCtrl::Both);

        if (comm->getRank() == 0) {
          OfflineRecorder::RecordMult(lhs.numel(),
                                      lhs.numel() * ((out_nbits + 7) / 8));
        }
#if !defined(ALKAID_USE_PRG_STATE) || !defined(ALKAID_USE_OFFLINE)
        std::fill(tmp0.begin(), tmp0.end(), 0);
        std::fill(tmp1.begin(), tmp1.end(), 0);
#endif
#ifdef ALKAID_USE_OFFLINE
        // dxy = dx & dy = (dx0 & dy0) ^ (dx0 & dy1) ^ (dx1 & dy0);
        // tmp0 is dxy0, tmp1 is dxy1.
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          tmp0[idx] = (l[1] & r[1]) ^ (l[1] & r[2]) ^ (l[2] & r[1]) ^
                    (tmp0[idx] ^ tmp1[idx]);
        });

        tmp1 = comm->rotate<out_el_t>(tmp0, "MrssAndBB, offline");  // comm => 1, k
// comm->addCommStatsManually(-1, -tmp0.size() * sizeof(out_el_t));
#endif

        pforeach(0, lhs.numel(), [&](int64_t idx) {
          const auto& l = _lhs[idx];
          const auto& r = _rhs[idx];
          out_shr_t& o = _out[idx];
          o[0] = ((l[0] & r[0])) ^ (l[0] & r[1]) ^ (l[1] & r[0]) ^
                 tmp0[idx];  // tmp0 is dxy0
          o[1] = ((l[0] & r[0])) ^ (l[0] & r[2]) ^ (l[2] & r[0]) ^
                 tmp1[idx];  // tmp1 is dxy1
        });
      });
    });
  });

  return out;
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
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* in_shty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_shty->nbits();
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTyMrss);
  const auto* out_shty = out_ty.as<BShrTyMrss>();
  NdArrayRef out(out_ty, in.shape());

  DISPATCH_UINT_PT_TYPES(in_shty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;
    NdArrayView<in_shr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 3>;
      NdArrayView<out_shr_t> _out(out);

      // correlated randomness for RSS based multiplication.
      std::vector<out_el_t> tmp0(in.numel(), 0);
      std::vector<out_el_t> tmp1(in.numel(), 0);
      prg_state->fillPrssPair(tmp0.data(), tmp1.data(), tmp0.size(),
                              PrgState::GenPrssCtrl::Both);
      #if !defined(ALKAID_USE_OFFLINE) || !defined(ALKAID_USE_PRG_STATE)
      std::fill(tmp0.begin(), tmp0.end(), 0);
      std::fill(tmp1.begin(), tmp1.end(), 0);
      #endif

      pforeach(0, in.numel(), [&](int64_t idx) {
        in_shr_t& i = _in[idx];
        out_shr_t& o = _out[idx];
        o[1] = tmp0[idx];
        o[2] = tmp1[idx];
        tmp0[idx] = i[0] ^ tmp0[idx];
      });

      tmp0 = comm->rotate2Next<out_el_t>(
          tmp0, "Resharing RSS to MSS, online");  // comm => 1, k

      pforeach(0, in.numel(), [&](int64_t idx) {
        in_shr_t& i = _in[idx];
        out_shr_t& o = _out[idx];
        o[0] = i[0] ^ i[1] ^ o[1] ^ o[2] ^ tmp0[idx];
      });
    });
  });

  return out;
}

// Resharing protocol from ASS to RSS.
// using RSS container to hold ASS.
NdArrayRef ResharingAss2Rss(KernelEvalContext* ctx, const NdArrayRef& in) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* in_shty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_shty->nbits();
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTy);
  const auto* out_shty = out_ty.as<BShrTy>();
  NdArrayRef out(out_ty, in.shape());

  DISPATCH_UINT_PT_TYPES(in_shty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;

    DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayView<in_shr_t> _in(in);
      NdArrayView<out_shr_t> _out(out);

      // generate zero-sharing.
      std::vector<out_el_t> tmp0(in.numel(), 0);
      std::vector<out_el_t> tmp1(in.numel(), 0); 
      prg_state->fillPrssPair(tmp0.data(), tmp1.data(), tmp0.size(),
                              PrgState::GenPrssCtrl::Both);
#if !defined(ALKAID_USE_OFFLINE) || !defined(ALKAID_USE_PRG_STATE)
      std::fill(tmp0.begin(), tmp0.end(), 0);
      std::fill(tmp1.begin(), tmp1.end(), 0);
#endif

      pforeach(0, in.numel(), [&](int64_t idx) {
        in_shr_t& i   = _in[idx];
        out_shr_t& o  = _out[idx];
        o[0]      = i[0] ^ tmp0[idx] ^ tmp1[idx];
        tmp0[idx] = o[0];
      });

      // TODO: not safe. should add a mask to tmp1.
      tmp0 = comm->rotate<out_el_t>(
          tmp0, "Resharing ASS to RSS, online");  // comm => 1, k

      pforeach(0, in.numel(), [&](int64_t idx) {
        out_shr_t& o = _out[idx];
        o[1] = tmp0[idx];
      });
    });
  });

  return out;
}

// Resharing protocol from ASS to MSS.
// using RSS container to hold ASS.
NdArrayRef ResharingAss2Mrss(KernelEvalContext* ctx, const NdArrayRef& in) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* in_shty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_shty->nbits();
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTyMrss);
  const auto* out_shty = out_ty.as<BShrTyMrss>();
  NdArrayRef out(out_ty, in.shape());

  DISPATCH_UINT_PT_TYPES(in_shty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;

    DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 3>;

      NdArrayView<in_shr_t> _in(in);
      NdArrayView<out_shr_t> _out(out);

      // generate rss(rx) and zero-sharing.
      std::vector<out_el_t> tmp0(in.numel(), 0);
      std::vector<out_el_t> tmp1(in.numel(), 0); 
      std::vector<out_el_t> tmp2(in.numel(), 0); 
      std::vector<out_el_t> tmp3(in.numel(), 0); 
      prg_state->fillPrssPair(tmp0.data(), tmp1.data(), tmp0.size(),
                              PrgState::GenPrssCtrl::Both);
      prg_state->fillPrssPair(tmp2.data(), tmp3.data(), tmp2.size(),
                              PrgState::GenPrssCtrl::Both);
#if !defined(ALKAID_USE_OFFLINE) || !defined(ALKAID_USE_PRG_STATE)
      std::fill(tmp0.begin(), tmp0.end(), 0);
      std::fill(tmp1.begin(), tmp1.end(), 0);
      std::fill(tmp2.begin(), tmp2.end(), 0);
      std::fill(tmp3.begin(), tmp3.end(), 0);
#endif

      pforeach(0, in.numel(), [&](int64_t idx) {
        in_shr_t& i   = _in[idx];
        out_shr_t& o  = _out[idx];
        o[1]          = tmp0[idx];
        o[2]          = tmp1[idx];
        tmp0[idx]     = i[0] ^ tmp0[idx] ^ tmp2[idx] ^ tmp3[idx];
      });

      comm->sendAsync<out_el_t>(
          comm->nextRank(), tmp0,
          "Resharing ASS to MSS, online, message 1");  // comm => 1, k
      comm->sendAsync<out_el_t>(
          comm->prevRank(), tmp0,
          "Resharing ASS to MSS, online, message 2");  // comm => 1, k
      tmp2 = comm->recv<out_el_t>(
          comm->prevRank(),
          "Resharing ASS to MSS, online, message 1");  // comm => 1, k
      tmp3 = comm->recv<out_el_t>(
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
        out_shr_t& o  = _out[idx];
        o[0]          = tmp0[idx] ^ tmp2[idx] ^ tmp3[idx];
      });
    });
  });

  return out;
}

// Resharing protocol from MSS to RSS.
NdArrayRef ResharingMrss2Rss(KernelEvalContext* ctx, const NdArrayRef& in) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* in_shty = in.eltype().as<BShrTyMrss>();

  const size_t out_nbits = in_shty->nbits();
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTy);
  const auto* out_shty = out_ty.as<BShrTy>();
  NdArrayRef out(out_ty, in.shape());

  DISPATCH_UINT_PT_TYPES(in_shty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 3>;

    DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayView<in_shr_t> _in(in);
      NdArrayView<out_shr_t> _out(out);

      // online.
      pforeach(0, in.numel(), [&](int64_t idx) {
        in_shr_t& i = _in[idx];
        out_shr_t& o = _out[idx];
        o[0] = i[0] ^ i[1];
        o[1] = i[0] ^ i[2];
      });
    });
  });

  return out;
}

NdArrayRef ResharingMrss2RssAri(KernelEvalContext* ctx, const NdArrayRef& in) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();

  const auto field = in.eltype().as<AShrTyMrss>()->field();
  NdArrayRef out(makeType<AShrTy>(field), in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using in_el_t = ring2k_t;
    using in_shr_t = std::array<in_el_t, 3>;

    using out_el_t = ring2k_t;
    using out_shr_t = std::array<out_el_t, 2>;

    NdArrayView<in_shr_t> _in(in);
    NdArrayView<out_shr_t> _out(out);

    // online.
    pforeach(0, in.numel(), [&](int64_t idx) {
      in_shr_t& i = _in[idx];
      out_shr_t& o = _out[idx];
      o[0] = (rank == 1) * i[0] - i[1];
      o[1] = (rank == 0) * i[0] - i[2];
    });
  });

  return out;
}

// Resharing protocol from RSS to ASS.
NdArrayRef ResharingRss2Ass(KernelEvalContext* ctx, const NdArrayRef& in) {
  [[maybe_unused]] auto* prg_state = ctx->getState<PrgState>();
  [[maybe_unused]] auto* comm = ctx->getState<Communicator>();

  const auto* in_shty = in.eltype().as<BShrTy>();

  const size_t out_nbits = in_shty->nbits();
  const Type out_ty = GET_BTYPE_FROM_BW(out_nbits, BShrTy);
  const auto* out_shty = out_ty.as<BShrTy>();
  NdArrayRef out(out_ty, in.shape());

  DISPATCH_UINT_PT_TYPES(in_shty->getBacktype(), [&]() {
    using in_el_t = ScalarT;
    using in_shr_t = std::array<in_el_t, 2>;

    DISPATCH_UINT_PT_TYPES(out_shty->getBacktype(), [&]() {
      using out_el_t = ScalarT;
      using out_shr_t = std::array<out_el_t, 2>;

      NdArrayView<in_shr_t> _in(in);
      NdArrayView<out_shr_t> _out(out);

      // zero-sharing for resharing from rss to ass.
      std::vector<out_el_t> tmp0(in.numel(), 0);
      std::vector<out_el_t> tmp1(in.numel(), 0);
      prg_state->fillPrssPair(tmp0.data(), tmp1.data(), tmp0.size(),
                              PrgState::GenPrssCtrl::Both);
#if !defined(ALKAID_USE_OFFLINE) || !defined(ALKAID_USE_PRG_STATE)
      std::fill(tmp0.begin(), tmp0.end(), 0);
      std::fill(tmp1.begin(), tmp1.end(), 0);
#endif

      pforeach(0, in.numel(), [&](int64_t idx) {
        in_shr_t& i = _in[idx];
        out_shr_t& o = _out[idx];
        o[0] = i[0] ^ tmp0[idx] ^ tmp1[idx];
        o[1] = 0;
      });
    });
  });

  return out;
}
}  // namespace spu::mpc::alkaid::core