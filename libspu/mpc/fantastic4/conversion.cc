// Copyright 2025 Ant Group Co., Ltd.
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

#include "libspu/mpc/fantastic4/conversion.h"
#include <functional>
#include "yacl/utils/platform_utils.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

#include "libspu/mpc/fantastic4/jmp.h"
#ifdef OPTIMIZED_F4
#define OPTIMIZED_CONVERSION
#endif

namespace spu::mpc::fantastic4 {

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_and_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(and_bb(ctx, WrapValue(x), WrapValue(y)));
}

#ifndef OPTIMIZED_CONVERSION
static std::array<NdArrayRef, 2> wrap_pfa_bb(SPUContext* ctx, const NdArrayRef& x,
  const NdArrayRef& y, const NdArrayRef& cin) {
  SPU_ENFORCE(x.shape() == y.shape());
  SPU_ENFORCE(x.shape() == cin.shape());
  auto res = pfa_bb(ctx, WrapValue(x), WrapValue(y), WrapValue(cin));
  std::array<NdArrayRef, 2> out = {UnwrapValue(res[0]), UnwrapValue(res[1])};
  return out;
}

static NdArrayRef wrap_lshift_b(SPUContext* ctx, const NdArrayRef& x, size_t k) {
  return UnwrapValue(lshift_b(ctx, WrapValue(x), {static_cast<int64_t>(k)}));
}

// Fantastic4 A2B based on Local Share Conversion
// [x] = (x0, x1, x2, x3)
//     [x0] = (x0, 0, 0, 0)
//     [x1] = (0, x1, 0, 0)
//     [x2] = (0, 0, x2, 0)
//     [x3] = (0, 0, 0, x3)
// Fantastic4 uses FA to reduce 4 operands to 2 operands
// Ref: Malicious A2B in ABY3, which is generalized to 4PC in Fantastic4 
//      P10 "...with the local share conversion for replicated secret sharing [4, 30], which we call share splitting..."
NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  const PtType out_btype = calcBShareBacktype(SizeOf(field) * 8);
  const auto out_ty = makeType<BShrTy>(out_btype, SizeOf(out_btype) * 8);
  auto numel = in.numel();

  auto shr0_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
  auto shr1_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
  auto shr2_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
  auto shr3_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
  memset(shr0_buf->data(), 0, in.shape().numel() * out_ty.size());
  memset(shr1_buf->data(), 0, in.shape().numel() * out_ty.size());
  memset(shr2_buf->data(), 0, in.shape().numel() * out_ty.size());
  memset(shr3_buf->data(), 0, in.shape().numel() * out_ty.size());

  NdArrayRef shr0(shr0_buf, out_ty, in.shape());
  NdArrayRef shr1(shr1_buf, out_ty, in.shape());
  NdArrayRef shr2(shr2_buf, out_ty, in.shape());
  NdArrayRef shr3(shr3_buf, out_ty, in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_t = std::array<ring2k_t, 3>;
    NdArrayView<ashr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 3>;

      NdArrayView<bshr_t> _shr0(shr0);
      NdArrayView<bshr_t> _shr1(shr1);
      NdArrayView<bshr_t> _shr2(shr2);
      NdArrayView<bshr_t> _shr3(shr3);

      if(rank == 0){
        pforeach(0, numel, [&](int64_t idx) {
            // P0 holds _in(x0, x1, x2)
            _shr0[idx][0] = _in[idx][0];
            _shr1[idx][1] = _in[idx][1];
            _shr2[idx][2] = _in[idx][2];
        });
      }
      else if(rank == 1){
        pforeach(0, numel, [&](int64_t idx) {
            // P1 holds _in(x1, x2, x3)
            _shr1[idx][0] = _in[idx][0];
            _shr2[idx][1] = _in[idx][1];
            _shr3[idx][2] = _in[idx][2];
        });
      }
      else if(rank == 2){
        pforeach(0, numel, [&](int64_t idx) {
            // P2 holds _in(x2, x3, x0)
            _shr2[idx][0] = _in[idx][0];
            _shr3[idx][1] = _in[idx][1];
            _shr0[idx][2] = _in[idx][2];
        });
      }
      else if(rank == 3){
        pforeach(0, numel, [&](int64_t idx) {
            // P3 holds _in(x3, x0, x1)
            _shr3[idx][0] = _in[idx][0];
            _shr0[idx][1] = _in[idx][1];
            _shr1[idx][2] = _in[idx][2];
        });
      }
    });
  });

  auto s_cout_0 = wrap_pfa_bb(ctx->sctx(), shr0, shr1, shr2);
  NdArrayRef s0 = s_cout_0[0];
  NdArrayRef cout0 = s_cout_0[1];
  auto cout0_times_2 = wrap_lshift_b(ctx->sctx(), cout0, 1);

  auto s_cout_1 = wrap_pfa_bb(ctx->sctx(), s0, cout0_times_2, shr3);
  NdArrayRef s1 = s_cout_1[0];
  NdArrayRef cout1 = s_cout_1[1];
  auto cout1_times_2 = wrap_lshift_b(ctx->sctx(), cout1, 1);

  return wrap_add_bb(ctx->sctx(), s1, cout1_times_2);
}

#else
// Optimized A2B By Ranyang Liu, Nankai University
// The semi-honest A2B of ABY3 fails in malicious settings since the corrupted party could contribute x1 + x2 + err while the honest party
// Fantastic4 adopts the malicious method of ABY3 that leverages local share conversion with FA and Binary Adders
// We take the advantage of 4-party replicated secret sharing
// Let
//      (P0, P1) share (x1 + x2)
//      (P2, P3) share (x0 + x3)
//      Jointly Compute Binary Adder, e.g. PPA((x1 + x2), (x0 + x3)) without the need of tree reduction (2k Full Adders)
NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();

  auto* comm = ctx->getState<Communicator>();

  auto rank = comm->getRank();

  const PtType out_btype = calcBShareBacktype(SizeOf(field) * 8);
  const auto out_ty = makeType<BShrTy>(out_btype, SizeOf(out_btype) * 8);

  auto x1_plus_x2_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
  auto x0_plus_x3_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
  memset(x1_plus_x2_buf->data(), 0, in.shape().numel() * out_ty.size());
  memset(x0_plus_x3_buf->data(), 0, in.shape().numel() * out_ty.size());
  NdArrayRef x1_plus_x2_shr(x1_plus_x2_buf, out_ty, in.shape());
  NdArrayRef x0_plus_x3_shr(x0_plus_x3_buf, out_ty, in.shape());

  auto numel = in.numel();

  DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_t = std::array<ring2k_t, 3>;
    NdArrayView<ashr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using bshr_el_t = ScalarT;

      std::vector<bshr_el_t> x1_plus_x2(numel);
      std::vector<bshr_el_t> x0_plus_x3(numel);
      if(rank == 0){
        pforeach(0, numel, [&](int64_t idx) {
            x1_plus_x2[idx] ^= _in[idx][1] + _in[idx][2];
        });
      }
      else if(rank == 1){
        pforeach(0, numel, [&](int64_t idx) {
            x1_plus_x2[idx] ^= _in[idx][0] + _in[idx][1];
        });
      }
      else if(rank == 2){
        pforeach(0, numel, [&](int64_t idx) {
            x0_plus_x3[idx] ^= _in[idx][1] + _in[idx][2];
        });
      }
      else if(rank == 3){
        pforeach(0, numel, [&](int64_t idx) {
            x0_plus_x3[idx] ^= _in[idx][0] + _in[idx][1];
        });
      }
      JointInputBool(ctx, x1_plus_x2, x1_plus_x2_shr, 0, 1, 2, 3);
      JointInputBool(ctx, x0_plus_x3, x0_plus_x3_shr, 3, 2, 1, 0);
    });
  });

  return wrap_add_bb(ctx->sctx(), x1_plus_x2_shr, x0_plus_x3_shr);
}

#endif

#ifndef OPTIMIZED_CONVERSION

static NdArrayRef wrap_mul_aa(SPUContext* ctx, const NdArrayRef& x,
  const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_a2b(SPUContext* ctx, const NdArrayRef& x) {

  return UnwrapValue(a2b(ctx, WrapValue(x)));
}

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
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
      NdArrayView<std::array<ring2k_t, 3>> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = 0;
        _out[idx][2] = 0;
      });
    });
    return out;
  }

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto rank = comm->getRank();

  const auto expanded_ty = makeType<BShrTy>(
    calcBShareBacktype(SizeOf(field) * 8), SizeOf(field) * 8);

  NdArrayRef x(expanded_ty, in.shape());
  NdArrayRef rand_ashr(out_ty, in.shape());
  NdArrayRef rand_bshr(expanded_ty, in.shape());

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using bshr_t = std::array<ScalarT, 3>;

    NdArrayView<bshr_t> _in(in);

    DISPATCH_ALL_FIELDS(field, [&]() {
      using ashr_el_t = ring2k_t;
      using ashr_t = std::array<ashr_el_t, 3>;
      NdArrayView<ashr_t> _out(out);
      NdArrayView<ashr_t> _x(x);

      if (in_nbits == 1) {
        std::vector<ashr_el_t> x1_xor_x2(numel);
        std::vector<ashr_el_t> x0_xor_x3(numel);

        auto x1_xor_x2_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
        memset(x1_xor_x2_buf->data(), 0, in.shape().numel() * out_ty.size());
        NdArrayRef x1_xor_x2_shr(x1_xor_x2_buf, out_ty, in.shape());
        NdArrayView<ashr_t> _x1_xor_x2_shr(x1_xor_x2_shr);

        auto x0_xor_x3_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
        memset(x0_xor_x3_buf->data(), 0, in.shape().numel() * out_ty.size());
        NdArrayRef x0_xor_x3_shr(x0_xor_x3_buf, out_ty, in.shape());
        NdArrayView<ashr_t> _x0_xor_x3_shr(x0_xor_x3_shr);

        pforeach(0, numel, [&](int64_t idx) {
          const auto& v = _in[idx];
          _x[idx][0] = v[0] & 0x1;
          _x[idx][1] = v[1] & 0x1;
          _x[idx][2] = v[2] & 0x1;
        });

        if (comm->getRank() == 0) {
          pforeach(0, numel, [&](int64_t idx) {
            x1_xor_x2[idx] = _x[idx][1] ^ _x[idx][2];
          });
        }
        else if (comm->getRank() == 1) {
          pforeach(0, numel, [&](int64_t idx) {
            x1_xor_x2[idx] = _x[idx][0] ^ _x[idx][1];
          });
        }
        else if (comm->getRank() == 2) {
          pforeach(0, numel, [&](int64_t idx) {
            x0_xor_x3[idx] = _x[idx][1] ^ _x[idx][2];
          });
        }
        else if (comm->getRank() == 3) {
          pforeach(0, numel, [&](int64_t idx) {
            x0_xor_x3[idx] = _x[idx][0] ^ _x[idx][1];
          });
        }
        JointInputArithSend(ctx, x1_xor_x2, x1_xor_x2_shr, 0, 1, 2, 3);
        JointInputArithSend(ctx, x0_xor_x3, x0_xor_x3_shr, 2, 3, 0, 1);

        JointInputArithRecv(ctx, x1_xor_x2, x1_xor_x2_shr, 0, 1, 2, 3);
        JointInputArithRecv(ctx, x0_xor_x3, x0_xor_x3_shr, 2, 3, 0, 1);

        auto mul_res = wrap_mul_aa(ctx->sctx(), x1_xor_x2_shr, x0_xor_x3_shr);
        NdArrayView<ashr_t> _mul_res(mul_res);

        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = _x1_xor_x2_shr[idx][0] + _x0_xor_x3_shr[idx][0] - 2 * _mul_res[idx][0];
          _out[idx][1] = _x1_xor_x2_shr[idx][1] + _x0_xor_x3_shr[idx][1] - 2 * _mul_res[idx][1];
          _out[idx][2] = _x1_xor_x2_shr[idx][2] + _x0_xor_x3_shr[idx][2] - 2 * _mul_res[idx][2];
        });
      }

      else {
        pforeach(0, numel, [&](int64_t idx) {
          const auto& v = _in[idx];
          _x[idx][0] = v[0];
          _x[idx][1] = v[1];
          _x[idx][2] = v[2];
        });
        // First Generate random arithmetic share
        std::vector<ashr_el_t> r0(numel);
        std::vector<ashr_el_t> r1(numel);
        std::vector<ashr_el_t> r2(numel);

        prg_state->fillPrssTuple<ashr_el_t>(r0.data(), nullptr, nullptr, r0.size(),
                                  PrgState::GenPrssCtrl::First);
        prg_state->fillPrssTuple<ashr_el_t>(nullptr, r1.data(), nullptr, r1.size(),
                                  PrgState::GenPrssCtrl::Second);
        prg_state->fillPrssTuple<ashr_el_t>(nullptr, nullptr, r2.data(), r2.size(),
                                  PrgState::GenPrssCtrl::Third);

        NdArrayView<ashr_t> _rand_ashr(rand_ashr);
        NdArrayView<ashr_t> _x(x);

        pforeach(0, numel, [&](int64_t idx) {
          _rand_ashr[idx][0] = r0[idx] ;
          _rand_ashr[idx][1] = r1[idx] ;
          _rand_ashr[idx][2] = r2[idx] ;

          // Expand the input to ensure the input bit-length = random value bit-length when input into binary adder
          const auto& v = _in[idx];
          _x[idx][0] = v[0];
          _x[idx][1] = v[1];
          _x[idx][2] = v[2];
        });

        // Compute Boolean shares of r (edabits)
        rand_bshr = wrap_a2b(ctx->sctx(), rand_ashr);

        // Compute Boolean shares of x + r
        auto x_plus_r_shr = wrap_add_bb(ctx->sctx(), x, rand_bshr);
        NdArrayView<ashr_t> _x_plus_r_shr(x_plus_r_shr);

        // Reveal x + r to all parties
        std::vector<ashr_el_t> second_shr(numel);
        std::vector<ashr_el_t> third_shr(numel);
        std::vector<ashr_el_t> x_plus_r_pub(numel);
        pforeach(0, numel, [&](int64_t idx) {
          second_shr[idx] = _x_plus_r_shr[idx][1];
          third_shr[idx] = _x_plus_r_shr[idx][2];
        });

        // Pass the third share to previous party
        auto fourth_shr = JointMsgRotate<ashr_el_t>(ctx, third_shr, second_shr);

        pforeach(0, numel, [&](int64_t idx) {
          x_plus_r_pub[idx] = _x_plus_r_shr[idx][0] ^ _x_plus_r_shr[idx][1] ^ _x_plus_r_shr[idx][2] ^ fourth_shr[idx];
        });

        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = -_rand_ashr[idx][0];
          _out[idx][1] = -_rand_ashr[idx][1];
          _out[idx][2] = -_rand_ashr[idx][2];
          if (rank == 0) {_out[idx][0] += x_plus_r_pub[idx];}
          if (rank == 2) {_out[idx][2] += x_plus_r_pub[idx];}
          if (rank == 3) {_out[idx][1] += x_plus_r_pub[idx];}
        });
      }
    });

  });
  return out;
}

#else
// Optimized B2A By Ranyang Liu, Nankai University
// Fantastic4 gives the 4-party generation of edabits based on its A2B
// Given edabitsthe B2A can be implemented using Binary Adders
// We take the advantage of 4-party replicated secret sharing
// We do not require r to be unknown to all party, instead r only unknown to other group
// Let
//      (P0, P1) share random r in both arithmetic and boolean form
//      Jointly Compute Binary Adder to obtain [x + r]_B and reveal x + r to (P2, P3)
//      (P2, P3) share x + r in arithmetic form
//      All parties set [x]_A = [x + r]_A - [r]_A
NdArrayRef Opt_MulAA(KernelEvalContext* ctx, const NdArrayRef& lhs, const NdArrayRef& rhs) {
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto next_rank = (rank + 1) % 4;

  return DISPATCH_ALL_FIELDS(field, [&]() {
  using el_t = ring2k_t;
  using shr_t = std::array<el_t, 3>;

  NdArrayView<shr_t> _lhs(lhs);
  NdArrayView<shr_t> _rhs(rhs);
  auto out_ty = makeType<AShrTy>(field);
  auto out_buf = std::make_shared<yacl::Buffer>(lhs.shape().numel() * out_ty.size());
  memset(out_buf->data(), 0, lhs.shape().numel() * out_ty.size());
  NdArrayRef out(out_buf ,out_ty, lhs.shape());
  NdArrayView<shr_t> _out(out);

  std::array<std::vector<el_t>, 5> a;

  for (auto& vec : a) {
    vec = std::vector<el_t>(lhs.numel());
  }

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    for(auto i =0; i<5;i++){
      a[i][idx] = 0;
    }
  });

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    a[rank][idx] = (_lhs[idx][0] + _lhs[idx][1]) * _rhs[idx][0] + _lhs[idx][0] * _rhs[idx][1];
    a[next_rank][idx] = (_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][1] + _lhs[idx][1] * _rhs[idx][2];
    a[4][idx] = _lhs[idx][0] * _rhs[idx][2] + _lhs[idx][2] * _rhs[idx][0];
    SPU_ENFORCE(a[3][idx] == 0);
    SPU_ENFORCE(a[1][idx] == 0);
  });

  // Do not send JointInputArith<el_t>(ctx, a[1], out, 0, 1, 3, 2);
  // Do not send JointInputArith<el_t>(ctx, a[3], out, 2, 3, 1, 0);
  JointInputArithSend<el_t>(ctx, a[0], out, 0, 3, 1, 2);
  JointInputArithSend<el_t>(ctx, a[2], out, 1, 2, 0, 3);
  JointInputArithSend<el_t>(ctx, a[4], out, 2, 0, 3, 1);
  JointInputArithSend<el_t>(ctx, a[4], out, 3, 1, 2, 0);

  JointInputArithRecv<el_t>(ctx, a[0], out, 0, 3, 1, 2);
  JointInputArithRecv<el_t>(ctx, a[2], out, 1, 2, 0, 3);
  JointInputArithRecv<el_t>(ctx, a[4], out, 2, 0, 3, 1);
  JointInputArithRecv<el_t>(ctx, a[4], out, 3, 1, 2, 0);

  return out;
  });
}


NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
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
        _out[idx][2] = 0;
      });
    });
    return out;
  }

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), [&]() {
    using bshr_t = std::array<ScalarT, 3>;
    NdArrayView<bshr_t> _in(in);

    DISPATCH_ALL_FIELDS(field, [&]() {
      using ashr_el_t = ring2k_t;
      using ashr_t = std::array<ashr_el_t, 3>;

      // first expand b share to a share length.
      const auto expanded_ty = makeType<BShrTy>(
          calcBShareBacktype(SizeOf(field) * 8), SizeOf(field) * 8);

      NdArrayRef x(expanded_ty, in.shape());
      NdArrayView<ashr_t> _x(x);

      NdArrayView<ashr_t> _out(out);

      if (in_nbits == 1) {
        std::vector<ashr_el_t> x1_xor_x2(numel);
        std::vector<ashr_el_t> x0_xor_x3(numel);

        auto x1_xor_x2_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
        memset(x1_xor_x2_buf->data(), 0, in.shape().numel() * out_ty.size());
        NdArrayRef x1_xor_x2_shr(x1_xor_x2_buf, out_ty, in.shape());
        NdArrayView<ashr_t> _x1_xor_x2_shr(x1_xor_x2_shr);

        auto x0_xor_x3_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * out_ty.size());
        memset(x0_xor_x3_buf->data(), 0, in.shape().numel() * out_ty.size());
        NdArrayRef x0_xor_x3_shr(x0_xor_x3_buf, out_ty, in.shape());
        NdArrayView<ashr_t> _x0_xor_x3_shr(x0_xor_x3_shr);

        pforeach(0, numel, [&](int64_t idx) {
          const auto& v = _in[idx];
          _x[idx][0] = v[0] & 0x1;
          _x[idx][1] = v[1] & 0x1;
          _x[idx][2] = v[2] & 0x1;
        });

        if (comm->getRank() == 0) {
          pforeach(0, numel, [&](int64_t idx) {
            x1_xor_x2[idx] = _x[idx][1] ^ _x[idx][2];
          });
        }
        else if (comm->getRank() == 1) {
          pforeach(0, numel, [&](int64_t idx) {
            x1_xor_x2[idx] = _x[idx][0] ^ _x[idx][1];
          });
        }
        else if (comm->getRank() == 2) {
          pforeach(0, numel, [&](int64_t idx) {
            x0_xor_x3[idx] = _x[idx][1] ^ _x[idx][2];
          });
        }
        else if (comm->getRank() == 3) {
          pforeach(0, numel, [&](int64_t idx) {
            x0_xor_x3[idx] = _x[idx][0] ^ _x[idx][1];
          });
        }

        JointInputArithSend(ctx, x1_xor_x2, x1_xor_x2_shr, 0, 1, 2, 3);
        JointInputArithSend(ctx, x0_xor_x3, x0_xor_x3_shr, 2, 3, 0, 1);

        JointInputArithRecv(ctx, x1_xor_x2, x1_xor_x2_shr, 0, 1, 2, 3);
        JointInputArithRecv(ctx, x0_xor_x3, x0_xor_x3_shr, 2, 3, 0, 1);
        // Above we let:
        //    P0 sends P2 x1^x2 - r1
        //    P2 sends P0 x0^x3 - r3
        // As the result:
        //    x1_xor_x2_shr = (0         ,- r1, x1^x2 - r1, 0)
        //    x0_xor_x3_shr = (x2^x3 - r3,   0,      0    , r3)
        // Recall that in 4PC MulAA:
        //    pforeach(0, lhs.numel(), [&](int64_t idx) {
        //      a[rank][idx] = (_lhs[idx][0] + _lhs[idx][1]) * _rhs[idx][0] + _lhs[idx][0] * _rhs[idx][1]; // xi*yi + xi*yj + xj*yi
        //      a[next_rank][idx] = (_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][1] + _lhs[idx][1] * _rhs[idx][2];  // xj*yj + xj*yg + xg*yj
        //      a[4][idx] = _lhs[idx][0] * _rhs[idx][2] + _lhs[idx][2] * _rhs[idx][0];                    // xi*yg + xg*yi
        //    });
        // Here we have:
        //    a[3] = 0
        //    a[1] = 0
        // For optimization, we do not send them in Opt_Mul.
        auto mul_res = Opt_MulAA(ctx, x1_xor_x2_shr, x0_xor_x3_shr);// Mind the order of lhs and rhs

        NdArrayView<ashr_t> _mul_res(mul_res);

        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = _x1_xor_x2_shr[idx][0] + _x0_xor_x3_shr[idx][0] - 2 * _mul_res[idx][0];
          _out[idx][1] = _x1_xor_x2_shr[idx][1] + _x0_xor_x3_shr[idx][1] - 2 * _mul_res[idx][1];
          _out[idx][2] = _x1_xor_x2_shr[idx][2] + _x0_xor_x3_shr[idx][2] - 2 * _mul_res[idx][2];
        });


      }
      else {
        pforeach(0, numel, [&](int64_t idx) {
          const auto& v = _in[idx];
          _x[idx][0] = v[0];
          _x[idx][1] = v[1];
          _x[idx][2] = v[2];
        });
        // P0 & P1 invoke PRG[1], PRG[2]
        // P2 invoke PRG[2], P3 invoke PRG[1]
        std::vector<ashr_el_t> r1(numel);
        std::vector<ashr_el_t> r2(numel);
        std::vector<ashr_el_t> r(numel);
        std::vector<ashr_el_t> neg_r(numel);

        auto neg_r_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * expanded_ty.size());
        auto r_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * expanded_ty.size());
        auto x_minus_r_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * expanded_ty.size());

        memset(neg_r_buf->data(), 0, in.shape().numel() * expanded_ty.size());
        memset(r_buf->data(), 0, in.shape().numel() * expanded_ty.size());
        memset(x_minus_r_buf->data(), 0, in.shape().numel() * expanded_ty.size());

        NdArrayRef neg_r_shr(neg_r_buf, expanded_ty, in.shape());
        NdArrayView<ashr_t> _neg_r_shr(neg_r_shr);

        NdArrayRef r_shr(r_buf, expanded_ty, in.shape());
        NdArrayView<ashr_t> _r_shr(r_shr);

        NdArrayRef x_minus_r_shr(x_minus_r_buf, expanded_ty, in.shape());
        NdArrayView<ashr_t> _x_minus_r_shr(x_minus_r_shr);

        if (comm->getRank() == 0) {
          // Sample r1, r2
          prg_state->fillPrssTuple<ashr_el_t>(nullptr, r1.data(), nullptr, r1.size(),
                                PrgState::GenPrssCtrl::Second);
          prg_state->fillPrssTuple<ashr_el_t>(nullptr, nullptr, r2.data(), r2.size(),
                                PrgState::GenPrssCtrl::Third);
          // r = r1 + r2
          pforeach(0, numel, [&](int64_t idx) {
              r[idx] = r1[idx] + r2[idx];
              neg_r[idx] = - r[idx];
          });

        } else if (comm->getRank() == 1) {
          prg_state->fillPrssTuple<ashr_el_t>(r1.data(), nullptr, nullptr, r1.size(),
                                PrgState::GenPrssCtrl::First);
          prg_state->fillPrssTuple<ashr_el_t>(nullptr, r2.data(), nullptr, r2.size(),
                                PrgState::GenPrssCtrl::Second);

          pforeach(0, numel, [&](int64_t idx) {
              r[idx] = r1[idx] + r2[idx];
              neg_r[idx] = - r[idx];
          });

        } else if (comm->getRank() == 2) {
          prg_state->fillPrssTuple<ashr_el_t>(r2.data(), nullptr, nullptr, r2.size(),
                                PrgState::GenPrssCtrl::First);

        } else if (comm->getRank() == 3) {
          prg_state->fillPrssTuple<ashr_el_t>(nullptr, nullptr, r1.data(), r1.size(),
                                PrgState::GenPrssCtrl::Third);

        }

        // P0, P1 share [-r]B
        JointInputArith(ctx, r, r_shr, 0, 1, 2, 3);

        JointInputBool(ctx, neg_r, neg_r_shr, 0, 1, 2, 3);

        // compute [x-r]B
        // comm => log(k) + 1, 2k(logk) + k
        auto x_minus_r = wrap_add_bb(ctx->sctx(), x, neg_r_shr);

        // reveal x-r to P2, P3
        NdArrayView<ashr_t> _x_minus_r(x_minus_r);

        std::vector<ashr_el_t> plaintext_x_minus_r(numel);

        // P1 serves as backup for both (P2-->P3) (P3-->P2)
        if(comm->getRank() == 1) {
          std::vector<ashr_el_t> shr_for_P3(numel);
          std::vector<ashr_el_t> shr_for_P2(numel);
          pforeach(0, numel, [&](int64_t idx) {
              shr_for_P2[idx] = _x_minus_r[idx][0];
              shr_for_P3[idx] = _x_minus_r[idx][1];
          });
          JointMsgPass(ctx, shr_for_P3, 2, 1, 3);
          JointMsgPass(ctx, shr_for_P2, 3, 1, 2);
        }
        if (comm->getRank() == 2) {
          // P2 send global shr[2] (own::shr[0]) to P3
          std::vector<ashr_el_t> shr_for_P3(numel);
          std::vector<ashr_el_t> shr_for_P2(numel);
          pforeach(0, numel, [&](int64_t idx) { shr_for_P3[idx] = _x_minus_r[idx][0]; });

          JointMsgPass(ctx, shr_for_P3, 2, 1, 3);
          JointMsgPass(ctx, shr_for_P2, 3, 1, 2);

          pforeach(0, numel,
                  [&](int64_t idx) { plaintext_x_minus_r[idx] = _x_minus_r[idx][0] ^ _x_minus_r[idx][1] ^ _x_minus_r[idx][2] ^ shr_for_P2[idx]; });

        }
        if (comm->getRank() == 3) {
          // P3 send global shr[1] (own::shr[2]) to P2
          std::vector<ashr_el_t> shr_for_P2(numel);
          std::vector<ashr_el_t> shr_for_P3(numel);
          pforeach(0, numel, [&](int64_t idx) { shr_for_P2[idx] = _x_minus_r[idx][2]; });

          JointMsgPass(ctx, shr_for_P2, 3, 1, 2);
          JointMsgPass(ctx, shr_for_P3, 2, 1, 3);

          pforeach(0, numel, [&](int64_t idx) { plaintext_x_minus_r[idx] = _x_minus_r[idx][0] ^ _x_minus_r[idx][1] ^ _x_minus_r[idx][2] ^ shr_for_P3[idx]; });

        }

        JointInputArith(ctx, plaintext_x_minus_r, x_minus_r_shr, 2, 3, 0, 1);

        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = _x_minus_r_shr[idx][0] + _r_shr[idx][0];
          _out[idx][1] = _x_minus_r_shr[idx][1] + _r_shr[idx][1];
          _out[idx][2] = _x_minus_r_shr[idx][2] + _r_shr[idx][2];
        });
      }
    });
  });

  return out;
}

#endif

#ifndef OPTIMIZED_CONVERSION
// Fantastic4 MSB based on Local Share Conversion
// [x] = (x0, x1, x2, x3)
//     [x0] = (x0, 0, 0, 0)
//     [x1] = (0, x1, 0, 0)
//     [x2] = (0, 0, x2, 0)
//     [x3] = (0, 0, 0, x3)
// Fantastic4 uses FA to reduce 4 operands to 2 operands
NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto numel = in.numel();
  const PtType out_btype = calcBShareBacktype(SizeOf(field) * 8);
  const auto out_ty = makeType<BShrTy>(out_btype, SizeOf(out_btype) * 8);

  NdArrayRef shr0(out_ty, in.shape());
  NdArrayRef shr1(out_ty, in.shape());
  NdArrayRef shr2(out_ty, in.shape());
  NdArrayRef shr3(out_ty, in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_t = std::array<ring2k_t, 3>;
    NdArrayView<ashr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(out_btype, [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 3>;

      NdArrayView<bshr_t> _shr0(shr0);
      NdArrayView<bshr_t> _shr1(shr1);
      NdArrayView<bshr_t> _shr2(shr2);
      NdArrayView<bshr_t> _shr3(shr3);

      if(rank == 0){
        pforeach(0, numel, [&](int64_t idx) {
            // P0 holds _in(x0, x1, x2)
            _shr0[idx][0] = _in[idx][0];
            _shr0[idx][1] = 0U;
            _shr0[idx][2] = 0U;

            _shr1[idx][0] = 0U;
            _shr1[idx][1] = _in[idx][1];
            _shr1[idx][2] = 0U;

            _shr2[idx][0] = 0U;
            _shr2[idx][1] = 0U;
            _shr2[idx][2] = _in[idx][2];

            _shr3[idx][0] = 0U;
            _shr3[idx][1] = 0U;
            _shr3[idx][2] = 0U;
        });
      }
      else if(rank == 1){
        pforeach(0, numel, [&](int64_t idx) {
            _shr0[idx][0] = 0U;
            _shr0[idx][1] = 0U;
            _shr0[idx][2] = 0U;
            // P1 holds _in(x1, x2, x3)

            _shr1[idx][0] = _in[idx][0];
            _shr1[idx][1] = 0U;
            _shr1[idx][2] = 0U;

            _shr2[idx][0] = 0U;
            _shr2[idx][1] = _in[idx][1];
            _shr2[idx][2] = 0U;

            _shr3[idx][0] = 0U;
            _shr3[idx][1] = 0U;
            _shr3[idx][2] = _in[idx][2];
        });
      }
      else if(rank == 2){
        pforeach(0, numel, [&](int64_t idx) {
            // P2 holds _in(x2, x3, x0)
            _shr2[idx][0] = _in[idx][0];
            _shr2[idx][1] = 0U;
            _shr2[idx][2] = 0U;

            _shr3[idx][0] = 0U;
            _shr3[idx][1] = _in[idx][1];
            _shr3[idx][2] = 0U;

            _shr0[idx][0] = 0U;
            _shr0[idx][1] = 0U;
            _shr0[idx][2] = _in[idx][2];

            _shr1[idx][0] = 0U;
            _shr1[idx][1] = 0U;
            _shr1[idx][2] = 0U;
        });
      }
      else if(rank == 3){
        pforeach(0, numel, [&](int64_t idx) {
            // P3 holds _in(x3, x0, x1)
            _shr3[idx][0] = _in[idx][0];
            _shr3[idx][1] = 0U;
            _shr3[idx][2] = 0U;

            _shr0[idx][0] = 0U;
            _shr0[idx][1] = _in[idx][1];
            _shr0[idx][2] = 0U;

            _shr1[idx][0] = 0U;
            _shr1[idx][1] = 0U;
            _shr1[idx][2] = _in[idx][2];

            _shr2[idx][0] = 0U;
            _shr2[idx][1] = 0U;
            _shr2[idx][2] = 0U;
        });
      }

    });
  });

  auto s_cout_0 = wrap_pfa_bb(ctx->sctx(), shr0, shr1, shr2);
  NdArrayRef s0 = s_cout_0[0];
  NdArrayRef cout0 = s_cout_0[1];

  auto cout0_times_2 = wrap_lshift_b(ctx->sctx(), cout0, 1);

  auto s_cout_1 = wrap_pfa_bb(ctx->sctx(), s0, cout0_times_2, shr3);
  NdArrayRef s1 = s_cout_1[0];
  NdArrayRef cout1 = s_cout_1[1];

  auto cout1_times_2 = wrap_lshift_b(ctx->sctx(), cout1, 1);

  // Compute the k-1'th carry bit.
  size_t nbits = SizeOf(field) * 8 - 1;
  auto* sctx = ctx->sctx();

  const Shape shape = {in.numel()};
  auto wrap_m = WrapValue(s1);
  auto wrap_n = WrapValue(cout1_times_2);
  {
    auto carry = carry_a2b(sctx, wrap_m, wrap_n, nbits);

    // Compute the k'th bit.
    //   (m^n)[k] ^ carry
    auto msb = xor_bb(sctx,
                      rshift_b(sctx, xor_bb(sctx, wrap_m, wrap_n),
                               {static_cast<int64_t>(nbits)}),
                      carry);
    return UnwrapValue(msb);
  }
}


#else
// Optimized MSB similar as the A2B without local conversion and 2k Full Adder Reduction
NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  const Type bshr_type = makeType<BShrTy>(GetStorageType(field), SizeOf(field) * 8);

  auto x1_plus_x2_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * bshr_type.size());
  auto x0_plus_x3_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * bshr_type.size());
  memset(x1_plus_x2_buf->data(), 0, in.shape().numel() * bshr_type.size());
  memset(x0_plus_x3_buf->data(), 0, in.shape().numel() * bshr_type.size());

  NdArrayRef x1_plus_x2_shr(x1_plus_x2_buf, bshr_type, in.shape());
  NdArrayRef x0_plus_x3_shr(x0_plus_x3_buf, bshr_type, in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _in(in);
    NdArrayView<shr_t> _x1_plus_x2_shr(x1_plus_x2_shr);
    NdArrayView<shr_t> _x0_plus_x3_shr(x0_plus_x3_shr);

    std::vector<el_t> x1_plus_x2(numel);
    std::vector<el_t> x0_plus_x3(numel);

    if(rank == 0){
      pforeach(0, numel, [&](int64_t idx) {
          x1_plus_x2[idx] ^= _in[idx][1] + _in[idx][2];
      });
    }
    else if(rank == 1){
      pforeach(0, numel, [&](int64_t idx) {
          x1_plus_x2[idx] ^= _in[idx][0] + _in[idx][1];
      });
    }
    else if(rank == 2){
      pforeach(0, numel, [&](int64_t idx) {
          x0_plus_x3[idx] ^= _in[idx][1] + _in[idx][2];
      });
    }
    else if(rank == 3){
      pforeach(0, numel, [&](int64_t idx) {
          x0_plus_x3[idx] ^= _in[idx][0] + _in[idx][1];
      });
    }
    JointInputBool(ctx, x1_plus_x2, x1_plus_x2_shr, 0, 1, 2, 3);
    JointInputBool(ctx, x0_plus_x3, x0_plus_x3_shr, 3, 2, 1, 0);
  });

  // Compute the k-1'th carry bit.
  size_t nbits = SizeOf(field) * 8 - 1;
  auto* sctx = ctx->sctx();
  const Shape shape = {in.numel()};
  auto wrap_x1_plus_x2_shr = WrapValue(x1_plus_x2_shr);
  auto wrap_x0_plus_x3_shr = WrapValue(x0_plus_x3_shr);

  // 2k + 16 * 2 bits
  auto carry = carry_a2b(sctx, wrap_x1_plus_x2_shr, wrap_x0_plus_x3_shr, nbits);

  // Compute the k'th bit.
  //   (m^n)[k] ^ carry
  auto msb = xor_bb(sctx,
                    rshift_b(sctx, xor_bb(sctx, wrap_x1_plus_x2_shr, wrap_x0_plus_x3_shr),
                              {static_cast<int64_t>(nbits)}),
                    carry);
  return UnwrapValue(msb);
}

#endif

static NdArrayRef wrap_rshift_b(SPUContext* ctx, const NdArrayRef& x,
  const Sizes& bits) {
  return UnwrapValue(rshift_b(ctx, WrapValue(x), bits));
}

#ifndef OPTIMIZED_CONVERSION
// Fantastic4 gives generation of edabits
// to enable edabits-based EQZ (Sec 3)
//    "For general conversion, Rotaru and Wood have established the concept of double-authenticated bits (daBits)..."
//    "On top of introducing the concept of edaBits and their applications to practical MPC, Escudero et al. have .... present more efficient protocols to preprocess edaBits in the context considered in our work"
// Ref. "New primitives for actively-secure MPC over rings with applications to private machine learning." https://eprint.iacr.org/2019/599

// Here we implement it by
//    generate edabits with A2B using local conversion
//    reveal x + r in public
//    locally compute ~(x + r) ^ r (see 'test_all_one' variable)
//    compute bit-wise AND with logk depth
NdArrayRef eqz(KernelEvalContext* ctx, const NdArrayRef& in) {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  auto rank = comm->getRank();

  // length-k boolean type
  const PtType in_bshr_btype = calcBShareBacktype(SizeOf(field) * 8);
  const PtType out_bshr_btype = calcBShareBacktype(8);

  // uint 8 for bit share
  NdArrayRef out(makeType<BShrTy>(out_bshr_btype, 8), in.shape());

  // length-k boolean share
  const auto expanded_ty = makeType<BShrTy>(in_bshr_btype, SizeOf(field) * 8);

  NdArrayRef rand_ashr(makeType<AShrTy>(field), in.shape());
  NdArrayRef rand_bshr(expanded_ty, in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    // First Generate random arithmetic share
    std::vector<ashr_el_t> r0(numel);
    std::vector<ashr_el_t> r1(numel);
    std::vector<ashr_el_t> r2(numel);

    prg_state->fillPrssTuple<ashr_el_t>(r0.data(), nullptr, nullptr, r0.size(),
                              PrgState::GenPrssCtrl::First);
    prg_state->fillPrssTuple<ashr_el_t>(nullptr, r1.data(), nullptr, r1.size(),
                              PrgState::GenPrssCtrl::Second);
    prg_state->fillPrssTuple<ashr_el_t>(nullptr, nullptr, r2.data(), r2.size(),
                              PrgState::GenPrssCtrl::Third);

    NdArrayView<ashr_t> _x(in);
    NdArrayView<ashr_t> _rand_ashr(rand_ashr);

    std::vector<ashr_el_t> plaintext_x_plus_r(numel);
    std::vector<ashr_el_t> second_shr(numel);
    std::vector<ashr_el_t> third_shr(numel);

    pforeach(0, numel, [&](int64_t idx) {
      _rand_ashr[idx][0] = r0[idx];
      _rand_ashr[idx][1] = r1[idx];
      _rand_ashr[idx][2] = r2[idx];

      plaintext_x_plus_r[idx] = _x[idx][0] + _rand_ashr[idx][0] + _x[idx][1] + _rand_ashr[idx][1] + _x[idx][2] + _rand_ashr[idx][2];

      second_shr[idx] = _x[idx][1] + _rand_ashr[idx][1];
      third_shr[idx] = _x[idx][2] + _rand_ashr[idx][2];
    });

    // Compute Boolean shares of r (edabits)
    rand_bshr = wrap_a2b(ctx->sctx(), rand_ashr);

    auto fourth_shr = JointMsgRotate<ashr_el_t>(ctx, third_shr, second_shr);

    pforeach(0, numel, [&](int64_t idx) { plaintext_x_plus_r[idx] += fourth_shr[idx];  });

    NdArrayRef test_all_one(makeType<BShrTy>(in_bshr_btype, SizeOf(in_bshr_btype) * 8), in.shape());

    DISPATCH_UINT_PT_TYPES(in_bshr_btype, [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 3>;

      NdArrayView<bshr_t> _test_all_one(test_all_one);
      NdArrayView<bshr_t> _rand_bshr(rand_bshr);

      if(rank == 0) {
        pforeach(0, numel, [&](int64_t idx) {
          _test_all_one[idx][0] = _rand_bshr[idx][0] ^ ~plaintext_x_plus_r[idx];
          _test_all_one[idx][1] = _rand_bshr[idx][1];
          _test_all_one[idx][2] = _rand_bshr[idx][2];
        });
      }

      if(rank == 1) {
        pforeach(0, numel, [&](int64_t idx) {
          _test_all_one[idx][0] = _rand_bshr[idx][0];
          _test_all_one[idx][1] = _rand_bshr[idx][1];
          _test_all_one[idx][2] = _rand_bshr[idx][2];
        });
      }

      if(rank == 2) {
        pforeach(0, numel, [&](int64_t idx) {
          _test_all_one[idx][0] = _rand_bshr[idx][0];
          _test_all_one[idx][1] = _rand_bshr[idx][1];
          _test_all_one[idx][2] = _rand_bshr[idx][2] ^ ~plaintext_x_plus_r[idx];
        });
      }

      if(rank == 3) {
        pforeach(0, numel, [&](int64_t idx) {
          _test_all_one[idx][0] = _rand_bshr[idx][0];
          _test_all_one[idx][1] = _rand_bshr[idx][1] ^ ~plaintext_x_plus_r[idx];
          _test_all_one[idx][2] = _rand_bshr[idx][2];
        });
      }

      std::vector<NdArrayRef> reduction_res;

      // Reduction start from length-k test_all_one
      reduction_res.push_back(test_all_one);

      int64_t cur_bits = SizeOf(in_bshr_btype) * 8;
      int64_t cur_ind = 0;
      while (cur_bits != 1) {
        cur_bits /= 2;
        cur_ind += 1;

        PtType cur_btype = calcBShareBacktype(cur_bits);
        NdArrayRef cur_out(makeType<BShrTy>(cur_btype, SizeOf(cur_btype) * 8), in.shape());

        // Use lshift_b api (not really to shift) to reduce the bit-length of back-type
        if (cur_bits > 8) {
          cur_out = wrap_and_bb(ctx->sctx(), wrap_rshift_b(ctx->sctx(), reduction_res[cur_ind - 1], {0}), wrap_rshift_b(ctx->sctx(), reduction_res[cur_ind - 1], {cur_bits}));
        }
        // Shortest type is UINT8, do not shift more
        else {
          cur_out = wrap_and_bb(ctx->sctx(), reduction_res[cur_ind - 1], wrap_rshift_b(ctx->sctx(), reduction_res[cur_ind - 1], {cur_bits}));
          NdArrayView<std::array<std::byte, 3>> _final_res(cur_out);
        }
        reduction_res.push_back(cur_out);
      }

      out = reduction_res[cur_ind];
    });
  });

  return out;
}

#else
// Optimized EQZ By Ranyang Liu, Nankai University
// Fantastic4 gives the 4-party generation of edabits based on its A2B
// Given edabits the EQZ can be implemented using Binary Adders
// We take the advantage of 4-party replicated secret sharing
// Each group holds half of x, if x = 0, then they are negative of each other
// Each group contributes it's half securely, otherwise the malicious party will be detected
// Let
//      (P0, P1) share ~(x1 + x2)
//      (P2, P3) share -(x0 + x3)
//      Locally Compute [s] = [~(x1 + x2) ^ -(x0 + x3)]
//      Jointly Compute AND of all bits of s, [s_0] & .... & [s_k-1] (k AND gates, logk rounds)

NdArrayRef eqz(KernelEvalContext* ctx, const NdArrayRef& in) {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  const PtType in_bshr_btype = calcBShareBacktype(SizeOf(field) * 8);
  const PtType out_bshr_btype = calcBShareBacktype(8);
  const auto numel = in.numel();
  // uint 8
  NdArrayRef out(makeType<BShrTy>(out_bshr_btype, 1), in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayView<ashr_t> _in(in);

    DISPATCH_UINT_PT_TYPES(in_bshr_btype, [&]() {
      using bshr_el_t = ScalarT;
      using bshr_t = std::array<bshr_el_t, 3>;

      auto in_ty = makeType<BShrTy>(in_bshr_btype, SizeOf(in_bshr_btype) * 8);

      auto x1_plus_x2_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * in_ty.size());
      auto x0_plus_x3_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * in_ty.size());
      auto test_all_one_buf = std::make_shared<yacl::Buffer>(in.shape().numel() * in_ty.size());
      memset(x1_plus_x2_buf->data(), 0, in.shape().numel() * in_ty.size());
      memset(x0_plus_x3_buf->data(), 0, in.shape().numel() * in_ty.size());
      memset(test_all_one_buf->data(), 0, in.shape().numel() * in_ty.size());

      NdArrayRef x1_plus_x2_shr(x1_plus_x2_buf, in_ty, in.shape());
      NdArrayView<bshr_t> _x1_plus_x2_shr(x1_plus_x2_shr);
      NdArrayRef x0_plus_x3_shr(x0_plus_x3_buf, in_ty, in.shape());
      NdArrayView<bshr_t> _x0_plus_x3_shr(x0_plus_x3_shr);

      NdArrayRef test_all_one(test_all_one_buf, in_ty, in.shape());
      NdArrayView<bshr_t> _test_all_one(test_all_one);

      std::vector<bshr_el_t> plaintext_x1_plus_x2(numel);
      std::vector<bshr_el_t> plaintext_x0_plus_x3(numel);

      if (comm->getRank() == 0) {
        pforeach(0, numel, [&](int64_t idx) {
          // ~(x1 + x2)
          plaintext_x1_plus_x2[idx] = ~(_in[idx][1] + _in[idx][2]);
        });
      }
      if (comm->getRank() == 1) {
        pforeach(0, numel, [&](int64_t idx) {
          // ~(x1 + x2)
          plaintext_x1_plus_x2[idx] = ~(_in[idx][0] + _in[idx][1]);
        });
      }
      if (comm->getRank() == 2) {
        pforeach(0, numel, [&](int64_t idx) {
          // -(x3 + x0)
          plaintext_x0_plus_x3[idx] = -(_in[idx][1] + _in[idx][2]);
        });
      }
      if (comm->getRank() == 3) {
        pforeach(0, numel, [&](int64_t idx) {
          // -(x3 + x0)
          plaintext_x0_plus_x3[idx] = -(_in[idx][0] + _in[idx][1]);
        });
      }

      // [m]B = [~(x1 + x2)]B
      // [n]B = [-(x0 + x3)]B
      // if x = 0, x1 + x2 = -(x0 + x3)
      //   < ---- >(x1 + x2)[i] == (- x0 - x3)[i], for all i
      //   < ---- >(x1 + x2)[i] XOR (- x0 - x3)[i] == 0, for all i
      //   < ---- >~(x1 + x2)[i] XOR (- x0 - x3)[i] == 1, for all i
      JointInputBool(ctx, plaintext_x1_plus_x2, x1_plus_x2_shr, 0, 1, 2, 3);
      JointInputBool(ctx, plaintext_x0_plus_x3, x0_plus_x3_shr, 2, 3, 0, 1);

      pforeach(0, numel, [&](int64_t idx) {
        _test_all_one[idx][0] = _x1_plus_x2_shr[idx][0] ^ _x0_plus_x3_shr[idx][0];
        _test_all_one[idx][1] = _x1_plus_x2_shr[idx][1] ^ _x0_plus_x3_shr[idx][1];
        _test_all_one[idx][2] = _x1_plus_x2_shr[idx][2] ^ _x0_plus_x3_shr[idx][2];
      });

      std::vector<NdArrayRef> reduction_res;

      // Reduction start from length-k test_all_one
      reduction_res.push_back(test_all_one);

      int64_t cur_bits = SizeOf(in_bshr_btype) * 8;
      int64_t cur_ind = 0;

      //    Theoreticall: k/2 + k/4 + k/8 + ... + 1 = k - 1 AND gates
      //    Actually: k/2 + k/4 + ... + 8 + 8 (4) + 8 (2) + 8 (1) = k + 16 AND gates
      while (cur_bits != 1) {
        cur_bits /= 2;
        cur_ind += 1;

        PtType cur_btype = calcBShareBacktype(cur_bits);
        NdArrayRef cur_out(makeType<BShrTy>(cur_btype, SizeOf(cur_btype) * 8), in.shape());

        // Use lshift_b api (not really to shift) to reduce the bit-length of back-type
        if (cur_bits > 8) {
          cur_out = wrap_and_bb(ctx->sctx(), wrap_rshift_b(ctx->sctx(), reduction_res[cur_ind - 1], {0}), wrap_rshift_b(ctx->sctx(), reduction_res[cur_ind - 1], {cur_bits}));
        }
        // Shortest type is UINT8, do not shift more
        else {
          cur_out = wrap_and_bb(ctx->sctx(), reduction_res[cur_ind - 1], wrap_rshift_b(ctx->sctx(), reduction_res[cur_ind - 1], {cur_bits}));
          NdArrayView<std::array<std::byte, 3>> _final_res(cur_out);
        }
        reduction_res.push_back(cur_out);
      }
      out = reduction_res[cur_ind];
    });
  });
  return out;
}

#endif

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
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      _out[idx][2] = _lhs[idx][2];
      if (rank == 0) _out[idx][0] -= _rhs[idx];
      if (rank == 2) _out[idx][2] -= _rhs[idx];
      if (rank == 3) _out[idx][1] -= _rhs[idx];
    });
    return out;
  });

  return eqz(ctx, out);
}

NdArrayRef EqualAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  NdArrayRef out(makeType<AShrTy>(field), lhs.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using shr_t = std::array<ring2k_t, 3>;
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] - _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] - _rhs[idx][1];
      _out[idx][2] = _lhs[idx][2] - _rhs[idx][2];
    });
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

} // namespace spu::mpc::fantastic4
