// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/swift/conversion.h"

#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/swift/arithmetic.h"
#include "libspu/mpc/swift/boolean.h"
#include "libspu/mpc/swift/type.h"
#include "libspu/mpc/swift/value.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::swift {

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_a2b(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(a2b(ctx, WrapValue(x)));
}

static NdArrayRef wrap_rshift_b(SPUContext* ctx, const NdArrayRef& x,
                                const Sizes& shift) {
  return UnwrapValue(rshift_b(ctx, WrapValue(x), shift));
}

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto numel = in.numel();

  const auto bty = makeType<BShrTy>(field);

  // v0 = beta, v1 = -alpha1, v2 = -alpha2
  NdArrayRef v0(bty, in.shape());
  NdArrayRef v1(bty, in.shape());
  NdArrayRef v2(bty, in.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _v0(v0);
    NdArrayView<shr_t> _v1(v1);
    NdArrayView<shr_t> _v2(v2);
    NdArrayView<shr_t> _in(in);

    // share of v0:
    // P0: 0,  0,  0
    // P1: 0,  v0, v0
    // P2: 0,  v0, v0

    // share of v1:
    // P0: v1,  0,  0
    // P1: v1,  0,  0
    // P2:  0,  0,  0

    // share of v2:
    // P0: 0,  v2,  0
    // P1: 0,   0,  0
    // P2: v2,  0,  0

    if (rank == 0) {
      pforeach(0, numel, [&](int64_t idx) {
        _v0[idx][0] = ring2k_t(0);
        _v0[idx][1] = ring2k_t(0);
        _v0[idx][2] = ring2k_t(0);

        _v1[idx][0] = -_in[idx][0];
        _v1[idx][1] = ring2k_t(0);
        _v1[idx][2] = ring2k_t(0);

        _v2[idx][0] = ring2k_t(0);
        _v2[idx][1] = -_in[idx][1];
        _v2[idx][2] = ring2k_t(0);
      });
    }
    if (rank == 1) {
      pforeach(0, numel, [&](int64_t idx) {
        _v0[idx][0] = ring2k_t(0);
        _v0[idx][1] = _in[idx][1];
        _v0[idx][2] = _in[idx][1];

        _v1[idx][0] = -_in[idx][0];
        _v1[idx][1] = ring2k_t(0);
        _v1[idx][2] = ring2k_t(0);

        _v2[idx][0] = ring2k_t(0);
        _v2[idx][1] = ring2k_t(0);
        _v2[idx][2] = ring2k_t(0);
      });
    }
    if (rank == 2) {
      pforeach(0, numel, [&](int64_t idx) {
        _v0[idx][0] = ring2k_t(0);
        _v0[idx][1] = _in[idx][1];
        _v0[idx][2] = _in[idx][1];

        _v1[idx][0] = ring2k_t(0);
        _v1[idx][1] = ring2k_t(0);
        _v1[idx][2] = ring2k_t(0);

        _v2[idx][0] = -_in[idx][0];
        _v2[idx][1] = ring2k_t(0);
        _v2[idx][2] = ring2k_t(0);
      });
    }
  });

  NdArrayRef res(bty, in.shape());

  res = wrap_add_bb(ctx->sctx(), v0, v1);
  res = wrap_add_bb(ctx->sctx(), res, v2);

  // if (rank == 0) {
  //   ring_print(spu::mpc::swift::getFirstShare(v0), "(B) P0: First Share");
  //   ring_print(spu::mpc::swift::getSecondShare(v0), "(B) P0: Second Share");
  //   ring_print(spu::mpc::swift::getThirdShare(v0), "(B) P0: Third Share");
  // }

  // if (rank == 1) {
  //   ring_print(spu::mpc::swift::getFirstShare(v0), "(B) P1: First Share");
  //   ring_print(spu::mpc::swift::getSecondShare(v0), "(B) P1: Second Share");
  //   ring_print(spu::mpc::swift::getThirdShare(v0), "(B) P1: Third Share");
  // }

  // if (rank == 2) {
  //   ring_print(spu::mpc::swift::getFirstShare(v0), "(B) P2: First Share");
  //   ring_print(spu::mpc::swift::getSecondShare(v0), "(B) P2: Second Share");
  //   ring_print(spu::mpc::swift::getThirdShare(v0), "(B) P2: Third Share");
  // }

  // test
  // auto a2p = A2P();
  // auto b2p = B2P();
  // auto input_rec = a2p.proc(ctx, in);
  // auto output_rec = b2p.proc(ctx, res);
  // if (rank == 0) {
  //   ring_print(input_rec, "A2B: input");
  //   ring_print(output_rec, "A2B: get output");
  // }
  return res.as(bty);
}

NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto bty = makeType<BShrTy>(field);

  // call A2B to get the boolean share
  auto bshr_res = wrap_a2b(ctx->sctx(), in);

  // get Msb from boolean share directly
  size_t k = SizeOf(field) * 8 - 1;
  auto msb_res =
      wrap_rshift_b(ctx->sctx(), bshr_res, {static_cast<int64_t>(k)});

  return msb_res.as(bty);
}

// In Swift, the Bit2A convert <Xi>B to <Xi>A
// but we need to implement B2A which convert <X>B to <X>A
// we can:
// 1. spilit <X>B to <Xi>B locally
// 2. convert <Xi>B to <Xi>A using the protocol of Bit2A in Swift
// 3. <X>A = sum: <Xi>A * 2^i
NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();

  const int64_t nbits = in.eltype().as<BShare>()->nbits();
  SPU_ENFORCE((size_t)nbits <= SizeOf(field) * 8, "invalid nbits = {}", nbits);

  auto res = NdArrayRef(makeType<AShrTy>(field), in.shape());
  auto numel = in.numel();

  if (nbits == 0) {
    // special case, it's known to be zero
    DISPATCH_ALL_FIELDS(field, [&]() {
      using el_t = ring2k_t;
      using ashr_t = std::array<el_t, 3>;

      NdArrayView<ashr_t> _res(res);
      pforeach(0, numel, [&](int64_t idx) {
        _res[idx][0] = el_t(0);
        _res[idx][1] = el_t(0);
        _res[idx][2] = el_t(0);
      });
    });
    return res.as(makeType<AShrTy>(field));
  }

  auto mult = MulAA();
  auto add = AddAA();
  auto neg = NegateA();
  auto a2p = A2P();

  auto decompose_numel = numel * static_cast<int64_t>(nbits);

  auto decompose_in = NdArrayRef(makeType<AShrTy>(field), {decompose_numel});

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using ashr_t = std::array<el_t, 3>;

    NdArrayView<ashr_t> _in(in);
    NdArrayView<ashr_t> _decompose_in(decompose_in);

    pforeach(0, numel, [&](int64_t idx) {
      for (int64_t bit_idx = 0; bit_idx < nbits; bit_idx++) {
        _decompose_in[idx * nbits + bit_idx][0] =
            static_cast<el_t>(_in[idx][0] >> bit_idx) & 0x1;
        _decompose_in[idx * nbits + bit_idx][1] =
            static_cast<el_t>(_in[idx][1] >> bit_idx) & 0x1;
        _decompose_in[idx * nbits + bit_idx][2] =
            static_cast<el_t>(_in[idx][2] >> bit_idx) & 0x1;
      }
    });
  });

  // Bit2A
  // Step0: joint share alpha_b1, alpha_b2, beta_b
  // Step1: alpha_b = (alpha_b1 ^ alpha_b2)
  //                = alpha_b1 + alpha_b2 - 2 * alpha_b1 * alpha_b2
  // Step2: b = (beta_b ^ alpha_b)
  //          = beta_b + alpha_b - 2 * beta_b * alpha_b

  NdArrayRef alpha_b1(makeType<AShrTy>(field), {decompose_numel});
  NdArrayRef alpha_b2(makeType<AShrTy>(field), {decompose_numel});
  NdArrayRef beta_b(makeType<AShrTy>(field), {decompose_numel});

  if (rank == 0) {
    alpha_b1 = getFirstShare(decompose_in);
    alpha_b2 = getSecondShare(decompose_in);
  }
  if (rank == 1) {
    alpha_b1 = getFirstShare(decompose_in);
    beta_b = getSecondShare(decompose_in);
  }
  if (rank == 2) {
    alpha_b2 = getFirstShare(decompose_in);
    beta_b = getSecondShare(decompose_in);
  }

  // P0, P1 joint share alpha_b1
  alpha_b1 = JointSharing(ctx, alpha_b1, 0, 1, "alpha_b1");

  // P0, P2 joint share alpha_b2
  alpha_b2 = JointSharing(ctx, alpha_b2, 0, 2, "alpha_b2");

  // P1, P2 joint share beta_b
  beta_b = JointSharing(ctx, beta_b, 1, 2, "beta_b");

  // auto alpha_b1_reconstruct = a2p.proc(ctx, alpha_b1);
  // auto alpha_b2_reconstruct = a2p.proc(ctx, alpha_b2);
  // auto beta_b_reconstruct = a2p.proc(ctx, beta_b);

  // if (rank == 0) {
  //   ring_print(alpha_b1_reconstruct, "( alpha_b1_reconstruct)");
  //   ring_print(alpha_b2_reconstruct, "( alpha_b2_reconstruct)");
  //   ring_print(beta_b_reconstruct, "( beta_b_reconstruct)");
  // }

  // Step1: alpha_b = (alpha_b1 ^ alpha_b2)
  //                = alpha_b1 + alpha_b2 - 2 * alpha_b1 * alpha_b2
  auto tmp = mult.proc(ctx, alpha_b1, alpha_b2);

  auto alpha_b = add.proc(ctx, alpha_b1, alpha_b2);
  auto neg_tmp = neg.proc(ctx, tmp);
  alpha_b = add.proc(ctx, alpha_b, neg_tmp);
  alpha_b = add.proc(ctx, alpha_b, neg_tmp);

  // auto alpha_b_reconstruct = a2p.proc(ctx, alpha_b);
  // if (rank == 0) {
  //   ring_print(alpha_b_reconstruct, "(alpha_b_reconstruct)");
  // }

  // Step2: b = (beta_b ^ alpha_b)
  //          = beta_b + alpha_b - 2 * beta_b * alpha_b
  tmp = mult.proc(ctx, beta_b, alpha_b);
  neg_tmp = neg.proc(ctx, tmp);
  auto decompose_out = add.proc(ctx, alpha_b, beta_b);
  decompose_out = add.proc(ctx, decompose_out, neg_tmp);
  decompose_out = add.proc(ctx, decompose_out, neg_tmp);

  auto b_reconstruct = a2p.proc(ctx, decompose_out);
  // if (rank == 0) {
  //   ring_print(b_reconstruct, "(b_reconstruct)");
  // }

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using ashr_t = std::array<el_t, 3>;

    NdArrayView<ashr_t> _res(res);
    NdArrayView<ashr_t> _decompose_out(decompose_out);

    el_t tmp_sum0, tmp_sum1, tmp_sum2;

    pforeach(0, numel, [&](int64_t idx) {
      // use _decompose_out[i*nbits, (i+1)*nbits) to construct res[i]
      tmp_sum0 = static_cast<el_t>(0);
      tmp_sum1 = static_cast<el_t>(0);
      tmp_sum2 = static_cast<el_t>(0);
      for (int64_t bit = 0; bit < nbits; bit++) {
        tmp_sum0 +=
            (static_cast<el_t>(_decompose_out[idx * nbits + bit][0]) << bit);
        tmp_sum1 +=
            (static_cast<el_t>(_decompose_out[idx * nbits + bit][1]) << bit);
        tmp_sum2 +=
            (static_cast<el_t>(_decompose_out[idx * nbits + bit][2]) << bit);
      }
      _res[idx][0] = tmp_sum0;
      _res[idx][1] = tmp_sum1;
      _res[idx][2] = tmp_sum2;
    });
  });

  // test
  // auto b2p = B2P();
  // auto input_rec = b2p.proc(ctx, in);
  // auto output_rec = a2p.proc(ctx, res);
  // if (rank == 0) {
  //   ring_print(input_rec, "B2A: input");
  //   ring_print(output_rec, "B2A: get output");
  // }
  return res.as(makeType<AShrTy>(field));
}

}  // namespace spu::mpc::swift