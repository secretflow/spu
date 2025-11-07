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

#include "libspu/mpc/cheetah/arithmetic.h"

#include <future>
#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"
#include "libspu/mpc/cheetah/nonlinear/ext_prot.h"
#include "libspu/mpc/cheetah/nonlinear/mix_mul_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        size_t bits, SignType sign) const {
  size_t n = x.numel();
  NdArrayRef out(x.eltype(), x.shape());
  if (n == 0) {
    return out;
  }

  return TiledDispatchOTFunc(
      ctx, x,
      [&](const NdArrayRef& input,
          const std::shared_ptr<BasicOTProtocols>& base_ot) {
        TruncateProtocol::Meta meta;
        meta.signed_arith = true;
        meta.sign = sign;
        meta.shift_bits = bits;
        meta.use_heuristic = true;
        TruncateProtocol prot(base_ot);
        return prot.Compute(input, meta);
      });
}

NdArrayRef TruncA2::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                         size_t bits, SignType sign, bool signed_arith) const {
  size_t n = x.numel();
  NdArrayRef out(x.eltype(), x.shape());
  if (n == 0) {
    return out;
  }
  const auto field = x.eltype().as<Ring2k>()->field();
  // only signed and full ring supported for heuristic
  bool use_heuristic =
      signed_arith &&
      ((x.fxp_bits() == 0) ||
       (x.fxp_bits() == static_cast<int64_t>(8 * SizeOf(field))));

  auto ret = TiledDispatchOTFunc(
      ctx, x,
      [&](const NdArrayRef& input,
          const std::shared_ptr<BasicOTProtocols>& base_ot) {
        TruncateProtocol::Meta meta;
        meta.exact = false;
        meta.signed_arith = signed_arith;
        meta.sign = sign;
        meta.shift_bits = bits;
        meta.use_heuristic = use_heuristic;
        TruncateProtocol prot(base_ot);
        return prot.Compute(input, meta);
      });

  if (x.fxp_bits() > 0) {
    ret.set_fxp_bits(x.fxp_bits());
  }

  return ret;
}

NdArrayRef TruncAE::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                         size_t bits, SignType sign, bool signed_arith) const {
  size_t n = x.numel();
  NdArrayRef out(x.eltype(), x.shape());
  if (n == 0) {
    return out;
  }
  const auto field = x.eltype().as<Ring2k>()->field();
  // only signed and full ring supported for heuristic
  bool use_heuristic =
      signed_arith &&
      ((x.fxp_bits() == 0) ||
       (x.fxp_bits() == static_cast<int64_t>(8 * SizeOf(field))));

  auto ret = TiledDispatchOTFunc(
      ctx, x,
      [&](const NdArrayRef& input,
          const std::shared_ptr<BasicOTProtocols>& base_ot) {
        TruncateProtocol::Meta meta;
        meta.exact = true;
        meta.signed_arith = signed_arith;
        meta.sign = sign;
        meta.shift_bits = bits;
        meta.use_heuristic = use_heuristic;
        TruncateProtocol prot(base_ot);
        return prot.Compute(input, meta);
      });

  if (x.fxp_bits() > 0) {
    ret.set_fxp_bits(x.fxp_bits());
  }

  return ret;
}

namespace {
// copied from ot_util.h
template <typename T>
T makeBitsMask(size_t nbits) {
  size_t max = sizeof(T) * 8;
  if (nbits == 0) {
    nbits = max;
  }
  SPU_ENFORCE(nbits <= max);
  T mask = static_cast<T>(-1);
  if (nbits < max) {
    mask = (static_cast<T>(1) << nbits) - 1;
  }
  return mask;
}
}  // namespace

// Math:
//  msb(x0 + x1 mod 2^k) = msb(x0) ^ msb(x1) ^ 1{(x0&msk + x1&msk) > 2^{k-1} -
//  1}
//  The carry bit 1{(x0&msk + x1&msk) > 2^{k - 1} - 1} = 1{x0&msk > 2^{k - 1}
//  - 1 - x1&msk} is computed using a Millionare protocol.
NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const int64_t numel = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  size_t nbits = nbits_ == 0 ? SizeOf(field) * 8 : nbits_;
  nbits = x.fxp_bits() > 0 ? static_cast<size_t>(x.fxp_bits()) : nbits;
  const size_t shft = nbits - 1;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  NdArrayRef out(x.eltype(), x.shape());
  if (numel == 0) {
    return out.as(makeType<BShrTy>(field, 1));
  }

  const int rank = ctx->getState<Communicator>()->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << shft) - 1;
    NdArrayRef adjusted = ring_zeros(field, x.shape());
    auto xinp = NdArrayView<const u2k>(x);
    auto xadj = NdArrayView<u2k>(adjusted);

    if (rank == 0) {
      // x0 mod 2^{k-1}
      pforeach(0, numel, [&](int64_t i) { xadj[i] = xinp[i] & mask; });
    } else {
      // (2^{k - 1} - 1 - x1 )mod 2^{k-1}
      pforeach(0, numel, [&](int64_t i) { xadj[i] = mask - (xinp[i] & mask); });
    }

    auto carry_bit = TiledDispatchOTFunc(
                         ctx, adjusted,
                         [&](const NdArrayRef& input,
                             const std::shared_ptr<BasicOTProtocols>& base_ot) {
                           CompareProtocol prot(base_ot);
                           return prot.Compute(input, /*greater*/ true, shft);
                         })
                         .as(x.eltype());

    // [msb(x)]_B <- [1{x0 + x1 > 2^{k- 1} - 1]_B ^ msb(x0)
    NdArrayView<u2k> _carry_bit(carry_bit);
    pforeach(0, numel,
             [&](int64_t i) { _carry_bit[i] ^= ((xinp[i] >> shft) & 1); });

    return carry_bit.as(makeType<BShrTy>(field, 1));
  });
}

namespace {

NdArrayRef wrap_xorbb(SPUContext* ctx, const NdArrayRef& x,
                      const NdArrayRef& y) {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  return UnwrapValue(xor_bb(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_andbb(SPUContext* ctx, const NdArrayRef& x,
                      const NdArrayRef& y) {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  return UnwrapValue(and_bb(ctx, WrapValue(x), WrapValue(y)));
}
}  // namespace

// Math:
// msk = 2^{k - 1} - 1
// msb(x0 + x1 mod 2^k) = msb(x0) ^ msb(x1) ^ 1{(x0&msk + x1&msk) >= 2^{k-1})
// I(x0+x1 mod 2^k = 0) = !msb & I(x0&msk + x1&msk = 0 mod 2^{k-1})
//                      = !msb & ((x0&msk ==0 && x1&msk == 0) ||
//                                (x0&msk == 2^{k-1} - x1&msk)
// let y0 = x0 mod 2^{k-1}, y1 = x1 mod 2^{k-1}
// then wrap = (y0 >= 2^{k-1} - y1) = (y0 > 2^{k-1} - y1) ^ (y0 == 2^{k-1} - y1)
std::vector<NdArrayRef> MsbEq::proc(KernelEvalContext* ctx,
                                    const NdArrayRef& x) const {
  const int64_t numel = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  const size_t nbits =
      x.fxp_bits() > 0 ? static_cast<size_t>(x.fxp_bits()) : SizeOf(field) * 8;
  const size_t shft = nbits - 1;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  NdArrayRef out(x.eltype(), x.shape());
  if (numel == 0) {
    return {out.as(makeType<BShrTy>(field, 1)),
            out.as(makeType<BShrTy>(field, 1))};
  }
  const int rank = ctx->getState<Communicator>()->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned_t<ring2k_t>;
    // msb, eq
    std::vector<NdArrayRef> out(2);

    const u2k mask = makeBitsMask<u2k>(shft);
    const u2k src_mask = makeBitsMask<u2k>(nbits);
    NdArrayRef adjusted = ring_zeros(field, x.shape());
    auto xinp = NdArrayView<const u2k>(x);
    auto xadj = NdArrayView<u2k>(adjusted);
    if (rank == 0) {
      // we must add another bit to express 2^{k-1}
      // y0 = x0 mod 2^{k-1} \in Z_{2^k}
      pforeach(0, numel, [&](int64_t i) {  //
        xadj[i] = xinp[i] & mask;
      });
    } else {
      // y1 = (2^{k-1}-x1) mod 2^{k-1}  \in Z_{2^k}
      pforeach(0, numel, [&](int64_t i) {  //
        xadj[i] = mask + 1 - (xinp[i] & mask);
        xadj[i] &= src_mask;
      });
    }

    // (gt, eq)
    auto gt_eq_arr = TiledDispatchOTFuncForMill(
        ctx, adjusted,
        [&](const NdArrayRef& input,
            const std::shared_ptr<BasicOTProtocols>& base_ot) {
          CompareProtocol prot(base_ot);
          // Note: the adjusted input and bw is different in `msb_a2b` kernel.
          return prot.ComputeWithEq(input, /*greater*/ true, nbits);
        });
    const auto btype = makeType<BShrTy>(field, 1);

    // ring_print(gt_eq_arr[0], "gt:" + std::to_string(rank));
    // ring_print(gt_eq_arr[1], "eq:" + std::to_string(rank));

    // wrap = y0 >= 2^{k-1} - y1
    auto wrap =
        wrap_xorbb(ctx->sctx(), gt_eq_arr[0].as(btype), gt_eq_arr[1].as(btype));
    // zero = (y0 == 0 && y1 == 0) ^ (y0 + y1 == 2^{k-1})
    // TODO: can use AndVV to replace AndBB
    NdArrayRef y0_eq_0 = ring_zeros(field, x.shape());
    NdArrayRef y1_eq_0 = ring_zeros(field, x.shape());

    if (rank == 0) {
      NdArrayView<u2k> _y0_eq_0(y0_eq_0);
      pforeach(0, numel, [&](int64_t idx) {  //
        _y0_eq_0[idx] = static_cast<u2k>((xinp[idx] & mask) == 0);
      });
    } else {
      NdArrayView<u2k> _y1_eq_0(y1_eq_0);
      pforeach(0, numel, [&](int64_t idx) {  //
        _y1_eq_0[idx] = static_cast<u2k>((xinp[idx] & mask) == 0);
      });
    }

    auto all_zero_test =
        wrap_andbb(ctx->sctx(), y0_eq_0.as(btype), y1_eq_0.as(btype));

    auto zero = wrap_xorbb(ctx->sctx(), all_zero_test.as(btype),
                           gt_eq_arr[1].as(btype));

    // msb = wrap ^ (x >> k-1)
    auto msb = ring_zeros(field, x.shape());
    NdArrayView<u2k> _msb(msb);
    NdArrayView<u2k> _wrap(wrap);
    pforeach(0, numel, [&](int64_t i) {  //
      _msb[i] = _wrap[i] ^ ((xinp[i] >> shft) & 1);
    });

    // ring_print(msb, "msb:" + std::to_string(rank));

    // eq = (!msb) && zero
    auto flip_msb = ring_zeros(field, x.shape());
    NdArrayView<u2k> _flip_msb(flip_msb);
    if (rank == 0) {
      pforeach(0, numel, [&](int64_t i) {  //
        _flip_msb[i] = (_msb[i] & 1) ^ 1;
      });
    } else {
      pforeach(0, numel, [&](int64_t i) {  //
        _flip_msb[i] = (_msb[i] & 1);
      });
    }

    auto eq = wrap_andbb(ctx->sctx(), flip_msb.as(btype), zero.as(btype));

    out[0] = msb.as(btype);
    out[1] = eq.as(btype);

    return out;
  });
}

NdArrayRef EqualAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y) const {
  EqualAA equal_aa;
  const auto field = x.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == y.eltype().as<Ring2k>()->field());
  if (x.fxp_bits() > 0 || y.fxp_bits() > 0) {
    SPU_ENFORCE(x.fxp_bits() == y.fxp_bits());
  }

  // TODO(juhou): Can we use any place holder to indicate the dummy 0s.
  if (0 == ctx->getState<Communicator>()->getRank()) {
    auto dummy = ring_zeros(field, x.shape());
    dummy.set_fxp_bits(x.fxp_bits());
    return equal_aa.proc(ctx, x, dummy);
  } else {
    return equal_aa.proc(ctx, x, y);
  }
}

NdArrayRef EqualAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());
  if (x.fxp_bits() > 0 || y.fxp_bits() > 0) {
    SPU_ENFORCE(x.fxp_bits() == y.fxp_bits());
  }

  const int64_t numel = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  size_t nbits = nbits_ == 0 ? SizeOf(field) * 8 : nbits_;
  nbits = x.fxp_bits() > 0 ? static_cast<size_t>(x.fxp_bits()) : nbits;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  NdArrayRef eq_bit(x.eltype(), x.shape());
  if (numel == 0) {
    return eq_bit.as(makeType<BShrTy>(field, 1));
  }

  const int rank = ctx->getState<Communicator>()->getRank();

  // SPDLOG_INFO("In equalAA, nbits: {}", nbits);
  // ring_print(x, "x:" + std::to_string(rank));
  // ring_print(y, "y:" + std::to_string(rank));

  //     x0 + x1 = y0 + y1 mod 2k
  // <=> x0 - y0 = y1 - x1 mod 2k
  NdArrayRef adjusted;
  if (rank == 0) {
    adjusted = ring_sub(x, y);
  } else {
    adjusted = ring_sub(y, x);
  }
  // ring_print(adjusted, "adjusted_before:" + std::to_string(rank));

  if (x.fxp_bits() > 0) {
    ring_reduce_(adjusted, nbits);
  }
  // ring_print(adjusted, "adjusted_after:" + std::to_string(rank));

  return TiledDispatchOTFunc(
             ctx, adjusted,
             [&](const NdArrayRef& input,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               EqualProtocol prot(base_ot);
               return prot.Compute(input, nbits);
             })
      .as(makeType<BShrTy>(field, 1));
}

NdArrayRef MulA1B::proc(KernelEvalContext* ctx, const NdArrayRef& ashr,
                        const NdArrayRef& bshr) const {
  SPU_ENFORCE_EQ(ashr.shape(), bshr.shape());
  const int64_t numel = ashr.numel();

  if (numel == 0) {
    return NdArrayRef(ashr.eltype(), ashr.shape());
  }

  return TiledDispatchOTFunc(
             ctx, ashr, bshr,
             [&](const NdArrayRef& input0, const NdArrayRef& input1,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               return base_ot->Multiplexer(input0, input1);
             })
      .as(ashr.eltype());
}

NdArrayRef MulA1BV::proc(KernelEvalContext* ctx, const NdArrayRef& ashr,
                         const NdArrayRef& bshr) const {
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();
  SPU_ENFORCE_EQ(ashr.shape(), bshr.shape());
  const int64_t numel = ashr.numel();
  const auto* ptype = bshr.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");

  const int owner = ptype->owner();

  NdArrayRef out(ashr.eltype(), ashr.shape());
  if (numel == 0) {
    return out;
  }

  if (rank != owner) {
    return TiledDispatchOTFunc(
               ctx, ashr,
               [&](const NdArrayRef& input,
                   const std::shared_ptr<BasicOTProtocols>& base_ot) {
                 return base_ot->PrivateMulxSend(input);
               })
        .as(ashr.eltype());
  }

  return TiledDispatchOTFunc(
             ctx, ashr, bshr,
             [&](const NdArrayRef& input0, const NdArrayRef& input1,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               return base_ot->PrivateMulxRecv(input0, input1);
             })
      .as(ashr.eltype());
}

NdArrayRef MulAV::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());
  const int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();
  const auto* ptype = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");
  const int owner = ptype->owner();

  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  mul_prot->LazyInitKeys(x.eltype().as<Ring2k>()->field());

  // (x0 * x1) * y
  // <x0 * y> + x1 * y
  auto fx = x.reshape({numel});
  NdArrayRef out;

  // compute <x0 * y>
  if (rank != owner) {
    out = mul_prot->MulOLE(fx, /*eval*/ true);
  } else {
    auto fy = y.reshape({numel});
    out = mul_prot->MulOLE(fy, /*eval*/ false);
    ring_add_(out, ring_mul(fx, fy));
  }

  return out.reshape(x.shape()).as(x.eltype());
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());
  SPU_ENFORCE_EQ(x.fxp_bits(), y.fxp_bits());

  int64_t batch_sze = ctx->getState<CheetahMulState>()->get()->OLEBatchSize();
  int64_t numel = x.numel();

  NdArrayRef ret;
  if (numel >= 2 * batch_sze) {
    ret = mulDirectly(ctx, x, y);
  } else {
    ret = mulWithBeaver(ctx, x, y);
  }

  if (x.fxp_bits() > 0) {
    ret.set_fxp_bits(x.fxp_bits());
    ring_reduce_(ret, x.fxp_bits());
  }

  return ret;
}

namespace {

FieldType adjust_to_field(FieldType to_field, size_t out_bw) {
  // 1. if out_bw=0 and to_field=FT_INVALID, raise error
  // 2. if to_field=FT_INVALID, but out_bw>0, then to_field is choosen as the
  // minimum field
  // 3. if out_bw=0, to_field != FT_INVALID, just use to_field
  // 4. else, first check if to_field is valid
  if (out_bw == 0 && to_field == FT_INVALID) {
    SPU_THROW("out_bw=0 and to_field=FT_INVALID are not allowed.");
  }

  if (to_field == FT_INVALID && out_bw > 0) {
    return FixGetProperFiled(out_bw);
  }

  if (out_bw == 0 && to_field != FT_INVALID) {
    return to_field;
  }

  SPU_ENFORCE(SizeOf(to_field) * 8 >= out_bw);
  return to_field;
}

NdArrayRef wrap_ext(SPUContext* ctx, const NdArrayRef& in, size_t bw,
                    FieldType to_field, SignType sign, bool signed_arith,
                    bool force) {
  return UnwrapValue(ring_cast_up_s(ctx, WrapValue(in), bw, to_field, sign,
                                    signed_arith, force));
}

[[maybe_unused]] NdArrayRef wrap_mul_aa(SPUContext* ctx, const NdArrayRef& x,
                                        const NdArrayRef& y) {
  return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
}

}  // namespace

// SIRNN's version
NdArrayRef MixMulAA::mulDirectly(KernelEvalContext* ctx, const NdArrayRef& x,
                                 const NdArrayRef& y, SignType sign_x,
                                 SignType sign_y, FieldType to_field,
                                 size_t out_bw, bool signed_arith) const {
  const auto x_field = x.eltype().as<Ring2k>()->field();
  const auto y_field = y.eltype().as<Ring2k>()->field();

  const auto x_bw = x.fxp_bits() > 0 ? x.fxp_bits() : SizeOf(x_field) * 8;
  const auto y_bw = y.fxp_bits() > 0 ? y.fxp_bits() : SizeOf(y_field) * 8;

  SPU_ENFORCE(SizeOf(x_field) * 8 >= x_bw);
  SPU_ENFORCE(SizeOf(y_field) * 8 >= y_bw);

  auto ret = TiledDispatchOTFunc(
      ctx, x, y,
      [&](const NdArrayRef& x, const NdArrayRef& y,
          const std::shared_ptr<BasicOTProtocols>& base_ot) {
        MixMulProtocol::Meta meta;
        meta.use_heuristic = false;  // not active yet.

        meta.signed_arith = signed_arith;

        meta.bw_x = x_bw;
        meta.bw_y = y_bw;
        meta.bw_out = out_bw;

        meta.field_x = x_field;
        meta.field_y = y_field;
        meta.field_out = to_field;

        meta.sign_x = sign_x;
        meta.sign_y = sign_y;

        MixMulProtocol prot(base_ot);
        return prot.Compute(x, y, meta);
      });

  ret.set_fxp_bits(out_bw);

  return ret;
}

// naive extend first then mul
NdArrayRef MixMulAA::mulNaively(KernelEvalContext* ctx, const NdArrayRef& x,
                                const NdArrayRef& y, SignType sign_x,
                                SignType sign_y, FieldType to_field,
                                size_t out_bw, bool signed_arith) const {
  // sanity check
  {
    const auto x_field = x.eltype().as<Ring2k>()->field();
    const auto y_field = y.eltype().as<Ring2k>()->field();

    const auto m = x.fxp_bits() > 0 ? x.fxp_bits() : SizeOf(x_field) * 8;
    const auto n = y.fxp_bits() > 0 ? y.fxp_bits() : SizeOf(y_field) * 8;
    const auto l = out_bw;

    // if uniform ring, should use 2pc mul
    if ((l == m) && (l == n)) {
      SPU_THROW("uniform ring should use mul_aa kernel.");
    }

    SPU_ENFORCE((m <= l) && (n <= l));
    SPU_ENFORCE(l <= m + n);

    SPU_ENFORCE((x_field <= to_field) && (y_field <= to_field));

    SPU_ENFORCE(m <= SizeOf(x_field) * 8);
    SPU_ENFORCE(n <= SizeOf(y_field) * 8);
    SPU_ENFORCE(l <= SizeOf(to_field) * 8);
  }

  auto ext_x =
      wrap_ext(ctx->sctx(), x, out_bw, to_field, sign_x, signed_arith, true);
  auto ext_y =
      wrap_ext(ctx->sctx(), y, out_bw, to_field, sign_y, signed_arith, true);

  auto ret = wrap_mul_aa(ctx->sctx(), ext_x, ext_y);
  ret.set_fxp_bits(out_bw);

  return ret;
}

NdArrayRef MixMulAA::proc(KernelEvalContext* ctx,                    //
                          const NdArrayRef& x, const NdArrayRef& y,  //
                          SignType sign_x, SignType sign_y,          //
                          FieldType to_field,                        //
                          size_t out_bw,                             //
                          bool signed_arith                          //
) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  const auto adj_field = adjust_to_field(to_field, out_bw);
  const auto adj_bw = out_bw == 0 ? SizeOf(adj_field) * 8 : out_bw;

  const auto out_ty = makeType<AShrTy>(adj_field);

  if (ctx->sctx()->config().cheetah_naive_mix_mul) {
    return mulNaively(ctx, x, y, sign_x, sign_y, adj_field, adj_bw,
                      signed_arith)
        .as(out_ty);
  } else {
    return mulDirectly(ctx, x, y, sign_x, sign_y, adj_field, adj_bw,
                       signed_arith)
        .as(out_ty);
  }
}

NdArrayRef SquareA::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  //   (x0 + x1) * (x0 + x1)
  // = x0^2 + 2*<x0*x1> + x1^2
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  mul_prot->LazyInitKeys(x.eltype().as<Ring2k>()->field());

  auto fx = x.reshape({numel});
  int64_t nhalf = numel <= 8192 ? numel : numel / 2;

  auto subtask = std::async([&]() -> spu::NdArrayRef {
    return mul_prot->MulOLE(fx.slice({0}, {nhalf}, {1}), rank == 0);
  });

  NdArrayRef mul1;
  if (nhalf < numel) {
    auto dupx = ctx->getState<CheetahMulState>()->duplx();
    mul1 = mul_prot->MulOLE(fx.slice({nhalf}, {numel}, {1}), dupx.get(),
                            rank == 1);
  }
  auto mul0 = subtask.get();

  NdArrayRef x0x1(x.eltype(), {numel});
  std::memcpy(&x0x1.at(0), &mul0.at(0), mul0.elsize() * nhalf);
  if (nhalf < numel) {
    std::memcpy(&x0x1.at(nhalf), &mul1.at(0), mul1.elsize() * mul1.numel());
  }
  ring_add_(x0x1, x0x1);
  x0x1 = x0x1.reshape(x.shape());

  return ring_add(x0x1, ring_mul(x, x)).as(x.eltype());
}

NdArrayRef MulAA::mulWithBeaver(KernelEvalContext* ctx, const NdArrayRef& x,
                                const NdArrayRef& y) const {
  const int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  // const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto field = x.eltype().as<Ring2k>()->field();
  auto [a, b, c] =
      ctx->getState<CheetahMulState>()->TakeCachedBeaver(field, numel);
  YACL_ENFORCE_EQ(a.numel(), numel);

  a = a.reshape(x.shape());
  b = b.reshape(x.shape());
  c = c.reshape(x.shape());

  auto* comm = ctx->getState<Communicator>();
  // Open x - a & y - b
  auto res = vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const NdArrayRef& s) {
    return comm->allReduce(ReduceOp::ADD, s, kBindName());
  });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(ring_mul(x_a, b), ring_mul(y_b, a));
  ring_add_(z, c);

  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(x_a, y_b));
  }

  return z.as(x.eltype());
}

#if 1
NdArrayRef MulAA::mulDirectly(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) const {
  const auto bw = x.fxp_bits();
  // Compute (x0 + x1) * (y0+ y1)
  auto* comm = ctx->getState<Communicator>();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  mul_prot->LazyInitKeys(x.eltype().as<Ring2k>()->field(), bw);

  auto fx = x.reshape({x.numel()});
  auto fy = y.reshape({y.numel()});
  const int64_t n = fx.numel();
  const int64_t nhalf = n / 2;
  const int rank = comm->getRank();

  // For long vectors, split into two subtasks.
  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    return mul_prot->MulShare(fx.slice({nhalf}, {n}, {1}),
                              fy.slice({nhalf}, {n}, {1}), dupx.get(),
                              /*evaluator*/ rank == 0, bw);
  });

  std::vector<NdArrayRef> out_slices(2);
  out_slices[0] = mul_prot->MulShare(fx.slice({0}, {nhalf}, {1}),
                                     fy.slice({0}, {nhalf}, {1}),
                                     /*evaluato*/ rank != 0, bw);
  out_slices[1] = task.get();

  NdArrayRef out(x.eltype(), x.shape());
  int64_t offset = 0;
  for (auto& out_slice : out_slices) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }
  return out;
}
#else
NdArrayRef MulAA::mulDirectly(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) const {
  const auto bw = x.fxp_bits();
  // (x0 + x1) * (y0+ y1)
  // Compute the cross terms x0*y1, x1*y0 homomorphically
  auto* comm = ctx->getState<Communicator>();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  mul_prot->LazyInitKeys(x.eltype().as<Ring2k>()->field(), bw);

  const int rank = comm->getRank();
  // auto fy = y.reshape({y.numel()});

  auto dupx = ctx->getState<CheetahMulState>()->duplx();

  // compute x0*y1
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return mul_prot->MulOLE(x, dupx.get(), true, bw);
    }
    return mul_prot->MulOLE(y, dupx.get(), false, bw);
  });

  NdArrayRef x1y0;
  if (rank == 0) {
    x1y0 = mul_prot->MulOLE(y, false, bw);
  } else {
    x1y0 = mul_prot->MulOLE(x, true, bw);
  }

  NdArrayRef x0y1 = task.get();
  return ring_add(x0y1, ring_add(x1y0, ring_mul(x, y))).as(x.eltype());
}
#endif

NdArrayRef MatMulVVS::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                           const NdArrayRef& y) const {
  auto out_type = makeType<cheetah::AShrTy>(ctx->sctx()->getField());
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(out_type, {x.shape()[0], y.shape()[1]});
  }
  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();

  const int self_rank = comm->getRank();
  auto lhs_owner = x.eltype().as<Priv2kTy>()->owner();

  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};
  if (self_rank == lhs_owner) {
    return dot_prot->DotOLE(x, dim3, /*is_lhs*/ true).as(out_type);
  } else {
    return dot_prot->DotOLE(y, dim3, /*is_lhs*/ false).as(out_type);
  }
}

// A is (M, K); B is (K, N)
NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  dot_prot->LazyInitKeys(x.eltype().as<Ring2k>()->field());

  const int rank = comm->getRank();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms homomorphically
  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};

  auto* conn = comm->lctx().get();
  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    // Compute x0*y1
    if (rank == 0) {
      return dot_prot->DotOLE(x, dupx.get(), dim3, true);
    } else {
      return dot_prot->DotOLE(y, dupx.get(), dim3, false);
    }
  });

  NdArrayRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->DotOLE(y, conn, dim3, false);
  } else {
    x1y0 = dot_prot->DotOLE(x, conn, dim3, true);
  }

  auto ret = ring_mmul(x, y);
  ring_add_(ret, x1y0);
  return ring_add(ret, task.get()).as(x.eltype());
}

NdArrayRef MatMulAV::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }
  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  dot_prot->LazyInitKeys(x.eltype().as<Ring2k>()->field());

  const int rank = comm->getRank();
  const auto* ptype = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");
  const int owner = ptype->owner();
  NdArrayRef out;
  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};
  // (x0 + x1)*y = <x0 * y>_0 + <x0 * y>_1 + x1 * y
  if (rank == owner) {
    // Compute <y * x0>
    out = dot_prot->DotOLE(y, dim3, false);
    auto local = ring_mmul(x, y);
    ring_add_(out, local);
  } else {
    out = dot_prot->DotOLE(x, dim3, true);
  }
  return out.as(x.eltype());
}

NdArrayRef LutAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                       const NdArrayRef& table, size_t bw,
                       FieldType field) const {
  const auto to_field =
      field == FieldType::FT_INVALID ? FixGetProperFiled(bw) : field;

  auto ret = TiledDispatchOTFuncForLUT(
                 ctx, in, table,
                 [&](const NdArrayRef& in, const NdArrayRef& table,
                     const std::shared_ptr<BasicOTProtocols>& base_ot) {
                   return base_ot->LookUpTable(in, table, bw, to_field);
                 })
                 .as(makeType<AShrTy>(field));
  ret.set_fxp_bits(bw);

  return ret;
}

}  // namespace spu::mpc::cheetah
