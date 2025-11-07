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

#include "libspu/mpc/cheetah/conversion.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/cheetah/nonlinear/ext_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_and_reduce_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

namespace {

NdArrayRef a2b_impl(KernelEvalContext* ctx, const NdArrayRef& x,
                    int64_t nbits) {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  const auto ring_bits = x.fxp_bits();

  int64_t valid_bits;
  if (ring_bits > 0) {
    SPU_ENFORCE(nbits == -1, "nbits={} inconsistent with ring fxp_bits={}",
                nbits, ring_bits);
    SPU_ENFORCE((size_t)ring_bits <= SizeOf(field) * 8,
                "ring fxp_bits={} invalid for field={}", ring_bits, field);
    valid_bits = ring_bits;
  } else {
    valid_bits = nbits == -1 ? SizeOf(field) * 8 : nbits;
  }
  std::vector<NdArrayRef> bshrs;

  const auto bty = makeType<BShrTy>(field, valid_bits);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto [r0, r1] =
        prg_state->genPrssPair(field, x.shape(), PrgState::GenPrssCtrl::Both);
    auto b = ring_xor(r0, r1);

    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }

    ring_reduce_(b, valid_bits);
    bshrs.push_back(b.as(bty));
  }

  NdArrayRef res = vreduce(bshrs.begin(), bshrs.end(),
                           [&](const NdArrayRef& xx, const NdArrayRef& yy) {
                             return wrap_add_bb(ctx->sctx(), xx, yy);
                           });

  if (nbits != -1 || ring_bits > 0) {
    ring_reduce_(res, valid_bits);
  }
  return res.as(bty);
}
}  // namespace

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  // full bits A2B
  return a2b_impl(ctx, x, -1);
}

NdArrayRef A2B_Bits::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          int64_t nbits) const {
  return a2b_impl(ctx, x, nbits);
}

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto numel = x.numel();
  if (numel == 0) {  // for empty input
    return NdArrayRef(makeType<AShrTy>(field), x.shape());
  }
  return TiledDispatchOTFunc(
             ctx, x,
             [&](const NdArrayRef& input,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               return base_ot->B2A(input);
             })
      .as(makeType<AShrTy>(field));
}

void CommonTypeV::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

  const auto* lhs_v = lhs.as<Priv2kTy>();
  const auto* rhs_v = rhs.as<Priv2kTy>();

  ctx->pushOutput(makeType<AShrTy>(std::max(lhs_v->field(), rhs_v->field())));
}

NdArrayRef RingCastDownA::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                               FieldType to_field) const {
  SPU_ENFORCE(in.eltype().isa<AShrTy>());
  const auto from_field = in.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(SizeOf(from_field) >= SizeOf(to_field),
              "from_field={} to_field={}", from_field, to_field);

  if (from_field == to_field) {
    return in;
  }

  NdArrayRef out(makeType<AShrTy>(to_field), in.shape());

  DISPATCH_ALL_FIELDS(from_field, [&]() {
    using FromT = std::make_unsigned_t<ring2k_t>;
    DISPATCH_ALL_FIELDS(to_field, [&]() {
      using ToT = std::make_unsigned_t<ring2k_t>;
      NdArrayView<FromT> _in(in);
      NdArrayView<ToT> _out(out);
      pforeach(0, in.numel(),
               [&](int64_t idx) { _out[idx] = static_cast<ToT>(_in[idx]); });
    });
  });

  return out;
}

NdArrayRef RingCastUp::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                            size_t bw, FieldType to_field, SignType sign,
                            bool signed_arith, bool force,
                            bool heuristic) const {
  SPU_ENFORCE(SizeOf(to_field) * 8 >= bw);
  // TODO: make force=true, heuristic=false;
  SPU_ENFORCE(force && (!heuristic));
  // for bshare, just change the type
  if (in.eltype().isa<BShrTy>()) {
    const auto from_ty = in.eltype().as<RingTy>()->field();
    const auto new_ty = makeType<BShrTy>(to_field, bw);
    NdArrayRef res(new_ty, in.shape());

    DISPATCH_ALL_FIELDS(from_ty, [&]() {
      using sT = ring2k_t;
      NdArrayView<const sT> src(in);

      DISPATCH_ALL_FIELDS(to_field, [&]() {
        using dT = ring2k_t;
        NdArrayView<dT> dst(res);
        pforeach(0, in.numel(),
                 [&](int64_t i) { dst[i] = static_cast<dT>(src[i]); });
      });
    });
  }

  SPU_ENFORCE(in.eltype().isa<AShrTy>(),
              "only BshrTy and AshrTy are supported.");
  const auto field = in.eltype().as<RingTy>()->field();
  const auto src_bw = in.fxp_bits() == 0 ? SizeOf(field) * 8 : in.fxp_bits();
  SPU_ENFORCE(src_bw <= bw);

  if (src_bw == bw) {
    SPU_ENFORCE(field == to_field,
                "Should be same field when bit width is same.");
    return in;
  }

  // if (!force && (field == to_field)) {
  //   SPDLOG_WARN("In Cheetah CastUp: not do casting up because field is
  //   same."); return in;
  // }

  const auto out_ty = makeType<AShrTy>(to_field);

  auto ret = TiledDispatchOTFunc(
                 ctx, in,
                 [&](const NdArrayRef& input,
                     const std::shared_ptr<BasicOTProtocols>& base_ot) {
                   RingExtendProtocol::Meta meta;

                   meta.sign = sign;
                   meta.signed_arith = signed_arith;

                   meta.src_width = src_bw;
                   meta.src_ring = field;
                   meta.dst_width = bw;
                   meta.dst_ring = to_field;

                   // use heuristic only for signed arith
                   meta.use_heuristic = heuristic;

                   RingExtendProtocol prot(base_ot);
                   return prot.Compute(input, meta);
                 })
                 .as(out_ty);

  ret.set_fxp_bits(bw);
  return ret;
}

NdArrayRef TruncateReduce::proc(
    KernelEvalContext* ctx, const NdArrayRef& in,
    const NdArrayRef& wrap_s,         //  only useful for exact truncation
    size_t bits, FieldType to_field,  //
    bool exact) const {
  SPU_ENFORCE(in.eltype().isa<AShrTy>());

  const auto field = in.eltype().as<RingTy>()->field();
  const auto src_bw = in.fxp_bits() == 0 ? SizeOf(field) * 8 : in.fxp_bits();
  const auto bw = src_bw - bits;

  SPU_ENFORCE(SizeOf(to_field) * 8 >= bw);
  SPU_ENFORCE(src_bw >= bw);

  if (src_bw == bw) {
    SPU_ENFORCE(field == to_field,
                "Should be same field when bit width is same.");
    return in;
  }

  const auto out_ty = makeType<AShrTy>(to_field);

  auto ret = TiledDispatchOTFunc(
                 ctx, in,
                 [&](const NdArrayRef& input,
                     const std::shared_ptr<BasicOTProtocols>& base_ot) {
                   RingTruncateAndReduceProtocol::Meta meta;

                   meta.exact = exact;
                   meta.src_width = src_bw;
                   meta.src_ring = field;
                   meta.dst_width = bw;
                   meta.dst_ring = to_field;

                   RingTruncateAndReduceProtocol prot(base_ot);
                   return prot.Compute(input, wrap_s, meta);
                 })
                 .as(out_ty);

  ret.set_fxp_bits(bw);
  return ret;
}

}  // namespace spu::mpc::cheetah
