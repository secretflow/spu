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

  std::vector<NdArrayRef> bshrs;
  int64_t valid_bits = nbits == -1 ? SizeOf(field) * 8 : nbits;

  const auto bty = makeType<BShrTy>(field, valid_bits);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto [r0, r1] =
        prg_state->genPrssPair(field, x.shape(), PrgState::GenPrssCtrl::Both);
    auto b = ring_xor(r0, r1);

    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }

    bshrs.push_back(b.as(bty));
  }

  NdArrayRef res = vreduce(bshrs.begin(), bshrs.end(),
                           [&](const NdArrayRef& xx, const NdArrayRef& yy) {
                             return wrap_add_bb(ctx->sctx(), xx, yy);
                           });

  if (nbits != -1) {
    ring_bitmask_(res, 0, valid_bits);
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

}  // namespace spu::mpc::cheetah
