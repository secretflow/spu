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

#include "libspu/core/memref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

MemRef A2B::proc(KernelEvalContext* ctx, const MemRef& x) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  std::vector<MemRef> bshrs;
  const auto bty =
      makeType<BoolShareTy>(x.eltype().semantic_type(), GetStorageType(field),
                            x.eltype().as<BaseRingType>()->valid_bits());
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    MemRef r0(x.eltype(), x.shape());
    MemRef r1(x.eltype(), x.shape());
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.elsize() * r0.numel());

    auto b = ring_xor(r0, r1).as(bty);

    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }
    bshrs.push_back(b.as(bty));
  }

  MemRef res = vreduce(bshrs.begin(), bshrs.end(),
                       [&](const MemRef& xx, const MemRef& yy) {
                         return add_bb(ctx->sctx(), xx, yy);
                       });
  return res.as(bty);
}

MemRef B2A::proc(KernelEvalContext* ctx, const MemRef& x) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  return TiledDispatchOTFunc(
             ctx, x,
             [&](const MemRef& input,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               return base_ot->B2A(input);
             })
      .as(makeType<ArithShareTy>(x.eltype().semantic_type(), field));
}

void CommonTypeV::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

  const auto* lhs_v = lhs.as<Priv2kTy>();
  const auto* rhs_v = rhs.as<Priv2kTy>();
  const auto valid_bits = std::max(lhs_v->valid_bits(), rhs_v->valid_bits());

  auto st = SizeOf(lhs_v->storage_type()) > SizeOf(rhs_v->storage_type())
                ? lhs_v->semantic_type()
                : rhs_v->semantic_type();
  ctx->pushOutput(makeType<ArithShareTy>(st, valid_bits));
}

MemRef RingCastS::proc(KernelEvalContext*, const MemRef& in,
                       SemanticType to_type) const {
  MemRef out(in);
  out.eltype().as<BaseRingType>()->set_semantic_type(to_type);
  return out;
}

std::vector<MemRef> BitDecompose::proc(KernelEvalContext* ctx,
                                       const MemRef& in) const {
  SPU_ENFORCE(in.eltype().as<BoolShare>());
  const auto* ty = in.eltype().as<BoolShareTy>();
  size_t nbits = ty->valid_bits();
  SPU_ENFORCE_GT(nbits, 0U);
  std::vector<MemRef> outs;
  auto bty = makeType<BoolShareTy>(ty->semantic_type(), ST_8, 1);
  DISPATCH_ALL_STORAGE_TYPES(ty->storage_type(), [&]() {
    using InT = ScalarT;
    MemRefView<InT> _in(in);
    DISPATCH_ALL_STORAGE_TYPES(bty.storage_type(), [&]() {
      using OutT = ScalarT;
      for (size_t i = 0; i < nbits; ++i) {
        MemRef bit(bty, in.shape());
        MemRefView<OutT> _bit(bit);
        pforeach(0, in.numel(), [&](int64_t idx) {
          _bit[idx] = static_cast<OutT>((_in[idx] >> i) & 0x1);
        });
        outs.push_back(std::move(bit));
      }
    });
  });
  return outs;
}

MemRef BitCompose::proc(KernelEvalContext* ctx,
                        const std::vector<MemRef>& in) const {
  SPU_ENFORCE(!in.empty());
  size_t nbits = in.size();
  auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto sst = GetStorageType(field);
  auto smt = in[0].eltype().semantic_type();
  const auto ty = makeType<BoolShareTy>(smt, sst, nbits);
  MemRef out(ty, in[0].shape());
  ring_zeros(out);
  DISPATCH_ALL_STORAGE_TYPES(sst, [&]() {
    using OutT = ScalarT;
    MemRefView<OutT> _out(out);
    for (size_t i = 0; i < nbits; ++i) {
      DISPATCH_ALL_STORAGE_TYPES(in[i].eltype().storage_type(), [&]() {
        using InT = ScalarT;
        MemRefView<InT> _in(in[i]);
        pforeach(0, in[0].numel(), [&](int64_t idx) {
          _out[idx] |= static_cast<OutT>(_in[idx] & 0x1) << i;
        });
      });
    }
  });
  return out;
}

}  // namespace spu::mpc::cheetah
