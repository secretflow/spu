// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/semi2k/permute.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

namespace {

NdArrayRef wrap_a2v(SPUContext* ctx, const NdArrayRef& x, size_t rank) {
  return UnwrapValue(a2v(ctx, WrapValue(x), rank));
}

inline bool isOwner(KernelEvalContext* ctx, const Type& type) {
  auto* comm = ctx->getState<Communicator>();
  return type.as<Priv2kTy>()->owner() == static_cast<int64_t>(comm->getRank());
}

inline int64_t getOwner(const NdArrayRef& x) {
  return x.eltype().as<Priv2kTy>()->owner();
}

Index ring2pv(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<Ring2k>(), "must be ring2k_type, got={}",
              x.eltype());
  const auto field = x.eltype().as<Ring2k>()->field();
  Index pv(x.numel());
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, x.numel(), [&](int64_t idx) { pv[idx] = int64_t(_x[idx]); });
  });
  return pv;
}

// Secure inverse permutation of x by perm_rank's permutation pv
// The idea here is:
// Input permutation pv, beaver generates perm pair {<A>, <B>} that
// InversePermute(A, pv) = B. So we can get <y> = InversePermute(open(<x> -
// <A>), pv) + <B> that y = InversePermute(x, pv).
NdArrayRef SecureInvPerm(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& perm, size_t perm_rank) {
  const auto lctx = ctx->lctx();
  const auto field = x.eltype().as<AShrTy>()->field();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  auto numel = x.numel();

  Index pv;
  if (perm.eltype().isa<PShare>() ||
      (perm.eltype().isa<Private>() && isOwner(ctx, perm.eltype()))) {
    pv = ring2pv(perm);
  }
  auto [a_buf, b_buf] = beaver->PermPair(field, numel, perm_rank, pv);

  NdArrayRef a(std::make_shared<yacl::Buffer>(std::move(a_buf)), x.eltype(),
               x.shape());
  NdArrayRef b(std::make_shared<yacl::Buffer>(std::move(b_buf)), x.eltype(),
               x.shape());

  auto t = wrap_a2v(ctx->sctx(), ring_sub(x, a).as(x.eltype()), perm_rank);

  if (lctx->Rank() == perm_rank) {
    SPU_ENFORCE(pv.size());
    ring_add_(b, applyInvPerm(t, pv));
    return b.as(x.eltype());
  } else {
    return b.as(x.eltype());
  }
}

}  // namespace

NdArrayRef RandPermM::proc(KernelEvalContext* ctx, const Shape& shape) const {
  NdArrayRef out(makeType<PShrTy>(), shape);

  auto* prg_state = ctx->getState<PrgState>();
  const auto perm_vector = prg_state->genPrivPerm(out.numel());

  const auto field = out.eltype().as<PShrTy>()->field();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, out.numel(),
             [&](int64_t idx) { _out[idx] = ring2k_t(perm_vector[idx]); });
  });

  return out;
}

NdArrayRef PermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  auto* comm = ctx->getState<Communicator>();

  NdArrayRef out(in);
  for (size_t i = 0; i < comm->getWorldSize(); ++i) {
    out = SecureInvPerm(ctx, out, perm, i);
  }
  return out;
}

NdArrayRef PermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  return applyPerm(in, perm);
}

NdArrayRef InvPermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  NdArrayRef out(in);
  auto inv_perm = genInversePerm(perm);
  for (int i = comm->getWorldSize() - 1; i >= 0; --i) {
    out = SecureInvPerm(ctx, out, inv_perm, i);
  }
  return out;
}

NdArrayRef InvPermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  return applyInvPerm(in, perm);
}

NdArrayRef InvPermAV::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  return SecureInvPerm(ctx, in, perm, getOwner(perm));
}

}  // namespace spu::mpc::semi2k