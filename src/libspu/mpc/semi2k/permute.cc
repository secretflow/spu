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

// Secure inverse permutation of x by perm_rank's permutation pv
NdArrayRef SecureInvPerm(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& perm, size_t perm_rank) {
  // INPUT: X and private perm owned by perm_rank
  const auto lctx = ctx->lctx();
  const auto field = x.eltype().as<AShrTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  auto numel = x.numel();

  if (lctx->Rank() == perm_rank) {
    SPU_ENFORCE(perm.numel() == numel);
    SPU_ENFORCE(perm.eltype().isa<PShare>() ||
                (perm.eltype().isa<Private>() && isOwner(ctx, perm.eltype())));
  }

  // beaver gives ai, bi, pr makes InvPerm(A, pr) = B
  // pr is a private random permutation owned by perm_rank.
  auto [a_buf, b_buf, pr] = beaver->PermPair(field, numel, perm_rank);

  NdArrayRef po;
  if (lctx->Rank() == perm_rank) {
    // mask perm by random permutation pr, get po = InvPerm(perm, pr)
    auto p = std::move(pr);
    po = applyInvPerm(perm, p);
    // so: InvPerm(B, po) = InvPerm(InvPerm(A, pr), po) = InvPerm(A, perm)
  }
  // broadcast po to all rank.
  po = comm->broadcast(po, perm_rank, perm.eltype(), perm.shape(),
                       "perm_open_perm");

  NdArrayRef a(std::make_shared<yacl::Buffer>(std::move(a_buf)), x.eltype(),
               x.shape());
  NdArrayRef b(std::make_shared<yacl::Buffer>(std::move(b_buf)), x.eltype(),
               x.shape());

  // reveal X-A to perm_rank
  auto x_a = wrap_a2v(ctx->sctx(), ring_sub(x, a).as(x.eltype()), perm_rank);

  if (lctx->Rank() == perm_rank) {
    // perm_rank get InvPerm(X-A, perm) + InvPerm(bi, po)
    b = applyInvPerm(b, po);
    ring_add_(b, applyInvPerm(x_a, perm));
    return b.as(x.eltype());
  } else {
    // others rank get InvPerm(bi, po)
    return applyInvPerm(b, po).as(x.eltype());
  }
  // finally get:
  // InvPerm(X-A, perm) + ∑InvPerm(bi, po) =
  // InvPerm(X, perm) - InvPerm(A, perm) + InvPerm(B, po) =
  // InvPerm(X, perm)
}

// Input: perm = (3,2,3,6,3,6), X = (4,1,8,2,7,9,5,5)
// Expected output: perm(X) = (2,8,2,5,2,5)
NdArrayRef GeneralSecurePerm(KernelEvalContext* ctx, const NdArrayRef& x,
                             const NdArrayRef& perm, size_t perm_rank) {
  const auto lctx = ctx->lctx();
  const auto field = x.eltype().as<AShrTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  // NOTE: len(x)=m may not equal to len(perm)=n
  auto numel = x.numel();

  if (lctx->Rank() == perm_rank) {
    SPU_ENFORCE(perm.eltype().isa<PShare>() ||
                (perm.eltype().isa<Private>() && isOwner(ctx, perm.eltype())));
  }

  // beaver gives ai, bi, pr makes pr(A) = B
  // po is a private random permutation owned by perm_rank.
  // po is the bijective permutation, so len(A) = len(B) = m
  auto [a_buf, b_buf, po] = beaver->GeneralPermPair(field, numel, perm_rank);

  NdArrayRef pr;
  if (lctx->Rank() == perm_rank) {
    auto p = std::move(po);
    // solve pr, s.t. perm = po ∘ pr
    // len(pr) = n
    pr = solvePerm(perm, p);
  }
  // broadcast po to all rank.
  pr = comm->broadcast(pr, perm_rank, perm.eltype(), perm.shape(),
                       "perm_open_perm");

  NdArrayRef a(std::make_shared<yacl::Buffer>(std::move(a_buf)), x.eltype(),
               x.shape());
  NdArrayRef b(std::make_shared<yacl::Buffer>(std::move(b_buf)), x.eltype(),
               x.shape());

  // reveal X-A to perm_rank
  auto x_a = wrap_a2v(ctx->sctx(), ring_sub(x, a).as(x.eltype()), perm_rank);

  if (lctx->Rank() == perm_rank) {
    auto ret = generalApplyPerm(b, pr);
    ring_add_(ret, generalApplyPerm(x_a, perm));
    return ret.as(x.eltype());
  } else {
    return generalApplyPerm(b, pr).as(x.eltype());
  }
}

}  // namespace

NdArrayRef RandPermM::proc(KernelEvalContext* ctx, const Shape& shape,
                           FieldType perm_field) const {
  NdArrayRef out(makeType<PShrTy>(perm_field), shape);

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

// compute perm(x), where perm is owned by perm_rank, x is secret shared.
// x: m elements
// perm: n elements
// output: perm(x), n elements
NdArrayRef PermAV2::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         const NdArrayRef& perm) const {
  return GeneralSecurePerm(ctx, in, perm, getOwner(perm));
}

NdArrayRef PermAP2::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         const NdArrayRef& perm) const {
  return generalApplyPerm(in, perm);
}

}  // namespace spu::mpc::semi2k