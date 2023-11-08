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

// Secure inverse permutation of x by perm_rank's permutation pv
// The idea here is:
// Input permutation pv, beaver generates perm pair {<A>, <B>} that
// InversePermute(A, pv) = B. So we can get <y> = InversePermute(open(<x> -
// <A>), pv) + <B> that y = InversePermute(x, pv).
NdArrayRef SecureInvPerm(KernelEvalContext* ctx, const NdArrayRef& x,
                         size_t perm_rank, absl::Span<const int64_t> pv) {
  const auto lctx = ctx->lctx();
  const auto field = x.eltype().as<AShrTy>()->field();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  auto perm_pair = beaver->PermPair(field, x.shape(), perm_rank, pv);

  auto t = wrap_a2v(ctx->sctx(), ring_sub(x, perm_pair.first).as(x.eltype()),
                    perm_rank);

  if (lctx->Rank() == perm_rank) {
    SPU_ENFORCE(pv.size());
    ring_add_(perm_pair.second, applyInvPerm(t, pv));
    return perm_pair.second.as(x.eltype());
  } else {
    return perm_pair.second.as(x.eltype());
  }
}

PermVector ring2pv(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<Ring2k>(), "must be ring2k_type, got={}",
              x.eltype());
  const auto field = x.eltype().as<Ring2k>()->field();
  PermVector pv(x.numel());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, x.numel(), [&](int64_t idx) { pv[idx] = int64_t(_x[idx]); });
  });
  return pv;
}

}  // namespace

NdArrayRef RandPermS::proc(KernelEvalContext* ctx, const Shape& shape) const {
  NdArrayRef out(makeType<PShrTy>(), shape);
  const auto field = out.eltype().as<PShrTy>()->field();
  const auto perm_vector = genRandomPerm(out.numel());

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, out.numel(),
             [&](int64_t idx) { _out[idx] = ring2k_t(perm_vector[idx]); });
  });

  return out;
}

NdArrayRef PermAS::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  auto* comm = ctx->getState<Communicator>();

  PermVector pv = ring2pv(perm);
  NdArrayRef out(in);
  for (size_t i = 0; i < comm->getWorldSize(); ++i) {
    out = SecureInvPerm(ctx, out, i, pv);
  }

  return out;
}

NdArrayRef PermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  PermVector pv = ring2pv(perm);
  auto out = applyPerm(in, pv);
  return out;
}

NdArrayRef InvPermAS::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  PermVector pv = ring2pv(perm);
  NdArrayRef out(in);
  auto inv_pv = genInversePerm(pv);
  for (int i = comm->getWorldSize() - 1; i >= 0; --i) {
    out = SecureInvPerm(ctx, out, i, inv_pv);
  }

  return out;
}

NdArrayRef InvPermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  PermVector pv = ring2pv(perm);
  auto out = applyInvPerm(in, pv);
  return out;
}

}  // namespace spu::mpc::semi2k