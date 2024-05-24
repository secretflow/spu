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
// The idea here is:
// Input permutation pv, beaver generates perm pair {<A>, <B>} that
// InversePermute(A, pv) = B. So we can get <y> = InversePermute(open(<x> -
// <A>), pv) + <B> that y = InversePermute(x, pv).
NdArrayRef SecureInvPerm(KernelEvalContext* ctx, const NdArrayRef& x,
                         size_t perm_rank, absl::Span<const int64_t> pv) {
  const auto lctx = ctx->lctx();
  const auto field = x.eltype().as<AShrTy>()->field();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  auto numel = x.numel();

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

  // generate a RandU64 as permutation seed
  auto* prg_state = ctx->getState<PrgState>();
  const auto seed = prg_state->genPriv(FieldType::FM64, {1});
  NdArrayView<uint64_t> _seed(seed);
  const auto perm_vector = genRandomPerm(out.numel(), _seed[0]);

  const auto field = out.eltype().as<PShrTy>()->field();
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, out.numel(),
             [&](int64_t idx) { _out[idx] = ring2k_t(perm_vector[idx]); });
  });

  return out;
}

NdArrayRef PermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
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

NdArrayRef InvPermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
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

NdArrayRef InvPermAV::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  PermVector pv;
  const auto lctx = ctx->lctx();
  if (isOwner(ctx, perm.eltype())) {
    pv = ring2pv(perm);
  }
  auto out = SecureInvPerm(ctx, in, getOwner(perm), pv);
  return out;
}

}  // namespace spu::mpc::semi2k