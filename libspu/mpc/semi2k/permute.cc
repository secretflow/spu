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

inline bool isOwner(KernelEvalContext* ctx, const Type& type) {
  auto* comm = ctx->getState<Communicator>();
  return type.as<Priv2kTy>()->owner() == static_cast<int64_t>(comm->getRank());
}

inline int64_t getOwner(const MemRef& x) {
  return x.eltype().as<Priv2kTy>()->owner();
}

// Secure inverse permutation of x by perm_rank's permutation pv
// The idea here is:
// Input permutation pv, beaver generates perm pair {<A>, <B>} that
// InversePermute(A, pv) = B. So we can get <y> = InversePermute(open(<x> -
// <A>), pv) + <B> that y = InversePermute(x, pv).
MemRef SecureInvPerm(KernelEvalContext* ctx, const MemRef& x, size_t perm_rank,
                     absl::Span<const int64_t> pv) {
  const auto lctx = ctx->lctx();
  const size_t field = ctx->getState<Z2kState>()->getDefaultField();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  auto numel = x.numel();

  auto [a_buf, b_buf] = beaver->PermPair(field, numel, perm_rank, pv);

  MemRef a(std::make_shared<yacl::Buffer>(std::move(a_buf)), x.eltype(),
           x.shape());
  MemRef b(std::make_shared<yacl::Buffer>(std::move(b_buf)), x.eltype(),
           x.shape());

  auto t = a2v(ctx->sctx(), ring_sub(x, a).as(x.eltype()), perm_rank);

  if (lctx->Rank() == perm_rank) {
    SPU_ENFORCE(pv.size());
    auto inv_t = applyInvPerm(t, pv);

    if (inv_t.eltype().storage_type() != x.eltype().storage_type()) {
      MemRef inv_t_cast(makeType<RingTy>(inv_t.eltype().semantic_type(), field),
                        inv_t.shape());
      ring_assign(inv_t_cast, inv_t);

      ring_add_(b, inv_t_cast);
    } else {
      ring_add_(b, inv_t);
    }
    return b.as(x.eltype());
  } else {
    return b.as(x.eltype());
  }
}

}  // namespace

MemRef RandPermM::proc(KernelEvalContext* ctx, const Shape& shape) const {
  MemRef out(makeType<PermShareTy>(), shape);

  // generate a RandU64 as permutation seed
  auto* prg_state = ctx->getState<PrgState>();
  uint64_t seed;
  prg_state->fillPriv(&seed, 8);
  const auto perm_vector = genRandomPerm(out.numel(), seed);

  DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _out(out);
    pforeach(0, out.numel(),
             [&](int64_t idx) { _out[idx] = ScalarT(perm_vector[idx]); });
  });

  return out;
}

MemRef PermAM::proc(KernelEvalContext* ctx, const MemRef& in,
                    const MemRef& perm) const {
  auto* comm = ctx->getState<Communicator>();

  PermVector pv = ring2pv(perm);
  MemRef out(in);
  for (size_t i = 0; i < comm->getWorldSize(); ++i) {
    out = SecureInvPerm(ctx, out, i, pv);
  }

  return out;
}

MemRef PermAP::proc(KernelEvalContext* ctx, const MemRef& in,
                    const MemRef& perm) const {
  PermVector pv = ring2pv(perm);
  auto out = applyPerm(in, pv);
  return out;
}

MemRef InvPermAM::proc(KernelEvalContext* ctx, const MemRef& in,
                       const MemRef& perm) const {
  auto* comm = ctx->getState<Communicator>();
  PermVector pv = ring2pv(perm);
  MemRef out(in);
  auto inv_pv = genInversePerm(pv);
  for (int i = comm->getWorldSize() - 1; i >= 0; --i) {
    out = SecureInvPerm(ctx, out, i, inv_pv);
  }

  return out;
}

MemRef InvPermAP::proc(KernelEvalContext* ctx, const MemRef& in,
                       const MemRef& perm) const {
  PermVector pv = ring2pv(perm);
  auto out = applyInvPerm(in, pv);
  return out;
}

MemRef InvPermAV::proc(KernelEvalContext* ctx, const MemRef& in,
                       const MemRef& perm) const {
  PermVector pv;
  const auto lctx = ctx->lctx();
  if (isOwner(ctx, perm.eltype())) {
    pv = ring2pv(perm);
  }
  auto out = SecureInvPerm(ctx, in, getOwner(perm), pv);
  return out;
}

}  // namespace spu::mpc::semi2k