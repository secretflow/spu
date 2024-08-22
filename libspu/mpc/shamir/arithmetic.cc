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

#include "libspu/mpc/shamir/arithmetic.h"

#include <functional>

#include "libspu/core/type_util.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/common/pv_gfmp.h"
#include "libspu/mpc/shamir/type.h"
#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::shamir {

namespace {

NdArrayRef wrap_a2p(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(a2p(ctx, WrapValue(x)));
}

// Ref: DN'07 protocol
//  https://www.iacr.org/archive/crypto2007/46220565/46220565.pdf
std::pair<NdArrayRef, NdArrayRef> gen_double_shares(KernelEvalContext* ctx,
                                                    int64_t numel) {
  auto* comm = ctx->getState<Communicator>();
  int64_t th = ctx->sctx()->config().sss_threshold();
  int64_t world_size = comm->getWorldSize();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto* prg_state = ctx->getState<PrgState>();
  auto rank = comm->getRank();

  // run one-time DN protocol we can generate (world_size-th) pairs of
  // double shares, so we need dn_times to generate multiplication double
  // shares
  auto dn_times = (numel - 1) / (world_size - th) + 1;
  auto ty = makeType<AShrTy>(field);
  auto r = gfmp_mod(prg_state->genPriv(field, {dn_times}).as(ty));
  auto t_sh_local = gfmp_rand_shamir_shares(r, world_size, th);
  auto t2_sh_local = gfmp_rand_shamir_shares(r, world_size, th * 2);

  std::vector<NdArrayRef> t_sh_global(world_size);
  std::vector<NdArrayRef> t2_sh_global(world_size);

  // Todo: optimize this ugly off-line code
  for (size_t i = 0; i < t_sh_local.size(); ++i) {
    if (i != rank) {
      auto share = t_sh_local[i].concatenate({t2_sh_local[i]}, 0);
      comm->sendAsync(i, share, "send share");
    }
  }
  for (size_t i = 0; i < t_sh_local.size(); ++i) {
    if (i != rank) {
      auto share = comm->recv(i, ty, "recv share");
      t_sh_global[i] = share.slice({0}, {dn_times}, {});
      t2_sh_global[i] = share.slice({dn_times}, {2 * dn_times}, {});
    }
  }
  t_sh_global[rank] = t_sh_local[rank];
  t2_sh_global[rank] = t2_sh_local[rank];

  std::pair<NdArrayRef, NdArrayRef> out{
      NdArrayRef(ty, {dn_times * (world_size - th)}),
      NdArrayRef(ty, {dn_times * (world_size - th)})};

  // generate random double shares
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _r_t(out.first);
    NdArrayView<ring2k_t> _r_2t(out.second);
    auto van = GenVandermondeMatrix<ring2k_t>(world_size, world_size - th);
    pforeach(0, dn_times, [&](int64_t idx) {
      // Optimize me: no copy need here
      GfmpMatrix<ring2k_t> s_t(1, world_size);
      GfmpMatrix<ring2k_t> s_2t(1, world_size);
      for (auto i = 0; i < world_size; ++i) {
        NdArrayView<ring2k_t> _share_t(t_sh_global[i]);
        NdArrayView<ring2k_t> _share_2t(t2_sh_global[i]);
        s_t(0, i) = Gfmp<ring2k_t>(_share_t[idx]);
        s_2t(0, i) = Gfmp<ring2k_t>(_share_2t[idx]);
      }
      auto ret_t = s_t * van;
      auto ret_2t = s_2t * van;

      for (auto i = 0; i < (world_size - th); ++i) {
        _r_t[idx * (world_size - th) + i] = ret_t(0, i).data();
        _r_2t[idx * (world_size - th) + i] = ret_2t(0, i).data();
      }
    });
  });
  return out;
}

}  // namespace

// Ref: DN'07 protocol for honesty majority
//  https://www.iacr.org/archive/crypto2007/46220565/46220565.pdf
NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  NdArrayRef out = ring_zeros(field, shape);
  int64_t world_size = comm->getWorldSize();
  int64_t th = ctx->sctx()->config().sss_threshold();
  int64_t numel = shape.numel();

  // run one-time DN protocol we can generate (world_size-th) random shares
  auto dn_times = (numel - 1) / (world_size - th) + 1;
  auto ty = makeType<GfmpTy>(field);
  auto r = gfmp_mod(prg_state->genPriv(field, {dn_times}).as(ty));
  auto shares_r = gfmp_rand_shamir_shares(r, world_size, th);
  auto rank = comm->getRank();

  std::vector<NdArrayRef> r_shrs(world_size);
  for (size_t i = 0; i < shares_r.size(); ++i) {
    if (i != rank) {
      comm->sendAsync(i, shares_r[i], "send r_share");
    }
  }
  for (size_t i = 0; i < shares_r.size(); ++i) {
    if (i != rank) {
      r_shrs[i] = comm->recv(i, shares_r[i].eltype(), "send r_share");
    }
  }
  r_shrs[rank] = shares_r[rank];
  return DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayRef r_t(ty, {dn_times * (world_size - th)});
    NdArrayView<ring2k_t> _r_t(r_t);
    auto van = GenVandermondeMatrix<ring2k_t>(world_size, world_size - th);
    pforeach(0, dn_times, [&](int64_t idx) {
      GfmpMatrix<ring2k_t> s_t(1, world_size);
      for (auto i = 0; i < world_size; ++i) {
        NdArrayView<ring2k_t> _r_shrs(r_shrs[i]);
        s_t(0, i) = Gfmp<ring2k_t>(_r_shrs[idx]);
      }
      auto ret_t = s_t * van;

      for (auto i = 0; i < (world_size - th); ++i) {
        _r_t[idx * (world_size - th) + i] = ret_t(0, i).data();
      }
    });
    auto out = r_t.slice({0}, {numel}, {});
    return out.as(makeType<AShrTy>(field));
  });
}

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  int64_t th = ctx->sctx()->config().sss_threshold();
  auto ty = makeType<PubGfmpTy>(field);
  auto r = prg_state->genPubl(field, {th * in.numel()}).as(ty);
  auto coeffs = gfmp_mod(r);
  auto shares = gfmp_rand_shamir_shares(in, coeffs, comm->getWorldSize(), th);
  return shares[comm->getRank()].as(makeType<AShrTy>(field));
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<GfmpTy>()->field();
  auto* comm = ctx->getState<Communicator>();
  // we choose rank 0 as the P_pking for reconstructing secrets
  auto arrays = comm->gather(in, 0, "send to pking");
  NdArrayRef out = ring_zeros(field, in.shape());
  if (comm->getRank() == 0) {
    out = gfmp_reconstruct_shamir_shares(arrays, comm->getWorldSize(),
                                         ctx->sctx()->config().sss_threshold());
  }
  out = comm->broadcast(out, 0, "distribute");
  return out.as(makeType<PubGfmpTy>(field));
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto out_ty = makeType<PrivGfmpTy>(field, rank);
  auto arrays = comm->gather(in, rank, "gather");
  if (comm->getRank() == rank) {
    SPU_ENFORCE(arrays.size() == comm->getWorldSize());
    auto out = gfmp_reconstruct_shamir_shares(
        arrays, comm->getWorldSize(), ctx->sctx()->config().sss_threshold());
    return out.as(out_ty);
  } else {
    return makeConstantArrayRef(out_ty, in.shape());
  }
}

NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<PrivGfmpTy>();
  const size_t owner_rank = in_ty->owner();
  const auto field = in_ty->field();
  auto* comm = ctx->getState<Communicator>();
  auto out_ty = makeType<AShrTy>(field);
  auto th = ctx->sctx()->config().sss_threshold();
  NdArrayRef out;
  if (comm->getRank() == owner_rank) {
    auto shares = gfmp_rand_shamir_shares(in, comm->getWorldSize(), th);
    for (size_t i = 0; i < shares.size(); ++i) {
      if (i != owner_rank) {
        comm->sendAsync(i, shares[i], "v2a");
      }
    }
    out = shares[owner_rank];
  } else {
    out = comm->recv(owner_rank, out_ty, "v2a");
  }
  return out.reshape(in.shape()).as(out_ty);
}

NdArrayRef NegateA::proc(KernelEvalContext*, const NdArrayRef& in) const {
  NdArrayRef out(in.eltype(), in.shape());
  const auto* ty = in.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = in.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _out(out);
    NdArrayView<ring2k_t> _in(in);
    pforeach(0, numel, [&](int64_t idx) { _out[idx] = add_inv(_in[idx]); });
  });
  return out;
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
NdArrayRef AddAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  auto ret = gfmp_add_mod(lhs, rhs);
  return ret.as(lhs.eltype());
}

NdArrayRef AddAA::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  SPU_ENFORCE(lhs.eltype() == rhs.eltype());
  return gfmp_add_mod(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
NdArrayRef MulAP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  return gfmp_mul_mod(lhs, rhs).as(lhs.eltype());
}

// Ref: DN'07 protocol
//  https://www.iacr.org/archive/crypto2007/46220565/46220565.pdf
NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  SPU_ENFORCE_EQ(lhs.eltype(), rhs.eltype());

  // local mul
  auto tmp_2t = gfmp_mul_mod(lhs, rhs).as(lhs.eltype());
  NdArrayRef out(lhs.eltype(), lhs.shape());

  // reduction degree
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto numel = lhs.numel();

  const auto& [r_t, r_2t] = gen_double_shares(ctx, numel);

  DISPATCH_ALL_FIELDS(field, [&]() {
    // generate double shares
    NdArrayView<ring2k_t> _r_t(r_t);
    NdArrayView<ring2k_t> _r_2t(r_2t);
    NdArrayView<ring2k_t> _out(out);
    NdArrayView<ring2k_t> _tmp_2t(tmp_2t);
    NdArrayRef d(lhs.eltype(), lhs.shape());
    NdArrayView<ring2k_t> _d(d);
    pforeach(0, numel,
             [&](int64_t idx) { _d[idx] = add_mod(_tmp_2t[idx], _r_2t[idx]); });
    auto revealed_d = wrap_a2p(ctx->sctx(), d);
    NdArrayView<ring2k_t> _revealed_d(revealed_d);
    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = add_mod(_revealed_d[idx], add_inv(_r_t[idx]));
    });
  });
  return out;
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
NdArrayRef MatMulAP::proc(KernelEvalContext*, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  return gfmp_mmul_mod(x, y).as(x.eltype());
}

NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  const auto field = x.eltype().as<GfmpTy>()->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    // local matmul
    NdArrayRef tmp_2t = gfmp_mmul_mod(x, y);
    NdArrayView<ring2k_t> _tmp_2t(tmp_2t);
    // degree reduction
    NdArrayRef out(tmp_2t.eltype(), tmp_2t.shape());
    const auto& [r_t, r_2t] = gen_double_shares(ctx, tmp_2t.numel());
    NdArrayView<ring2k_t> _r_t(r_t);
    NdArrayView<ring2k_t> _r_2t(r_2t);
    NdArrayView<ring2k_t> _out(out);
    NdArrayRef d(out.eltype(), out.shape());
    NdArrayView<ring2k_t> _d(d);
    pforeach(0, tmp_2t.numel(),
             [&](int64_t idx) { _d[idx] = add_mod(_tmp_2t[idx], _r_2t[idx]); });
    auto revealed_d = wrap_a2p(ctx->sctx(), d);
    NdArrayView<ring2k_t> _revealed_d(revealed_d);
    pforeach(0, out.numel(), [&](int64_t idx) {
      _out[idx] = add_mod(_revealed_d[idx], add_inv(_r_t[idx]));
    });
    return out;
  });
};

}  // namespace spu::mpc::shamir
