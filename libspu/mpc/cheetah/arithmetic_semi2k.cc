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
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
namespace spu::mpc::cheetah {
/// Kernels that identical to semi2k ///
MemRef RandA::proc(KernelEvalContext* ctx, SemanticType type,
                   const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  MemRef ret(makeType<ArithShareTy>(type, field), shape);
  prg_state->fillPriv(ret.data(), ret.elsize() * ret.numel());
  ring_rshift_(ret, {2});
  return ret;
}

MemRef P2A::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* ty = in.eltype().as<BaseRingType>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  MemRef r0(makeType<RingTy>(in.eltype().semantic_type(), field), in.shape());
  MemRef r1(makeType<RingTy>(in.eltype().semantic_type(), field), in.shape());

  prg_state->fillPrssPair(r0.data(), r1.data(), r0.elsize() * r0.numel());
  auto x = ring_sub(r0, r1);

  if (comm->getRank() == 0) {
    if (x.eltype().storage_type() != in.eltype().storage_type()) {
      MemRef in_cast(x.eltype(), in.shape());
      ring_assign(in_cast, in);
      ring_add_(x, in_cast);
    } else {
      ring_add_(x, in);
    }
  }

  return x.as(makeType<ArithShareTy>(ty->semantic_type(), field));
}

MemRef A2P::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* ty = in.eltype().as<BaseRingType>();
  auto* comm = ctx->getState<Communicator>();
  auto tmp = comm->allReduce(ReduceOp::ADD, in, kBindName());
  MemRef out(makeType<Pub2kTy>(ty->semantic_type()), in.shape());
  ring_assign(out, tmp);
  return out;
}

MemRef A2V::proc(KernelEvalContext* ctx, const MemRef& in, size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  auto out_ty = makeType<Priv2kTy>(in.eltype().semantic_type(),
                                   in.eltype().storage_type(), rank);

  auto numel = in.numel();

  return DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using ring2k_t = ScalarT;
    std::vector<ring2k_t> share(numel);
    MemRefView<ring2k_t> _in(in);
    pforeach(0, numel, [&](int64_t idx) { share[idx] = _in[idx]; });

    std::vector<std::vector<ring2k_t>> shares =
        comm->gather<ring2k_t>(share, rank, "a2v");  // comm => 1, k
    if (comm->getRank() == rank) {
      SPU_ENFORCE(shares.size() == comm->getWorldSize());
      MemRef out(out_ty, in.shape());
      MemRefView<ScalarT> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        ScalarT s = 0;
        for (auto& share : shares) {
          s += share[idx];
        }
        _out[idx] = s;
      });
      return out;
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  });
}

MemRef V2A::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const size_t owner_rank = in_ty->owner();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  MemRef r0(makeType<RingTy>(in_ty->semantic_type(), field), in.shape());
  MemRef r1(makeType<RingTy>(in_ty->semantic_type(), field), in.shape());

  prg_state->fillPrssPair(r0.data(), r1.data(), r0.elsize() * r0.numel());
  auto x = ring_sub(r0, r1).as(
      makeType<ArithShareTy>(in.eltype().semantic_type(), field));

  if (comm->getRank() == owner_rank) {
    if (x.eltype().storage_type() != in.eltype().storage_type()) {
      MemRef in_cast(x.eltype(), in.shape());
      ring_assign(in_cast, in);
      ring_add_(x, in_cast);
    } else {
      ring_add_(x, in);
    }
  }
  return x.as(makeType<ArithShareTy>(in_ty->semantic_type(), field));
}

MemRef NegateA::proc(KernelEvalContext* ctx, const MemRef& in) const {
  auto res = ring_neg(in);
  return res.as(in.eltype());
}

MemRef AddAP::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  auto* comm = ctx->getState<Communicator>();

  if (comm->getRank() == 0) {
    if (lhs.eltype().storage_type() != rhs.eltype().storage_type()) {
      MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                       SizeOf(lhs.eltype().storage_type()) * 8),
                      rhs.shape());
      ring_assign(rhs_cast, rhs);
      return ring_add(lhs, rhs_cast).as(lhs.eltype());
    }
    return ring_add(lhs, rhs).as(lhs.eltype());
  }

  return lhs;
}

MemRef AddAA::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  SPU_ENFORCE(lhs.eltype().storage_type() == rhs.eltype().storage_type(),
              "lhs {} vs rhs {}", lhs.eltype(), rhs.eltype());

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto ty = makeType<ArithShareTy>(
      std::max(lhs.eltype().semantic_type(), rhs.eltype().semantic_type()),
      field);

  return ring_add(lhs, rhs).as(ty);
}

MemRef MulAP::proc(KernelEvalContext*, const MemRef& lhs,
                   const MemRef& rhs) const {
  if (lhs.eltype().storage_type() != rhs.eltype().storage_type()) {
    MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                     SizeOf(lhs.eltype().storage_type()) * 8),
                    rhs.shape());
    ring_assign(rhs_cast, rhs);
    return ring_mul(lhs, rhs_cast).as(lhs.eltype());
  }
  return ring_mul(lhs, rhs).as(lhs.eltype());
}

MemRef MatMulAP::proc(KernelEvalContext*, const MemRef& lhs,
                      const MemRef& rhs) const {
  if (lhs.eltype().storage_type() != rhs.eltype().storage_type()) {
    MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                     SizeOf(lhs.eltype().storage_type()) * 8),
                    rhs.shape());
    ring_assign(rhs_cast, rhs);
    return ring_mmul(lhs, rhs_cast).as(lhs.eltype());
  }
  return ring_mmul(lhs, rhs).as(lhs.eltype());
}

MemRef LShiftA::proc(KernelEvalContext*, const MemRef& in,
                     const Sizes& bits) const {
  return ring_lshift(in, bits).as(in.eltype());
}

}  // namespace spu::mpc::cheetah