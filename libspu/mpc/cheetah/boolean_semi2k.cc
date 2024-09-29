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

#include "libspu/mpc/cheetah/boolean.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
namespace {

size_t getNumBits(const MemRef& in) {
  if (in.eltype().isa<Pub2kTy>()) {
    return DISPATCH_ALL_STORAGE_TYPES(
        in.eltype().storage_type(), [&]() { return maxBitWidth<ScalarT>(in); });
  } else if (in.eltype().isa<BoolShareTy>()) {
    return in.eltype().as<BoolShareTy>()->valid_bits();
  } else {
    SPU_THROW("should not be here, {}", in.eltype());
  }
}

MemRef makeBShare(const MemRef& r, SemanticType set, StorageType stt,
                  size_t nbits) {
  const auto ty = makeType<BoolShareTy>(set, stt, nbits);
  return r.as(ty);
}
}  // namespace

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  const auto lhs_st = lhs.as<BoolShareTy>()->storage_type();
  const auto rhs_st = rhs.as<BoolShareTy>()->storage_type();
  const size_t lhs_nbits = lhs.as<BoolShareTy>()->valid_bits();
  const size_t rhs_nbits = rhs.as<BoolShareTy>()->valid_bits();

  SPU_ENFORCE(lhs_st == rhs_st,
              "cheetah always use same bshare field, lhs={}, rhs={}", lhs_st,
              rhs_st);

  ctx->pushOutput(
      makeType<BoolShareTy>(std::max(lhs.as<BoolShareTy>()->semantic_type(),
                                     rhs.as<BoolShareTy>()->semantic_type()),
                            rhs_st, std::max(lhs_nbits, rhs_nbits)));
}

MemRef CastTypeB::proc(KernelEvalContext*, const MemRef& in,
                       const Type& to_type) const {
  return in.as(to_type);
}

MemRef B2P::proc(KernelEvalContext* ctx, const MemRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::XOR, in, kBindName());
  MemRef ret(makeType<Pub2kTy>(in.eltype().semantic_type()), out.shape());
  DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _out(out);
    DISPATCH_ALL_STORAGE_TYPES(ret.eltype().storage_type(), [&]() {
      MemRefView<ScalarT> _ret(ret);
      pforeach(0, ret.numel(),
               [&](int64_t i) { _ret[i] = static_cast<ScalarT>(_out[i]); });
    });
  });

  return ret;
}

MemRef P2B::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* ty = in.eltype().as<BaseRingType>();
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  auto* comm = ctx->getState<Communicator>();

  MemRef r0(makeType<RingTy>(in.eltype().semantic_type(), field), in.shape());
  MemRef r1(makeType<RingTy>(in.eltype().semantic_type(), field), in.shape());

  prg_state->fillPrssPair(r0.data(), r1.data(), r0.elsize() * r0.numel());

  auto x = ring_xor(r0, r1).as(makeType<BoolShareTy>(
      ty->semantic_type(), r0.eltype().storage_type(), 0));

  if (comm->getRank() == 0) {
    if (x.eltype().storage_type() != in.eltype().storage_type()) {
      MemRef in_cast(makeType<RingTy>(in.eltype().semantic_type(), field),
                     in.shape());
      ring_assign(in_cast, in);
      ring_xor_(x, in_cast);
    } else {
      ring_xor_(x, in);
    }
  }

  return makeBShare(x, x.eltype().semantic_type(), x.eltype().storage_type(),
                    getNumBits(in));
}

MemRef AndBP::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE(lhs.shape() == rhs.shape());

  const size_t out_nbits = std::min(getNumBits(lhs), getNumBits(rhs));
  MemRef out(makeType<BoolShareTy>(lhs.eltype().semantic_type(),
                                   lhs.eltype().storage_type(), out_nbits),
             lhs.shape());

  // FIXME: this seems to be not right if lhs/rhs/out use different storage
  // types. Currently semi2k always use the same storage type.
  DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
    using LT = ScalarT;
    DISPATCH_ALL_STORAGE_TYPES(rhs.eltype().storage_type(), [&]() {
      using RT = ScalarT;
      MemRefView<LT> _lhs(lhs);
      MemRefView<RT> _rhs(rhs);
      MemRefView<LT> _out(out);

      pforeach(0, lhs.numel(), [&](int64_t idx) {
        _out[idx] = _lhs[idx] & static_cast<LT>(_rhs[idx]);
      });
    });
  });
  return out;
}

MemRef XorBP::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  auto* comm = ctx->getState<Communicator>();

  const size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));

  if (comm->getRank() == 0) {
    if (lhs.eltype().storage_type() != rhs.eltype().storage_type()) {
      MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                       SizeOf(lhs.eltype().storage_type()) * 8),
                      rhs.shape());
      ring_assign(rhs_cast, rhs);
      return makeBShare(ring_xor(lhs, rhs_cast), lhs.eltype().semantic_type(),
                        lhs.eltype().storage_type(), out_nbits);
    }

    return makeBShare(ring_xor(lhs, rhs), lhs.eltype().semantic_type(),
                      lhs.eltype().storage_type(), out_nbits);
  }

  return makeBShare(lhs, lhs.eltype().semantic_type(),
                    lhs.eltype().storage_type(), out_nbits);
}

MemRef XorBB::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  const size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));
  return makeBShare(ring_xor(lhs, rhs), lhs.eltype().semantic_type(),
                    lhs.eltype().storage_type(), out_nbits);
}

}  // namespace spu::mpc::cheetah
