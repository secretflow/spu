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

#include "libspu/mpc/securenn/boolean.h"

#include <functional>

#include "libspu/core/bit_utils.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/securenn/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::securenn {
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

  SPU_ENFORCE(lhs == rhs,
              "securenn always use same bshare type, lhs={}, rhs={}", lhs, rhs);

  ctx->pushOutput(lhs);
}

MemRef CastTypeB::proc(KernelEvalContext* ctx, const MemRef& in,
                       const Type& to_type) const {
  SPU_ENFORCE(in.eltype() == to_type,
              "securenn always use same bshare type, lhs={}, rhs={}",
              in.eltype(), to_type);
  return in;
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

MemRef AndBB::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE(lhs.shape() == rhs.shape());

  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  const size_t out_nbits = std::min(getNumBits(lhs), getNumBits(rhs));
  const auto backtype = GetStorageType(out_nbits);
  const int64_t numel = lhs.numel();

  MemRef out(makeType<BoolShareTy>(lhs.eltype().semantic_type(),
                                   lhs.eltype().storage_type(), out_nbits),
             lhs.shape());
  const auto ty = out.eltype();
  DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
    using T = ScalarT;
    MemRefView<T> _lhs(lhs);
    MemRefView<T> _rhs(rhs);

    DISPATCH_ALL_STORAGE_TYPES(backtype, [&]() {
      using V = ScalarT;

      int64_t numBytes = numel * SizeOf(backtype);
      int64_t numField = numBytes / SizeOf(field);
      if (numBytes % SizeOf(field)) numField += 1;
      MemRef a(out.eltype(), {numField});
      MemRef b(out.eltype(), {numField});
      MemRef c(out.eltype(), {numField});

      ring_zeros(a);
      ring_zeros(b);
      ring_zeros(c);

      // P2 to be the beaver generator
      if (rank == 2) {
        // P2 generate a0, a1, b0, b1, c0 by PRF
        // and calculate c1
        MemRef a1(out.eltype(), {numField});
        MemRef a0(out.eltype(), {numField});
        prg_state->fillPrssPair(a1.data(), a0.data(),
                                out.elsize() * a0.numel());

        MemRef b1(out.eltype(), {numField});
        MemRef b0(out.eltype(), {numField});
        prg_state->fillPrssPair(b1.data(), b0.data(),
                                out.elsize() * b0.numel());

        MemRef c0(out.eltype(), {numField});
        prg_state->fillPrssPair(nullptr, c0.data(), c0.elsize() * c0.numel());

        // c1 = (a0 ^ a1) & (b0 ^ b1) ^ c0
        auto c1 = ring_xor(ring_and(ring_xor(a0, a1), ring_xor(b0, b1)), c0);

        comm->sendAsync(1, c1, "c");  // 1 latency, k
      }
      if (rank == 0) {
        prg_state->fillPrssPair(a.data(), nullptr, out.elsize() * a.numel());
        prg_state->fillPrssPair(b.data(), nullptr, out.elsize() * b.numel());
        prg_state->fillPrssPair(c.data(), nullptr, out.elsize() * c.numel());
      }
      if (rank == 1) {
        prg_state->fillPrssPair(nullptr, a.data(), out.elsize() * a.numel());
        prg_state->fillPrssPair(nullptr, b.data(), out.elsize() * b.numel());
        c = comm->recv(2, ty, "c");
        c = c.reshape({numField});
      }

      MemRefView<V> _a(a);
      MemRefView<V> _b(b);
      MemRefView<V> _c(c);

      // first half mask x^a, second half mask y^b.
      std::vector<V> mask(numel * 2, 0);
      pforeach(0, numel, [&](int64_t idx) {
        mask[idx] = _lhs[idx] ^ _a[idx];
        mask[numel + idx] = _rhs[idx] ^ _b[idx];
      });

      mask = comm->allReduce<V, std::bit_xor>(mask, "open(x^a,y^b)");

      // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
      MemRefView<T> _z(out);
      pforeach(0, numel, [&](int64_t idx) {
        _z[idx] = _c[idx];
        _z[idx] ^= mask[idx] & _b[idx];
        _z[idx] ^= mask[numel + idx] & _a[idx];
        if (comm->getRank() == 0) {
          _z[idx] ^= mask[idx] & mask[numel + idx];
        }
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

}  // namespace spu::mpc::securenn
