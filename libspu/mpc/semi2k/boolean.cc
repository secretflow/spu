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

#include "libspu/mpc/semi2k/boolean.h"

#include <functional>

#include "libspu/core/bit_utils.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {
namespace {

size_t getNumBits(const ArrayRef& in) {
  if (in.eltype().isa<Pub2kTy>()) {
    const auto field = in.eltype().as<Pub2kTy>()->field();
    return DISPATCH_ALL_FIELDS(field, "_", [&]() {
      auto _in = ArrayView<ring2k_t>(in);
      return _in.maxBitWidth();
    });
  } else if (in.eltype().isa<BShrTy>()) {
    return in.eltype().as<BShrTy>()->nbits();
  } else {
    SPU_THROW("should not be here, {}", in.eltype());
  }
}

ArrayRef makeBShare(const ArrayRef& r, FieldType field, size_t nbits) {
  const auto ty = makeType<BShrTy>(field, nbits);
  return r.as(ty);
}

// TODO: DRY
PtType getBacktype(size_t nbits) {
  if (nbits <= 8) {
    return PT_U8;
  }
  if (nbits <= 16) {
    return PT_U16;
  }
  if (nbits <= 32) {
    return PT_U32;
  }
  if (nbits <= 64) {
    return PT_U64;
  }
  if (nbits <= 128) {
    return PT_U128;
  }
  SPU_THROW("invalid number of bits={}", nbits);
}

}  // namespace

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_ENFORCE(lhs == rhs, "semi2k always use same bshare type, lhs={}, rhs={}",
              lhs, rhs);

  ctx->setOutput(lhs);
}

ArrayRef CastTypeB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                         const Type& to_type) const {
  SPU_ENFORCE(in.eltype() == to_type,
              "semi2k always use same bshare type, lhs={}, rhs={}", in.eltype(),
              to_type);
  return in;
}

ArrayRef B2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::XOR, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

ArrayRef P2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();

  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] = prg_state->genPrssPair(field, in.numel());
  auto x = ring_xor(r0, r1).as(makeType<BShrTy>(field, 0));

  if (comm->getRank() == 0) {
    ring_xor_(x, in);
  }

  return makeBShare(x, field, getNumBits(in));
}

ArrayRef AndBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::min(getNumBits(lhs), getNumBits(rhs));
  ArrayRef out(makeType<BShrTy>(field, out_nbits), lhs.numel());

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    auto _lhs = ArrayView<T>(lhs);
    auto _rhs = ArrayView<T>(rhs);
    auto _out = ArrayView<T>(out);
    pforeach(0, lhs.numel(),
             [&](int64_t idx) { _out[idx] = _lhs[idx] & _rhs[idx]; });
  });
  return out;
}

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  const size_t out_nbits = std::min(getNumBits(lhs), getNumBits(rhs));
  const PtType backtype = getBacktype(out_nbits);
  const size_t numel = lhs.numel();

  // semi2k always use the same storage type.
  ArrayRef out(makeType<BShrTy>(field, out_nbits), numel);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;
    DISPATCH_UINT_PT_TYPES(backtype, "_", [&]() {
      using V = ScalarT;

      // TODO: redefine beaver interface, generate variadic beaver and bits.
      size_t numBytes = numel * SizeOf(backtype);
      size_t numField = numBytes / SizeOf(field);
      if (numBytes % SizeOf(field)) numField += 1;

      auto [a, b, c] = beaver->And(field, numField);
      SPU_ENFORCE(a.buf()->size() >= static_cast<int64_t>(numBytes));

      ArrayView<V> _a(&a.at<V>(0), 1, numel);
      ArrayView<V> _b(&b.at<V>(0), 1, numel);
      ArrayView<V> _c(&c.at<V>(0), 1, numel);

      ArrayView<T> _x(lhs);
      ArrayView<T> _y(rhs);
      ArrayView<T> _z(out);

      // first half mask x^a, second half mask y^b.
      std::vector<V> mask(numel * 2, 0);
      pforeach(0, numel, [&](int64_t idx) {
        mask[idx] = _x[idx] ^ _a[idx];
        mask[numel + idx] = _y[idx] ^ _b[idx];
      });

      mask = comm->allReduce<V, std::bit_xor>(mask, "open(x^a,y^b)");

      // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
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

ArrayRef XorBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  auto* comm = ctx->getState<Communicator>();

  const auto field = lhs.eltype().as<Ring2k>()->field();
  const size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));

  if (comm->getRank() == 0) {
    return makeBShare(ring_xor(lhs, rhs), field, out_nbits);
  }

  return makeBShare(lhs, field, out_nbits);
}

ArrayRef XorBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));
  return makeBShare(ring_xor(lhs, rhs), field, out_nbits);
}

ArrayRef LShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  size_t out_nbits = in.eltype().as<BShare>()->nbits() + shift;
  out_nbits = std::clamp(out_nbits, static_cast<size_t>(0), SizeOf(field) * 8);

  return makeBShare(ring_lshift(in, shift), field, out_nbits);
}

ArrayRef RShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  size_t nbits = in.eltype().as<BShare>()->nbits();
  size_t out_nbits = nbits - std::min(nbits, shift);
  SPU_ENFORCE(nbits <= SizeOf(field) * 8);

  return makeBShare(ring_rshift(in, shift), field, out_nbits);
}

ArrayRef ARShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  // arithmetic right shift expects to work on ring, or the behaviour is
  // undefined.
  return makeBShare(ring_arshift(in, shift), field, SizeOf(field) * 8);
}

ArrayRef BitrevB::proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                       size_t end) const {
  const auto field = in.eltype().as<Ring2k>()->field();

  SPU_ENFORCE(start <= end);
  SPU_ENFORCE(end <= SizeOf(field) * 8);
  const size_t out_nbits = std::max(getNumBits(in), end);

  // TODO: more accurate bits.
  return makeBShare(ring_bitrev(in, start, end), field, out_nbits);
}

ArrayRef BitIntlB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t stride) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = getNumBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits));

  ArrayRef out = in.clone();
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;
    pforeach(0, in.numel(), [&](int64_t idx) {
      out.at<T>(idx) = BitIntl<T>(in.at<T>(idx), stride, nbits);
    });
  });

  return out;
}

ArrayRef BitDeintlB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                          size_t stride) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = getNumBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits));

  ArrayRef out = in.clone();
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;
    pforeach(0, in.numel(), [&](int64_t idx) {
      out.at<T>(idx) = BitDeintl<T>(in.at<T>(idx), stride, nbits);
    });
  });

  return out;
}

}  // namespace spu::mpc::semi2k
