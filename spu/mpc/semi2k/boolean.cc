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

#include "spu/mpc/semi2k/boolean.h"

#include "spu/core/profile.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/interfaces.h"
#include "spu/mpc/kernel.h"
#include "spu/mpc/semi2k/object.h"
#include "spu/mpc/semi2k/type.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::semi2k {
namespace {

size_t maxNumBits(const ArrayRef& lhs, const ArrayRef& rhs) {
  size_t res = std::max(lhs.eltype().as<BShare>()->nbits(),
                        rhs.eltype().as<BShare>()->nbits());
  YASL_ENFORCE(res <= SizeOf(lhs.eltype().as<Ring2k>()->field()) * 8);
  return res;
}

ArrayRef makeBShare(const ArrayRef& r, FieldType field,
                    size_t nbits = std::numeric_limits<size_t>::max()) {
  const auto ty = makeType<BShrTy>(field, nbits);
  return r.as(ty);
}

}  // namespace

ArrayRef ZeroB::proc(KernelEvalContext* ctx, FieldType field,
                     size_t size) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, size);

  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto [r0, r1] = prg_state->genPrssPair(field, size);

  return makeBShare(ring_xor(r0, r1), field);
}

ArrayRef B2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::XOR, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

ArrayRef P2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto x = zero_b(ctx->caller(), field, in.numel());

  if (comm->getRank() == 0) {
    ring_xor_(x, in);
  }

  return makeBShare(x, field);
}

ArrayRef AndBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  return makeBShare(ring_and(lhs, rhs), field);
}

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

  // generate beaver and triple.
  auto [a, b, c] = beaver->And(field, lhs.numel());

  // open x^a, y^b
  auto res =
      vectorize({ring_xor(lhs, a), ring_xor(rhs, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::XOR, s, kBindName);
      });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
  auto z = ring_xor(ring_xor(ring_and(x_a, b), ring_and(y_b, a)), c);
  if (comm->getRank() == 0) {
    ring_xor_(z, ring_and(x_a, y_b));
  }

  return makeBShare(z, field, maxNumBits(lhs, rhs));
}

ArrayRef XorBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  auto* comm = ctx->caller()->getState<Communicator>();

  const auto field = lhs.eltype().as<Ring2k>()->field();

  if (comm->getRank() == 0) {
    return makeBShare(ring_xor(lhs, rhs), field);
  }

  return makeBShare(lhs, field);
}

ArrayRef XorBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  return makeBShare(ring_xor(lhs, rhs), field, maxNumBits(lhs, rhs));
}

ArrayRef LShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  size_t nbits = in.eltype().as<BShare>()->nbits() + bits;
  nbits = std::clamp(nbits, (size_t)0, SizeOf(field) * 8);

  return makeBShare(ring_lshift(in, bits), field, nbits);
}

ArrayRef RShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  size_t nbits = in.eltype().as<BShare>()->nbits();
  nbits -= std::min(nbits, bits);
  YASL_ENFORCE(nbits <= SizeOf(field) * 8);

  return makeBShare(ring_rshift(in, bits), field, nbits);
}

ArrayRef ARShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  return makeBShare(ring_arshift(in, bits), field);
}

ArrayRef BitrevB::proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                       size_t end) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, start, end);
  const auto field = in.eltype().as<Ring2k>()->field();

  YASL_ENFORCE(start <= end);
  YASL_ENFORCE(end <= SizeOf(field) * 8);

  // TODO: more accurate bits.
  return makeBShare(ring_bitrev(in, start, end), field);
}

}  // namespace spu::mpc::semi2k
