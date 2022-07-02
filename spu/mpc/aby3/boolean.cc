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

#include "spu/mpc/aby3/boolean.h"

#include <algorithm>

#include "spu/core/profile.h"
#include "spu/mpc/aby3/type.h"
#include "spu/mpc/aby3/value.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/interfaces.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::aby3 {
namespace {

size_t maxNumBits(const ArrayRef& lhs, const ArrayRef& rhs) {
  size_t res = std::max(lhs.eltype().as<BShare>()->nbits(),
                        rhs.eltype().as<BShare>()->nbits());
  YASL_ENFORCE(res <= SizeOf(lhs.eltype().as<Ring2k>()->field()) * 8);
  return res;
}

}  // namespace

ArrayRef B2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();
  const auto field = in.eltype().as<Ring2k>()->field();

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  const auto& x3 = comm->rotate(x2, kBindName);  // comm => 1, k

  // ret
  auto res = ring_xor(ring_xor(x1, x2), x3);

  return res.as(makeType<Pub2kTy>(field));
}

ArrayRef P2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto numel = in.numel();

  const auto& zeros = ring_zeros(field, numel);

  // ArrayRef& in is public
  if (comm->getRank() == 0) {
    return makeBShare(in, zeros, field);
  } else if (comm->getRank() == 2) {
    return makeBShare(zeros, in, field);
  } else {
    return makeBShare(zeros, zeros, field);
  }
}

ArrayRef AndBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x1 = getFirstShare(lhs);
  const auto& x2 = getSecondShare(lhs);

  // ret
  auto z1 = ring_and(x1, rhs);
  auto z2 = ring_and(x2, rhs);

  return makeBShare(z1, z2, field);
}

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  // lhs
  const auto& x1 = getFirstShare(lhs);
  const auto& x2 = getSecondShare(lhs);

  // rhs
  const auto& y1 = getFirstShare(rhs);
  const auto& y2 = getSecondShare(rhs);

  auto [r0, r1] = prg_state->genPrssPair(field, lhs.numel());
  auto r = ring_xor(r0, r1);

  // ret
  auto z1 = ring_xor(ring_and(x1, y1), ring_and(x1, y2));
  ring_xor_(z1, ring_and(x2, y1));
  ring_xor_(z1, r);

  const auto& z2 = comm->rotate(z1, kBindName);  // comm => 1, k

  return makeBShare(z1, z2, field, maxNumBits(lhs, rhs));
}

ArrayRef XorBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x1 = getFirstShare(lhs);
  const auto& x2 = getSecondShare(lhs);

  // ret
  const auto z1 = ring_xor(x1, rhs);
  const auto z2 = ring_xor(x2, rhs);

  return makeBShare(z1, z2, field);
}

ArrayRef XorBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x1 = getFirstShare(lhs);
  const auto& x2 = getSecondShare(lhs);

  // rhs
  const auto& y1 = getFirstShare(rhs);
  const auto& y2 = getSecondShare(rhs);

  // ret
  const auto z1 = ring_xor(x1, y1);
  const auto z2 = ring_xor(x2, y2);

  return makeBShare(z1, z2, field, maxNumBits(lhs, rhs));
}

ArrayRef LShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  // ret
  const auto z1 = ring_lshift(x1, bits);
  const auto z2 = ring_lshift(x2, bits);

  size_t nbits = in.eltype().as<BShare>()->nbits() + bits;
  nbits = std::clamp(nbits, (size_t)0, SizeOf(field) * 8);

  return makeBShare(z1, z2, field, nbits);
}

ArrayRef RShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  // ret
  const auto z1 = ring_rshift(x1, bits);
  const auto z2 = ring_rshift(x2, bits);

  size_t nbits = in.eltype().as<BShare>()->nbits();
  nbits -= std::min(nbits, bits);
  YASL_ENFORCE(nbits <= SizeOf(field) * 8);

  return makeBShare(z1, z2, field, nbits);
}

ArrayRef ARShiftB::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  // ret
  const auto z1 = ring_arshift(x1, bits);
  const auto z2 = ring_arshift(x2, bits);

  return makeBShare(z1, z2, field);
}

ArrayRef BitrevB::proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                       size_t end) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, start, end);

  const auto field = in.eltype().as<Ring2k>()->field();

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  // ret
  const auto z1 = ring_bitrev(x1, start, end);
  const auto z2 = ring_bitrev(x2, start, end);

  // OPTIMIZE: calc number of valid bits.
  return makeBShare(z1, z2, field);
}

}  // namespace spu::mpc::aby3
