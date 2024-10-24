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

size_t getNumBits(const NdArrayRef& in) {
  if (in.eltype().isa<Pub2kTy>()) {
    const auto field = in.eltype().as<Pub2kTy>()->field();
    return DISPATCH_ALL_FIELDS(field, "_",
                               [&]() { return maxBitWidth<ring2k_t>(in); });
  } else if (in.eltype().isa<BShrTy>()) {
    return in.eltype().as<BShrTy>()->nbits();
  } else {
    SPU_THROW("should not be here, {}", in.eltype());
  }
}

NdArrayRef makeBShare(const NdArrayRef& r, FieldType field, size_t nbits) {
  const auto ty = makeType<BShrTy>(field, nbits);
  return r.as(ty);
}
}  // namespace

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_ENFORCE(lhs == rhs, "cheetah always use same bshare type, lhs={}, rhs={}",
              lhs, rhs);

  ctx->setOutput(lhs);
}

NdArrayRef CastTypeB::proc(KernelEvalContext*, const NdArrayRef& in,
                           const Type& to_type) const {
  SPU_ENFORCE(in.eltype() == to_type,
              "cheetah always use same bshare type, lhs={}, rhs={}",
              in.eltype(), to_type);
  return in;
}

NdArrayRef B2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::XOR, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

NdArrayRef P2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();

  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] =
      prg_state->genPrssPair(field, in.shape(), PrgState::GenPrssCtrl::Both);
  auto x = ring_xor(r0, r1).as(makeType<BShrTy>(field, 0));

  if (comm->getRank() == 0) {
    ring_xor_(x, in);
  }

  return makeBShare(x, field, getNumBits(in));
}

NdArrayRef AndBP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.shape() == rhs.shape());

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::min(getNumBits(lhs), getNumBits(rhs));
  NdArrayRef out(makeType<BShrTy>(field, out_nbits), lhs.shape());

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _lhs(lhs);
    NdArrayView<ring2k_t> _rhs(rhs);
    NdArrayView<ring2k_t> _out(out);

    pforeach(0, lhs.numel(),
             [&](int64_t idx) { _out[idx] = _lhs[idx] & _rhs[idx]; });
  });
  return out;
}

NdArrayRef XorBP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  auto* comm = ctx->getState<Communicator>();

  const auto field = lhs.eltype().as<Ring2k>()->field();
  const size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));

  if (comm->getRank() == 0) {
    return makeBShare(ring_xor(lhs, rhs), field, out_nbits);
  }

  return makeBShare(lhs, field, out_nbits);
}

NdArrayRef XorBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));
  return makeBShare(ring_xor(lhs, rhs), field, out_nbits);
}

NdArrayRef LShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                         size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  size_t out_nbits = in.eltype().as<BShare>()->nbits() + shift;
  out_nbits = std::clamp(out_nbits, static_cast<size_t>(0), SizeOf(field) * 8);

  return makeBShare(ring_lshift(in, shift), field, out_nbits);
}

NdArrayRef RShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                         size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  size_t nbits = in.eltype().as<BShare>()->nbits();
  size_t out_nbits = nbits - std::min(nbits, shift);
  SPU_ENFORCE(nbits <= SizeOf(field) * 8);

  return makeBShare(ring_rshift(in, shift), field, out_nbits);
}

NdArrayRef ARShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                          size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  // arithmetic right shift expects to work on ring, or the behaviour is
  // undefined.
  return makeBShare(ring_arshift(in, shift), field, SizeOf(field) * 8);
}

NdArrayRef BitrevB::proc(KernelEvalContext*, const NdArrayRef& in, size_t start,
                         size_t end) const {
  const auto field = in.eltype().as<Ring2k>()->field();

  SPU_ENFORCE(start <= end);
  SPU_ENFORCE(end <= SizeOf(field) * 8);
  const size_t out_nbits = std::max(getNumBits(in), end);

  // TODO: more accurate bits.
  return makeBShare(ring_bitrev(in, start, end), field, out_nbits);
}

NdArrayRef BitIntlB::proc(KernelEvalContext*, const NdArrayRef& in,
                          size_t stride) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = getNumBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  auto numel = in.numel();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _in(in);
    NdArrayView<ring2k_t> _out(out);

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = BitIntl<ring2k_t>(_in[idx], stride, nbits);
    });
  });

  return out;
}

NdArrayRef BitDeintlB::proc(KernelEvalContext*, const NdArrayRef& in,
                            size_t stride) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = getNumBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  auto numel = in.numel();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _in(in);
    NdArrayView<ring2k_t> _out(out);

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = BitDeintl<ring2k_t>(_in[idx], stride, nbits);
    });
  });

  return out;
}
}  // namespace spu::mpc::cheetah