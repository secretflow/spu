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

#include "spu/mpc/cheetah/arithmetic.h"

#include "spu/core/profile.h"
#include "spu/core/vectorize.h"
#include "spu/mpc/cheetah/object.h"
#include "spu/mpc/cheetah/utils.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/interfaces.h"
#include "spu/mpc/semi2k/type.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::cheetah {

ArrayRef TruncPrA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, bits);
  auto* primitives = ctx->caller()->getState<CheetahState>()->primitives();
  size_t size = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  ArrayRef y(makeType<RingTy>(field), size);

  if (heuristic) {
    // Use heuristic optimization from SecureQ8: Add a large positive to make
    // sure the value is always positive
    ArrayRef adjusted_x =
        ring_add(x, ring_lshift(ring_ones(field, size), x.elsize() * 8 - 5));

    DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
      using U = typename std::make_unsigned<ring2k_t>::type;
      auto x_ptr = adjusted_x.getOrCreateCompactBuf()->data<U>();
      auto y_ptr = y.getOrCreateCompactBuf()->data<U>();
      primitives->nonlinear()->truncate_msb0(y_ptr, x_ptr, size, bits,
                                             sizeof(U) * 8);
      primitives->nonlinear()->flush();
    });
    ring_sub_(y,
              ring_lshift(ring_ones(field, size), x.elsize() * 8 - 5 - bits));
  } else {
    DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
      using U = typename std::make_unsigned<ring2k_t>::type;
      auto x_ptr = x.getOrCreateCompactBuf()->data<U>();
      auto y_ptr = y.getOrCreateCompactBuf()->data<U>();
      primitives->nonlinear()->truncate(y_ptr, x_ptr, size, bits,
                                        sizeof(U) * 8);
      primitives->nonlinear()->flush();
    });
  }
  return y.as(x.eltype());
}

ArrayRef MsbA::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x);
  auto* primitives = ctx->caller()->getState<CheetahState>()->primitives();

  size_t size = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  ArrayRef y(makeType<RingTy>(field), size);

  DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
    using U = typename std::make_unsigned<ring2k_t>::type;
    auto x_ptr = x.getOrCreateCompactBuf()->data<U>();
    auto y_ptr = y.getOrCreateCompactBuf()->data<U>();
    yasl::Buffer msb_buf(size);
    primitives->nonlinear()->msb(msb_buf.data<uint8_t>(), x_ptr, size,
                                 sizeof(U) * 8);
    primitives->nonlinear()->flush();
    cast(y_ptr, msb_buf.data<uint8_t>(), size);
  });
  // Enforce it to be a boolean sharing
  return y.as(makeType<semi2k::BShrTy>(field, 1));
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<CheetahState>()->beaver();
  auto [a, b, c] = beaver->Mul(field, lhs.numel());

  // Open x-a & y-b
  auto res =
      vectorize({ring_sub(lhs, a), ring_sub(rhs, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::ADD, s, kBindName);
      });

  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(ring_add(ring_mul(x_a, b), ring_mul(y_b, a)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(x_a, y_b));
  }

  return z.as(lhs.eltype());
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<CheetahState>()->beaver();

  // generate beaver multiple triple.
  auto [a, b, c] = beaver->Dot(field, M, N, K);

  // Open x-a & y-b
  auto res =
      vectorize({ring_sub(x, a), ring_sub(y, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::ADD, s, kBindName);
      });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) dot Bi + Ai dot (Y - B) + <(X - A) dot (Y - B)>
  auto z = ring_add(
      ring_add(ring_mmul(x_a, b, M, N, K), ring_mmul(a, y_b, M, N, K)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mmul(x_a, y_b, M, N, K));
  }
  return z.as(x.eltype());
}

}  // namespace spu::mpc::cheetah
