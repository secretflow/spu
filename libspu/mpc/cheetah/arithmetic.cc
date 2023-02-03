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

#include <future>

#include "xtensor/xview.hpp"

#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/object.h"
#include "libspu/mpc/cheetah/utils.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/util/ring_ops.h"

namespace spu::mpc::cheetah {

ArrayRef TruncA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                      size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, x, bits);
  auto primitives = ctx->getState<CheetahState>()->beaver()->OTPrimitives();
  size_t size = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  ArrayRef y(makeType<RingTy>(field), size);

  if (heuristic_) {
    // Use heuristic optimization from SecureQ8: Add a large positive to make
    // sure the value is always positive
    ArrayRef adjusted_x =
        ring_add(x, ring_lshift(ring_ones(field, size), x.elsize() * 8 - 5));

    DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
      using U = ring2k_t;
      auto x_buf = adjusted_x.getOrCreateCompactBuf();
      auto y_buf = y.getOrCreateCompactBuf();
      primitives->nonlinear()->truncate_msb0(y_buf->data<U>(), x_buf->data<U>(),
                                             size, bits, sizeof(U) * 8);
      primitives->nonlinear()->flush();
    });
    ring_sub_(y,
              ring_lshift(ring_ones(field, size), x.elsize() * 8 - 5 - bits));
  } else {
    DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
      using U = ring2k_t;
      auto x_buf = x.getOrCreateCompactBuf();
      auto y_buf = y.getOrCreateCompactBuf();
      primitives->nonlinear()->truncate(y_buf->data<U>(), x_buf->data<U>(),
                                        size, bits, sizeof(U) * 8);
      primitives->nonlinear()->flush();
    });
  }
  return y.as(x.eltype());
}

ArrayRef MsbA2B::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_TRACE_MPC_LEAF(ctx, x);
  auto primitives = ctx->getState<CheetahState>()->beaver()->OTPrimitives();

  size_t size = x.numel();
  const auto field = x.eltype().as<Ring2k>()->field();
  ArrayRef y(makeType<RingTy>(field), size);

  DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
    using U = ring2k_t;
    auto x_buf = x.getOrCreateCompactBuf();
    auto y_buf = y.getOrCreateCompactBuf();
    yacl::Buffer msb_buf(size);
    primitives->nonlinear()->msb(msb_buf.data<uint8_t>(), x_buf->data<U>(),
                                 size, sizeof(U) * 8);
    primitives->nonlinear()->flush();
    cast(y_buf->data<U>(), msb_buf.data<uint8_t>(), size);
  });
  // Enforce it to be a boolean sharing
  return y.as(makeType<semi2k::BShrTy>(field, 1));
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
  // (lhs0 + lhs1) * (rhs0 + rhs1)
  // lhs0*rhs0 + lhs0*rhs1 + lhs1*rhs0 + lhs1*rhs1
  // Compute the cross terms lhs0*rhs1, lhs1*rhs0 homomorphically
  auto comm = ctx->getState<Communicator>();
  auto beaver = ctx->getState<CheetahState>()->beaver();
  int rank = comm->getRank();

  auto dupx = comm->lctx()->Spawn();
  // NOTE(juhou): we suppose rank0 and rank1 have the same level of computation
  // power. So we parallel the two computation by switching the role of
  // evaluator.
  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return beaver->MulOLE(lhs, dupx.get(), /*evaluator*/ true);
    } else {
      return beaver->MulOLE(rhs, dupx.get(), false);
    }
  });

  ArrayRef cross0;
  yacl::link::Context* conn = comm->lctx().get();
  if (rank == 0) {
    cross0 = beaver->MulOLE(rhs, conn, false);
  } else {
    cross0 = beaver->MulOLE(lhs, conn, true);
  }
  ArrayRef cross1 = task.get();

  return ring_add(cross0, ring_add(cross1, ring_mul(lhs, rhs)))
      .as(lhs.eltype());
}

static ArrayRef TransposeMat(const ArrayRef& mat, size_t nrows, size_t ncols) {
  auto cpy = mat.clone();
  YACL_ENFORCE_EQ((size_t)mat.numel(), nrows * ncols);
  const auto field = mat.eltype().as<Ring2k>()->field();
  auto ans = ring_zeros(field, nrows * ncols);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto src_mat = xt_adapt<ring2k_t>(mat);
    src_mat.reshape({nrows, ncols});

    auto xmatT = xt::eval(xt::transpose(src_mat));
    auto dst_mat = xt_mutable_adapt<ring2k_t>(ans);
    std::copy_n(xmatT.begin(), xmatT.size(), dst_mat.data());
  });
  return ans;
}

// A is (M, K); B is (K, N)
ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);

  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<CheetahState>()->beaver();
  int rank = comm->getRank();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms homomorphically
  auto dupx = comm->lctx()->Spawn();
  size_t lhs_dim = std::max(M, N);
  size_t rhs_dim = std::min(M, N);

  ArrayRef lhs;
  ArrayRef rhs;
  // DotOLE needs RHS is given in the column-major order
  if (lhs_dim == M) {
    // Case: LHS = x, RHS = y
    lhs = x;
    rhs = TransposeMat(y, K, N);
  } else {
    // Case: LHS = y, RHS = x
    // But, in this case, we compute y^t * x^t
    // So we tranpose the LHS only
    lhs = TransposeMat(y, K, N);
    rhs = x;
  }

  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return beaver->DotOLE(lhs, dupx.get(), lhs_dim, rhs_dim, K, false);
    } else {
      return beaver->DotOLE(rhs, dupx.get(), lhs_dim, rhs_dim, K, true);
    }
  });

  ArrayRef x0y1;
  yacl::link::Context* conn = comm->lctx().get();
  if (rank == 0) {
    x0y1 = beaver->DotOLE(rhs, conn, lhs_dim, rhs_dim, K, true);
  } else {
    x0y1 = beaver->DotOLE(lhs, conn, lhs_dim, rhs_dim, K, false);
  }
  ArrayRef x1y0 = task.get();

  if (lhs_dim == M) {
    // x*y is given in the transposed form (see DotOLE)
    x0y1 = TransposeMat(x0y1, N, M);
    x1y0 = TransposeMat(x1y0, N, M);
  } else {
    // Nothing to do in this case
    // We compute y^t * x^t and resulting at the transposed form.
    // That is x*y already.
  }
  return ring_add(x0y1, ring_add(x1y0, ring_mmul(x, y, M, N, K)))
      .as(x.eltype());
}

}  // namespace spu::mpc::cheetah
