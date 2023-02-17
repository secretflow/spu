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

#include "libspu/core/trace.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/object.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

ArrayRef TruncA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                      size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, x);
  TruncateProtocol prot(ctx->getState<CheetahOTState>()->get());
  TruncateProtocol::Meta meta;
  meta.signed_arith = true;
  return prot.Compute(x, meta, bits);
}

ArrayRef MsbA2B::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_TRACE_MPC_LEAF(ctx, x);

  const int rank = ctx->getState<Communicator>()->getRank();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t n = x.numel();
  const size_t shft = SizeOf(field) * 8 - 1;

  CompareProtocol compare_prot(ctx->getState<CheetahOTState>()->get());

  return DISPATCH_ALL_FIELDS(field, "", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << shft) - 1;
    ArrayRef adjusted = ring_zeros(field, n);
    auto xinp = ArrayView<const u2k>(x);
    auto xadj = ArrayView<u2k>(adjusted);

    if (rank == 0) {
      pforeach(0, n, [&](int64_t i) { xadj[i] = xinp[i] & mask; });
    } else {
      pforeach(0, n, [&](int64_t i) { xadj[i] = (mask - xinp[i]) & mask; });
    }

    // NOTE(juhou): CompareProtocol returns BShr type
    ArrayRef cmp_bit = compare_prot.Compute(adjusted, /*gt*/ true);
    auto xcmp = ArrayView<u2k>(cmp_bit);
    pforeach(0, n, [&](int64_t i) { xcmp[i] ^= (xinp[i] >> shft); });

    return cmp_bit;
  });
}

ArrayRef MulA1B::proc(KernelEvalContext* ctx, const ArrayRef& x,
                      const ArrayRef& y) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);
  SPU_ENFORCE_EQ(x.numel(), y.numel());
  auto ot_prot = ctx->getState<CheetahOTState>()->get();
  return ot_prot->Multiplexer(x, y).as(x.eltype());
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                     const ArrayRef& y) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);
  size_t batch_sze = ctx->getState<CheetahMulState>()->get()->OLEBatchSize();
  size_t numel = x.numel();
  if (numel >= batch_sze) {
    // TODO(juhou): combine mulWithBeaver and mulDirectly to save time
    return mulDirectly(ctx, x, y);
  }
  return mulWithBeaver(ctx, x, y);
}

ArrayRef MulAA::mulWithBeaver(KernelEvalContext* ctx, const ArrayRef& x,
                              const ArrayRef& y) const {
  const int64_t numel = x.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto [a, b, c] =
      ctx->getState<CheetahMulState>()->TakeCachedBeaver(field, numel);
  YACL_ENFORCE_EQ(a.numel(), numel);

  auto* comm = ctx->caller()->getState<Communicator>();
  // Open x-a & y - b
  auto res =
      vectorize({ring_sub(x, a), ring_sub(y, b)}, [&](const ArrayRef& s) {
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

  return z.as(x.eltype());
}

ArrayRef MulAA::mulDirectly(KernelEvalContext* ctx, const ArrayRef& x,
                            const ArrayRef& y) const {
  // (x0 + x1) * (y0+ y1)
  // Compute the cross terms x0*y1, x1*y0 homomorphically
  auto* comm = ctx->getState<Communicator>();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  const int rank = comm->getRank();

  auto* conn = comm->lctx().get();
  auto dupx = conn->Spawn();
  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return mul_prot->MulOLE(x, dupx.get(), true);
    }
    return mul_prot->MulOLE(y, dupx.get(), false);
  });

  ArrayRef x0y1;
  ArrayRef x1y0;
  if (rank == 0) {
    x1y0 = mul_prot->MulOLE(y, conn, false);
  } else {
    x1y0 = mul_prot->MulOLE(x, conn, true);
  }
  x0y1 = task.get();

  return ring_add(x0y1, ring_add(x1y0, ring_mul(x, y))).as(x.eltype());
}

// A is (M, K); B is (K, N)
ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t m, size_t n, size_t k) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const int rank = comm->getRank();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms homomorphically
  const Shape3D dim3 = {static_cast<int64_t>(m), static_cast<int64_t>(k),
                        static_cast<int64_t>(n)};

  auto* conn = comm->lctx().get();
  auto dupx = conn->Spawn();
  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return dot_prot->DotOLE(x, dupx.get(), dim3, true);
    } else {
      return dot_prot->DotOLE(y, dupx.get(), dim3, false);
    }
  });

  ArrayRef x0y1;
  ArrayRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->DotOLE(y, conn, dim3, false);
  } else {
    x1y0 = dot_prot->DotOLE(x, conn, dim3, true);
  }
  x0y1 = task.get();

  return ring_add(x0y1, ring_add(x1y0, ring_mmul(x, y, m, n, k)))
      .as(x.eltype());
}

}  // namespace spu::mpc::cheetah
