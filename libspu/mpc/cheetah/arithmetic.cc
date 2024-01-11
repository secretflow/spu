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

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        size_t bits, SignType sign) const {
  size_t n = x.numel();
  NdArrayRef out(x.eltype(), x.shape());
  if (n == 0) {
    return out;
  }

  size_t nworker = InitOTState(ctx, n);
  size_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);

  TruncateProtocol::Meta meta;
  meta.signed_arith = true;
  meta.sign = sign;
  meta.shift_bits = bits;
  meta.use_heuristic = true;

  // Operate on 1D array
  auto flatten_x = x.reshape({x.numel()});
  TiledDispatch(ctx, nworker, [&](int64_t job) {
    int64_t slice_bgn = std::min<int64_t>(job * work_load, n);
    int64_t slice_end = std::min<int64_t>(slice_bgn + work_load, n);
    if (slice_end == slice_bgn) {
      return;
    }

    TruncateProtocol prot(ctx->getState<CheetahOTState>()->get(job));
    auto out_slice =
        prot.Compute(flatten_x.slice({slice_bgn}, {slice_end}, {1}), meta);
    std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                out_slice.numel() * out_slice.elsize());
  });

  return out;
}

// Math:
//  msb(x0 + x1 mod 2^k) = msb(x0) ^ msb(x1) ^ 1{(x0 + x1) > 2^{k-1} - 1}
//  The carry bit
//     1{(x0 + x1) > 2^{k - 1} - 1} = 1{x0 > 2^{k - 1} - 1 - x1}
//  is computed using a Millionare protocol.
NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const int64_t numel = x.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t nbits = nbits_ == 0 ? SizeOf(field) * 8 : nbits_;
  const size_t shft = nbits - 1;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  NdArrayRef out(x.eltype(), x.shape());
  if (numel == 0) {
    return out.as(makeType<BShrTy>(field, 1));
  }

  const int64_t nworker = InitOTState(ctx, numel);
  const int64_t work_load = nworker == 0 ? 0 : CeilDiv(numel, nworker);
  const int rank = ctx->getState<Communicator>()->getRank();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << shft) - 1;
    NdArrayRef adjusted = ring_zeros(field, {numel});
    auto xinp = NdArrayView<const u2k>(x);
    auto xadj = NdArrayView<u2k>(adjusted);

    if (rank == 0) {
      // x0
      pforeach(0, numel, [&](int64_t i) { xadj[i] = xinp[i] & mask; });
    } else {
      // 2^{k - 1} - 1 - x1
      pforeach(0, numel, [&](int64_t i) { xadj[i] = (mask - xinp[i]) & mask; });
    }

    NdArrayRef carry_bit(x.eltype(), x.shape());
    TiledDispatch(ctx, nworker, [&](int64_t job) {
      int64_t slice_bgn = std::min(job * work_load, numel);
      int64_t slice_end = std::min(slice_bgn + work_load, numel);
      if (slice_end == slice_bgn) {
        return;
      }

      CompareProtocol prot(ctx->getState<CheetahOTState>()->get(job));

      // 1{x0 > 2^{k - 1} - 1 - x1}
      auto out_slice =
          prot.Compute(adjusted.slice({slice_bgn}, {slice_end}, {1}),
                       /*greater*/ true);

      std::memcpy(&carry_bit.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    });

    // [msb(x)]_B <- [1{x0 + x1 > 2^{k- 1} - 1]_B ^ msb(x0)
    NdArrayView<u2k> _carry_bit(carry_bit);
    pforeach(0, numel, [&](int64_t i) { _carry_bit[i] ^= (xinp[i] >> shft); });

    return carry_bit.as(makeType<BShrTy>(field, 1));
  });
}

NdArrayRef EqualAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y) const {
  EqualAA equal_aa;
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  // TODO(juhou): Can we use any place holder to indicate the dummy 0s.
  if (0 == ctx->getState<Communicator>()->getRank()) {
    return equal_aa.proc(ctx, x, ring_zeros(field, x.shape()));
  } else {
    return equal_aa.proc(ctx, x, y);
  }
}

NdArrayRef EqualAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  const int64_t numel = x.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t nbits = nbits_ == 0 ? SizeOf(field) * 8 : nbits_;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  NdArrayRef eq_bit(x.eltype(), x.shape());
  if (numel == 0) {
    return eq_bit.as(makeType<BShrTy>(field, 1));
  }

  const int64_t nworker = InitOTState(ctx, numel);
  const int64_t work_load = nworker == 0 ? 0 : CeilDiv(numel, nworker);
  const int rank = ctx->getState<Communicator>()->getRank();

  //     x0 + x1 = y0 + y1 mod 2k
  // <=> x0 - y0 = y1 - x1 mod 2k
  NdArrayRef adjusted;
  if (rank == 0) {
    adjusted = ring_sub(x, y);
  } else {
    adjusted = ring_sub(y, x);
  }

  // Need 1D array
  adjusted = adjusted.reshape({adjusted.numel()});
  TiledDispatch(ctx, nworker, [&](int64_t job) {
    int64_t slice_bgn = std::min(job * work_load, numel);
    int64_t slice_end = std::min(slice_bgn + work_load, numel);
    if (slice_end == slice_bgn) {
      return;
    }

    EqualProtocol prot(ctx->getState<CheetahOTState>()->get(job));
    auto out_slice =
        prot.Compute(adjusted.slice({slice_bgn}, {slice_end}, {1}), nbits);

    std::memcpy(&eq_bit.at(slice_bgn), &out_slice.at(0),
                out_slice.numel() * out_slice.elsize());
  });

  return eq_bit.as(makeType<BShrTy>(field, 1));
}

NdArrayRef MulA1B::proc(KernelEvalContext* ctx, const NdArrayRef& ashr,
                        const NdArrayRef& bshr) const {
  SPU_ENFORCE_EQ(ashr.shape(), bshr.shape());
  const int64_t numel = ashr.numel();
  NdArrayRef out(ashr.eltype(), ashr.shape());

  if (numel == 0) {
    return out;
  }

  const int64_t nworker = InitOTState(ctx, numel);
  const int64_t work_load = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  // Need 1D Array
  auto flatten_a = ashr.reshape({ashr.numel()});
  auto flatten_b = bshr.reshape({bshr.numel()});
  TiledDispatch(ctx, nworker, [&](int64_t job) {
    int64_t slice_bgn = std::min(job * work_load, numel);
    int64_t slice_end = std::min(slice_bgn + work_load, numel);
    if (slice_end == slice_bgn) {
      return;
    }

    auto out_slice = ctx->getState<CheetahOTState>()->get(job)->Multiplexer(
        flatten_a.slice({slice_bgn}, {slice_end}, {1}),
        flatten_b.slice({slice_bgn}, {slice_end}, {1}));

    std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                out_slice.numel() * out_slice.elsize());
  });

  return out;
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  int64_t batch_sze = ctx->getState<CheetahMulState>()->get()->OLEBatchSize();
  int64_t numel = x.numel();

  if (numel >= batch_sze) {
    return mulDirectly(ctx, x, y);
  }
  return mulWithBeaver(ctx, x, y);
}

NdArrayRef MulAA::mulWithBeaver(KernelEvalContext* ctx, const NdArrayRef& x,
                                const NdArrayRef& y) const {
  const int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto [a, b, c] =
      ctx->getState<CheetahMulState>()->TakeCachedBeaver(field, numel);
  YACL_ENFORCE_EQ(a.numel(), numel);

  a = a.reshape(x.shape());
  b = b.reshape(x.shape());
  c = c.reshape(x.shape());

  auto* comm = ctx->getState<Communicator>();
  // Open x - a & y - b
  auto res = vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const NdArrayRef& s) {
    return comm->allReduce(ReduceOp::ADD, s, kBindName);
  });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(ring_mul(x_a, b), ring_mul(y_b, a));
  ring_add_(z, c);

  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(x_a, y_b));
  }

  return z.as(x.eltype());
}

NdArrayRef MulAA::mulDirectly(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) const {
  // (x0 + x1) * (y0+ y1)
  // Compute the cross terms x0*y1, x1*y0 homomorphically
  auto* comm = ctx->getState<Communicator>();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  const int rank = comm->getRank();
  auto fx = x.reshape({x.numel()});
  auto fy = y.reshape({y.numel()});

  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return mul_prot->MulOLE(fx, dupx.get(), true);
    }
    return mul_prot->MulOLE(fy, dupx.get(), false);
  });

  NdArrayRef x1y0;
  if (rank == 0) {
    x1y0 = mul_prot->MulOLE(fy, false);
  } else {
    x1y0 = mul_prot->MulOLE(fx, true);
  }

  x1y0 = x1y0.reshape(x.shape());
  NdArrayRef x0y1 = task.get().reshape(x.shape());
  return ring_add(x0y1, ring_add(x1y0, ring_mul(x, y))).as(x.eltype());
}

// A is (M, K); B is (K, N)
NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const int rank = comm->getRank();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms homomorphically
  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};

  auto* conn = comm->lctx().get();
  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    // Compute x0*y1
    if (rank == 0) {
      return dot_prot->DotOLE(x, dupx.get(), dim3, true);
    } else {
      return dot_prot->DotOLE(y, dupx.get(), dim3, false);
    }
  });

  NdArrayRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->DotOLE(y, conn, dim3, false);
  } else {
    x1y0 = dot_prot->DotOLE(x, conn, dim3, true);
  }

  auto ret = ring_mmul(x, y);
  ring_add_(ret, x1y0);
  return ring_add(ret, task.get()).as(x.eltype());
}

// LHS is a share type (A); RHS is a private type (V)
NdArrayRef MatMulAV::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const auto* priv_type = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(priv_type != nullptr, "RHS should be a private type");
  const int rank = comm->getRank();
  const int owner = priv_type->owner();

  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};
  NdArrayRef out;
  if (rank == owner) {
    out = dot_prot->DotOLE(y, dim3, false);
    auto tmp = ring_mmul(x, y);
    ring_add_(out, tmp);
  } else {
    out = dot_prot->DotOLE(x, dim3, true);
  }
  return out.as(x.eltype());
}

void BatchMatMulAA::evaluate(KernelEvalContext* ctx) const {
  // NOTE(lwj): overwrite the shape check in the MatmulKernel
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);
  const auto& lhs_shape = lhs.shape();
  const auto& rhs_shape = rhs.shape();
  SPU_ENFORCE(lhs_shape.ndim() == rhs_shape.ndim(),
              "ndim mismatch: lhs={}, rhs={}", lhs_shape, rhs_shape);
  SPU_ENFORCE(lhs_shape[0] == rhs_shape[0], "batch mismatch: lhs={}, rhs={}",
              lhs_shape, rhs_shape);
  SPU_ENFORCE(lhs_shape[2] == rhs_shape[1], "shape mismatch: lhs={}, rhs={}",
              lhs_shape, rhs_shape);
  ctx->setOutput(WrapValue(proc(ctx, lhs.data(), rhs.data())));
}

NdArrayRef BatchMatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                               const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const int rank = comm->getRank();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms homomorphically
  const Shape4D dim4 = {x.shape()[0], x.shape()[1], x.shape()[2], y.shape()[2]};

  auto* conn = comm->lctx().get();
  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    // Compute x0*y1
    if (rank == 0) {
      return dot_prot->BatchDotOLE(x, dupx.get(), dim4, true);
    } else {
      return dot_prot->BatchDotOLE(y, dupx.get(), dim4, false);
    }
  });

  NdArrayRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->BatchDotOLE(y, conn, dim4, false);
  } else {
    x1y0 = dot_prot->BatchDotOLE(x, conn, dim4, true);
  }

  // local batch mmul
  const Strides strides(x.shape().size(), 1);
  Index lhs_slice_end(x.shape().begin(), x.shape().end());
  Index rhs_slice_end(y.shape().begin(), y.shape().end());
  Index lhs_slice_begin(3, 0);
  Index rhs_slice_begin(3, 0);
  NdArrayRef out(x.eltype(), {dim4[0], dim4[1], dim4[3]});
  for (int64_t batch = 0; batch < dim4[0]; ++batch) {
    lhs_slice_begin[0] = batch;
    lhs_slice_end[0] = batch + 1;
    rhs_slice_begin[0] = batch;
    rhs_slice_end[0] = batch + 1;
    auto lhs_slice = x.slice(lhs_slice_begin, lhs_slice_end, strides)
                         .reshape({dim4[1], dim4[2]});
    auto rhs_slice = y.slice(rhs_slice_begin, rhs_slice_end, strides)
                         .reshape({dim4[2], dim4[3]});

    auto out_slice =
        out.slice({batch, 0, 0}, {batch + 1, dim4[1], dim4[3]}, strides);
    out_slice = out_slice.reshape({dim4[1], dim4[3]});
    ring_mmul_(out_slice, lhs_slice, rhs_slice);
  }

  ring_add_(out, x1y0);
  ring_add_(out, task.get());
  return out;
}

void BatchMatMulAV::evaluate(KernelEvalContext* ctx) const {
  // NOTE(lwj): overwrite the shape check in the MatmulKernel
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);
  const auto& lhs_shape = lhs.shape();
  const auto& rhs_shape = rhs.shape();
  SPU_ENFORCE(lhs_shape.ndim() == rhs_shape.ndim(),
              "ndim mismatch: lhs={}, rhs={}", lhs_shape, rhs_shape);
  SPU_ENFORCE(lhs_shape[0] == rhs_shape[0], "batch mismatch: lhs={}, rhs={}",
              lhs_shape, rhs_shape);
  SPU_ENFORCE(lhs_shape[2] == rhs_shape[1], "shape mismatch: lhs={}, rhs={}",
              lhs_shape, rhs_shape);
  ctx->setOutput(WrapValue(proc(ctx, lhs.data(), rhs.data())));
}

NdArrayRef BatchMatMulAV::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                               const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const auto* priv_type = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(priv_type != nullptr, "RHS should be a private type");
  const int rank = comm->getRank();
  const int owner = priv_type->owner();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms homomorphically
  const Shape4D dim4 = {x.shape()[0], x.shape()[1], x.shape()[2], y.shape()[2]};

  NdArrayRef out;
  if (rank != owner) {
    out = dot_prot->BatchDotOLE(x, comm->lctx().get(), dim4, true);
  } else {
    out = dot_prot->BatchDotOLE(y, comm->lctx().get(), dim4, false);
    // local batch mmul
    const Strides strides(x.shape().size(), 1);
    Index lhs_slice_end(x.shape().begin(), x.shape().end());
    Index rhs_slice_end(y.shape().begin(), y.shape().end());
    Index lhs_slice_begin(3, 0);
    Index rhs_slice_begin(3, 0);
    NdArrayRef out(x.eltype(), {dim4[0], dim4[1], dim4[3]});
    for (int64_t batch = 0; batch < dim4[0]; ++batch) {
      lhs_slice_begin[0] = batch;
      lhs_slice_end[0] = batch + 1;
      rhs_slice_begin[0] = batch;
      rhs_slice_end[0] = batch + 1;
      auto lhs_slice = x.slice(lhs_slice_begin, lhs_slice_end, strides)
                           .reshape({dim4[1], dim4[2]});
      auto rhs_slice = y.slice(rhs_slice_begin, rhs_slice_end, strides)
                           .reshape({dim4[2], dim4[3]});
      auto local = ring_mmul(lhs_slice, rhs_slice);

      auto out_slice =
          out.slice({batch, 0, 0}, {batch + 1, dim4[1], dim4[3]}, strides);
      out_slice = out_slice.reshape({dim4[1], dim4[3]});
      ring_add_(out_slice, local);
    }
  }
  return out;
}

}  // namespace spu::mpc::cheetah
