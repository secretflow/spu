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
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
constexpr size_t kMinWorkSize = 5000;

ArrayRef TruncAWithSign::proc(KernelEvalContext* ctx, const ArrayRef& x,
                              size_t bits, bool is_positive) const {
  SPU_TRACE_MPC_LEAF(ctx, x);
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  size_t n = x.numel();
  size_t nworker =
      std::min(ot_state->parallel_size(), CeilDiv(n, kMinWorkSize));
  size_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);
  for (size_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  ArrayRef out(x.eltype(), n);
  TruncateProtocol::Meta meta;
  meta.signed_arith = true;
  meta.msb = is_positive ? TruncateProtocol::MSB_st::zero
                         : TruncateProtocol::MSB_st::one;
  meta.shift_bits = bits;
  yacl::parallel_for(0, nworker, 1, [&](size_t bgn, size_t end) {
    for (size_t job = bgn; job < end; ++job) {
      size_t slice_bgn = std::min(job * work_load, n);
      size_t slice_end = std::min(slice_bgn + work_load, n);
      if (slice_end == slice_bgn) {
        break;
      }

      TruncateProtocol prot(ctx->getState<CheetahOTState>()->get(job));
      auto out_slice = prot.Compute(x.slice(slice_bgn, slice_end), meta);
      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return out;
}

ArrayRef TruncA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                      size_t bits) const {
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  size_t n = x.numel();
  size_t nworker =
      std::min(ot_state->parallel_size(), CeilDiv(n, kMinWorkSize));
  size_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);
  for (size_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  ArrayRef out(x.eltype(), n);
  TruncateProtocol::Meta meta;
  meta.signed_arith = true;
  meta.msb = TruncateProtocol::MSB_st::unknown;
  meta.shift_bits = bits;
  yacl::parallel_for(0, nworker, 1, [&](size_t bgn, size_t end) {
    for (size_t job = bgn; job < end; ++job) {
      size_t slice_bgn = std::min(job * work_load, n);
      size_t slice_end = std::min(slice_bgn + work_load, n);
      if (slice_end == slice_bgn) {
        break;
      }

      TruncateProtocol prot(ctx->getState<CheetahOTState>()->get(job));
      auto out_slice = prot.Compute(x.slice(slice_bgn, slice_end), meta);
      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return out;
}

ArrayRef MsbA2B::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  size_t n = x.numel();
  size_t nworker =
      std::min(ot_state->parallel_size(), CeilDiv(n, kMinWorkSize));
  size_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);
  for (size_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  // Math:
  //  msb(x0 + x1 mod 2^k) = msb(x0) ^ msb(x1) ^ 1{(x0 + x1) > 2^{k-1} - 1}
  //  The carry bit
  //     1{(x0 + x1) > 2^{k - 1} - 1} = 1{x0 > 2^{k - 1} - 1 - x1}
  //  is computed using a Millionare protocol.
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const int rank = comm->getRank();
  const size_t shft = SizeOf(field) * 8 - 1;
  return DISPATCH_ALL_FIELDS(field, "", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << shft) - 1;
    ArrayRef adjusted = ring_zeros(field, n);
    auto xinp = ArrayView<const u2k>(x);
    auto xadj = ArrayView<u2k>(adjusted);

    if (rank == 0) {
      // x0
      pforeach(0, n, [&](int64_t i) { xadj[i] = xinp[i] & mask; });
    } else {
      // 2^{k - 1} - 1 - x1
      pforeach(0, n, [&](int64_t i) { xadj[i] = (mask - xinp[i]) & mask; });
    }

    ArrayRef carry_bit(x.eltype(), n);
    yacl::parallel_for(0, nworker, 1, [&](size_t bgn, size_t end) {
      for (size_t job = bgn; job < end; ++job) {
        size_t slice_bgn = std::min(job * work_load, n);
        size_t slice_end = std::min(slice_bgn + work_load, n);

        if (slice_end == slice_bgn) {
          break;
        }

        CompareProtocol prot(ot_state->get(job));
        // 1{x0 > 2^{k - 1} - 1 - x1}
        auto out_slice = prot.Compute(adjusted.slice(slice_bgn, slice_end),
                                      /*greater*/ true);

        std::memcpy(&carry_bit.at(slice_bgn), &out_slice.at(0),
                    out_slice.numel() * out_slice.elsize());
      }
    });

    auto xcarry = ArrayView<u2k>(carry_bit);
    // [msb(x)]_B <- [1{x0 + x1 > 2^{k- 1} - 1]_B ^ msb(x0)
    pforeach(0, n, [&](int64_t i) { xcarry[i] ^= (xinp[i] >> shft); });

    return carry_bit.as(makeType<semi2k::BShrTy>(field, 1));
  });
}

ArrayRef EqualAP::proc(KernelEvalContext* ctx, const ArrayRef& x,
                       const ArrayRef& y) const {
  EqualAA equal_aa;
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  // TODO(juhou): Can we use any place holder to indicate the dummy 0s.
  if (0 == ctx->getState<Communicator>()->getRank()) {
    return equal_aa.proc(ctx, x, ring_zeros(field, x.numel()));
  } else {
    return equal_aa.proc(ctx, x, y);
  }
}

ArrayRef EqualAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                       const ArrayRef& y) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);
  SPU_ENFORCE_EQ(x.numel(), y.numel());

  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  size_t n = x.numel();
  size_t nworker =
      std::min(ot_state->parallel_size(), CeilDiv(n, kMinWorkSize));
  size_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);
  for (size_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const int rank = comm->getRank();

  //     x0 + x1 = y0 + y1 mod 2k
  // <=> x0 - y0 = y1 - x1 mod 2k
  ArrayRef adjusted;
  if (rank == 0) {
    adjusted = ring_sub(x, y);
  } else {
    adjusted = ring_sub(y, x);
  }

  ArrayRef eq_bit(x.eltype(), n);
  yacl::parallel_for(0, nworker, 1, [&](size_t bgn, size_t end) {
    for (size_t job = bgn; job < end; ++job) {
      size_t slice_bgn = std::min(job * work_load, n);
      size_t slice_end = std::min(slice_bgn + work_load, n);

      if (slice_end == slice_bgn) {
        break;
      }

      EqualProtocol prot(ot_state->get(job));
      auto out_slice = prot.Compute(adjusted.slice(slice_bgn, slice_end));

      std::memcpy(&eq_bit.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return eq_bit.as(makeType<semi2k::BShrTy>(field, 1));
}

ArrayRef MulA1B::proc(KernelEvalContext* ctx, const ArrayRef& x,
                      const ArrayRef& y) const {
  SPU_ENFORCE_EQ(x.numel(), y.numel());
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  size_t n = x.numel();
  size_t nworker =
      std::min(ot_state->parallel_size(), CeilDiv(n, kMinWorkSize));
  size_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);

  for (size_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  ArrayRef out(x.eltype(), n);
  yacl::parallel_for(0, nworker, 1, [&](size_t bgn, size_t end) {
    for (size_t job = bgn; job < end; ++job) {
      size_t slice_bgn = std::min(job * work_load, n);
      size_t slice_end = std::min(slice_bgn + work_load, n);

      if (slice_end == slice_bgn) {
        break;
      }

      auto out_slice = ot_state->get(job)->Multiplexer(
          x.slice(slice_bgn, slice_end), y.slice(slice_bgn, slice_end));

      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return out;
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                     const ArrayRef& y) const {
  SPU_ENFORCE_EQ(x.numel(), y.numel());

  size_t batch_sze = ctx->getState<CheetahMulState>()->get()->OLEBatchSize();
  size_t numel = x.numel();
  if (numel >= batch_sze) {
    return mulDirectly(ctx, x, y);
  }
  return mulWithBeaver(ctx, x, y);
}

ArrayRef MulAA::mulWithBeaver(KernelEvalContext* ctx, const ArrayRef& x,
                              const ArrayRef& y) const {
  const int64_t numel = x.numel();
  if (numel == 0) {
    return ArrayRef(x.eltype(), 0);
  }

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto [a, b, c] =
      ctx->getState<CheetahMulState>()->TakeCachedBeaver(field, numel);
  YACL_ENFORCE_EQ(a.numel(), numel);

  auto* comm = ctx->getState<Communicator>();
  // Open x - a & y - b
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
  if (0 == x.numel() || 0 == y.numel()) {
    return ArrayRef(x.eltype(), 0);
  }

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

  ArrayRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->DotOLE(y, conn, dim3, false);
  } else {
    x1y0 = dot_prot->DotOLE(x, conn, dim3, true);
  }

  ArrayRef ret = ring_add(x1y0, ring_mmul(x, y, m, n, k));
  return ring_add(ret, task.get()).as(x.eltype());
}

ArrayRef Conv2DAA::proc(KernelEvalContext* ctx, const ArrayRef& tensor,
                        const ArrayRef& filter, size_t N, size_t H, size_t W,
                        size_t C, size_t O, size_t h, size_t w, size_t stride_h,
                        size_t stride_w) const {
  SPU_TRACE_MPC_LEAF(ctx, tensor, filter);
  if (0 == tensor.numel() || 0 == filter.numel()) {
    return ArrayRef(tensor.eltype(), 0);
  }

  int64_t tensor_sze = N * H * W * C;
  int64_t filter_sze = h * w * C * O;
  SPU_ENFORCE_EQ(tensor.numel(), tensor_sze);
  SPU_ENFORCE_EQ(filter.numel(), filter_sze);
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();

  Shape3D tensor_shape;
  Shape3D filter_shape;
  Shape2D window_strides;
  tensor_shape[0] = static_cast<int64_t>(H);
  tensor_shape[1] = static_cast<int64_t>(W);
  tensor_shape[2] = static_cast<int64_t>(C);
  filter_shape[0] = static_cast<int64_t>(h);
  filter_shape[1] = static_cast<int64_t>(w);
  filter_shape[2] = static_cast<int64_t>(C);
  window_strides[0] = static_cast<int64_t>(stride_h);
  window_strides[1] = static_cast<int64_t>(stride_w);
  auto* conv2d_prot = ctx->getState<CheetahDotState>()->get();

  auto* conn = comm->lctx().get();
  auto dupx = conn->Spawn();
  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return conv2d_prot->Conv2dOLE(tensor, dupx.get(), N, tensor_shape, O,
                                    filter_shape, window_strides, true);
    } else {
      return conv2d_prot->Conv2dOLE(filter, dupx.get(), N, tensor_shape, O,
                                    filter_shape, window_strides, false);
    }
  });

  ArrayRef x1y0;
  if (rank == 0) {
    x1y0 = conv2d_prot->Conv2dOLE(filter, comm->lctx().get(), N, tensor_shape,
                                  O, filter_shape, window_strides, false);

  } else {
    x1y0 = conv2d_prot->Conv2dOLE(tensor, comm->lctx().get(), N, tensor_shape,
                                  O, filter_shape, window_strides, true);
  }

  ArrayRef ret = ring_add(x1y0, ring_conv2d(tensor, filter, N, tensor_shape, O,
                                            filter_shape, window_strides));
  return ring_add(ret, task.get()).as(tensor.eltype());
}

}  // namespace spu::mpc::cheetah
