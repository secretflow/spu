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
namespace {

constexpr int64_t kMinWorkSize = 5000;

TruncateProtocol::MSB_st getMsbType(SignType sign) {
  switch (sign) {
    case SignType::Positive:
      return TruncateProtocol::MSB_st::zero;
    case SignType::Negative:
      return TruncateProtocol::MSB_st::one;
    case SignType::Unknown:
      return TruncateProtocol::MSB_st::unknown;
    default:
      SPU_THROW("should not be here");
  }
}

}  // namespace

NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        size_t bits, SignType sign) const {
  SPU_TRACE_MPC_LEAF(ctx, x);
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  int64_t n = x.numel();
  int64_t nworker = std::min(static_cast<int64_t>(ot_state->parallel_size()),
                             CeilDiv(n, kMinWorkSize));
  int64_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);
  for (int64_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  NdArrayRef out(x.eltype(), x.shape());
  TruncateProtocol::Meta meta;
  meta.signed_arith = true;
  meta.msb = getMsbType(sign);
  meta.shift_bits = bits;
  auto f_x = flatten(x);
  yacl::parallel_for(0, nworker, 1, [&](int64_t bgn, int64_t end) {
    for (int64_t job = bgn; job < end; ++job) {
      int64_t slice_bgn = std::min(job * work_load, n);
      int64_t slice_end = std::min(slice_bgn + work_load, n);
      if (slice_end == slice_bgn) {
        break;
      }

      TruncateProtocol prot(ctx->getState<CheetahOTState>()->get(job));
      auto out_slice = prot.Compute(f_x.slice(slice_bgn, slice_end), meta);
      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return out;
}

NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  int64_t n = x.numel();
  int64_t nworker = std::min(static_cast<int64_t>(ot_state->parallel_size()),
                             CeilDiv(n, kMinWorkSize));
  int64_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);
  for (int64_t w = 0; w < nworker; ++w) {
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

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << shft) - 1;
    NdArrayRef adjusted = ring_zeros(field, {n});
    // auto xinp = ArrayView<const u2k>(x);
    // auto xadj = ArrayView<u2k>(adjusted);

    if (rank == 0) {
      // x0
      pforeach(0, n,
               [&](int64_t i) { adjusted.at<u2k>(i) = x.at<u2k>(i) & mask; });
    } else {
      // 2^{k - 1} - 1 - x1
      pforeach(0, n, [&](int64_t i) {
        adjusted.at<u2k>(i) = (mask - x.at<u2k>(i)) & mask;
      });
    }

    NdArrayRef carry_bit(x.eltype(), x.shape());
    auto f_adjusted = flatten(adjusted);
    yacl::parallel_for(0, nworker, 1, [&](int64_t bgn, int64_t end) {
      for (int64_t job = bgn; job < end; ++job) {
        int64_t slice_bgn = std::min(job * work_load, n);
        int64_t slice_end = std::min(slice_bgn + work_load, n);

        if (slice_end == slice_bgn) {
          break;
        }

        CompareProtocol prot(ot_state->get(job));
        // 1{x0 > 2^{k - 1} - 1 - x1}
        auto out_slice = prot.Compute(f_adjusted.slice(slice_bgn, slice_end),
                                      /*greater*/ true);

        std::memcpy(&carry_bit.at(slice_bgn), &out_slice.at(0),
                    out_slice.numel() * out_slice.elsize());
      }
    });

    // [msb(x)]_B <- [1{x0 + x1 > 2^{k- 1} - 1]_B ^ msb(x0)
    pforeach(0, n, [&](int64_t i) {
      carry_bit.at<u2k>(i) ^= (x.at<u2k>(i) >> shft);
    });

    return carry_bit.as(makeType<semi2k::BShrTy>(field, 1));
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
  SPU_TRACE_MPC_LEAF(ctx, x, y);
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  int64_t n = x.numel();
  int64_t nworker = std::min(static_cast<int64_t>(ot_state->parallel_size()),
                             CeilDiv(n, kMinWorkSize));
  int64_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);
  for (int64_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const int rank = comm->getRank();

  //     x0 + x1 = y0 + y1 mod 2k
  // <=> x0 - y0 = y1 - x1 mod 2k
  NdArrayRef adjusted;
  if (rank == 0) {
    adjusted = ring_sub(x, y);
  } else {
    adjusted = ring_sub(y, x);
  }

  NdArrayRef eq_bit(x.eltype(), x.shape());
  auto f_adjusted = flatten(adjusted);
  yacl::parallel_for(0, nworker, 1, [&](int64_t bgn, int64_t end) {
    for (int64_t job = bgn; job < end; ++job) {
      int64_t slice_bgn = std::min(job * work_load, n);
      int64_t slice_end = std::min(slice_bgn + work_load, n);

      if (slice_end == slice_bgn) {
        break;
      }

      EqualProtocol prot(ot_state->get(job));
      auto out_slice = prot.Compute(f_adjusted.slice(slice_bgn, slice_end));

      std::memcpy(&eq_bit.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return eq_bit.as(makeType<semi2k::BShrTy>(field, 1));
}

NdArrayRef MulA1B::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  int64_t n = x.numel();
  int64_t nworker = std::min(static_cast<int64_t>(ot_state->parallel_size()),
                             CeilDiv(n, kMinWorkSize));
  int64_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);

  for (int64_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  NdArrayRef out(x.eltype(), x.shape());
  auto f_x = flatten(x);
  auto f_y = flatten(y);
  yacl::parallel_for(0, nworker, 1, [&](int64_t bgn, int64_t end) {
    for (int64_t job = bgn; job < end; ++job) {
      int64_t slice_bgn = std::min(job * work_load, n);
      int64_t slice_end = std::min(slice_bgn + work_load, n);

      if (slice_end == slice_bgn) {
        break;
      }

      auto out_slice = ot_state->get(job)->Multiplexer(
          f_x.slice(slice_bgn, slice_end), f_y.slice(slice_bgn, slice_end));

      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return out;
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.numel(), y.numel());

  size_t batch_sze = ctx->getState<CheetahMulState>()->get()->OLEBatchSize();
  size_t numel = x.numel();
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

  auto* comm = ctx->getState<Communicator>();
  // Open x - a & y - b
  auto res = vmap({ring_sub(x, unflatten(a, x.shape())),
                   ring_sub(y, unflatten(b, y.shape()))},
                  [&](const NdArrayRef& s) {
                    return comm->allReduce(ReduceOp::ADD, s, kBindName);
                  });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(ring_mul(x_a, unflatten(b, x_a.shape())),
                    ring_mul(y_b, unflatten(a, y_b.shape())));
  ring_add_(z, unflatten(c, z.shape()));

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

  auto* conn = comm->lctx().get();
  auto dupx = conn->Spawn();
  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return mul_prot->MulOLE(flatten(x), dupx.get(), true);
    }
    return mul_prot->MulOLE(flatten(y), dupx.get(), false);
  });

  ArrayRef x0y1;
  ArrayRef x1y0;
  if (rank == 0) {
    x1y0 = mul_prot->MulOLE(flatten(y), conn, false);
  } else {
    x1y0 = mul_prot->MulOLE(flatten(x), conn, true);
  }
  x0y1 = task.get();

  return ring_add(unflatten(x0y1, x.shape()),
                  ring_add(unflatten(x1y0, x.shape()), ring_mul(x, y)))
      .as(x.eltype());
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
  const Shape3D dim3 = {static_cast<int64_t>(x.shape()[0]),
                        static_cast<int64_t>(x.shape()[1]),
                        static_cast<int64_t>(y.shape()[1])};

  auto* conn = comm->lctx().get();
  auto dupx = conn->Spawn();
  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return dot_prot->DotOLE(flatten(x), dupx.get(), dim3, true);
    } else {
      return dot_prot->DotOLE(flatten(y), dupx.get(), dim3, false);
    }
  });

  ArrayRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->DotOLE(flatten(y), conn, dim3, false);
  } else {
    x1y0 = dot_prot->DotOLE(flatten(x), conn, dim3, true);
  }

  auto ret = ring_mmul(x, y);
  ring_add_(ret, unflatten(x1y0, ret.shape()));
  return ring_add(ret, unflatten(task.get(), ret.shape())).as(x.eltype());
}

NdArrayRef Conv2DAA::proc(KernelEvalContext* ctx, const NdArrayRef& tensor,
                          const NdArrayRef& filter, int64_t stride_h,
                          int64_t stride_w) const {
  SPU_TRACE_MPC_LEAF(ctx, tensor, filter);
  if (0 == tensor.numel() || 0 == filter.numel()) {
    return NdArrayRef(tensor.eltype(), {0});
  }

  auto N = tensor.shape()[0];
  auto C = tensor.shape()[3];
  auto H = tensor.shape()[1];
  auto W = tensor.shape()[2];

  auto h = filter.shape()[0];
  auto w = filter.shape()[1];
  auto O = filter.shape()[3];

  int64_t tensor_sze = N * H * W * C;
  int64_t filter_sze = h * w * C * O;
  SPU_ENFORCE_EQ(tensor.numel(), tensor_sze);
  SPU_ENFORCE_EQ(filter.numel(), filter_sze);
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();

  Shape3D tensor_shape{H, W, C};
  Shape3D filter_shape{h, w, C};
  Shape2D window_strides{stride_h, stride_w};

  auto* conv2d_prot = ctx->getState<CheetahDotState>()->get();

  auto* conn = comm->lctx().get();
  auto dupx = conn->Spawn();

  auto f_tensor = flatten(tensor);
  auto f_filter = flatten(filter);
  std::future<ArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return conv2d_prot->Conv2dOLE(f_tensor, dupx.get(), N, tensor_shape, O,
                                    filter_shape, window_strides, true);
    } else {
      return conv2d_prot->Conv2dOLE(f_filter, dupx.get(), N, tensor_shape, O,
                                    filter_shape, window_strides, false);
    }
  });

  ArrayRef x1y0;
  if (rank == 0) {
    x1y0 = conv2d_prot->Conv2dOLE(f_filter, comm->lctx().get(), N, tensor_shape,
                                  O, filter_shape, window_strides, false);

  } else {
    x1y0 = conv2d_prot->Conv2dOLE(f_tensor, comm->lctx().get(), N, tensor_shape,
                                  O, filter_shape, window_strides, true);
  }

  auto ret = ring_conv2d(tensor, filter, N, tensor_shape, O, filter_shape,
                         window_strides);
  ring_add_(ret, unflatten(x1y0, ret.shape()));
  return ring_add(ret, unflatten(task.get(), ret.shape())).as(tensor.eltype());
}

}  // namespace spu::mpc::cheetah
