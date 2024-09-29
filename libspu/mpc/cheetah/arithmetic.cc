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
#include <memory>

#include "libspu/core/memref.h"
#include "libspu/core/trace.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

MemRef TruncA::proc(KernelEvalContext* ctx, const MemRef& x, size_t bits,
                    SignType sign) const {
  size_t n = x.numel();
  MemRef out(x.eltype(), x.shape());
  if (n == 0) {
    return out;
  }

  return TiledDispatchOTFunc(
      ctx, x,
      [&](const MemRef& input,
          const std::shared_ptr<BasicOTProtocols>& base_ot) {
        TruncateProtocol::Meta meta;
        meta.signed_arith = true;
        meta.sign = sign;
        meta.shift_bits = bits;
        meta.use_heuristic = true;
        TruncateProtocol prot(base_ot);
        return prot.Compute(input, meta);
      });
}

// Math:
//  msb(x0 + x1 mod 2^k) = msb(x0) ^ msb(x1) ^ 1{(x0 + x1) > 2^{k-1} - 1}
//  The carry bit
//     1{(x0 + x1) > 2^{k - 1} - 1} = 1{x0 > 2^{k - 1} - 1 - x1}
//  is computed using a Millionaire protocol.
MemRef MsbA2B::proc(KernelEvalContext* ctx, const MemRef& x) const {
  const int64_t numel = x.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t nbits = nbits_ == 0 ? SizeOf(field) * 8 : nbits_;
  const size_t shft = nbits - 1;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  MemRef out(x.eltype(), x.shape());
  if (numel == 0) {
    return out.as(makeType<BoolShareTy>(SE_1, GetStorageType(field), 1));
  }

  const int rank = ctx->getState<Communicator>()->getRank();

  return DISPATCH_ALL_STORAGE_TYPES(GetStorageType(field), [&]() {
    using u2k = std::make_unsigned<ScalarT>::type;
    const u2k mask = (static_cast<u2k>(1) << shft) - 1;
    MemRef adjusted(makeType<RingTy>(SE_INVALID, field), x.shape());
    ring_zeros(adjusted);
    auto xinp = MemRefView<const u2k>(x);
    auto xadj = MemRefView<u2k>(adjusted);

    if (rank == 0) {
      // x0
      pforeach(0, numel, [&](int64_t i) { xadj[i] = xinp[i] & mask; });
    } else {
      // 2^{k - 1} - 1 - x1
      pforeach(0, numel, [&](int64_t i) { xadj[i] = (mask - xinp[i]) & mask; });
    }

    auto carry_bit = TiledDispatchOTFunc(
                         ctx, adjusted,
                         [&](const MemRef& input,
                             const std::shared_ptr<BasicOTProtocols>& base_ot) {
                           CompareProtocol prot(base_ot);
                           return prot.Compute(input, /*greater*/ true);
                         })
                         .as(x.eltype());
    // [msb(x)]_B <- [1{x0 + x1 > 2^{k- 1} - 1]_B ^ msb(x0)
    MemRefView<u2k> _carry_bit(carry_bit);
    pforeach(0, numel, [&](int64_t i) { _carry_bit[i] ^= (xinp[i] >> shft); });

    return carry_bit.as(makeType<BoolShareTy>(SE_1, GetStorageType(field), 1));
  });
}

MemRef EqualAP::proc(KernelEvalContext* ctx, const MemRef& x,
                     const MemRef& y) const {
  EqualAA equal_aa;
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  // TODO(juhou): Can we use any place holder to indicate the dummy 0s.
  if (0 == ctx->getState<Communicator>()->getRank()) {
    MemRef zeros(makeType<RingTy>(SE_INVALID, field), x.shape());
    ring_zeros(zeros);
    return equal_aa.proc(ctx, x, zeros);
  } else {
    if (x.eltype().storage_type() != y.eltype().storage_type()) {
      MemRef y_cast(makeType<RingTy>(x.eltype().semantic_type(),
                                     SizeOf(x.eltype().storage_type()) * 8),
                    y.shape());
      ring_assign(y_cast, y);
      return equal_aa.proc(ctx, x, y_cast);
    } else {
      return equal_aa.proc(ctx, x, y);
    }
  }
}

MemRef EqualAA::proc(KernelEvalContext* ctx, const MemRef& x,
                     const MemRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  const int64_t numel = x.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t nbits = nbits_ == 0 ? SizeOf(field) * 8 : nbits_;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  MemRef eq_bit(makeType<BoolShareTy>(SE_1, GetStorageType(field), 1),
                x.shape());
  if (numel == 0) {
    return eq_bit;
  }

  const int rank = ctx->getState<Communicator>()->getRank();

  //     x0 + x1 = y0 + y1 mod 2k
  // <=> x0 - y0 = y1 - x1 mod 2k
  MemRef adjusted;
  if (rank == 0) {
    adjusted = ring_sub(x, y);
  } else {
    adjusted = ring_sub(y, x);
  }
  return TiledDispatchOTFunc(
             ctx, adjusted,
             [&](const MemRef& input,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               EqualProtocol prot(base_ot);
               return prot.Compute(input, nbits);
             })
      .as(makeType<BoolShareTy>(SE_1, GetStorageType(field), 1));
}

MemRef MulA1B::proc(KernelEvalContext* ctx, const MemRef& ashr,
                    const MemRef& bshr) const {
  SPU_ENFORCE_EQ(ashr.shape(), bshr.shape());
  const int64_t numel = ashr.numel();

  if (numel == 0) {
    return MemRef(ashr.eltype(), ashr.shape());
  }

  return TiledDispatchOTFunc(
             ctx, ashr, bshr,
             [&](const MemRef& input0, const MemRef& input1,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               return base_ot->Multiplexer(input0, input1);
             })
      .as(ashr.eltype());
}

MemRef MulA1BV::proc(KernelEvalContext* ctx, const MemRef& ashr,
                     const MemRef& bshr) const {
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();
  SPU_ENFORCE_EQ(ashr.shape(), bshr.shape());
  const int64_t numel = ashr.numel();
  const auto* ptype = bshr.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");

  const int owner = ptype->owner();

  MemRef out(ashr.eltype(), ashr.shape());
  if (numel == 0) {
    return out;
  }

  if (rank != owner) {
    return TiledDispatchOTFunc(
               ctx, ashr,
               [&](const MemRef& input,
                   const std::shared_ptr<BasicOTProtocols>& base_ot) {
                 return base_ot->PrivateMulxSend(input);
               })
        .as(ashr.eltype());
  }

  auto b_ = bshr;
  if (bshr.eltype().storage_type() != ashr.eltype().storage_type()) {
    MemRef bshr_casted(makeType<Priv2kTy>(bshr.eltype().semantic_type(),
                                          ashr.eltype().storage_type(), owner),
                       bshr.shape());
    ring_assign(bshr_casted, bshr);
    b_ = bshr_casted;
  }

  return TiledDispatchOTFunc(
             ctx, ashr, b_,
             [&](const MemRef& input0, const MemRef& input1,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               return base_ot->PrivateMulxRecv(input0, input1);
             })
      .as(ashr.eltype());
}

MemRef MulAV::proc(KernelEvalContext* ctx, const MemRef& x,
                   const MemRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());
  const int64_t numel = x.numel();
  if (numel == 0) {
    return MemRef(x.eltype(), x.shape());
  }
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();
  const auto* ptype = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");
  const int owner = ptype->owner();

  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  mul_prot->LazyInitKeys(field);

  // (x0 * x1) * y
  // <x0 * y> + x1 * y
  auto fx = x.reshape({numel});
  MemRef out;

  // compute <x0 * y>
  if (rank != owner) {
    out = mul_prot->MulOLE(fx, /*eval*/ true);
  } else {
    MemRef fy;
    if (y.eltype().storage_type() != x.eltype().storage_type()) {
      MemRef y_casted(makeType<RingTy>(y.eltype().semantic_type(),
                                       SizeOf(x.eltype().storage_type()) * 8),
                      y.shape());
      ring_assign(y_casted, y);
      fy = y_casted.reshape({numel});
    } else {
      fy = y.reshape({numel});
    }
    out = mul_prot->MulOLE(fy, /*eval*/ false);
    ring_add_(out, ring_mul(fx, fy));
  }

  return out.reshape(x.shape()).as(x.eltype());
}

MemRef MulAA::proc(KernelEvalContext* ctx, const MemRef& x,
                   const MemRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  int64_t batch_size = ctx->getState<CheetahMulState>()->get()->OLEBatchSize();
  int64_t numel = x.numel();

  if (numel >= 2 * batch_size) {
    return mulDirectly(ctx, x, y);
  }
  return mulWithBeaver(ctx, x, y);
}

MemRef SquareA::proc(KernelEvalContext* ctx, const MemRef& x) const {
  const int64_t numel = x.numel();
  if (numel == 0) {
    return MemRef(x.eltype(), x.shape());
  }

  //   (x0 + x1) * (x0 + x1)
  // = x0^2 + 2*<x0*x1> + x1^2
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  mul_prot->LazyInitKeys(field);

  auto fx = x.reshape({numel});
  int64_t nhalf = numel <= 8192 ? numel : numel / 2;

  auto subtask = std::async([&]() -> spu::MemRef {
    return mul_prot->MulOLE(fx.slice({0}, {nhalf}, {1}), rank == 0);
  });

  MemRef mul1;
  if (nhalf < numel) {
    auto dupx = ctx->getState<CheetahMulState>()->duplx();
    mul1 = mul_prot->MulOLE(fx.slice({nhalf}, {numel - nhalf}, {1}), dupx.get(),
                            rank == 1);
  }
  auto mul0 = subtask.get();

  MemRef x0x1(x.eltype(), {numel});
  std::memcpy(&x0x1.at(0), &mul0.at(0), mul0.elsize() * nhalf);
  if (nhalf < numel) {
    std::memcpy(&x0x1.at(nhalf), &mul1.at(0), mul1.elsize() * mul1.numel());
  }
  ring_add_(x0x1, x0x1);
  x0x1 = x0x1.reshape(x.shape());

  return ring_add(x0x1, ring_mul(x, x)).as(x.eltype());
}

MemRef MulAA::mulWithBeaver(KernelEvalContext* ctx, const MemRef& x,
                            const MemRef& y) const {
  const int64_t numel = x.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  const auto ty = makeType<ArithShareTy>(
      std::max(x.eltype().semantic_type(), y.eltype().semantic_type()), field);
  if (numel == 0) {
    return MemRef(ty, x.shape());
  }

  auto [a, b, c] =
      ctx->getState<CheetahMulState>()->TakeCachedBeaver(field, numel);
  YACL_ENFORCE_EQ(a.numel(), numel);

  a = a.reshape(x.shape());
  b = b.reshape(x.shape());
  c = c.reshape(x.shape());

  auto* comm = ctx->getState<Communicator>();
  // Open x - a & y - b
  auto res = vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const MemRef& s) {
    return comm->allReduce(ReduceOp::ADD, s, kBindName());
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

  return z.as(ty);
}

#if 1
MemRef MulAA::mulDirectly(KernelEvalContext* ctx, const MemRef& x,
                          const MemRef& y) const {
  // Compute (x0 + x1) * (y0+ y1)
  auto* comm = ctx->getState<Communicator>();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  mul_prot->LazyInitKeys(ctx->getState<Z2kState>()->getDefaultField());

  auto fx = x.reshape({x.numel()});
  auto fy = y.reshape({y.numel()});
  const int64_t n = fx.numel();
  const int64_t nhalf = n / 2;
  const int rank = comm->getRank();

  // For long vectors, split into two subtasks.
  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<MemRef> task = std::async(std::launch::async, [&] {
    return mul_prot->MulShare(fx.slice({nhalf}, {n}, {1}),
                              fy.slice({nhalf}, {n}, {1}), dupx.get(),
                              /*evaluator*/ rank == 0);
  });

  std::vector<MemRef> out_slices(2);
  out_slices[0] =
      mul_prot->MulShare(fx.slice({0}, {nhalf}, {1}),
                         fy.slice({0}, {nhalf}, {1}), /*evaluato*/ rank != 0);
  out_slices[1] = task.get();

  MemRef out(x.eltype(), x.shape());
  int64_t offset = 0;
  for (auto& out_slice : out_slices) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }
  return out;
}
#else
// Old code for MulAA using two OLEs which commnuicate about 30% more than the
// above version.
NdArrayRef MulAA::mulDirectly(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) const {
  // (x0 + x1) * (y0+ y1)
  // Compute the cross terms x0*y1, x1*y0 homomorphically
  auto* comm = ctx->getState<Communicator>();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  const auto ty = makeType<ArithShareTy>(
      std::max(x.eltype().semantic_type(), y.eltype().semantic_type()), field);
  mul_prot->LazyInitKeys(field);

  const int rank = comm->getRank();

  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<MemRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return mul_prot->MulOLE(x, dupx.get(), true);
    }
    return mul_prot->MulOLE(y, dupx.get(), false);
  });

  MemRef x1y0;
  if (rank == 0) {
    x1y0 = mul_prot->MulOLE(y, false);
  } else {
    x1y0 = mul_prot->MulOLE(x, true);
  }

  MemRef x0y1 = task.get();
  return ring_add(x0y1, ring_add(x1y0, ring_mul(x, y))).as(ty);
}
#endif

MemRef MatMulVVS::proc(KernelEvalContext* ctx, const MemRef& x,
                       const MemRef& y) const {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto out_type = makeType<cheetah::ArithShareTy>(
      std::max(x.eltype().semantic_type(), y.eltype().semantic_type()), field);
  if (0 == x.numel() || 0 == y.numel()) {
    return MemRef(out_type, {x.shape()[0], y.shape()[1]});
  }
  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();

  const int self_rank = comm->getRank();
  auto lhs_owner = x.eltype().as<Priv2kTy>()->owner();

  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};
  if (self_rank == lhs_owner) {
    if (out_type.storage_type() != x.eltype().storage_type()) {
      MemRef x_casted(makeType<RingTy>(x.eltype().semantic_type(),
                                       SizeOf(out_type.storage_type()) * 8),
                      x.shape());
      ring_assign(x_casted, x);
      return dot_prot->DotOLE(x_casted, dim3, /*is_lhs*/ true).as(out_type);
    }
    return dot_prot->DotOLE(x, dim3, /*is_lhs*/ true).as(out_type);
  } else {
    if (out_type.storage_type() != y.eltype().storage_type()) {
      MemRef y_casted(makeType<RingTy>(y.eltype().semantic_type(),
                                       SizeOf(out_type.storage_type()) * 8),
                      y.shape());
      ring_assign(y_casted, y);
      return dot_prot->DotOLE(y_casted, dim3, /*is_lhs*/ false).as(out_type);
    }
    return dot_prot->DotOLE(y, dim3, /*is_lhs*/ false).as(out_type);
  }
}

// A is (M, K); B is (K, N)
MemRef MatMulAA::proc(KernelEvalContext* ctx, const MemRef& x,
                      const MemRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return MemRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  dot_prot->LazyInitKeys(field);

  const int rank = comm->getRank();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms homomorphically
  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};

  auto* conn = comm->lctx().get();
  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<MemRef> task = std::async(std::launch::async, [&] {
    // Compute x0*y1
    if (rank == 0) {
      return dot_prot->DotOLE(x, dupx.get(), dim3, true);
    } else {
      return dot_prot->DotOLE(y, dupx.get(), dim3, false);
    }
  });

  MemRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->DotOLE(y, conn, dim3, false);
  } else {
    x1y0 = dot_prot->DotOLE(x, conn, dim3, true);
  }

  auto ret = ring_mmul(x, y);
  ring_add_(ret, x1y0);
  return ring_add(ret, task.get()).as(x.eltype());
}

MemRef MatMulAV::proc(KernelEvalContext* ctx, const MemRef& x,
                      const MemRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return MemRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }
  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  dot_prot->LazyInitKeys(field);

  const int rank = comm->getRank();
  const auto* ptype = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");
  const int owner = ptype->owner();
  MemRef out;
  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};
  // (x0 + x1)*y = <x0 * y>_0 + <x0 * y>_1 + x1 * y
  if (rank == owner) {
    if (y.eltype().storage_type() != x.eltype().storage_type()) {
      MemRef y_casted(makeType<RingTy>(y.eltype().semantic_type(),
                                       SizeOf(x.eltype().storage_type()) * 8),
                      y.shape());
      ring_assign(y_casted, y);
      out = dot_prot->DotOLE(y_casted, dim3, false);
    } else {
      out = dot_prot->DotOLE(y, dim3, false);
    }

    // Compute <y * x0>
    auto local = ring_mmul(x, y);
    ring_add_(out, local);
  } else {
    out = dot_prot->DotOLE(x, dim3, true);
  }
  return out.as(x.eltype());
}

}  // namespace spu::mpc::cheetah
