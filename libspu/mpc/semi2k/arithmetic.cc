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

#include "libspu/mpc/semi2k/arithmetic.h"

#include <functional>

#include "libspu/core/type_util.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

void CommonTypeA::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  const size_t width = std::max(SizeOf(lhs.as<ArithShareTy>()->storage_type()),
                                SizeOf(rhs.as<ArithShareTy>()->storage_type()));

  ctx->pushOutput(makeType<ArithShareTy>(
      std::max(lhs.semantic_type(), rhs.semantic_type()), width * 8));
}

MemRef CastTypeA::proc(KernelEvalContext*, const MemRef& in,
                       const Type& to_type) const {
  SPU_ENFORCE(in.eltype().storage_type() == to_type.storage_type(),
              "in = {}, to = {}", in.eltype(), to_type);

  return in.as(to_type);
}

MemRef RandA::proc(KernelEvalContext* ctx, SemanticType type,
                   const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  // NOTES for ring_rshift to 2 bits.
  // Refer to:
  // New Primitives for Actively-Secure MPC over Rings with Applications to
  // Private Machine Learning
  // - https://eprint.iacr.org/2019/599.pdf
  // It's safer to keep the number within [-2**(k-2), 2**(k-2)) for comparison
  // operations.
  MemRef ret(makeType<ArithShareTy>(type, field), shape);
  prg_state->fillPriv(ret.data(), ret.elsize() * ret.numel());
  ring_rshift_(ret, {2});
  return ret;
}

MemRef P2A::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* ty = in.eltype().as<BaseRingType>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  MemRef r0(makeType<RingTy>(in.eltype().semantic_type(), field), in.shape());
  MemRef r1(makeType<RingTy>(in.eltype().semantic_type(), field), in.shape());

  prg_state->fillPrssPair(r0.data(), r1.data(), r0.elsize() * r0.numel());

  auto x = ring_sub(r0, r1);
  if (comm->getRank() == 0) {
    if (x.eltype().storage_type() != in.eltype().storage_type()) {
      MemRef in_cast(x.eltype(), in.shape());
      ring_assign(in_cast, in);
      ring_add_(x, in_cast);
    } else {
      ring_add_(x, in);
    }
  }

  return x.as(makeType<ArithShareTy>(ty->semantic_type(), field));
}

MemRef A2P::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* ty = in.eltype().as<BaseRingType>();
  auto* comm = ctx->getState<Communicator>();
  auto tmp = comm->allReduce(ReduceOp::ADD, in, kBindName());
  MemRef out(makeType<Pub2kTy>(ty->semantic_type()), in.shape());
  ring_assign(out, tmp);
  return out;
}

MemRef A2V::proc(KernelEvalContext* ctx, const MemRef& in, size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  auto out_ty = makeType<Priv2kTy>(in.eltype().semantic_type(),
                                   in.eltype().storage_type(), rank);

  auto numel = in.numel();

  return DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using ring2k_t = ScalarT;
    std::vector<ring2k_t> share(numel);
    MemRefView<ring2k_t> _in(in);
    pforeach(0, numel, [&](int64_t idx) { share[idx] = _in[idx]; });

    std::vector<std::vector<ring2k_t>> shares =
        comm->gather<ring2k_t>(share, rank, "a2v");  // comm => 1, k
    if (comm->getRank() == rank) {
      SPU_ENFORCE(shares.size() == comm->getWorldSize());
      MemRef out(out_ty, in.shape());
      MemRefView<ScalarT> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        ScalarT s = 0;
        for (auto& share : shares) {
          s += share[idx];
        }
        _out[idx] = s;
      });
      return out;
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  });
}

MemRef V2A::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const size_t owner_rank = in_ty->owner();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  MemRef r0(makeType<RingTy>(in_ty->semantic_type(), field), in.shape());
  MemRef r1(makeType<RingTy>(in_ty->semantic_type(), field), in.shape());

  prg_state->fillPrssPair(r0.data(), r1.data(), r0.elsize() * r0.numel());
  auto x = ring_sub(r0, r1).as(
      makeType<ArithShareTy>(in.eltype().semantic_type(), field));

  if (comm->getRank() == owner_rank) {
    if (x.eltype().storage_type() != in.eltype().storage_type()) {
      MemRef in_cast(x.eltype(), in.shape());
      ring_assign(in_cast, in);
      ring_add_(x, in_cast);
    } else {
      ring_add_(x, in);
    }
  }
  return x.as(makeType<ArithShareTy>(in_ty->semantic_type(), field));
}

MemRef NegateA::proc(KernelEvalContext* ctx, const MemRef& in) const {
  auto res = ring_neg(in);
  return res.as(in.eltype());
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
MemRef AddAP::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  auto* comm = ctx->getState<Communicator>();

  if (comm->getRank() == 0) {
    if (lhs.eltype().storage_type() != rhs.eltype().storage_type()) {
      MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                       SizeOf(lhs.eltype().storage_type()) * 8),
                      rhs.shape());
      ring_assign(rhs_cast, rhs);
      return ring_add(lhs, rhs_cast).as(lhs.eltype());
    }
    return ring_add(lhs, rhs).as(lhs.eltype());
  }

  return lhs;
}

MemRef AddAA::proc(KernelEvalContext*, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  SPU_ENFORCE(lhs.eltype().storage_type() == rhs.eltype().storage_type(),
              "lhs {} vs rhs {}", lhs.eltype(), rhs.eltype());

  return ring_add(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
MemRef MulAP::proc(KernelEvalContext*, const MemRef& lhs,
                   const MemRef& rhs) const {
  if (lhs.eltype().storage_type() != rhs.eltype().storage_type()) {
    MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                     SizeOf(lhs.eltype().storage_type()) * 8),
                    rhs.shape());
    ring_assign(rhs_cast, rhs);
    return ring_mul(lhs, rhs_cast).as(lhs.eltype());
  }
  return ring_mul(lhs, rhs).as(lhs.eltype());
}

namespace {

MemRef UnflattenBuffer(yacl::Buffer&& buf, const Type& t, const Shape& s) {
  return MemRef(std::make_shared<yacl::Buffer>(std::move(buf)), t, s);
}

MemRef UnflattenBuffer(yacl::Buffer&& buf, const MemRef& x) {
  return MemRef(std::make_shared<yacl::Buffer>(std::move(buf)), x.eltype(),
                x.shape());
}

std::tuple<MemRef, MemRef, MemRef, MemRef, MemRef> MulOpen(
    KernelEvalContext* ctx, const MemRef& x, const MemRef& y, bool mmul) {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  auto* beaver_cache = ctx->getState<Semi2kState>()->beaver_cache();
  auto x_cache = beaver_cache->GetCache(x, mmul);
  auto y_cache = beaver_cache->GetCache(y, mmul);

  // can't init on same array twice
  if (x == y && x_cache.enabled && x_cache.replay_desc.status == Beaver::Init) {
    // FIXME: how to avoid open on same array twice (x * x  or  x.t dot x)
    y_cache.enabled = false;
  }

  Shape z_shape;
  if (mmul) {
    SPU_ENFORCE(x.shape()[1] == y.shape()[0]);
    z_shape = Shape{x.shape()[0], y.shape()[1]};
  } else {
    SPU_ENFORCE(x.shape() == y.shape());
    z_shape = x.shape();
  }

  // generate beaver multiple triple.
  MemRef a;
  MemRef b;
  MemRef c;
  if (mmul) {
    auto [a_buf, b_buf, c_buf] =
        beaver->Dot(field, x.shape()[0], y.shape()[1], x.shape()[1],  //
                    x_cache.enabled ? &x_cache.replay_desc : nullptr,
                    y_cache.enabled ? &y_cache.replay_desc : nullptr);
    SPU_ENFORCE(static_cast<size_t>(a_buf.size()) == x.numel() * SizeOf(field));
    SPU_ENFORCE(static_cast<size_t>(b_buf.size()) == y.numel() * SizeOf(field));
    SPU_ENFORCE(static_cast<size_t>(c_buf.size()) ==
                z_shape.numel() * SizeOf(field));

    a = UnflattenBuffer(std::move(a_buf), x);
    b = UnflattenBuffer(std::move(b_buf), y);
    c = UnflattenBuffer(std::move(c_buf), x.eltype(), z_shape);
  } else {
    const size_t numel = x.shape().numel();
    auto [a_buf, b_buf, c_buf] =
        beaver->Mul(field, numel,  //
                    x_cache.enabled ? &x_cache.replay_desc : nullptr,
                    y_cache.enabled ? &y_cache.replay_desc : nullptr);
    SPU_ENFORCE(static_cast<size_t>(a_buf.size()) == numel * SizeOf(field));
    SPU_ENFORCE(static_cast<size_t>(b_buf.size()) == numel * SizeOf(field));
    SPU_ENFORCE(static_cast<size_t>(c_buf.size()) == numel * SizeOf(field));

    a = UnflattenBuffer(std::move(a_buf), x);
    b = UnflattenBuffer(std::move(b_buf), y);
    c = UnflattenBuffer(std::move(c_buf), x);
  }

  // Open x-a & y-b
  MemRef x_a;
  MemRef y_b;

  auto x_hit_cache = x_cache.replay_desc.status != Beaver::Init;
  auto y_hit_cache = y_cache.replay_desc.status != Beaver::Init;

  if (ctx->sctx()->config().experimental_disable_vectorization() ||
      x_hit_cache || y_hit_cache) {
    if (x_hit_cache) {
      x_a = std::move(x_cache.open_cache);
    } else {
      x_a = comm->allReduce(ReduceOp::ADD, ring_sub(x, a), "open(x-a)");
    }
    if (y_hit_cache) {
      y_b = std::move(y_cache.open_cache);
    } else {
      y_b = comm->allReduce(ReduceOp::ADD, ring_sub(y, b), "open(y-b)");
    }
  } else {
    auto res = vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const MemRef& s) {
      return comm->allReduce(ReduceOp::ADD, s, "open(x-a,y-b)");
    });
    x_a = std::move(res[0]);
    y_b = std::move(res[1]);
  }

  if (x_cache.enabled && x_cache.replay_desc.status == Beaver::Init) {
    beaver_cache->SetCache(x, x_cache.replay_desc, x_a);
  }
  if (y_cache.enabled && y_cache.replay_desc.status == Beaver::Init) {
    beaver_cache->SetCache(y, y_cache.replay_desc, y_b);
  }

  return {std::move(a), std::move(b), std::move(c), std::move(x_a),
          std::move(y_b)};
}

}  // namespace

MemRef MulAA::proc(KernelEvalContext* ctx, const MemRef& x,
                   const MemRef& y) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto ty = makeType<ArithShareTy>(
      std::max(x.eltype().semantic_type(), y.eltype().semantic_type()), field);

  auto [a, b, c, x_a, y_b] = MulOpen(ctx, x, y, false);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(
      ring_add(ring_mul(std::move(b), x_a), ring_mul(std::move(a), y_b)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(std::move(x_a), y_b));
  }
  return z.as(ty);
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
MemRef MatMulAP::proc(KernelEvalContext*, const MemRef& lhs,
                      const MemRef& rhs) const {
  if (lhs.eltype().storage_type() != rhs.eltype().storage_type()) {
    MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                     SizeOf(lhs.eltype().storage_type()) * 8),
                    rhs.shape());
    ring_assign(rhs_cast, rhs);
    return ring_mmul(lhs, rhs_cast).as(lhs.eltype());
  }
  return ring_mmul(lhs, rhs).as(lhs.eltype());
}

MemRef MatMulAA::proc(KernelEvalContext* ctx, const MemRef& x,
                      const MemRef& y) const {
  auto* comm = ctx->getState<Communicator>();

  auto [a, b, c, x_a, y_b] = MulOpen(ctx, x, y, true);

  // Zi = Ci + (X - A) dot Bi + Ai dot (Y - B) + <(X - A) dot (Y - B)>
  auto z = ring_add(ring_add(ring_mmul(x_a, b), ring_mmul(a, y_b)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mmul(x_a, y_b));
  }
  return z.as(x.eltype());
}

MemRef LShiftA::proc(KernelEvalContext*, const MemRef& in,
                     const Sizes& bits) const {
  return ring_lshift(in, bits).as(in.eltype());
}

MemRef TruncA::proc(KernelEvalContext* ctx, const MemRef& x, size_t bits,
                    SignType sign) const {
  auto* comm = ctx->getState<Communicator>();

  (void)sign;  // TODO: optimize me.

  // TODO: add truncation method to options.
  if (comm->getWorldSize() == 2) {
    // SecureML, local truncation.
    // Ref: Theorem 1. https://eprint.iacr.org/2017/396.pdf
    return ring_arshift(x, {static_cast<int64_t>(bits)}).as(x.eltype());
  } else {
    // ABY3, truncation pair method.
    // Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
    auto* beaver = ctx->getState<Semi2kState>()->beaver();

    const auto field = ctx->getState<Z2kState>()->getDefaultField();
    auto [r_buf, rb_buf] = beaver->Trunc(field, x.shape().numel(), bits);

    MemRef r(std::make_shared<yacl::Buffer>(std::move(r_buf)), x.eltype(),
             x.shape());
    MemRef rb(std::make_shared<yacl::Buffer>(std::move(rb_buf)), x.eltype(),
              x.shape());

    // open x - r
    auto x_r = comm->allReduce(ReduceOp::ADD, ring_sub(x, r), kBindName());
    auto res = rb;
    if (comm->getRank() == 0) {
      ring_add_(res, ring_arshift(x_r, {static_cast<int64_t>(bits)}));
    }

    // res = [x-r] + [r], x which [*] is truncation operation.
    return res.as(x.eltype());
  }
}

MemRef TruncAPr::proc(KernelEvalContext* ctx, const MemRef& in, size_t bits,
                      SignType sign) const {
  (void)sign;  // TODO: optimize me.
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  const auto numel = in.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t k = SizeOf(field) * 8;

  MemRef out(in.eltype(), in.shape());
  DISPATCH_ALL_STORAGE_TYPES(in.eltype().storage_type(), [&]() {
    using U = ScalarT;
    auto [r, rc, rb] = beaver->TruncPr(field, numel, bits);
    SPU_ENFORCE(static_cast<size_t>(r.size()) == numel * SizeOf(field));
    SPU_ENFORCE(static_cast<size_t>(rc.size()) == numel * SizeOf(field));
    SPU_ENFORCE(static_cast<size_t>(rb.size()) == numel * SizeOf(field));

    MemRefView<U> _in(in);
    absl::Span<const U> _r(r.data<U>(), numel);
    absl::Span<const U> _rc(rc.data<U>(), numel);
    absl::Span<const U> _rb(rb.data<U>(), numel);
    MemRefView<U> _out(out);

    std::vector<U> c;
    {
      std::vector<U> x_plus_r(numel);

      pforeach(0, numel, [&](int64_t idx) {
        auto x = _in[idx];
        // handle negative number.
        // assume secret x in [-2^(k-2), 2^(k-2)), by
        // adding 2^(k-2) x' = x + 2^(k-2) in [0, 2^(k-1)), with msb(x') ==
        // 0
        if (comm->getRank() == 0) {
          x += U(1) << (k - 2);
        }
        // mask x with r
        x_plus_r[idx] = x + _r[idx];
      });
      // open <x> + <r> = c
      c = comm->allReduce<U, std::plus>(x_plus_r, kBindName());
    }

    pforeach(0, numel, [&](int64_t idx) {
      auto ck_1 = c[idx] >> (k - 1);

      U y;
      if (comm->getRank() == 0) {
        // <b> = <rb> ^ c{k-1} = <rb> + c{k-1} - 2*c{k-1}*<rb>
        auto b = _rb[idx] + ck_1 - 2 * ck_1 * _rb[idx];
        // c_hat = c/2^m mod 2^(k-m-1) = (c << 1) >> (1+m)
        auto c_hat = (c[idx] << 1) >> (1 + bits);
        // y = c_hat - <rc> + <b> * 2^(k-m-1)
        y = c_hat - _rc[idx] + (b << (k - 1 - bits));
        // re-encode negative numbers.
        // from https://eprint.iacr.org/2020/338.pdf, section 5.1
        // y' = y - 2^(k-2-m)
        y -= (U(1) << (k - 2 - bits));
      } else {
        auto b = _rb[idx] + 0 - 2 * ck_1 * _rb[idx];
        y = 0 - _rc[idx] + (b << (k - 1 - bits));
      }

      _out[idx] = y;
    });
  });

  return out;
}

void BeaverCacheKernel::evaluate(KernelEvalContext* ctx) const {
  const auto& v = ctx->getParam(0);
  const auto& enable_cache = ctx->getParam<bool>(1);

  auto* beaver_cache = ctx->getState<Semi2kState>()->beaver_cache();

  if (enable_cache) {
    beaver_cache->EnableCache(v);
  } else {
    beaver_cache->DisableCache(v);
  }
}

}  // namespace spu::mpc::semi2k
