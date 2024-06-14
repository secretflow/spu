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

NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  // NOTES for ring_rshift to 2 bits.
  // Refer to:
  // New Primitives for Actively-Secure MPC over Rings with Applications to
  // Private Machine Learning
  // - https://eprint.iacr.org/2019/599.pdf
  // It's safer to keep the number within [-2**(k-2), 2**(k-2)) for comparison
  // operations.
  return ring_rshift(prg_state->genPriv(field, shape), 2)
      .as(makeType<AShrTy>(field));
}

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] =
      prg_state->genPrssPair(field, in.shape(), PrgState::GenPrssCtrl::Both);
  auto x = ring_sub(r0, r1).as(makeType<AShrTy>(field));

  if (comm->getRank() == 0) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::ADD, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto out_ty = makeType<Priv2kTy>(field, rank);

  auto numel = in.numel();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    std::vector<ring2k_t> share(numel);
    NdArrayView<ring2k_t> _in(in);
    pforeach(0, numel, [&](int64_t idx) { share[idx] = _in[idx]; });

    std::vector<std::vector<ring2k_t>> shares =
        comm->gather<ring2k_t>(share, rank, "a2v");  // comm => 1, k
    if (comm->getRank() == rank) {
      SPU_ENFORCE(shares.size() == comm->getWorldSize());
      NdArrayRef out(out_ty, in.shape());
      NdArrayView<ring2k_t> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        ring2k_t s = 0;
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

NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const size_t owner_rank = in_ty->owner();
  const auto field = in_ty->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] =
      prg_state->genPrssPair(field, in.shape(), PrgState::GenPrssCtrl::Both);
  auto x = ring_sub(r0, r1).as(makeType<AShrTy>(field));

  if (comm->getRank() == owner_rank) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

NdArrayRef NotA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  // First, let's show negate could be locally processed.
  //   let X = sum(Xi)     % M
  //   let Yi = neg(Xi) = M-Xi
  //
  // we get
  //   Y = sum(Yi)         % M
  //     = n*M - sum(Xi)   % M
  //     = -sum(Xi)        % M
  //     = -X              % M
  //
  // 'not' could be processed accordingly.
  //   not(X)
  //     = M-1-X           # by definition, not is the complement of 2^k
  //     = neg(X) + M-1
  //
  auto res = ring_neg(in);
  if (comm->getRank() == 0) {
    const auto field = in.eltype().as<Ring2k>()->field();
    ring_add_(res, ring_not(ring_zeros(field, in.shape())));
  }

  return res.as(in.eltype());
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
NdArrayRef AddAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  auto* comm = ctx->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_add(lhs, rhs).as(lhs.eltype());
  }

  return lhs;
}

NdArrayRef AddAA::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  SPU_ENFORCE(lhs.eltype() == rhs.eltype());

  return ring_add(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
NdArrayRef MulAP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  return ring_mul(lhs, rhs).as(lhs.eltype());
}

namespace {

NdArrayRef UnflattenBuffer(yacl::Buffer&& buf, const Type& t, const Shape& s) {
  return NdArrayRef(std::make_shared<yacl::Buffer>(std::move(buf)), t, s);
}

NdArrayRef UnflattenBuffer(yacl::Buffer&& buf, const NdArrayRef& x) {
  return NdArrayRef(std::make_shared<yacl::Buffer>(std::move(buf)), x.eltype(),
                    x.shape());
}

std::tuple<NdArrayRef, NdArrayRef, NdArrayRef, NdArrayRef, NdArrayRef> MulOpen(
    KernelEvalContext* ctx, const NdArrayRef& x, const NdArrayRef& y,
    bool mmul) {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  auto* beaver_cache = ctx->getState<Semi2kState>()->beaver_cache();
  auto x_cache = beaver_cache->GetCache(x, mmul);
  auto y_cache = beaver_cache->GetCache(y, mmul);

  // can't init on same array twice
  if (x == y && x_cache.enabled && x_cache.replay_desc.status == Beaver::Init) {
    // FIXME: how to avoid open on same array twice (x.t dot x)
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
  NdArrayRef a;
  NdArrayRef b;
  NdArrayRef c;
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
  NdArrayRef x_a;
  NdArrayRef y_b;

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
    auto res = vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const NdArrayRef& s) {
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

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  auto* comm = ctx->getState<Communicator>();

  auto [a, b, c, x_a, y_b] = MulOpen(ctx, x, y, false);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(
      ring_add(ring_mul(std::move(b), x_a), ring_mul(std::move(a), y_b)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(std::move(x_a), y_b));
  }
  return z.as(x.eltype());
}

NdArrayRef SquareA::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  auto* beaver_cache = ctx->getState<Semi2kState>()->beaver_cache();
  auto x_cache = beaver_cache->GetCache(x, false);

  // generate beaver Square pair.
  NdArrayRef a;
  NdArrayRef b;
  const size_t numel = x.shape().numel();
  auto [a_buf, b_buf] =
      beaver->Square(field, numel,  //
                     x_cache.enabled ? &x_cache.replay_desc : nullptr);
  SPU_ENFORCE(static_cast<size_t>(a_buf.size()) == numel * SizeOf(field));
  SPU_ENFORCE(static_cast<size_t>(b_buf.size()) == numel * SizeOf(field));

  a = UnflattenBuffer(std::move(a_buf), x);
  b = UnflattenBuffer(std::move(b_buf), x);

  // Open x-a
  NdArrayRef x_a;

  if (x_cache.replay_desc.status != Beaver::Init) {
    x_a = std::move(x_cache.open_cache);
  } else {
    x_a = comm->allReduce(ReduceOp::ADD, ring_sub(x, a), "open(x-a)");
  }

  if (x_cache.enabled && x_cache.replay_desc.status == Beaver::Init) {
    beaver_cache->SetCache(x, x_cache.replay_desc, x_a);
  }

  // Zi = Bi + 2 * (X - A) * Ai + <(X - A) * (X - A)>
  auto z = ring_add(ring_mul(ring_mul(std::move(a), x_a), 2), b);
  if (comm->getRank() == 0) {
    // z += (X - A) * (X - A);
    ring_add_(z, ring_mul(x_a, x_a));
  }
  return z.as(x.eltype());
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
NdArrayRef MatMulAP::proc(KernelEvalContext*, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  return ring_mmul(x, y).as(x.eltype());
}

NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
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

NdArrayRef LShiftA::proc(KernelEvalContext*, const NdArrayRef& in,
                         size_t bits) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  return ring_lshift(in, bits).as(in.eltype());
}

NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        size_t bits, SignType sign) const {
  auto* comm = ctx->getState<Communicator>();

  (void)sign;  // TODO: optimize me.

  // TODO: add truncation method to options.
  if (comm->getWorldSize() == 2) {
    // SecureML, local truncation.
    // Ref: Theorem 1. https://eprint.iacr.org/2017/396.pdf
    return ring_arshift(x, bits).as(x.eltype());
  } else {
    // ABY3, truncation pair method.
    // Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
    auto* beaver = ctx->getState<Semi2kState>()->beaver();

    const auto field = x.eltype().as<Ring2k>()->field();
    auto [r_buf, rb_buf] = beaver->Trunc(field, x.shape().numel(), bits);

    NdArrayRef r(std::make_shared<yacl::Buffer>(std::move(r_buf)), x.eltype(),
                 x.shape());
    NdArrayRef rb(std::make_shared<yacl::Buffer>(std::move(rb_buf)), x.eltype(),
                  x.shape());

    // open x - r
    auto x_r = comm->allReduce(ReduceOp::ADD, ring_sub(x, r), kBindName);
    auto res = rb;
    if (comm->getRank() == 0) {
      ring_add_(res, ring_arshift(x_r, bits));
    }

    // res = [x-r] + [r], x which [*] is truncation operation.
    return res.as(x.eltype());
  }
}

NdArrayRef TruncAPr::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t bits, SignType sign) const {
  (void)sign;  // TODO: optimize me.
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  const auto numel = in.numel();
  const auto field = in.eltype().as<Ring2k>()->field();
  const size_t k = SizeOf(field) * 8;

  NdArrayRef out(in.eltype(), in.shape());

  DISPATCH_ALL_FIELDS(field, "semi2k.truncpr", [&]() {
    using U = ring2k_t;
    auto [r, rc, rb] = beaver->TruncPr(field, numel, bits);
    SPU_ENFORCE(static_cast<size_t>(r.size()) == numel * SizeOf(field));
    SPU_ENFORCE(static_cast<size_t>(rc.size()) == numel * SizeOf(field));
    SPU_ENFORCE(static_cast<size_t>(rb.size()) == numel * SizeOf(field));

    NdArrayView<U> _in(in);
    absl::Span<const U> _r(r.data<U>(), numel);
    absl::Span<const U> _rc(rc.data<U>(), numel);
    absl::Span<const U> _rb(rb.data<U>(), numel);
    NdArrayView<U> _out(out);

    std::vector<U> c;
    {
      std::vector<U> x_plus_r(numel);

      pforeach(0, numel, [&](int64_t idx) {
        auto x = _in[idx];
        // handle negative number.
        // assume secret x in [-2^(k-2), 2^(k-2)), by
        // adding 2^(k-2) x' = x + 2^(k-2) in [0, 2^(k-1)), with msb(x') == 0
        if (comm->getRank() == 0) {
          x += U(1) << (k - 2);
        }
        // mask x with r
        x_plus_r[idx] = x + _r[idx];
      });
      // open <x> + <r> = c
      c = comm->allReduce<U, std::plus>(x_plus_r, kBindName);
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
  const auto& v = ctx->getParam<Value>(0);
  const auto& enable_cache = ctx->getParam<bool>(1);

  auto* beaver_cache = ctx->getState<Semi2kState>()->beaver_cache();

  if (enable_cache) {
    beaver_cache->EnableCache(v.data());
    if (v.isComplex()) {
      beaver_cache->EnableCache(v.imag().value());
    }
  } else {
    beaver_cache->DisableCache(v.data());
    if (v.isComplex()) {
      beaver_cache->DisableCache(v.imag().value());
    }
  }
  // dummy output
  ctx->pushOutput(Value());
}

}  // namespace spu::mpc::semi2k
