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
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

ArrayRef RandA::proc(KernelEvalContext* ctx, size_t size) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  // NOTES for ring_rshift to 2 bits.
  // Refer to:
  // New Primitives for Actively-Secure MPC over Rings with Applications to
  // Private Machine Learning
  // - https://eprint.iacr.org/2019/599.pdf
  // It's safer to keep the number within [-2**(k-2), 2**(k-2)) for comparison
  // operations.
  return ring_rshift(prg_state->genPriv(field, size), 2)
      .as(makeType<AShrTy>(field));
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] = prg_state->genPrssPair(field, in.numel());
  auto x = ring_sub(r0, r1).as(makeType<AShrTy>(field));

  if (comm->getRank() == 0) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::ADD, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

ArrayRef A2V::proc(KernelEvalContext* ctx, const ArrayRef& in,
                   size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto out_ty = makeType<Priv2kTy>(field, rank);
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    std::vector<ring2k_t> share(in.numel());
    pforeach(0, in.numel(),
             [&](int64_t idx) { share[idx] = in.at<ring2k_t>(idx); });

    std::vector<std::vector<ring2k_t>> shares =
        comm->gather<ring2k_t>(share, rank, "a2v");  // comm => 1, k
    if (comm->getRank() == rank) {
      SPU_ENFORCE(shares.size() == comm->getWorldSize());
      ArrayRef out(out_ty, in.numel());
      auto _out = ArrayView<ring2k_t>(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = 0;
        for (auto& share : shares) {
          _out[idx] += share[idx];
        }
      });
      return out;
    } else {
      return makeConstantArrayRef(out_ty, in.numel());
    }
  });
}

ArrayRef V2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const size_t owner_rank = in_ty->owner();
  const auto field = in_ty->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] = prg_state->genPrssPair(field, in.numel());
  auto x = ring_sub(r0, r1).as(makeType<AShrTy>(field));

  if (comm->getRank() == owner_rank) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

ArrayRef NotA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
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
    ring_add_(res, ring_not(ring_zeros(field, in.numel())));
  }

  return res.as(in.eltype());
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  auto* comm = ctx->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_add(lhs, rhs).as(lhs.eltype());
  }

  return lhs;
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  SPU_ENFORCE(lhs.eltype() == rhs.eltype());

  return ring_add(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  return ring_mul(lhs, rhs).as(lhs.eltype());
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  auto res = ArrayRef(makeType<AShrTy>(field), lhs.numel());
  DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
    using U = ring2k_t;
    auto [a, b, c] = beaver->Mul(field, lhs.numel());

    auto _x = ArrayView<U>(lhs);
    auto _y = ArrayView<U>(rhs);
    auto _z = ArrayView<U>(res);

    auto _a = ArrayView<U>(a);
    auto _b = ArrayView<U>(b);
    auto _c = ArrayView<U>(c);

    std::vector<U> eu(a.numel() * 2);
    absl::Span<U> e(eu.data(), _x.numel());
    absl::Span<U> u(eu.data() + _x.numel(), _x.numel());

    pforeach(0, a.numel(), [&](int64_t idx) {
      e[idx] = _x[idx] - _a[idx];  // e = x - a;
      u[idx] = _y[idx] - _b[idx];  // u = y - b;
    });

    // open x-a & y-b
    if (ctx->sctx()->config().experimental_disable_vectorization()) {
      auto ee = comm->allReduce<U, std::plus>(e, "open(x-a)");
      auto uu = comm->allReduce<U, std::plus>(u, "open(y-b)");
      std::copy(ee.begin(), ee.end(), e.begin());
      std::copy(uu.begin(), uu.end(), u.begin());
    } else {
      eu = comm->allReduce<U, std::plus>(eu, "open(x-a,y-b)");
    }

    e = absl::Span<U>(eu.data(), _x.numel());
    u = absl::Span<U>(eu.data() + _x.numel(), _x.numel());

    // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
    pforeach(0, a.numel(), [&](int64_t idx) {
      _z[idx] = _c[idx] + e[idx] * _b[idx] + u[idx] * _a[idx];
      if (comm->getRank() == 0) {
        // z += (X-A) * (Y-B);
        _z[idx] += e[idx] * u[idx];
      }
    });
  });
  return res;
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t m, size_t n, size_t k) const {
  return ring_mmul(x, y, m, n, k).as(x.eltype());
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t m, size_t n, size_t k) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  // generate beaver multiple triple.
  auto [a, b, c] = beaver->Dot(field, m, n, k);

  // Open x-a & y-b
  ArrayRef x_a;
  ArrayRef y_b;

  if (ctx->sctx()->config().experimental_disable_vectorization()) {
    x_a = comm->allReduce(ReduceOp::ADD, ring_sub(x, a), "open(x-a)");
    y_b = comm->allReduce(ReduceOp::ADD, ring_sub(y, b), "open(y-b)");
  } else {
    auto res =
        vectorize({ring_sub(x, a), ring_sub(y, b)}, [&](const ArrayRef& s) {
          return comm->allReduce(ReduceOp::ADD, s, "open(x-a,y-b)");
        });
    x_a = std::move(res[0]);
    y_b = std::move(res[1]);
  }

  // Zi = Ci + (X - A) dot Bi + Ai dot (Y - B) + <(X - A) dot (Y - B)>
  auto z = ring_add(
      ring_add(ring_mmul(x_a, b, m, n, k), ring_mmul(a, y_b, m, n, k)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mmul(x_a, y_b, m, n, k));
  }
  return z.as(x.eltype());
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  return ring_lshift(in, bits).as(in.eltype());
}

ArrayRef TruncA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                      size_t bits) const {
  auto* comm = ctx->getState<Communicator>();

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
    const auto& [r, rb] = beaver->Trunc(field, x.numel(), bits);

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

ArrayRef TruncAPr::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();
  const auto numel = in.numel();
  const auto field = in.eltype().as<Ring2k>()->field();
  const size_t k = SizeOf(field) * 8;

  ArrayRef r;
  ArrayRef rc;
  ArrayRef rb;
  std::tie(r, rc, rb) = beaver->TruncPr(field, numel, bits);

  ArrayRef out(in.eltype(), numel);
  DISPATCH_ALL_FIELDS(field, "semi2k.truncpr", [&]() {
    using U = ring2k_t;

    std::vector<U> c;
    {
      std::vector<U> x_plus_r(numel);
      pforeach(0, numel, [&](int64_t idx) {
        auto x = in.at<U>(idx);
        // handle negative number.
        // assume secret x in [-2^(k-2), 2^(k-2)), by
        // adding 2^(k-2) x' = x + 2^(k-2) in [0, 2^(k-1)), with msb(x') == 0
        if (comm->getRank() == 0) {
          x += U(1) << (k - 2);
        }
        // mask x with r
        x_plus_r[idx] = x + r.at<U>(idx);
      });
      // open <x> + <r> = c
      c = comm->allReduce<U, std::plus>(x_plus_r, kBindName);
    }

    pforeach(0, numel, [&](int64_t idx) {
      auto ck_1 = c[idx] >> (k - 1);

      U y;
      if (comm->getRank() == 0) {
        // <b> = <rb> ^ c{k-1} = <rb> + c{k-1} - 2*c{k-1}*<rb>
        auto b = rb.at<U>(idx) + ck_1 - 2 * ck_1 * rb.at<U>(idx);
        // c_hat = c/2^m mod 2^(k-m-1) = (c << 1) >> (1+m)
        auto c_hat = (c[idx] << 1) >> (1 + bits);
        // y = c_hat - <rc> + <b> * 2^(k-m-1)
        y = c_hat - rc.at<U>(idx) + (b << (k - 1 - bits));
        // re-encode negative numbers.
        // from https://eprint.iacr.org/2020/338.pdf, section 5.1
        // y' = y - 2^(k-2-m)
        y -= (U(1) << (k - 2 - bits));
      } else {
        auto b = rb.at<U>(idx) + 0 - 2 * ck_1 * rb.at<U>(idx);
        y = 0 - rc.at<U>(idx) + (b << (k - 1 - bits));
      }

      out.at<U>(idx) = y;
    });
  });

  return out;
}

}  // namespace spu::mpc::semi2k
