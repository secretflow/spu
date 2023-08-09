// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/securenn/arithmetic.h"

#include <array>
#include <functional>
#include <random>

#include "libspu/core/type_util.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/securenn/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::securenn {

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
  auto res = ring_zeros(field, in.shape());

  auto [r0, r1] = prg_state->genPrssPair(field, in.shape());
  // SPU_ENFORCE(owner_rank != 2);
  if (owner_rank == 2) {
    auto x = ring_sub(r0, r1).as(makeType<AShrTy>(field));
    if (comm->getRank() == 2) {
      comm->sendAsync(0, ring_add(x, in).as(makeType<AShrTy>(field)), "s");
    }
    if (comm->getRank() == 0) {
      auto tmp = comm->recv(2, makeType<AShrTy>(field), "s");
      tmp = tmp.reshape(in.shape());
      res = ring_add(x, tmp);
    }
    if (comm->getRank() == 1) {
      res = x;
    }
  } else {
    // P0.r1 = P1.r0
    if (comm->getRank() == 0) res = r1.as(makeType<AShrTy>(field));
    if (comm->getRank() == 1) res = ring_neg(r0).as(makeType<AShrTy>(field));

    if (comm->getRank() == owner_rank) {
      ring_add_(res, in);
    }
  }
  return res.as(makeType<AShrTy>(field));
}

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
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto res = ring_zeros(field, in.shape());
  auto [r0, r1] = prg_state->genPrssPair(field, in.shape());
  // P0.r1 = P1.r0
  if (comm->getRank() == 0) res = r1;
  if (comm->getRank() == 1) res = ring_sub(in, r0);

  return res.as(makeType<AShrTy>(field));
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::ADD, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
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
  SPU_ENFORCE(lhs.shape() == rhs.shape());
  auto* comm = ctx->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
  return lhs;
}

NdArrayRef AddAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.shape() == rhs.shape());
  SPU_ENFORCE(lhs.eltype() == rhs.eltype());

  return ring_add(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
NdArrayRef MulAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  return ring_mul(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
NdArrayRef MatMulAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  return ring_mmul(x, y).as(x.eltype());
}

NdArrayRef LShiftA::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         size_t bits) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  return ring_lshift(in, bits).as(in.eltype());
}

NdArrayRef TruncAPr::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t bits, SignType sign) const {
  (void)sign;  // TODO: optimize me.

  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  const auto numel = in.numel();
  const auto field = in.eltype().as<Ring2k>()->field();
  const size_t k = SizeOf(field) * 8;

  // NdArrayRef r(in.eltype(), in.shape());
  // NdArrayRef rc(in.eltype(), in.shape());
  // NdArrayRef rb(in.eltype(), in.shape());
  // std::tie(r, rc, rb) = beaver->TruncPr(field, in.shape(), bits);

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_ALL_FIELDS(field, "securenn.truncpr", [&]() {
    using U = ring2k_t;

    auto r = prg_state->genPriv(field, in.shape());
    auto rc = prg_state->genPriv(field, in.shape());
    auto rb = prg_state->genPriv(field, in.shape());

    // reconstruct r, rc, rb
    auto r_recon = comm->reduce(ReduceOp::ADD, r, 2, "r");
    auto rc_recon = comm->reduce(ReduceOp::ADD, rc, 2, "rc");
    auto rb_recon = comm->reduce(ReduceOp::ADD, rb, 2, "rb");

    if (rank == 2) {
      auto adjust1 =
          ring_sub(ring_rshift(ring_lshift(r_recon, 1), bits + 1), rc_recon);
      auto adjust2 = ring_sub(ring_rshift(r_recon, k - 1), rb_recon);
      comm->sendAsync(0, adjust1, "adjust1");
      comm->sendAsync(0, adjust2, "adjust2");
    }
    if (rank == 0) {
      auto adjust1 = comm->recv(2, makeType<AShrTy>(field), "adjust1");
      adjust1 = adjust1.reshape(in.shape());
      auto adjust2 = comm->recv(2, makeType<AShrTy>(field), "adjust2");
      adjust2 = adjust2.reshape(in.shape());
      ring_add_(rc, adjust1);
      ring_add_(rb, adjust2);
    }

    SPU_ENFORCE(r.isCompact() && rc.isCompact() && rb.isCompact(),
                "beaver triple must be compact");

    NdArrayView<U> _in(in);
    NdArrayView<U> _r(r);
    NdArrayView<U> _rb(rb);
    NdArrayView<U> _rc(rc);
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
  // P2 send share to P0
  const auto kComm = in.elsize();
  comm->addCommStatsManually(1, kComm);
  if (rank == 2) {
    comm->sendAsync(0, out, "out");
    out = ring_zeros(field, in.shape()).as(makeType<AShrTy>(field));
  }
  if (rank == 0) {
    auto tmp = comm->recv(2, makeType<AShrTy>(field), "out");
    tmp = tmp.reshape(in.shape());
    out = ring_add(out, tmp);
  }

  return out.as(makeType<AShrTy>(field));
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  SPU_ENFORCE(x.shape() == y.shape());

  auto size = x.numel();
  auto ty = makeType<AShrTy>(field);
  auto z = ring_zeros(field, x.shape());

  const auto kComm = x.elsize() * size;
  comm->addCommStatsManually(1, kComm * 4);
  // P2 to be the beaver generator
  if (rank == 2) {
    // P2 generate a0, a1, b0, b1, c0 by PRF
    // and calculate c1
    auto [a1, a0] = prg_state->genPrssPair(field, x.shape());
    auto [b1, b0] = prg_state->genPrssPair(field, x.shape());
    auto c0 = prg_state->genPrssPair(field, x.shape(), true, false).second;

    // c1 = (a0 + a1) * (b0 + b1) - c0
    auto c1 = ring_sub(ring_mul(ring_add(a0, a1), ring_add(b0, b1)), c0);

    comm->sendAsync(1, c1, "c");  // 1 latency, k
  }
  if (rank <= 1) {
    NdArrayRef a(ty, x.shape());
    NdArrayRef b(ty, x.shape());
    NdArrayRef c(ty, x.shape());
    if (rank == 0) {
      a = prg_state->genPrssPair(field, x.shape(), false, true).first;
      b = prg_state->genPrssPair(field, x.shape(), false, true).first;
      c = prg_state->genPrssPair(field, x.shape(), false, true).first;
    }
    if (rank == 1) {
      a = prg_state->genPrssPair(field, x.shape(), true, false).second;
      b = prg_state->genPrssPair(field, x.shape(), true, false).second;
      prg_state->genPrssPair(field, x.shape(), true, true);
      c = comm->recv(2, ty, "c");
      c = c.reshape(x.shape());
    }

    // Open x-a & y-b
    auto send_x_a = ring_sub(x, a).as(ty);
    auto send_y_b = ring_sub(y, b).as(ty);
    // 1 latency, 2 * 2k
    comm->sendAsync((rank + 1) % 2, send_x_a, "x_a");
    comm->sendAsync((rank + 1) % 2, send_y_b, "y_b");
    auto recv_x_a = comm->recv((rank + 1) % 2, ty, "x_a");
    auto recv_y_b = comm->recv((rank + 1) % 2, ty, "y_b");
    recv_x_a = recv_x_a.reshape(x.shape());
    recv_y_b = recv_y_b.reshape(x.shape());
    auto x_a = ring_add(send_x_a, recv_x_a);
    auto y_b = ring_add(send_y_b, recv_y_b);

    // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
    z = ring_add(ring_add(ring_mul(x_a, b), ring_mul(y_b, a)), c);
    if (rank == 0) {
      // z += (X-A) * (Y-B);
      z = ring_add(z, ring_mul(x_a, y_b));
    }
  }

  // P0 and P1 add the share of zero
  // P0.zero_1 = P1.zero_0
  auto [zero_0, zero_1] = prg_state->genPrssPair(field, x.shape());
  if (rank == 0) {
    z = ring_sub(z, zero_1);
  }
  if (rank == 1) {
    z = ring_add(z, zero_0);
  }

  return z.as(ty);
}

NdArrayRef MatMulAA_simple::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                                 const NdArrayRef& y) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto ty = makeType<AShrTy>(field);
  auto shape1 = x.shape();
  auto shape2 = y.shape();
  auto shape3 = ring_mmul(x, y).shape();
  auto z = ring_zeros(field, shape3);

  const auto kComm = x.elsize();
  comm->addCommStatsManually(
      2, (2 * shape1[0] * shape1[1] + 2 * shape2[0] * shape2[1]) * kComm);
  // P2 to be the beaver generator
  if (rank == 2) {
    auto a0 = ring_rand(field, shape1);
    auto a1 = ring_rand(field, shape1);
    auto b0 = ring_rand(field, shape2);
    auto b1 = ring_rand(field, shape2);
    auto c0 = ring_rand(field, shape3);
    auto c1 = ring_sub(ring_mmul(ring_add(a0, a1), ring_add(b0, b1)), c0);
    // 1 latency, 2 * (m * n + m * k + n * k) * kComm (offline)
    comm->sendAsync(0, a0, "a");
    comm->sendAsync(0, b0, "b");
    comm->sendAsync(0, c0, "c");
    comm->sendAsync(1, a1, "a");
    comm->sendAsync(1, b1, "b");
    comm->sendAsync(1, c1, "c");
  }

  if (rank <= 1) {
    auto a = comm->recv(2, ty, "a");
    auto b = comm->recv(2, ty, "b");
    auto c = comm->recv(2, ty, "c");
    a = a.reshape(shape1);
    b = b.reshape(shape2);
    c = c.reshape(shape3);

    // Open x-a & y-b
    auto send_x_a = ring_sub(x, a);
    auto send_y_b = ring_sub(y, b);
    // 1 latency, 2 * (m * k * kComm + k * n * kComm)
    comm->sendAsync((rank + 1) % 2, send_x_a, "x_a");
    comm->sendAsync((rank + 1) % 2, send_y_b, "y_b");
    auto recv_x_a = comm->recv((rank + 1) % 2, ty, "x_a");
    auto recv_y_b = comm->recv((rank + 1) % 2, ty, "y_b");
    recv_x_a = recv_x_a.reshape(shape1);
    recv_y_b = recv_y_b.reshape(shape2);

    auto x_a = ring_add(send_x_a, recv_x_a);
    auto y_b = ring_add(send_y_b, recv_y_b);

    // Zi = Ci + (X - A) dot Bi + Ai dot (Y - B) + <(X - A) dot (Y - B)>
    z = ring_add(ring_add(ring_mmul(x_a, b), ring_mmul(a, y_b)), c);
    if (rank == 0) {
      // z += (X-A) * (Y-B);
      z = ring_add(z, ring_mmul(x_a, y_b));
    }
  }

  // P0 and P1 add the share of zero
  // P0.zero_1 = P1.zero_0
  auto [zero_0, zero_1] = prg_state->genPrssPair(field, shape3);
  if (rank == 0) {
    z = ring_sub(z, zero_1);
  }
  if (rank == 1) {
    z = ring_add(z, zero_0);
  }

  return z.as(ty);
}

NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto shape1 = x.shape();
  auto shape2 = y.shape();
  auto shape3 = ring_mmul(x, y).shape();
  auto ty = makeType<AShrTy>(field);
  auto z = ring_zeros(field, shape3);

  const auto kComm = x.elsize();
  comm->addCommStatsManually(
      2, (2 * shape1[0] * shape1[1] + 2 * shape2[0] * shape2[1]) * kComm);
  // P2 to be the beaver generator
  if (rank == 2) {
    // P2 generate a0, a1, b0, b1, c0 by PRF
    // and calculate c1
    auto [a1, a0] = prg_state->genPrssPair(field, shape1);
    auto [b1, b0] = prg_state->genPrssPair(field, shape2);
    auto c0 = prg_state->genPrssPair(field, shape3, true, false).second;

    // c1 = (a0 + a1) * (b0 + b1) - c0
    auto c1 = ring_sub(ring_mmul(ring_add(a0, a1), ring_add(b0, b1)), c0);
    comm->sendAsync(1, c1, "c");  // 1 latency, m * n * kComm (offline)
  }

  if (rank <= 1) {
    NdArrayRef a(ty, shape1);
    NdArrayRef b(ty, shape2);
    NdArrayRef c(ty, shape3);
    if (rank == 0) {
      a = prg_state->genPrssPair(field, shape1, false, true).first;
      b = prg_state->genPrssPair(field, shape2, false, true).first;
      c = prg_state->genPrssPair(field, shape3, false, true).first;
    }
    if (rank == 1) {
      a = prg_state->genPrssPair(field, shape1, true, false).second;
      b = prg_state->genPrssPair(field, shape2, true, false).second;
      prg_state->genPrssPair(field, shape3, true, true);

      c = comm->recv(2, ty, "c");
      c = c.reshape(shape3);
    }

    // Open x-a & y-b
    auto send_x_a = ring_sub(x, a);
    auto send_y_b = ring_sub(y, b);
    // 1 latency, 2 * (m * k * kComm + k * n * kComm)
    comm->sendAsync((rank + 1) % 2, send_x_a, "x_a");
    comm->sendAsync((rank + 1) % 2, send_y_b, "y_b");
    auto recv_x_a = comm->recv((rank + 1) % 2, ty, "x_a");
    auto recv_y_b = comm->recv((rank + 1) % 2, ty, "y_b");
    recv_x_a = recv_x_a.reshape(shape1);
    recv_y_b = recv_y_b.reshape(shape2);

    auto x_a = ring_add(send_x_a, recv_x_a);
    auto y_b = ring_add(send_y_b, recv_y_b);

    // Zi = Ci + (X - A) dot Bi + Ai dot (Y - B) + <(X - A) dot (Y - B)>
    z = ring_add(ring_add(ring_mmul(x_a, b), ring_mmul(a, y_b)), c);
    if (rank == 0) {
      // z += (X-A) * (Y-B);
      z = ring_add(z, ring_mmul(x_a, y_b));
    }
  }

  // P0 and P1 add the share of zero
  // P0.zero_1 = P1.zero_0
  auto [zero_0, zero_1] = prg_state->genPrssPair(field, shape3);
  if (rank == 0) {
    z = ring_sub(z, zero_1);
  }
  if (rank == 1) {
    z = ring_add(z, zero_0);
  }

  return z.as(ty);
}

template <typename T>
static std::vector<uint8_t> bitDecompose(T in, size_t nbits) {
  std::vector<uint8_t> res;
  for (size_t bit = 0; bit < nbits; bit++) {
    res.push_back(static_cast<uint8_t>((in >> bit) & 0x1));
  }
  return res;
}

NdArrayRef ShareConvert::proc(KernelEvalContext* ctx,
                              const NdArrayRef& a) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = a.eltype().as<AShrTy>()->field();
  const int64_t k = SizeOf(field) * 8;
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  const int64_t size = a.numel();
  const int p = 131;
  const auto log_p = 9;
  const auto ty = makeType<AShrTy>(field);
  const auto one = ring_ones(field, a.shape());
  auto res = ring_zeros(field, a.shape()).as(makeType<AShrTy>(field));
  const auto kComm = a.elsize() * size;
  comm->addCommStatsManually(4, 4 * log_p * kComm + 6 * kComm);

  DISPATCH_ALL_FIELDS(field, "securenn.sc", [&]() {
    using U = ring2k_t;
    const U L_1 = (U)(~0);  // 2^k - 1
    // P0 and P1 add the share of zero
    // P0.zero_1 = P1.zero_0
    auto [zero_0, zero_1] = prg_state->genPrssPair(field, a.shape());
    NdArrayView<U> _zero_0(zero_0);
    NdArrayView<U> _zero_1(zero_1);
    NdArrayView<U> _res(res);

    // P0 and P1 hold eta__ by PRF
    auto [eta__0, eta__1] = prg_state->genPrssPair(field, a.shape());
    // P0 and P1 hold r and share  it into r0 and r1
    // which means P0 and P1 hold r0 and r1
    // P0.r0_1 = P1.r0_0 = r0
    // P0.r1_1 = P1.r1_0 = r1
    auto [r0_0, r0_1] = prg_state->genPrssPair(field, a.shape());
    auto [r1_0, r1_1] = prg_state->genPrssPair(field, a.shape());

    // random for PC
    auto [s_r0, s_r1] = prg_state->genPrssPair(field, {size * k});
    auto [u_r0, u_r1] = prg_state->genPrssPair(field, {size * k});

    if (rank <= 1) {
      auto beta = ring_zeros(field, a.shape());
      NdArrayRef r_share(ty, a.shape());
      NdArrayRef r(ty, a.shape());
      auto alpha = ring_zeros(field, a.shape());
      NdArrayView<U> _alpha(alpha);

      if (rank == 0) {
        r_share = r0_1;
        r = ring_add(r0_1, r1_1);
      }
      if (rank == 1) {
        r_share = r1_0;
        r = ring_add(r0_0, r1_0);
      }

      NdArrayView<U> _r_share(r_share);
      NdArrayView<U> _r(r);

      auto a_ = ring_add(a, r_share);
      NdArrayView<U> _a(a);
      NdArrayView<U> _a_(a_);
      NdArrayView<U> _beta(beta);

      // beta_rank = wrap(a_rank, r_rank, 2^k)
      // alpha = wrap(r_0, r_1, L)
      pforeach(0, size, [&](int64_t idx) {
        if (_a_[idx] < _a[idx]) _beta[idx] = (U)1;
        if (_r[idx] < _r_share[idx]) _alpha[idx] = (U)1;
      });

      comm->sendAsync(2, a_, "a_");  // 1 lentancy, 2k

      auto dp_x = comm->recv(2, ty, "dp_x");
      auto delta = comm->recv(2, ty, "delta");
      dp_x = dp_x.reshape({size * k});
      delta = delta.reshape(a.shape());
      NdArrayView<U> _dp_x(dp_x);
      NdArrayView<U> _delta(delta);

      NdArrayRef eta__(makeType<RingTy>(field), a.shape());
      if (rank == 0) eta__ = eta__1;
      if (rank == 1) eta__ = eta__0;

      // & ring_ones
      NdArrayView<U> _eta__(eta__);
      for (int64_t i = 0; i < size; i++) {
        _eta__[i] = _eta__[i] & 0x1;
      }

      // Private Compare
      auto t = r;
      r = ring_sub(r, one);

      NdArrayView<U> _t(t);

      NdArrayRef u(ty, {size * k});
      NdArrayRef s(ty, {size * k});
      if (rank == 0) {
        u = u_r1;
        s = s_r1;
      }
      if (rank == 1) {
        u = u_r0;
        s = s_r0;
      }
      NdArrayView<U> _u(u);
      NdArrayView<U> _s(s);

      NdArrayRef c(ty, {size * k});
      NdArrayView<U> _c(c);

      size_t w;
      size_t w_total;

      pforeach(0, size, [&](int64_t idx) {
        auto r_bits = bitDecompose(_r[idx], k);
        auto t_bits = bitDecompose(_t[idx], k);

        w_total = 0;
        for (int i = (int)(k - 1); i >= 0; i--) {
          if (_eta__[idx] == 0) {
            w = (p + _dp_x[idx * k + i] + rank * r_bits[i] -
                 2 * r_bits[i] * _dp_x[idx * k + i]) %
                p;
            _c[idx * k + i] =
                (p + rank * r_bits[i] - _dp_x[idx * k + i] + rank + w_total) %
                p;
            w_total = (w_total + w) % p;
          } else if (_eta__[idx] == 1 && _r[idx] != L_1) {
            w = (p + _dp_x[idx * k + i] + rank * t_bits[i] -
                 2 * t_bits[i] * _dp_x[idx * k + i]) %
                p;
            _c[idx * k + i] =
                (p - rank * t_bits[i] + _dp_x[idx * k + i] + rank + w_total) %
                p;
            w_total = (w_total + w) % p;
          } else {
            // r = 2 ^ k - 1 bigger than everything else in the ring
            // c = [0, 1,..., 1]
            if (i != 1) {
              _u[idx] = _u[idx] % p;
              _c[idx * k + i] =
                  (1 - rank) * (_u[idx * k + i] + 1) - rank * _u[idx * k + i];
            } else {
              _u[idx] = _u[idx] % p;
              if (rank == 0) _c[idx * k + i] = _u[idx * k + i];
              if (rank == 1) _c[idx * k + i] = -_u[idx * k + i];
            }
          }
          _s[idx * k + i] = (_s[idx * k + i] % (p - 1)) + 1;  //[1, p-1]
          _c[idx * k + i] = (_s[idx * k + i] * _c[idx * k + i]) % p;
        }
      });  // end foreach

      comm->sendAsync(2, c, "d");  // 1 latency, 2 * logp * k
      // Private Compare end

      auto eta_ = comm->recv(2, ty, "eta_");
      eta_ = eta_.reshape(a.shape());
      NdArrayView<U> _eta_(eta_);

      NdArrayRef eta(ty, a.shape());
      NdArrayRef theta(ty, a.shape());
      NdArrayView<U> _eta(eta);
      NdArrayView<U> _theta(theta);

      pforeach(0, size, [&](int64_t idx) {
        // eta = eta_ + (1 - rank) * eta__ - 2 * eta__ * eta_  mod L_1
        if (_eta__[idx] == 0) _eta[idx] = _eta_[idx];
        if (_eta__[idx] == 1) {
          if (_eta_[idx] == 0)
            _eta[idx] = (1 - rank);
          else
            _eta[idx] = L_1 - _eta_[idx] + (1 - rank);
        }

        // theta = beta + (1 - rank) * ( - alpha - 1) + delta + eta mod L_1
        _theta[idx] = _delta[idx] + _eta[idx] + _beta[idx];
        if (_theta[idx] < _delta[idx]) _theta[idx] += (U)1;  // when overflow
        auto tmp = _theta[idx];
        _theta[idx] += (1 - rank) * (-_alpha[idx] - 1);
        if (_theta[idx] > tmp) _theta[idx] -= (U)1;  // when overflow

        _res[idx] = _a[idx] - _theta[idx];
        if (_a[idx] < _theta[idx]) _res[idx] -= (U)1;

        // share of 0
        if (rank == 0) {
          _res[idx] += _zero_1[idx];
          if (_res[idx] < _zero_1[idx]) _res[idx] += (U)1;
        }
        if (rank == 1) {
          tmp = _res[idx];
          _res[idx] -= _zero_0[idx];
          if (tmp < _zero_0[idx]) _res[idx] -= (U)1;
        }
      });

    }  // P0 and P1 end execute

    if (rank == 2) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<U> dis(0, L_1 - 1);

      auto a_0 = comm->recv(0, ty, "a_");
      auto a_1 = comm->recv(1, ty, "a_");
      a_0 = a_0.reshape(a.shape());
      a_1 = a_1.reshape(a.shape());
      auto x = ring_add(a_0, a_1);

      NdArrayView<U> _a_0(a_0);
      NdArrayView<U> _x(x);

      auto delta = ring_zeros(field, a.shape());
      NdArrayView<U> _delta(delta);

      // delta = wrap(a_0, a_1, 2^k)
      pforeach(0, size, [&](int64_t idx) {
        if (_x[idx] < _a_0[idx]) _delta[idx] = (U)1;
      });

      NdArrayRef dp_x_p0(ty, {size * k});
      NdArrayRef dp_x_p1(ty, {size * k});
      NdArrayView<U> _dp_x_p0(dp_x_p0);
      NdArrayView<U> _dp_x_p1(dp_x_p1);

      NdArrayRef delta_p0(ty, a.shape());
      NdArrayRef delta_p1(ty, a.shape());
      NdArrayView<U> _delta_p0(delta_p0);
      NdArrayView<U> _delta_p1(delta_p1);

      pforeach(0, size, [&](int64_t idx) {
        auto dp_x = bitDecompose(_x[idx], k);  // vector<uint8_t>

        // split bitDecompose(x) into dp_x_p0 and dp_x_p1

        auto rand_Zp = ring_rand_range(field, {k}, 0, p - 1);
        NdArrayView<U> _rand_Zp(rand_Zp);
        for (int64_t bit = 0; bit < k; bit++) {
          _dp_x_p0[idx * k + bit] = (_rand_Zp[bit]);
          _dp_x_p1[idx * k + bit] =
              (U)(dp_x[bit] + p - _dp_x_p0[idx * k + bit]);
        }

        // split delta in Z_(L_1)
        _delta_p0[idx] = dis(gen);
        _delta_p1[idx] = _delta[idx] - _delta_p0[idx];
        if (_delta[idx] < _delta_p0[idx])
          _delta_p1[idx] -= (U)1;  // when overflow
      });                          // end foreach

      // 1 latency, 2 * k + 2 * k * logp
      comm->sendAsync(0, dp_x_p0, "dp_x");
      comm->sendAsync(1, dp_x_p1, "dp_x");
      comm->sendAsync(0, delta_p0, "delta");
      comm->sendAsync(1, delta_p1, "delta");

      // split eta_ in Z_(L_1)
      NdArrayRef eta_p0(ty, a.shape());
      NdArrayRef eta_p1(ty, a.shape());
      NdArrayView<U> _eta_p0(eta_p0);
      NdArrayView<U> _eta_p1(eta_p1);

      // Private Compare
      auto d0 = comm->recv(0, ty, "d");
      auto d1 = comm->recv(1, ty, "d");
      d0 = d0.reshape({size * k});
      d1 = d1.reshape({size * k});
      NdArrayView<U> _d0(d0);
      NdArrayView<U> _d1(d1);

      auto eta_ = ring_zeros(field, a.shape());
      NdArrayView<U> _eta_(eta_);
      NdArrayRef d(ty, {size * k});
      NdArrayView<U> _d(d);
      pforeach(0, size, [&](int64_t idx) {
        for (int64_t i = 0; i < k; i++) {
          _d[idx * k + i] = (_d0[idx * k + i] + _d1[idx * k + i]) % p;
          if (_d[idx * k + i] == 0) {
            _eta_[idx] = U(1);
            break;
          }
        }

        // split eta_ in Z_(L_1)
        _eta_p0[idx] = dis(gen);
        _eta_p1[idx] = _eta_[idx] - _eta_p0[idx];
        if (_eta_[idx] < _eta_p0[idx]) _eta_p1[idx] -= (U)1;  // when overflow
      });                                                     // end pforeach

      // Private Compare end

      // 1 latency, 2 * k
      comm->sendAsync(0, eta_p0, "eta_");
      comm->sendAsync(1, eta_p1, "eta_");
    }  // P2 end execute
  });

  return res;
}

NdArrayRef Msb::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = in.eltype().as<AShrTy>()->field();
  const int64_t k = SizeOf(field) * 8;
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  const int64_t size = in.numel();
  const int p = 131;
  const auto log_p = 9;
  const auto ty = makeType<AShrTy>(field);
  const auto one = ring_ones(field, in.shape());
  auto res = ring_zeros(field, in.shape()).as(makeType<AShrTy>(field));

  const auto kComm = in.elsize() * size;
  comm->addCommStatsManually(5, 13 * kComm + 4 * kComm * log_p);

  DISPATCH_ALL_FIELDS(field, "securenn.msb", [&]() {
    using U = ring2k_t;
    const U L_1 = (U)(~0);

    NdArrayRef gamma(ty, in.shape());
    NdArrayRef delta(ty, in.shape());
    // P0 and P1 hold beta by PRF
    auto [beta0, beta1] = prg_state->genPrssPair(field, in.shape());
    auto [s_r0, s_r1] = prg_state->genPrssPair(field, {size * k});
    auto [u_r0, u_r1] = prg_state->genPrssPair(field, {size * k});
    if (rank == 2) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<U> dis(0, L_1 - 1);

      // random for beaver
      // P2 generate a0, a1, b0, b1, c0 by PRF
      // and calculate c1
      auto [a1, a0] = prg_state->genPrssPair(field, in.shape());
      auto [b1, b0] = prg_state->genPrssPair(field, in.shape());
      auto c0 = prg_state->genPrssPair(field, in.shape(), true, false).second;
      // c1 = (a0 + a1) * (b0 + b1) - c0
      auto c1 = ring_sub(ring_mul(ring_add(a0, a1), ring_add(b0, b1)), c0);
      // end beaver  (c1 will be sent with x to reduce one round latency)

      NdArrayRef x(ty, in.shape());
      NdArrayView<U> _x(x);

      // split x into x_p0 and x_p1 in Z_(L-1), (L=2^k)

      NdArrayRef x_p0(ty, in.shape());
      NdArrayRef x_p1(ty, in.shape());
      NdArrayView<U> _x_p0(x_p0);
      NdArrayView<U> _x_p1(x_p1);

      // split bitDecompose(x) into dp_x_p0 and dp_x_p1 (vector<vector<size_t>>)
      NdArrayRef dp_x_p0(ty, {size * k});
      NdArrayRef dp_x_p1(ty, {size * k});
      NdArrayView<U> _dp_x_p0(dp_x_p0);
      NdArrayView<U> _dp_x_p1(dp_x_p1);

      // split lsb(x)
      // when you add / sub in ring2k_t,the overflow part will be thrown away,
      // which equivalents to mod 2^k, when you want to mod 2^k - 1:
      // add : if overflow : res = res + 1
      // sub : if overflow : res = res - 1
      NdArrayRef lsb_x(ty, in.shape());
      NdArrayView<U> _lsb_x(lsb_x);
      pforeach(0, size, [&](int64_t idx) {
        _x[idx] = dis(gen);
        auto dp_x = bitDecompose(_x[idx], k);  // vector<uint8_t>

        // split x
        _x_p0[idx] = dis(gen);
        _x_p1[idx] = _x[idx] - _x_p0[idx];
        if (_x[idx] < _x_p0[idx]) _x_p1[idx] -= (U)1;  // when overflow

        // split each bit of x
        auto rand_Zp = ring_rand_range(field, {k}, 0, p - 1);
        NdArrayView<U> _rand_Zp(rand_Zp);
        for (int64_t bit = 0; bit < k; bit++) {
          _dp_x_p0[idx * k + bit] = (_rand_Zp[bit]);
          _dp_x_p1[idx * k + bit] =
              (U)(dp_x[bit] + p - _dp_x_p0[idx * k + bit]);
        }

        // split lsb(x)
        _lsb_x[idx] = static_cast<U>(dp_x[0]);
      });  // end foreach
      auto lsb_x_split = ring_rand_additive_splits(lsb_x, 2);

      // 1 latency
      comm->sendAsync(1, c1, "beaver_c");   // k
      comm->sendAsync(0, x_p0, "x");        // k
      comm->sendAsync(1, x_p1, "x");        // k
      comm->sendAsync(0, dp_x_p0, "dp_x");  // k * log p
      comm->sendAsync(1, dp_x_p1, "dp_x");  // k * log p

      comm->sendAsync(0, lsb_x_split[0], "lsb_x");  // k
      comm->sendAsync(1, lsb_x_split[1], "lsb_x");  // k

      // Private Compare
      auto d0 = comm->recv(0, ty, "d");
      auto d1 = comm->recv(1, ty, "d");
      SPU_ENFORCE(d0.shape() == d1.shape());
      NdArrayView<U> _d0(d0);
      NdArrayView<U> _d1(d1);

      auto beta_ = ring_zeros(field, in.shape());
      NdArrayView<U> _beta_(beta_);
      NdArrayRef d(ty, {size * k});
      NdArrayView<U> _d(d);
      pforeach(0, size, [&](int64_t idx) {
        for (int64_t i = 0; i < k; i++) {
          _d[idx * k + i] = (_d0[idx * k + i] + _d1[idx * k + i]) % p;
          if (_d[idx * k + i] == 0) {
            _beta_[idx] = U(1);
            break;
          }
        }
      });  // end pforeach
      // Private Compare end

      // split beta_ into beta_0 and beta_1
      auto beta_split = ring_rand_additive_splits(beta_, 2);

      // 1 latency
      comm->sendAsync(0, beta_split[0].as(ty), "beta_");  // k
      comm->sendAsync(1, beta_split[1].as(ty), "beta_");  // k

    }  // P2 execute end

    if (rank <= 1) {
      // random for beaver
      NdArrayRef beaver_a(ty, in.shape());
      NdArrayRef beaver_b(ty, in.shape());
      NdArrayRef beaver_c(ty, in.shape());
      if (rank == 0) {
        beaver_a = prg_state->genPrssPair(field, in.shape(), false, true).first;
        beaver_b = prg_state->genPrssPair(field, in.shape(), false, true).first;
        beaver_c = prg_state->genPrssPair(field, in.shape(), false, true).first;
      }
      if (rank == 1) {
        beaver_a =
            prg_state->genPrssPair(field, in.shape(), true, false).second;
        beaver_b =
            prg_state->genPrssPair(field, in.shape(), true, false).second;
        prg_state->genPrssPair(field, in.shape(), true, true);
        beaver_c = comm->recv(2, ty, "beaver_c");
        beaver_c = beaver_c.reshape(in.shape());
      }
      // end beaver

      auto x = comm->recv(2, ty, "x");
      auto dp_x = comm->recv(2, ty, "dp_x");
      auto lsb_x = comm->recv(2, ty, "lsb_x");
      x = x.reshape(in.shape());
      dp_x = dp_x.reshape({size * k});
      lsb_x = lsb_x.reshape(in.shape());

      NdArrayRef y(ty, in.shape());
      NdArrayRef r1(ty, in.shape());
      NdArrayRef r(ty, in.shape());
      NdArrayRef lsb_r(makeType<RingTy>(field), in.shape());
      NdArrayView<U> _y(y);
      NdArrayView<U> _r1(r1);
      NdArrayView<U> _r(r);
      NdArrayView<U> _a(in);
      NdArrayView<U> _x(x);
      NdArrayView<U> _lsb_r(lsb_r);
      NdArrayView<U> _dp_x(dp_x);

      for (int64_t i = 0; i < size; i++) {
        _y[i] = _a[i] * 2;
        if (_y[i] < _a[i]) _y[i] += (U)1;
        _r1[i] = _y[i] + _x[i];
        if (_r1[i] < _y[i]) _r1[i] += (U)1;
      }

      // P0 and P1 reconstruct r
      // 1 latency, 2 * k
      comm->sendAsync((rank + 1) % 2, r1, "r1");
      auto r2 = comm->recv((rank + 1) % 2, ty, "r1");
      r2 = r2.reshape(in.shape());
      NdArrayView<U> _r2(r2);
      for (int64_t i = 0; i < size; i++) {
        _r[i] = _r1[i] + _r2[i];
        if (_r[i] < _r1[i]) _r[i] += (U)1;
      }

      // P0 and P1 hold beta by PRF
      NdArrayRef beta(makeType<RingTy>(field), in.shape());
      if (rank == 0) beta = beta1;
      if (rank == 1) beta = beta0;

      NdArrayView<U> _beta(beta);
      for (int64_t i = 0; i < size; i++) {
        _beta[i] = _beta[i] & 0x1;
      }

      // Private Compare
      auto t = ring_add(r, one);
      NdArrayView<U> _t(t);

      NdArrayRef u(ty, {size * k});
      NdArrayRef s(ty, {size * k});
      if (rank == 0) {
        u = u_r1;
        s = s_r1;
      }
      if (rank == 1) {
        u = u_r0;
        s = s_r0;
      }
      NdArrayView<U> _u(u);
      NdArrayView<U> _s(s);

      NdArrayRef c(ty, {size * k});
      NdArrayView<U> _c(c);

      size_t w;
      size_t w_total;

      pforeach(0, in.numel(), [&](int64_t idx) {
        auto r_bits = bitDecompose(_r[idx], k);
        auto t_bits = bitDecompose(_t[idx], k);
        _lsb_r[idx] = static_cast<U>(r_bits[0]);
        w_total = 0;
        for (int i = (int)(k - 1); i >= 0; i--) {
          if (_beta[idx] == 0) {
            w = (p + _dp_x[idx * k + i] + rank * r_bits[i] -
                 2 * r_bits[i] * _dp_x[idx * k + i]) %
                p;
            _c[idx * k + i] =
                (p + rank * r_bits[i] - _dp_x[idx * k + i] + rank + w_total) %
                p;
            w_total = (w_total + w) % p;
          } else if (_beta[idx] == 1 && _r[idx] != L_1) {
            w = (p + _dp_x[idx * k + i] + rank * t_bits[i] -
                 2 * t_bits[i] * _dp_x[idx * k + i]) %
                p;
            _c[idx * k + i] =
                (p - rank * t_bits[i] + _dp_x[idx * k + i] + rank + w_total) %
                p;
            w_total = (w_total + w) % p;
          } else {
            // r = 2 ^ k - 1 bigger than everything else in the ring
            // c = [0, 1,..., 1]
            if (i != 1) {
              _u[idx] = _u[idx] % p;
              _c[idx * k + i] =
                  (1 - rank) * (_u[idx * k + i] + 1) - rank * _u[idx * k + i];
            } else {
              _u[idx] = _u[idx] % p;
              if (rank == 0) _c[idx * k + i] = _u[idx * k + i];
              if (rank == 1) _c[idx * k + i] = -_u[idx * k + i];
            }
          }
          _s[idx * k + i] = (_s[idx * k + i] % (p - 1)) + 1;  //[1, p-1]
          _c[idx * k + i] = (_s[idx * k + i] * _c[idx * k + i]) % p;
        }
      });  // end foreach

      // 1 latency, 2 * log p * k
      comm->sendAsync(2, c, "d");
      // Private Compare end

      auto beta_ = comm->recv(2, ty, "beta_");
      beta_ = beta_.reshape(in.shape());

      // gamma = beta_ + rank * beta - 2 * beta * beta_
      // delta = lsb(x) + rank * lsb(r) - 2 * lsb(x) * lsb(r)
      gamma = ring_sub(ring_sub(beta_, ring_mul(beta, beta_)),
                       ring_mul(beta, beta_));
      delta = ring_sub(ring_sub(lsb_x, ring_mul(lsb_x, lsb_r)),
                       ring_mul(lsb_x, lsb_r));
      if (rank == 1) {
        gamma = ring_add(gamma, beta);
        delta = ring_add(delta, lsb_r);
      }

      // mulaa start  theta = gamma * delta
      // Open x-a & y-b
      auto send_gamma_a = ring_sub(gamma, beaver_a).as(ty);
      auto send_delta_b = ring_sub(delta, beaver_b).as(ty);
      // 1 latency, 2 * 2k
      comm->sendAsync((rank + 1) % 2, send_gamma_a, "gamma_a");
      comm->sendAsync((rank + 1) % 2, send_delta_b, "delta_b");
      auto recv_gamma_a = comm->recv((rank + 1) % 2, ty, "gamma_a");
      auto recv_delta_b = comm->recv((rank + 1) % 2, ty, "delta_b");
      recv_gamma_a = recv_gamma_a.reshape(in.shape());
      recv_delta_b = recv_delta_b.reshape(in.shape());
      auto gamma_a = ring_add(send_gamma_a, recv_gamma_a);
      auto delta_b = ring_add(send_delta_b, recv_delta_b);

      // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
      auto theta = ring_add(
          ring_add(ring_mul(gamma_a, beaver_b), ring_mul(delta_b, beaver_a)),
          beaver_c);
      if (rank == 0)
        // z += (X-A) * (Y-B);
        theta = ring_add(theta, ring_mul(gamma_a, delta_b));
      // mulaa end

      res = ring_sub(ring_sub(ring_add(gamma, delta), theta), theta);

    }  // P0 and P1 execute end
  });

  // P0 and P1 add the share of zero
  // P0.zero_1 = P1.zero_0
  auto [zero_0, zero_1] = prg_state->genPrssPair(field, in.shape());
  if (rank == 0) {
    res = ring_sub(res, zero_1);
  }
  if (rank == 1) {
    res = ring_add(res, zero_0);
  }
  return res;
}

NdArrayRef Msb_opt::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = in.eltype().as<AShrTy>()->field();
  const int64_t k = SizeOf(field) * 8;
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  const int64_t size = in.numel();
  const int p = 131;
  const auto log_p = 9;
  const auto ty = makeType<AShrTy>(field);
  const auto one = ring_ones(field, in.shape());
  auto res = ring_zeros(field, in.shape()).as(makeType<AShrTy>(field));

  const auto kComm = in.elsize() * size;
  comm->addCommStatsManually(5, 9 * kComm + 3 * kComm * log_p);

  DISPATCH_ALL_FIELDS(field, "securenn.msb", [&]() {
    using U = ring2k_t;
    const U L_1 = (U)(~0);

    NdArrayRef gamma(ty, in.shape());
    NdArrayRef delta(ty, in.shape());
    // P0 and P1 hold beta by PRF
    auto [beta0, beta1] = prg_state->genPrssPair(field, in.shape());
    // random for pc
    auto [s_r0, s_r1] = prg_state->genPrssPair(field, {size * k});
    auto [u_r0, u_r1] = prg_state->genPrssPair(field, {size * k});
    // using PRF for reduce some comm
    auto [prf_x0, prf_x1] = prg_state->genPrssPair(field, in.shape());
    auto [prf_dpx0, prf_dpx1] = prg_state->genPrssPair(field, {size * k});
    auto [prf_lsbx0, prf_lsbx1] = prg_state->genPrssPair(field, in.shape());
    auto [beta_0, beta_1] = prg_state->genPrssPair(field, in.shape());
    if (rank == 2) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<U> dis(0, L_1 - 1);

      // random for beaver
      // P2 generate a0, a1, b0, b1, c0 by PRF
      // and calculate c1
      auto [a1, a0] = prg_state->genPrssPair(field, in.shape());
      auto [b1, b0] = prg_state->genPrssPair(field, in.shape());
      auto c0 = prg_state->genPrssPair(field, in.shape(), true, false).second;
      // c1 = (a0 + a1) * (b0 + b1) - c0
      auto c1 = ring_sub(ring_mul(ring_add(a0, a1), ring_add(b0, b1)), c0);
      // end beaver  (c1 will be sent with x to reduce one round latency)

      NdArrayRef x(ty, in.shape());
      NdArrayView<U> _x(x);

      // split x into x_p0 and x_p1 in Z_(L-1), (L=2^k)

      auto x_p0 = prf_x0;
      auto x_p1 = prf_x1;
      NdArrayView<U> _x_p0(x_p0);
      NdArrayView<U> _x_p1(x_p1);

      // split bitDecompose(x) into dp_x_p0 and dp_x_p1 (vector<vector<size_t>>)
      auto dp_x_p0 = prf_dpx1;
      NdArrayRef dp_x_p1(ty, {size * k});
      NdArrayView<U> _dp_x_p0(dp_x_p0);
      NdArrayView<U> _dp_x_p1(dp_x_p1);

      // split lsb(x)
      // when you add / sub in ring2k_t,the overflow part will be thrown away,
      // which equivalents to mod 2^k, when you want to mod 2^k - 1:
      // add : if overflow : res = res + 1
      // sub : if overflow : res = res - 1
      NdArrayRef lsb_x(ty, in.shape());
      NdArrayView<U> _lsb_x(lsb_x);
      pforeach(0, size, [&](int64_t idx) {
        // reconstruct x
        if (_x_p0[idx] == L_1) _x_p0[idx] = (U)0;
        if (_x_p1[idx] == L_1) _x_p1[idx] = (U)0;
        _x[idx] = _x_p0[idx] + _x_p1[idx];
        if (_x[idx] < _x_p0[idx]) _x[idx] += (U)1;  // when overflow

        // split each bit of x
        auto dp_x = bitDecompose(_x[idx], k);  // vector<uint8_t>

        for (int64_t bit = 0; bit < k; bit++) {
          _dp_x_p0[idx * k + bit] = _dp_x_p0[idx * k + bit] % p;
          _dp_x_p1[idx * k + bit] =
              (U)(dp_x[bit] + p - _dp_x_p0[idx * k + bit]);
        }

        // split lsb(x)
        _lsb_x[idx] = static_cast<U>(dp_x[0]);
      });  // end foreach
      auto lsb_x0 = prf_lsbx1;
      auto lsb_x1 = ring_sub(lsb_x, lsb_x0);

      // 1 latency
      comm->sendAsync(1, c1, "beaver_c");   // k
      comm->sendAsync(1, dp_x_p1, "dp_x");  // k * log p

      comm->sendAsync(1, lsb_x1, "lsb_x");  // k

      // Private Compare
      auto d0 = comm->recv(0, ty, "d");
      auto d1 = comm->recv(1, ty, "d");
      d0 = d0.reshape({size * k});
      d1 = d1.reshape({size * k});
      NdArrayView<U> _d0(d0);
      NdArrayView<U> _d1(d1);

      auto beta_ = ring_zeros(field, in.shape());
      NdArrayView<U> _beta_(beta_);
      NdArrayRef d(ty, {size * k});
      NdArrayView<U> _d(d);
      pforeach(0, size, [&](int64_t idx) {
        for (int64_t i = 0; i < k; i++) {
          _d[idx * k + i] = (_d0[idx * k + i] + _d1[idx * k + i]) % p;
          if (_d[idx * k + i] == 0) {
            _beta_[idx] = U(1);
            break;
          }
        }
      });  // end pforeach
      // Private Compare end

      // split beta_ into beta_0 and beta_1
      // beta_x0 = beta_1;
      auto beta_x1 = ring_sub(beta_, beta_1);

      // 1 latency
      comm->sendAsync(1, beta_x1.as(ty), "beta_");  // k

    }  // P2 execute end

    if (rank <= 1) {
      // random for beaver
      NdArrayRef beaver_a(ty, in.shape());
      NdArrayRef beaver_b(ty, in.shape());
      NdArrayRef beaver_c(ty, in.shape());
      if (rank == 0) {
        beaver_a = prg_state->genPrssPair(field, in.shape(), false, true).first;
        beaver_b = prg_state->genPrssPair(field, in.shape(), false, true).first;
        beaver_c = prg_state->genPrssPair(field, in.shape(), false, true).first;
      }
      if (rank == 1) {
        beaver_a =
            prg_state->genPrssPair(field, in.shape(), true, false).second;
        beaver_b =
            prg_state->genPrssPair(field, in.shape(), true, false).second;
        prg_state->genPrssPair(field, in.shape(), true, true);
        beaver_c = comm->recv(2, ty, "beaver_c");
        beaver_c = beaver_c.reshape(in.shape());
      }
      // end beaver

      NdArrayRef x(ty, in.shape());
      if (rank == 0) x = prf_x0;
      if (rank == 1) x = prf_x1;

      NdArrayRef dp_x(ty, {size * k});
      if (rank == 1) {
        dp_x = comm->recv(2, ty, "dp_x");
        dp_x = dp_x.reshape({size * k});
      }
      if (rank == 0) dp_x = prf_dpx0;
      NdArrayView<U> _dp_x(dp_x);

      NdArrayRef lsb_x(ty, in.shape());
      if (rank == 0) lsb_x = prf_lsbx0;
      if (rank == 1) {
        lsb_x = comm->recv(2, ty, "lsb_x");
        lsb_x = lsb_x.reshape(in.shape());
      }

      NdArrayRef y(ty, in.shape());
      NdArrayRef r1(ty, in.shape());
      NdArrayRef r(ty, in.shape());
      NdArrayRef lsb_r(makeType<RingTy>(field), in.shape());
      NdArrayView<U> _y(y);
      NdArrayView<U> _r1(r1);
      NdArrayView<U> _r(r);
      NdArrayView<U> _a(in);
      NdArrayView<U> _x(x);
      NdArrayView<U> _lsb_r(lsb_r);

      for (int64_t i = 0; i < size; i++) {
        _y[i] = _a[i] * 2;
        if (_y[i] < _a[i]) _y[i] += (U)1;
        if (_x[i] == L_1) _x[i] = (U)0;
        _r1[i] = _y[i] + _x[i];
        if (_r1[i] < _y[i]) _r1[i] += (U)1;
      }

      // P0 and P1 reconstruct r
      // 1 latency, 2 * k
      comm->sendAsync((rank + 1) % 2, r1, "r1");
      auto r2 = comm->recv((rank + 1) % 2, ty, "r1");
      r2 = r2.reshape(in.shape());
      NdArrayView<U> _r2(r2);
      for (int64_t i = 0; i < size; i++) {
        _r[i] = _r1[i] + _r2[i];
        if (_r[i] < _r1[i]) _r[i] += (U)1;
      }

      // P0 and P1 hold beta by PRF
      NdArrayRef beta(makeType<RingTy>(field), in.shape());
      if (rank == 0) beta = beta1;
      if (rank == 1) beta = beta0;

      NdArrayView<U> _beta(beta);
      for (int64_t i = 0; i < size; i++) {
        _beta[i] = _beta[i] & 0x1;
      }

      // Private Compare
      auto t = ring_add(r, one);
      NdArrayView<U> _t(t);

      NdArrayRef u(ty, {size * k});
      NdArrayRef s(ty, {size * k});
      if (rank == 0) {
        u = u_r1;
        s = s_r1;
      }
      if (rank == 1) {
        u = u_r0;
        s = s_r0;
      }
      NdArrayView<U> _u(u);
      NdArrayView<U> _s(s);

      NdArrayRef c(ty, {size * k});
      NdArrayView<U> _c(c);

      size_t w;
      size_t w_total;

      pforeach(0, in.numel(), [&](int64_t idx) {
        auto r_bits = bitDecompose(_r[idx], k);
        auto t_bits = bitDecompose(_t[idx], k);
        _lsb_r[idx] = static_cast<U>(r_bits[0]);
        w_total = 0;
        for (int i = (int)(k - 1); i >= 0; i--) {
          if (rank == 0) _dp_x[idx * k + i] = _dp_x[idx * k + i] % p;
          if (_beta[idx] == 0) {
            w = (p + _dp_x[idx * k + i] + rank * r_bits[i] -
                 2 * r_bits[i] * _dp_x[idx * k + i]) %
                p;
            _c[idx * k + i] =
                (p + rank * r_bits[i] - _dp_x[idx * k + i] + rank + w_total) %
                p;
            w_total = (w_total + w) % p;
          } else if (_beta[idx] == 1 && _r[idx] != L_1) {
            w = (p + _dp_x[idx * k + i] + rank * t_bits[i] -
                 2 * t_bits[i] * _dp_x[idx * k + i]) %
                p;
            _c[idx * k + i] =
                (p - rank * t_bits[i] + _dp_x[idx * k + i] + rank + w_total) %
                p;
            w_total = (w_total + w) % p;
          } else {
            // r = 2 ^ k - 1 bigger than everything else in the ring
            // c = [0, 1,..., 1]
            if (i != 1) {
              _u[idx] = _u[idx] % p;
              _c[idx * k + i] =
                  (1 - rank) * (_u[idx * k + i] + 1) - rank * _u[idx * k + i];
            } else {
              _u[idx] = _u[idx] % p;
              if (rank == 0) _c[idx * k + i] = _u[idx * k + i];
              if (rank == 1) _c[idx * k + i] = -_u[idx * k + i];
            }
          }
          _s[idx * k + i] = (_s[idx * k + i] % (p - 1)) + 1;  //[1, p-1]
          _c[idx * k + i] = (_s[idx * k + i] * _c[idx * k + i]) % p;
        }
      });  // end foreach

      // 1 latency, 2 * log p * k
      comm->sendAsync(2, c, "d");
      // Private Compare end

      NdArrayRef beta_(ty, in.shape());
      if (rank == 0) beta_ = beta_0;
      if (rank == 1) {
        beta_ = comm->recv(2, ty, "beta_");
        beta_ = beta_.reshape(in.shape());
      }

      // gamma = beta_ + rank * beta - 2 * beta * beta_
      // delta = lsb(x) + rank * lsb(r) - 2 * lsb(x) * lsb(r)
      gamma = ring_sub(ring_sub(beta_, ring_mul(beta, beta_)),
                       ring_mul(beta, beta_));
      delta = ring_sub(ring_sub(lsb_x, ring_mul(lsb_x, lsb_r)),
                       ring_mul(lsb_x, lsb_r));
      if (rank == 1) {
        gamma = ring_add(gamma, beta);
        delta = ring_add(delta, lsb_r);
      }

      // mulaa start  theta = gamma * delta
      // Open x-a & y-b
      auto send_gamma_a = ring_sub(gamma, beaver_a).as(ty);
      auto send_delta_b = ring_sub(delta, beaver_b).as(ty);
      // 1 latency, 2 * 2k
      comm->sendAsync((rank + 1) % 2, send_gamma_a, "gamma_a");
      comm->sendAsync((rank + 1) % 2, send_delta_b, "delta_b");
      auto recv_gamma_a = comm->recv((rank + 1) % 2, ty, "gamma_a");
      auto recv_delta_b = comm->recv((rank + 1) % 2, ty, "delta_b");
      recv_gamma_a = recv_gamma_a.reshape(in.shape());
      recv_delta_b = recv_delta_b.reshape(in.shape());
      auto gamma_a = ring_add(send_gamma_a, recv_gamma_a);
      auto delta_b = ring_add(send_delta_b, recv_delta_b);

      // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
      auto theta = ring_add(
          ring_add(ring_mul(gamma_a, beaver_b), ring_mul(delta_b, beaver_a)),
          beaver_c);
      if (rank == 0)
        // z += (X-A) * (Y-B);
        theta = ring_add(theta, ring_mul(gamma_a, delta_b));
      // mulaa end

      res = ring_sub(ring_sub(ring_add(gamma, delta), theta), theta);

    }  // P0 and P1 execute end
  });

  // P0 and P1 add the share of zero
  // P0.zero_1 = P1.zero_0
  auto [zero_0, zero_1] = prg_state->genPrssPair(field, in.shape());
  if (rank == 0) {
    res = ring_sub(res, zero_1);
  }
  if (rank == 1) {
    res = ring_add(res, zero_0);
  }
  return res;
}

}  // namespace spu::mpc::securenn
