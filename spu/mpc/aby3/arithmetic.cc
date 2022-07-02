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

#include "spu/mpc/aby3/arithmetic.h"

#include <future>

#include "spdlog/spdlog.h"

#include "spu/core/profile.h"
#include "spu/mpc/aby3/type.h"
#include "spu/mpc/aby3/value.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::aby3 {

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();
  const auto field = in.eltype().as<Ring2k>()->field();

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  const auto& x3 = comm->rotate(x2, kBindName);  // comm => 1, k

  // ret
  auto z = ring_add(ring_add(x1, x2), x3);
  auto ty = makeType<Pub2kTy>(field);

  return z.as(ty);
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto numel = in.numel();

#ifdef ENABLE_MASK_DURING_ABY3_P2A
  auto [r1, r2] =
      ctx->caller()->getState<PrgState>()->genPrssPair(field, numel);

  auto x1 = ring_sub(r1, r2);

  if (comm->getRank() == 0) {
    ring_add_(x1, in);
  }

  const auto& x2 = comm->rotate(x1, kBindName);

  return makeAShare(x1, x2, field);
#else
  const auto& zeros = ring_zeros(field, numel);

  // ArrayRef& in is ``public''
  if (comm->getRank() == 0) {
    return makeAShare(in, zeros, field);
  } else if (comm->getRank() == 2) {
    return makeAShare(zeros, in, field);
  } else {
    return makeAShare(zeros, zeros, field);
  }
#endif
}

ArrayRef NotA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  // compute n_x
  auto n_x1 = ring_neg(x1);
  auto n_x2 = ring_neg(x2);

  // add public M-1
  auto* comm = ctx->caller()->getState<Communicator>();
  const auto ones = ring_not(ring_zeros(field, in.numel()));
  if (comm->getRank() == 0) {
    ring_add_(n_x2, ones);
  } else if (comm->getRank() == 1) {
    ring_add_(n_x1, ones);
  }

  return makeAShare(n_x1, n_x2, field);
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();

  // lhs
  const auto& x1 = getFirstShare(lhs);
  const auto& x2 = getSecondShare(lhs);

  // remember that rhs is public
  if (comm->getRank() == 0) {
    return makeAShare(x1, ring_add(x2, rhs), field);
  } else if (comm->getRank() == 1) {
    return makeAShare(ring_add(x1, rhs), x2, field);
  } else {
    return lhs;
  }
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x1 = getFirstShare(lhs);
  const auto& x2 = getSecondShare(lhs);

  // rhs
  const auto& y1 = getFirstShare(rhs);
  const auto& y2 = getSecondShare(rhs);

  // ret
  const auto& z1 = ring_add(x1, y1);
  const auto& z2 = ring_add(x2, y2);

  return makeAShare(z1, z2, field);
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x1 = getFirstShare(lhs);
  const auto& x2 = getSecondShare(lhs);

  // ret
  const auto& z1 = ring_mul(x1, rhs);
  const auto& z2 = ring_mul(x2, rhs);

  return makeAShare(z1, z2, field);
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  auto r_future = std::async([&] {
    auto [r0, r1] = prg_state->genPrssPair(field, lhs.numel());
    return ring_sub(r0, r1);
  });

  // lhs
  const auto& x1 = getFirstShare(lhs);
  const auto& x2 = getSecondShare(lhs);

  // rhs
  const auto& y1 = getFirstShare(rhs);
  const auto& y2 = getSecondShare(rhs);

  // z = x1*y1 + x1*y2 + x2*y1 + r
  auto z1 = ring_sum({ring_mul(x1, y1), ring_mul(x1, y2), ring_mul(x2, y1)});
  ring_add_(z1, r_future.get());

  const auto& z2 = comm->rotate(z1, kBindName);  // comm => 1, k

  return makeAShare(z1, z2, field);
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto z1 = ring_mmul(x1, y, M, N, K);
  auto z2 = ring_mmul(x2, y, M, N, K);

  return makeAShare(z1, z2, field);
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  auto r_future = std::async([&] {
    auto [r0, r1] = prg_state->genPrssPair(field, M * N);
    return ring_sub(r0, r1);
  });

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto y1 = getFirstShare(y);
  auto y2 = getSecondShare(y);

  // z1 := x1*y1 + x1*y2 + x2*y1 + k1
  // z2 := x2*y2 + x2*y3 + x3*y2 + k2
  // z3 := x3*y3 + x3*y1 + x1*y3 + k3
  auto z1 = ring_sum({ring_mmul(x1, y1, M, N, K),  //
                      ring_mmul(x1, y2, M, N, K),  //
                      ring_mmul(x2, y1, M, N, K)});
  ring_add_(z1, r_future.get());

  auto z2 = comm->rotate(z1, kBindName);  // comm => 1, k

  return makeAShare(z1, z2, field);
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  // ret
  const auto z1 = ring_lshift(x1, bits);
  const auto z2 = ring_lshift(x2, bits);

  return makeAShare(z1, z2, field);
}

// Refer to:
// Share Truncation I, 5.1 Fixed-point Arithmetic, P13,
// ABY3: A Mixed Protocol Framework for Machine Learning
// - https://eprint.iacr.org/2018/403.pdf
ArrayRef TruncPrA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto* comm = ctx->caller()->getState<Communicator>();

  auto r_future =
      std::async([&] { return prg_state->genPrssPair(field, in.numel()); });

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  const auto kComm = x1.elsize() * x1.numel();

  // we only record the maximum communication, we need to manually add comm
  comm->addCommStatsManually(1, kComm);  // comm => 1, 2

  // ret
  switch (comm->getRank()) {
    case 0: {
      const auto z1 = ring_arshift(x1, bits);
      const auto z2 = comm->recv(1, x1.eltype(), kBindName);
      return makeAShare(z1, z2, field);
    }

    case 1: {
      auto r1 = r_future.get().second;
      const auto z1 = ring_sub(ring_arshift(ring_add(x1, x2), bits), r1);
      comm->sendAsync(0, z1, kBindName);
      return makeAShare(z1, r1, field);
    }

    case 2: {
      const auto z2 = ring_arshift(x2, bits);
      return makeAShare(r_future.get().first, z2, field);
    }

    default:
      YASL_THROW("Party number exceeds 3!");
  }
}

// PRECISE VERSION
// Refer to:
// 3.2.2 Truncation by a public value, P10,
// Secure Evaluation of Quantized Neural Networks
// - https://arxiv.org/pdf/1910.12435.pdf
ArrayRef TruncPrAPrecise::proc(KernelEvalContext* ctx, const ArrayRef& in,
                               size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto* comm = ctx->caller()->getState<Communicator>();

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  const auto kComm = x1.elsize() * x1.numel();
  auto numel = in.numel();

  auto open = [&](const ArrayRef& arr) {
    const auto peer_rank = 1 - comm->getRank();
    comm->sendAsync(peer_rank, arr, "_");
    ArrayRef peer_arr = comm->recv(peer_rank, arr.eltype(), "_");
    return ring_add(arr, peer_arr);
  };

  const size_t K = SizeOf(field) * 8;
  switch (comm->getRank()) {
    case 0: {
      auto [ra, _] = prg_state->genPrssPair(field, numel);

      auto cr = comm->recv(2, x1.eltype(), "cr0");
      comm->addCommStatsManually(1, 2 * kComm);

      // let ra = r, rb = r{k-1}, rc = sum(r{i-m}<<(i-m)) for i in range(m, k-2)
      auto rb = cr.slice(0, numel);
      auto rc = cr.slice(numel, 2 * numel);

      // convert to 2-out-of-2 share.
      auto x = ring_add(x1, x2);

      // assume secret x in [-2^(K-2), 2^(K-2)), by adding 2^(K-2)
      // x' = x + 2^(K-2) in [0, 2^(K-1)), with msb(x') == 0
      const auto kOne = ring_ones(field, numel);
      ring_add_(x, ring_lshift(kOne, K - 2));

      // c = open(<x> + <r>)
      auto c = open(ring_add(x, ra));
      comm->addCommStatsManually(1, kComm);

      // c_hat = c/2^m mod 2^(k-m-1) = (c << 1) >> (1+m)
      auto c_hat = ring_rshift(ring_lshift(c, 1), 1 + bits);

      // <b> = <rb> ^ c{k-1} = <rb> + c{k-1} - 2*c{k-1}*<rb>
      // note: <rb> is a randbit (in r^2k)
      const auto ck_1 = ring_rshift(c, K - 1);
      auto b = ring_sub(rb, ring_mul(ring_lshift(ck_1, 1), rb));
      ring_add_(b, ck_1);

      // y_dash = c_hat - <rc> + <b> * 2^(k-m-1)
      auto y_dash = ring_sub(ring_lshift(b, K - 1 - bits), rc);
      ring_add_(y_dash, c_hat);

      // re-encode negative numbers.
      // from https://eprint.iacr.org/2020/338.pdf, section 5.1
      // y' = y - 2^(K-2-m)
      ring_sub_(y_dash, ring_lshift(kOne, K - 2 - bits));

      // reconstruct 2-out-of-3 share.
      auto [y1, __] = prg_state->genPrssPair(field, numel);
      auto y2 = ring_sub(y_dash, y1);
      comm->sendAsync(1, y2, "y2");
      y2 = ring_add(y2, comm->recv(1, y1.eltype(), "y1"));
      comm->addCommStatsManually(1, kComm);
      return makeAShare(y1, y2, field);
    }
    case 1: {
      auto [_, ra] = prg_state->genPrssPair(field, numel);

      auto cr = comm->recv(2, x1.eltype(), "r1");
      comm->addCommStatsManually(1, 2 * kComm);

      // let ra = r, rb = r{k-1}, rc = sum(r{i-m}<<(i-m)) for i in range(m, k-2)
      auto rb = cr.slice(0, numel);
      auto rc = cr.slice(numel, 2 * numel);

      // convert to 2-out-of-2 share
      const auto x = x2;

      // c = open(<x> + <r>)
      auto c = open(ring_add(x, ra));
      comm->addCommStatsManually(1, kComm);

      // c_hat = c/2^m mod 2^(k-m-1) = (c << 1) >> (1+m)
      auto c_hat = ring_rshift(ring_lshift(c, 1), 1 + bits);

      // <b> = <rb> ^ c{k-1} = <rb> + c{k-1} - 2*c{k-1}*<rb>
      // note: <rb> is a randbit (in r^2k)
      const auto ck_1 = ring_rshift(c, K - 1);
      auto b = ring_sub(rb, ring_mul(ring_lshift(ck_1, 1), rb));

      // y_dash = c_hat - <rc> + <b> * 2^(k-m-1)
      auto y_dash = ring_sub(ring_lshift(b, K - 1 - bits), rc);

      // reconstruct 2-out-of-3 share.
      auto [__, y2] = prg_state->genPrssPair(field, numel);
      auto y1 = ring_sub(y_dash, y2);
      comm->sendAsync(0, y1, "y1");
      y1 = ring_add(y1, comm->recv(0, y1.eltype(), "y2"));
      comm->addCommStatsManually(1, kComm);
      return makeAShare(y1, y2, field);
    }
    case 2: {
      auto [r0, r1] = prg_state->genPrssPair(field, numel);

      // r = r0 + r1.
      const auto r = ring_add(r0, r1);
      // rb = r{k-1}
      const auto rb = ring_rshift(r, K - 1);
      // rc = sum(r{i-m} << (i-m)) for i in range(m, k-2)
      //    = (r<<1)>>(m+1)
      const auto rc = ring_rshift(ring_lshift(r, 1), bits + 1);

      // Now P0 knows r0, P1 knows r1, but they do not know rb, rc, so we send
      // rb, rb slice to each other.
      // Optimization point, we can send the diff to only one party.
      auto cr0 = prg_state->genPriv(field, 2 * numel);
      auto cr1 = ring_neg(cr0);

      // rb = r{k-1}
      auto rb0 = cr0.slice(0, numel);
      ring_add_(rb0, ring_rshift(r, K - 1));

      // rc = sum(r{i-m} << (i-m)) for i in range(m, k-2)
      //    = (r<<1)>>(m+1)
      auto rc0 = cr0.slice(numel, 2 * numel);
      ring_add_(rc0, ring_rshift(ring_lshift(r, 1), bits + 1));

      comm->sendAsync(0, cr0, "cr0");
      comm->sendAsync(1, cr1, "cr1");

      // TODO: cost model is asymmetric, but test framework requires the same.
      comm->addCommStatsManually(3, kComm * 4);

      auto [y1, y2] = prg_state->genPrssPair(field, numel);
      return makeAShare(y1, y2, field);
    }
    default:
      YASL_THROW("Party number exceeds 3!");
  }
}

}  // namespace spu::mpc::aby3
