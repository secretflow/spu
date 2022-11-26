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

#include "spu/core/array_ref.h"
#include "spu/core/trace.h"
#include "spu/mpc/aby3/ot.h"
#include "spu/mpc/aby3/type.h"
#include "spu/mpc/aby3/value.h"
#include "spu/mpc/common/abprotocol.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/util/circuits.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/linalg.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::aby3 {
namespace {

std::vector<ArrayRef> a1b_offline(size_t sender, const ArrayRef& a,
                                  FieldType field, size_t self_rank,
                                  PrgState* prg_state, size_t numel,
                                  const ArrayRef& b1, const ArrayRef& b2) {
  if (self_rank == sender) {
    auto c1 = prg_state->genPrssPair(field, numel, false, true).first;
    auto c2 = prg_state->genPrssPair(field, numel, true).second;

    auto m0 = ring_zeros(field, numel);
    {
      ring_xor_(m0, b1);
      ring_xor_(m0, b2);
      ring_mul_(m0, a);
      ring_sub_(m0, c1);
      ring_sub_(m0, c2);
    }

    auto m1 = ring_ones(field, numel);
    {
      ring_xor_(m1, b1);
      ring_xor_(m1, b2);
      ring_mul_(m1, a);
      ring_sub_(m1, c1);
      ring_sub_(m1, c2);
    }

    return {c1, c2, m0, m1};
  } else if (self_rank == (sender + 1) % 3) {
    prg_state->genPrssPair(field, numel, true, true);
    auto c1 = prg_state->genPrssPair(field, numel, false, true).first;

    return {c1};
  } else {
    auto c2 = prg_state->genPrssPair(field, numel, true, false).second;
    prg_state->genPrssPair(field, numel, true, true);

    return {c2};
  }
}

}  // namespace

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using PShrT = ring2k_t;
    using AShrT = ring2k_t;

    ArrayRef out(makeType<Pub2kTy>(field), in.numel());
    auto _in = ArrayView<std::array<AShrT, 2>>(in);
    auto _out = ArrayView<PShrT>(out);

    std::vector<AShrT> x2(in.numel());

    pforeach(0, in.numel(), [&](int64_t idx) {  //
      x2[idx] = _in[idx][1];
    });

    auto x3 = comm->rotate<AShrT>(x2, "a2p");  // comm => 1, k

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
    });

    return out;
  });
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();

  // TODO: we should expect Pub2kTy instead of Ring2k
  const auto* in_ty = in.eltype().as<Ring2k>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using AShrT = ring2k_t;
    using PShrT = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), in.numel());
    auto _in = ArrayView<PShrT>(in);
    auto _out = ArrayView<std::array<AShrT, 2>>(out);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = rank == 0 ? _in[idx] : 0;
      _out[idx][1] = rank == 2 ? _in[idx] : 0;
    });

// for debug purpose, randomize the inputs to avoid corner cases.
#ifdef ENABLE_MASK_DURING_ABY3_P2A
    std::vector<AShrT> r0(in.numel());
    std::vector<AShrT> r1(in.numel());
    auto* prg_state = ctx->caller()->getState<PrgState>();
    prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      r0[idx] = r0[idx] - r1[idx];
    }
    r1 = comm->rotate<AShrT>(r0, "p2b.zero");

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      _out[idx][0] += r0[idx];
      _out[idx][1] += r1[idx];
    }
#endif

    return out;
  });
}

ArrayRef NotA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using S = std::make_unsigned_t<ring2k_t>;

    ArrayRef out(makeType<AShrTy>(field), in.numel());
    auto _in = ArrayView<std::array<S, 2>>(in);
    auto _out = ArrayView<std::array<S, 2>>(out);

    // neg(x) = not(x) + 1
    // not(x) = neg(x) - 1
    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = -_in[idx][0];
      _out[idx][1] = -_in[idx][1];
      if (rank == 0) {
        _out[idx][1] -= 1;
      } else if (rank == 1) {
        _out[idx][0] -= 1;
      }
    });

    return out;
  });
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  auto* comm = ctx->caller()->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  YACL_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), lhs.numel());

    auto _lhs = ArrayView<std::array<U, 2>>(lhs);
    auto _rhs = ArrayView<U>(rhs);
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      if (rank == 0) _out[idx][1] += _rhs[idx];
      if (rank == 1) _out[idx][0] += _rhs[idx];
    });
    return out;
  });
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  YACL_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), lhs.numel());

    auto _lhs = ArrayView<std::array<U, 2>>(lhs);
    auto _rhs = ArrayView<std::array<U, 2>>(rhs);
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] + _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] + _rhs[idx][1];
    });
    return out;
  });
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  YACL_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), lhs.numel());

    auto _lhs = ArrayView<std::array<U, 2>>(lhs);
    auto _rhs = ArrayView<U>(rhs);
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] * _rhs[idx];
      _out[idx][1] = _lhs[idx][1] * _rhs[idx];
    });
    return out;
  });
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  return DISPATCH_ALL_FIELDS(field, "aby3.mulAA", [&]() {
    using U = ring2k_t;

    std::vector<U> r0(lhs.numel());
    std::vector<U> r1(lhs.numel());
    prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

    auto _lhs = ArrayView<std::array<U, 2>>(lhs);
    auto _rhs = ArrayView<std::array<U, 2>>(rhs);

    // z1 = (x1 * y1) + (x1 * y2) + (x2 * y1) + (r0 - r1);
    pforeach(0, lhs.numel(), [&](int64_t idx) {
      r0[idx] = (_lhs[idx][0] * _rhs[idx][0]) +  //
                (_lhs[idx][0] * _rhs[idx][1]) +  //
                (_lhs[idx][1] * _rhs[idx][0]) +  //
                (r0[idx] - r1[idx]);
    });

    r1 = comm->rotate<U>(r0, "mulaa");  // comm => 1, k

    ArrayRef out(makeType<AShrTy>(field), lhs.numel());
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = r0[idx];
      _out[idx][1] = r1[idx];
    });

    return out;
  });
}

// Refer to:
// 5.4 Computing [a]A[b]B = [ab]A, P19,
// ABY3: A Mixed Protocol Framework for Machine Learning
// - https://eprint.iacr.org/2018/403.pdf
ArrayRef MulA1B::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                      const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  YACL_ENFORCE(lhs.numel() == rhs.numel());
  YACL_ENFORCE(lhs.eltype().isa<AShare>());
  YACL_ENFORCE(rhs.eltype().isa<BShare>() &&
               rhs.eltype().as<BShare>()->nbits() == 1);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  const auto numel = lhs.numel();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  // lhs
  const auto& a1 = getFirstShare(lhs);
  const auto& a2 = getSecondShare(lhs);

  // rhs
  auto b1 = getFirstShare(rhs);
  auto b2 = getSecondShare(rhs);

  // leave only lsb, in case the boolean value is randomized in a larger
  // domain.
  const auto kOne = ring_ones(field, numel);
  b1 = ring_and(b1, kOne);
  b2 = ring_and(b2, kOne);

  auto self_rank = comm->getRank();
  const auto kComm = a1.elsize() * a1.numel();

  // split BShr * Pub into offline + online.
  // offline part: prepare rand data use in online part.

  auto get_ot = [&](size_t sender) {
    Ot3 ot(field, numel,
           Ot3::RoleRanks{sender, (sender + 2) % 3, (sender + 1) % 3}, comm,
           prg_state, false);
    return ot;
  };

  // TODO: optimization for large input.
  // online part: tasks two rounds latency. do 3-parties OT.
  auto offline = [&](size_t sender, const ArrayRef& a) {
    return a1b_offline(sender, a, field, self_rank, prg_state, numel, b1, b2);
  };

  // parallel online: parallel two 3-parties OT.
  auto parallel_online =
      [&](size_t sender1, const std::vector<ArrayRef>& data1, size_t sender2,
          const std::vector<ArrayRef>& data2) -> std::pair<ArrayRef, ArrayRef> {
    auto ot1 = get_ot(sender1);
    auto ot2 = get_ot(sender2);

    std::pair<ArrayRef, ArrayRef> r1;
    std::pair<ArrayRef, ArrayRef> r2;

    // asymmetric cost.
    comm->addCommStatsManually(2, kComm * 8);

    if (self_rank == sender1) {
      ot1.send(data1[2], data1[3]);  // 2k
      r1 = {data1[0], data1[1]};
    }
    if (self_rank == sender2) {
      ot2.send(data2[2], data2[3]);  // 2k
      r2 = {data2[0], data2[1]};
    }

    if (self_rank == (sender1 + 1) % 3) {
      ot1.help(ring_cast_boolean(b2));  // 1k
      r1.first = data1[0];
    }
    if (self_rank == (sender2 + 1) % 3) {
      ot2.help(ring_cast_boolean(b2));  // 1k
      r2.first = data2[0];
    }

    if (self_rank == (sender1 + 2) % 3) {
      // 1 latency
      auto c1 = ot1.recv(ring_cast_boolean(b1));
      comm->sendAsync((sender1 + 1) % 3, c1, "ABY3-MUL-R1C1");  // 1k
      r1 = {c1, data1[0]};
    }
    if (self_rank == (sender2 + 2) % 3) {
      // 1 latency overlapping with "ABY3-MUL-R1C1"
      auto c1 = ot2.recv(ring_cast_boolean(b1));
      comm->sendAsync((sender2 + 1) % 3, c1, "ABY3-MUL-R2C1");  // 1k
      r2 = {c1, data2[0]};
    }

    if (self_rank == (sender1 + 1) % 3) {
      // 1 latency
      r1.second = comm->recv((sender1 + 2) % 3, a1.eltype(), "ABY3-MUL-R1C1");
    }
    if (self_rank == (sender2 + 1) % 3) {
      // 1 latency overlapping with "ABY3-MUL-R1C1"
      r2.second = comm->recv((sender2 + 2) % 3, a1.eltype(), "ABY3-MUL-R2C1");
    }

    ring_add_(r1.first, r2.first);
    ring_add_(r1.second, r2.second);

    return r1;
  };

  // parallel online, two rounds latency.
  auto data1 = offline(0, a2);
  // only sender access a1 + a2, avoid useless add for other two parties.
  auto data2 = offline(2, self_rank == 2 ? ring_add(a1, a2) : a2);
  auto ret = parallel_online(0, data1, 2, data2);
  return makeAShare(ret.first, ret.second, field);
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);

  const auto field = x.eltype().as<Ring2k>()->field();

  ArrayRef z(makeType<AShrTy>(field), M * N);

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto z1 = getFirstShare(z);
  auto z2 = getSecondShare(z);

  ring_mmul_(z1, x1, y, M, N, K);
  ring_mmul_(z2, x2, y, M, N, K);

  return z;
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  auto r = std::async([&] {
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
  ArrayRef out(makeType<AShrTy>(field), M * N);
  auto o1 = getFirstShare(out);
  auto o2 = getSecondShare(out);

  auto t2 = std::async(ring_mmul, x2, y1, M, N, K);
  auto t0 = ring_mmul(x1, ring_add(y1, y2), M, N, K);  //
  auto z1 = ring_sum({t0, t2.get(), r.get()});

  auto f = std::async([&] { ring_assign(o1, z1); });
  ring_assign(o2, comm->rotate(z1, kBindName));  // comm => 1, k
  f.get();
  return out;
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), in.numel());
    auto _in = ArrayView<std::array<U, 2>>(in);
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = _in[idx][0] << bits;
      _out[idx][1] = _in[idx][1] << bits;
    });

    return out;
  });
}

// Refer to:
// Share Truncation I, 5.1 Fixed-point Arithmetic, P13,
// ABY3: A Mixed Protocol Framework for Machine Learning
// - https://eprint.iacr.org/2018/403.pdf
ArrayRef TruncPrA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

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
      YACL_THROW("Party number exceeds 3!");
  }
}

template <typename T>
std::vector<T> openWith(Communicator* comm, size_t peer_rank,
                        absl::Span<T const> in) {
  comm->sendAsync(peer_rank, in, "_");
  auto peer = comm->recv<T>(peer_rank, "_");
  YACL_ENFORCE(peer.size() == in.size());
  std::vector<T> out(in.size());

  pforeach(0, in.size(), [&](int64_t idx) {  //
    out[idx] = in[idx] + peer[idx];
  });

  return out;
}

// PRECISE VERSION
// Refer to:
// 3.2.2 Truncation by a public value, P10,
// Secure Evaluation of Quantized Neural Networks
// - https://arxiv.org/pdf/1910.12435.pdf
ArrayRef TruncPrAPrecise::proc(KernelEvalContext* ctx, const ArrayRef& in,
                               size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  const size_t k = SizeOf(field) * 8;

  auto* prg_state = ctx->caller()->getState<PrgState>();
  auto* comm = ctx->caller()->getState<Communicator>();

  // TODO: cost model is asymmetric, but test framework requires the same.
  comm->addCommStatsManually(3, 4 * SizeOf(field) * numel);

  ArrayRef out(in.eltype(), numel);
  DISPATCH_ALL_FIELDS(field, "aby3.truncpr", [&]() {
    using U = ring2k_t;

    auto _in = ArrayView<std::array<U, 2>>(in);
    auto _out = ArrayView<std::array<U, 2>>(out);

    // 1. P0 & P1 samples r together.
    // 2. P2 knows r and compute correlated random r{k-1} & sum(r{m~(k-2)})
    switch (comm->getRank()) {
      case 0: {
        std::vector<U> r(numel);
        prg_state->fillPrssPair(absl::MakeSpan(r), {}, false, true);

        std::vector<U> x_plus_r(numel);
        pforeach(0, numel, [&](int64_t idx) {
          // convert to 2-outof-2 share.
          auto x = _in[idx][0] + _in[idx][1];

          // handle negative number.
          // assume secret x in [-2^(k-2), 2^(k-2)), by
          // adding 2^(k-2) x' = x + 2^(k-2) in [0,
          // 2^(k-1)), with msb(x') == 0
          x += U(1) << (k - 2);

          // mask it with ra
          x_plus_r[idx] = x + r[idx];
        });

        // open c = <x> + <r>
        auto c = openWith<U>(comm, 1, x_plus_r);

        // get correlated randomness from P2
        // let rb = r{k-1},
        //     rc = sum(r{i-m}<<(i-m)) for i in range(m, k-2)
        auto cr = comm->recv<U>(2, "cr0");
        auto rb = absl::MakeSpan(cr).subspan(0, numel);
        auto rc = absl::MakeSpan(cr).subspan(numel, numel);

        std::vector<U> y2(numel);  // the 2-out-of-2 truncation result
        pforeach(0, numel, [&](int64_t idx) {
          // c_hat = c/2^m mod 2^(k-m-1) = (c << 1) >> (1+m)
          auto c_hat = (c[idx] << 1) >> (1 + bits);

          // <b> = <rb> ^ c{k-1} = <rb> + c{k-1} - 2*c{k-1}*<rb>
          // note: <rb> is a randbit (in r^2k)
          const auto ck_1 = c[idx] >> (k - 1);
          auto b = rb[idx] + ck_1 - 2 * ck_1 * rb[idx];

          // y = c_hat - <rc> + <b> * 2^(k-m-1)
          auto y = c_hat - rc[idx] + (b << (k - 1 - bits));

          // re-encode negative numbers.
          // from https://eprint.iacr.org/2020/338.pdf, section 5.1
          // y' = y - 2^(k-2-m)
          y2[idx] = y - (U(1) << (k - 2 - bits));
        });

        //
        std::vector<U> y1(numel);
        prg_state->fillPrssPair(absl::MakeSpan(y1), {}, false, true);
        pforeach(0, numel, [&](int64_t idx) {  //
          y2[idx] -= y1[idx];
        });

        comm->sendAsync<U>(1, y2, "2to3");
        auto tmp = comm->recv<U>(1, "2to3");

        // rebuild the final result.
        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = y1[idx];
          _out[idx][1] = y2[idx] + tmp[idx];
        });
        break;
      }
      case 1: {
        std::vector<U> r(numel);
        prg_state->fillPrssPair({}, absl::MakeSpan(r), true, false);

        std::vector<U> x_plus_r(numel);
        pforeach(0, numel, [&](int64_t idx) {
          // let t as 2-out-of-2 share, mask it with ra.
          x_plus_r[idx] = _in[idx][1] + r[idx];
        });

        // open c = <x> + <r>
        auto c = openWith<U>(comm, 0, x_plus_r);

        // get correlated randomness from P2
        // let rb = r{k-1},
        //     rc = sum(r{i-m}<<(i-m)) for i in range(m, k-2)
        auto cr = comm->recv<U>(2, "cr1");
        auto rb = absl::MakeSpan(cr).subspan(0, numel);
        auto rc = absl::MakeSpan(cr).subspan(numel, numel);

        std::vector<U> y2(numel);  // the 2-out-of-2 truncation result
        pforeach(0, numel, [&](int64_t idx) {
          // <b> = <rb> ^ c{k-1} = <rb> + c{k-1} - 2*c{k-1}*<rb>
          // note: <rb> is a randbit (in r^2k)
          const auto ck_1 = c[idx] >> (k - 1);
          auto b = rb[idx] + 0 - 2 * ck_1 * rb[idx];

          // y = c_hat - <rc> + <b> * 2^(k-m-1)
          y2[idx] = 0 - rc[idx] + (b << (k - 1 - bits));
        });

        std::vector<U> y3(numel);
        prg_state->fillPrssPair({}, absl::MakeSpan(y3), true, false);
        pforeach(0, numel, [&](int64_t idx) {  //
          y2[idx] -= y3[idx];
        });
        comm->sendAsync<U>(0, y2, "2to3");
        auto tmp = comm->recv<U>(0, "2to3");

        // rebuild the final result.
        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = y2[idx] + tmp[idx];
          _out[idx][1] = y3[idx];
        });
        break;
      }
      case 2: {
        std::vector<U> r0(numel);
        std::vector<U> r1(numel);
        prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

        std::vector<U> cr0(2 * numel);
        std::vector<U> cr1(2 * numel);
        auto rb0 = absl::MakeSpan(cr0).subspan(0, numel);
        auto rc0 = absl::MakeSpan(cr0).subspan(numel, numel);
        auto rb1 = absl::MakeSpan(cr1).subspan(0, numel);
        auto rc1 = absl::MakeSpan(cr1).subspan(numel, numel);

        prg_state->fillPriv(absl::MakeSpan(cr0));
        pforeach(0, numel, [&](int64_t idx) {
          // let <rb> = <rc> = 0
          rb1[idx] = -rb0[idx];
          rc1[idx] = -rc0[idx];

          auto r = r0[idx] + r1[idx];

          // <rb> = r{k-1}
          rb0[idx] += r >> (k - 1);

          // rc = sum(r{i-m} << (i-m)) for i in range(m,
          // k-2)
          //    = (r<<1)>>(m+1)
          rc0[idx] += (r << 1) >> (bits + 1);
        });

        comm->sendAsync<U>(0, cr0, "cr0");
        comm->sendAsync<U>(1, cr1, "cr1");

        std::vector<U> y3(numel);
        std::vector<U> y1(numel);
        prg_state->fillPrssPair(absl::MakeSpan(y3), absl::MakeSpan(y1));
        pforeach(0, numel, [&](int64_t idx) {
          _out[idx][0] = y3[idx];
          _out[idx][1] = y1[idx];
        });
        break;
      }
      default:
        YACL_THROW("Party number exceeds 3!");
    };
  });

  return out;
}

namespace {
// split even and odd bits. e.g.
//   xAyBzCwD -> (xyzw, ABCD)
std::pair<ArrayRef, ArrayRef> bit_split(const ArrayRef& in) {
  constexpr std::array<uint128_t, 6> kSwapMasks = {{
      yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
      yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
      yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
      yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
      yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
      yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
  }};
  constexpr std::array<uint128_t, 6> kKeepMasks = {{
      yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
      yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
      yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
      yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
      yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
      yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
  }};

  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();
  YACL_ENFORCE(in_nbits != 0 && in_nbits % 2 == 0, "in_nbits={}", in_nbits);
  const size_t out_nbits = in_nbits / 2;
  const auto out_backtype = calcBShareBacktype(out_nbits);
  const auto out_type = makeType<BShrTy>(out_backtype, out_nbits);

  ArrayRef lo(out_type, in.numel());
  ArrayRef hi(out_type, in.numel());
  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using InT = ScalarT;
    auto _in = ArrayView<std::array<InT, 2>>(in);
    DISPATCH_UINT_PT_TYPES(out_backtype, "_", [&]() {
      using OutT = ScalarT;
      auto _lo = ArrayView<std::array<OutT, 2>>(lo);
      auto _hi = ArrayView<std::array<OutT, 2>>(hi);

      pforeach(0, in.numel(), [&](int64_t idx) {
        InT r0 = _in[idx][0];
        InT r1 = _in[idx][1];
        // algorithm:
        //      0101010101010101
        // swap  ^^  ^^  ^^  ^^
        //      0011001100110011
        // swap   ^^^^    ^^^^
        //      0000111100001111
        // swap     ^^^^^^^^
        //      0000000011111111
        for (int k = 0; k + 1 < log2Ceil(in_nbits); k++) {
          InT keep = static_cast<InT>(kKeepMasks[k]);
          InT move = static_cast<InT>(kSwapMasks[k]);
          int shift = 1 << k;

          r0 = (r0 & keep) ^ ((r0 >> shift) & move) ^ ((r0 & move) << shift);
          r1 = (r1 & keep) ^ ((r1 >> shift) & move) ^ ((r1 & move) << shift);
        }
        InT mask = (InT(1) << (in_nbits / 2)) - 1;
        _lo[idx][0] = static_cast<OutT>(r0) & mask;
        _hi[idx][0] = static_cast<OutT>(r0 >> (in_nbits / 2)) & mask;
        _lo[idx][1] = static_cast<OutT>(r1) & mask;
        _hi[idx][1] = static_cast<OutT>(r1 >> (in_nbits / 2)) & mask;
      });
    });
  });

  return std::make_pair(lo, hi);
};

// compute the k'th bit of x + y
ArrayRef carry_out(Object* ctx, const ArrayRef& x, const ArrayRef& y,
                   size_t k) {
  // init P & G
  auto P = xor_bb(ctx, x, y);
  auto G = and_bb(ctx, x, y);

  // Use kogge stone layout.
  while (k > 1) {
    if (k % 2 != 0) {
      k += 1;
      P = lshift_b(ctx, P, 1);
      G = lshift_b(ctx, G, 1);
    }
    auto [P0, P1] = bit_split(P);
    auto [G0, G1] = bit_split(G);

    // Calculate next-level of P, G
    //   P = P1 & P0
    //   G = G1 | (P1 & G0)
    //     = G1 ^ (P1 & G0)
    std::vector<ArrayRef> v = vectorize(
        {P0, G0}, {P1, P1}, [&](const ArrayRef& xx, const ArrayRef& yy) {
          return and_bb(ctx, xx, yy);
        });
    P = std::move(v[0]);
    G = xor_bb(ctx, G1, v[1]);
    k >>= 1;
  }

  return G;
}

}  // namespace

ArrayRef MsbA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* prg_state = ctx->caller()->getState<PrgState>();

  // First construct 2 boolean shares.
  // Let
  //   X = [(x0, x1), (x1, x2), (x2, x0)] as input.
  //   Z = (z0, z1, z2) as boolean zero share.
  //
  // Construct M, N as boolean shares,
  //   M = [((x0+x1)^z0, z1), (z1, z2), (z2, (x0+x1)^z0)]
  //   N = [(0,          0),  (0,  x2), (x2, 0         )]
  //
  // That
  //  M + N = (x0+x1)^z0^z1^z2 + x2
  //        = x0 + x1 + x2 = X
  const Type bshr_type =
      makeType<BShrTy>(GetStorageType(field), SizeOf(field) * 8);
  ArrayRef m(bshr_type, in.numel());
  ArrayRef n(bshr_type, in.numel());
  DISPATCH_ALL_FIELDS(field, "aby3.msb.split", [&]() {
    using U = ring2k_t;

    auto _in = ArrayView<std::array<U, 2>>(in);
    auto _m = ArrayView<std::array<U, 2>>(m);
    auto _n = ArrayView<std::array<U, 2>>(n);

    std::vector<U> r0(numel);
    std::vector<U> r1(numel);
    prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

    pforeach(0, in.numel(), [&](int64_t idx) {
      r0[idx] = r0[idx] ^ r1[idx];
      if (comm->getRank() == 0) {
        r0[idx] ^= (_in[idx][0] + _in[idx][1]);
      }
    });

    r1 = comm->rotate<U>(r0, "m");

    pforeach(0, in.numel(), [&](int64_t idx) {
      _m[idx][0] = r0[idx];
      _m[idx][1] = r1[idx];
      _n[idx][0] = comm->getRank() == 2 ? _in[idx][0] : 0;
      _n[idx][1] = comm->getRank() == 1 ? _in[idx][1] : 0;
    });
  });

  // Compute the k-1'th carry bit.
  size_t nbits = SizeOf(field) * 8 - 1;
  auto carry = carry_out(ctx->caller(), m, n, nbits);

  // Compute the k'th bit.
  //   (m^n)[k] ^ carry
  auto* obj = ctx->caller();
  return xor_bb(obj, rshift_b(obj, xor_bb(obj, m, n), nbits), carry);
}

}  // namespace spu::mpc::aby3
