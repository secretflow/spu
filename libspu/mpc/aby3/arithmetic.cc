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

#include "libspu/mpc/aby3/arithmetic.h"

#include <functional>
#include <future>

#include "spdlog/spdlog.h"

#include "libspu/core/array_ref.h"
#include "libspu/mpc/aby3/ot.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/linalg.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {
namespace {

// [zx]: Adapt this to new semantics of boolean sharing
std::vector<ArrayRef> a1b_offline(size_t sender, const ArrayRef& a,
                                  FieldType field, size_t self_rank,
                                  PrgState* prg_state, size_t numel,
                                  const ArrayRef& b) {
  SPU_ENFORCE(a.eltype().isa<RingTy>());
  SPU_ENFORCE(b.eltype().isa<BShrTy>());

  return DISPATCH_ALL_FIELDS(field, "_", [&]() -> std::vector<ArrayRef> {
    using AShrT = ring2k_t;

    ArrayRef m0(makeType<RingTy>(field), numel);
    ArrayRef m1(makeType<RingTy>(field), numel);
    linalg::setConstantValue(numel, &m0.at<AShrT>(0), m0.stride(), AShrT(0));
    linalg::setConstantValue(numel, &m1.at<AShrT>(0), m1.stride(), AShrT(1));

    auto _m0 = ArrayView<std::array<AShrT, 1>>(m0);
    auto _m1 = ArrayView<std::array<AShrT, 1>>(m1);
    auto _a = ArrayView<std::array<AShrT, 1>>(a);

    return DISPATCH_UINT_PT_TYPES(
        b.eltype().as<BShrTy>()->getBacktype(), "_",
        [&]() -> std::vector<ArrayRef> {
          using BSharT = ScalarT;
          if (self_rank == sender) {
            auto c1 = prg_state->genPrssPair(field, numel, false, true).first;
            auto c2 = prg_state->genPrssPair(field, numel, true).second;

            auto _b = ArrayView<std::array<BSharT, 2>>(b);

            auto _c1 = ArrayView<std::array<AShrT, 1>>(c1);
            auto _c2 = ArrayView<std::array<AShrT, 1>>(c2);

            // (i \xor b1 \xor b2) * a - c1 - c2
            pforeach(0, numel, [&](int64_t idx) {
              _m0[idx][0] =
                  (_m0[idx][0] ^ (_b[idx][0] & 0x1) ^ (_b[idx][1] & 0x1)) *
                      _a[idx][0] -
                  _c1[idx][0] - _c2[idx][0];
            });

            pforeach(0, numel, [&](int64_t idx) {
              _m1[idx][0] =
                  (_m1[idx][0] ^ (_b[idx][0] & 0x1) ^ (_b[idx][1] & 0x1)) *
                      _a[idx][0] -
                  _c1[idx][0] - _c2[idx][0];
            });

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
        });
  });
}

std::vector<uint8_t> ring_cast_boolean(const ArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<PtTy>(), "expect PtTy type, got={}", x.eltype());
  const size_t numel = x.numel();
  std::vector<uint8_t> res(numel);

  DISPATCH_UINT_PT_TYPES(x.eltype().as<PtTy>()->pt_type(), "_", [&]() {
    using BShrT = ScalarT;
    auto _x = ArrayView<std::array<BShrT, 1>>(x);
    pforeach(0, numel, [&](int64_t idx) {
      res[idx] = static_cast<uint8_t>(_x[idx][0] & 0x1);
    });
  });

  return res;
}

}  // namespace

ArrayRef RandA::proc(KernelEvalContext* ctx, size_t size) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  ArrayRef out(makeType<AShrTy>(field), size);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using AShrT = ring2k_t;

    std::vector<AShrT> r0(size);
    std::vector<AShrT> r1(size);
    prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

    auto _out = ArrayView<std::array<AShrT, 2>>(out);
    pforeach(0, size, [&](int64_t idx) {
      // Comparison only works for [-2^(k-2), 2^(k-2)).
      // TODO: Move this constrait to upper layer, saturate it here.
      _out[idx][0] = r0[idx] >> 2;
      _out[idx][1] = r1[idx] >> 2;
    });
  });

  return out;
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
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
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Pub2kTy>();
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
    auto* prg_state = ctx->getState<PrgState>();
    prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      r0[idx] = r0[idx] - r1[idx];
    }
    r1 = comm->rotate<AShrT>(r0, "p2a.zero");

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      _out[idx][0] += r0[idx];
      _out[idx][1] += r1[idx];
    }
#endif

    return out;
  });
}

ArrayRef A2V::proc(KernelEvalContext* ctx, const ArrayRef& in,
                   size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using VShrT = ring2k_t;
    using AShrT = ring2k_t;

    auto _in = ArrayView<std::array<AShrT, 2>>(in);
    auto out_ty = makeType<Priv2kTy>(field, rank);

    if (comm->getRank() == rank) {
      auto x3 = comm->recv<AShrT>(comm->nextRank(), "a2v");  // comm => 1, k
                                                             //
      ArrayRef out(out_ty, in.numel());
      auto _out = ArrayView<VShrT>(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
      });
      return out;

    } else if (comm->getRank() == (rank + 1) % 3) {
      std::vector<AShrT> x2(in.numel());

      pforeach(0, in.numel(), [&](int64_t idx) {  //
        x2[idx] = _in[idx][1];
      });

      comm->sendAsync<AShrT>(comm->prevRank(), x2, "a2v");  // comm => 1, k
      return makeConstantArrayRef(out_ty, in.numel());
    } else {
      return makeConstantArrayRef(out_ty, in.numel());
    }
  });
}

ArrayRef V2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const auto field = in_ty->field();

  size_t owner_rank = in_ty->owner();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using AShrT = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), in.numel());
    auto _out = ArrayView<std::array<AShrT, 2>>(out);
    if (comm->getRank() == owner_rank) {
      auto splits = ring_rand_additive_splits(in, 2);
      comm->sendAsync(comm->nextRank(), splits[1], "v2a");  // comm => 1, k
      comm->sendAsync(comm->prevRank(), splits[0], "v2a");  // comm => 1, k

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = splits[0].at<AShrT>(idx);
        _out[idx][1] = splits[1].at<AShrT>(idx);
      });
    } else if (comm->getRank() == (owner_rank + 1) % 3) {
      auto x0 = comm->recv<AShrT>(comm->prevRank(), "v2a");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = x0[idx];
        _out[idx][1] = 0;
      });
    } else {
      auto x1 = comm->recv<AShrT>(comm->nextRank(), "v2a");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = x1[idx];
      });
    }

    return out;
  });
}

ArrayRef NotA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
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
  auto* comm = ctx->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
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
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
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
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
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
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

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
  SPU_ENFORCE(lhs.numel() == rhs.numel());
  SPU_ENFORCE(lhs.eltype().isa<AShrTy>());
  SPU_ENFORCE(rhs.eltype().isa<BShrTy>() &&
              rhs.eltype().as<BShrTy>()->nbits() == 1);

  const auto field = lhs.eltype().as<AShrTy>()->field();
  const size_t in_nbits = rhs.eltype().as<BShrTy>()->nbits();

  SPU_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);

  const auto numel = rhs.numel();

  const auto b_ty = *rhs.eltype().as<BShrTy>();

  ArrayRef out(makeType<AShrTy>(field), numel);

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  // lhs
  const auto& a1 = getFirstShare(lhs);
  const auto& a2 = getSecondShare(lhs);

  // rhs
  auto b1 = getFirstShare(rhs);
  auto b2 = getSecondShare(rhs);

  // leave only lsb, in case the boolean value is randomized in a larger
  // domain.
  // NOTE: This is useless since we have n_bits to indicate valid bits
  // const auto kOne = ring_ones(back_type, rhs.numel());
  // b1 = ring_and(b1, kOne);
  // b2 = ring_and(b2, kOne);

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
    return a1b_offline(sender, a, field, self_rank, prg_state, numel, rhs);
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

    // c1, c3, m0, m1
    if (self_rank == sender1) {
      ot1.send(data1[2], data1[3]);  // 2k
      r1 = {data1[0], data1[1]};
    }
    if (self_rank == sender2) {
      ot2.send(data2[2], data2[3]);  // 2k
      r2 = {data2[0], data2[1]};
    }

    // helper send wc to receiver
    if (self_rank == (sender1 + 1) % 3) {
      ot1.help(ring_cast_boolean(b2));  // 1k
      r1.first = data1[0];
    }
    if (self_rank == (sender2 + 1) % 3) {
      ot2.help(ring_cast_boolean(b2));  // 1k
      r2.first = data2[0];
    }

    // receiver recv c2 and send c2 to helper
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

    // // NOTE: here two sequential rounds are required
    if (self_rank == (sender1 + 1) % 3) {
      // 1 latency
      r1.second = comm->recv((sender1 + 2) % 3, a1.eltype(), "ABY3-MUL-R1C1");
    }
    if (self_rank == (sender2 + 1) % 3) {
      // 1 latency overlapping with "ABY3-MUL-R1C1"
      r2.second = comm->recv((sender2 + 2) % 3, a1.eltype(), "ABY3-MUL-R2C1");
    }

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using AShrT = ring2k_t;

      auto _r1_first = ArrayView<std::array<AShrT, 1>>(r1.first);
      auto _r1_second = ArrayView<std::array<AShrT, 1>>(r1.second);
      auto _r2_first = ArrayView<std::array<AShrT, 1>>(r2.first);
      auto _r2_second = ArrayView<std::array<AShrT, 1>>(r2.second);

      // r1.first = r1.first + r2.first
      // r1.second = r1.second + r2.second
      pforeach(0, r1.first.numel(), [&](int64_t idx) {
        _r1_first[idx][0] = _r1_first[idx][0] + _r2_first[idx][0];
        _r1_second[idx][0] = _r1_second[idx][0] + _r2_second[idx][0];
      });
    });
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
                        const ArrayRef& y, size_t m, size_t n, size_t k) const {
  const auto field = x.eltype().as<Ring2k>()->field();

  ArrayRef z(makeType<AShrTy>(field), m * n);

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto z1 = getFirstShare(z);
  auto z2 = getSecondShare(z);

  ring_mmul_(z1, x1, y, m, n, k);
  ring_mmul_(z2, x2, y, m, n, k);

  return z;
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t m, size_t n, size_t k) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  auto r = std::async([&] {
    auto [r0, r1] = prg_state->genPrssPair(field, m * n);
    return ring_sub(r0, r1);
  });

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto y1 = getFirstShare(y);
  auto y2 = getSecondShare(y);

  // z1 := x1*y1 + x1*y2 + x2*y1 + k1
  // z2 := x2*y2 + x2*y3 + x3*y2 + k2
  // z3 := x3*y3 + x3*y1 + x1*y3 + k3
  ArrayRef out(makeType<AShrTy>(field), m * n);
  auto o1 = getFirstShare(out);
  auto o2 = getSecondShare(out);

  auto t2 = std::async(ring_mmul, x2, y1, m, n, k);
  auto t0 = ring_mmul(x1, ring_add(y1, y2), m, n, k);  //
  auto z1 = ring_sum({t0, t2.get(), r.get()});

  auto f = std::async([&] { ring_assign(o1, z1); });
  ring_assign(o2, comm->rotate(z1, kBindName));  // comm => 1, k
  f.get();
  return out;
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
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
ArrayRef TruncA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                      size_t bits) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

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
      SPU_THROW("Party number exceeds 3!");
  }
}

template <typename T>
std::vector<T> openWith(Communicator* comm, size_t peer_rank,
                        absl::Span<T const> in) {
  comm->sendAsync(peer_rank, in, "_");
  auto peer = comm->recv<T>(peer_rank, "_");
  SPU_ENFORCE(peer.size() == in.size());
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
ArrayRef TruncAPr::proc(KernelEvalContext* ctx, const ArrayRef& in,
                        size_t bits) const {
  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  const size_t k = SizeOf(field) * 8;

  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  // TODO: cost model is asymmetric, but test framework requires the same.
  comm->addCommStatsManually(3, 4 * SizeOf(field) * numel);

  // 1. P0 & P1 samples r together.
  // 2. P2 knows r and compute correlated random r{k-1} & sum(r{m~(k-2)})

  size_t pivot;
  prg_state->fillPubl(absl::MakeSpan(&pivot, 1));
  size_t P0 = pivot % 3;
  size_t P1 = (pivot + 1) % 3;
  size_t P2 = (pivot + 2) % 3;

  ArrayRef out(in.eltype(), numel);
  DISPATCH_ALL_FIELDS(field, "aby3.truncpr", [&]() {
    using U = ring2k_t;

    auto _in = ArrayView<std::array<U, 2>>(in);
    auto _out = ArrayView<std::array<U, 2>>(out);

    if (comm->getRank() == P0) {
      std::vector<U> r(numel);
      prg_state->fillPrssPair(absl::MakeSpan(r), {}, false, true);

      std::vector<U> x_plus_r(numel);
      pforeach(0, numel, [&](int64_t idx) {
        // convert to 2-outof-2 share.
        auto x = _in[idx][0] + _in[idx][1];

        // handle negative number.
        // assume secret x in [-2^(k-2), 2^(k-2)), by
        // adding 2^(k-2) x' = x + 2^(k-2) in [0, 2^(k-1)), with msb(x') == 0
        x += U(1) << (k - 2);

        // mask it with ra
        x_plus_r[idx] = x + r[idx];
      });

      // open c = <x> + <r>
      auto c = openWith<U>(comm, P1, x_plus_r);

      // get correlated randomness from P2
      // let rb = r{k-1},
      //     rc = sum(r{i-m}<<(i-m)) for i in range(m, k-2)
      auto cr = comm->recv<U>(P2, "cr0");
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

      comm->sendAsync<U>(P1, y2, "2to3");
      auto tmp = comm->recv<U>(P1, "2to3");

      // rebuild the final result.
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = y1[idx];
        _out[idx][1] = y2[idx] + tmp[idx];
      });
    } else if (comm->getRank() == P1) {
      std::vector<U> r(numel);
      prg_state->fillPrssPair({}, absl::MakeSpan(r), true, false);

      std::vector<U> x_plus_r(numel);
      pforeach(0, numel, [&](int64_t idx) {
        // let t as 2-out-of-2 share, mask it with ra.
        x_plus_r[idx] = _in[idx][1] + r[idx];
      });

      // open c = <x> + <r>
      auto c = openWith<U>(comm, P0, x_plus_r);

      // get correlated randomness from P2
      // let rb = r{k-1},
      //     rc = sum(r{i-m}<<(i-m)) for i in range(m, k-2)
      auto cr = comm->recv<U>(P2, "cr1");
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
      comm->sendAsync<U>(P0, y2, "2to3");
      auto tmp = comm->recv<U>(P0, "2to3");

      // rebuild the final result.
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = y2[idx] + tmp[idx];
        _out[idx][1] = y3[idx];
      });
    } else if (comm->getRank() == P2) {
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

        // rc = sum(r{i-m} << (i-m)) for i in range(m, k-2)
        //    = (r<<1)>>(m+1)
        rc0[idx] += (r << 1) >> (bits + 1);
      });

      comm->sendAsync<U>(P0, cr0, "cr0");
      comm->sendAsync<U>(P1, cr1, "cr1");

      std::vector<U> y3(numel);
      std::vector<U> y1(numel);
      prg_state->fillPrssPair(absl::MakeSpan(y3), absl::MakeSpan(y1));
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = y3[idx];
        _out[idx][1] = y1[idx];
      });
    } else {
      SPU_THROW("Party number exceeds 3!");
    }
  });

  return out;
}

}  // namespace spu::mpc::aby3
