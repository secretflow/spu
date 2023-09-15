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
std::vector<NdArrayRef> a1b_offline(size_t sender, const NdArrayRef& a,
                                    FieldType field, size_t self_rank,
                                    PrgState* prg_state, const NdArrayRef& b) {
  SPU_ENFORCE(a.eltype().isa<RingTy>());
  SPU_ENFORCE(b.eltype().isa<BShrTy>());

  auto numel = a.numel();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() -> std::vector<NdArrayRef> {
    using ashr_el_t = ring2k_t;
    NdArrayRef m0(makeType<RingTy>(field), a.shape());
    NdArrayRef m1(makeType<RingTy>(field), a.shape());
    ring_set_value(m0, ashr_el_t(0));
    ring_set_value(m1, ashr_el_t(1));

    NdArrayView<ashr_el_t> _m0(m0);
    NdArrayView<ashr_el_t> _m1(m1);

    return DISPATCH_UINT_PT_TYPES(
        b.eltype().as<BShrTy>()->getBacktype(), "_",
        [&]() -> std::vector<NdArrayRef> {
          using bshr_t = std::array<ScalarT, 2>;
          if (self_rank == sender) {
            auto c1 = prg_state
                          ->genPrssPair(field, a.shape(),
                                        PrgState::GenPrssCtrl::First)
                          .first;
            auto c2 = prg_state
                          ->genPrssPair(field, a.shape(),
                                        PrgState::GenPrssCtrl::Second)
                          .second;

            NdArrayView<ashr_el_t> _a(a);
            NdArrayView<bshr_t> _b(b);
            NdArrayView<ashr_el_t> _c1(c1);
            NdArrayView<ashr_el_t> _c2(c2);

            // (i \xor b1 \xor b2) * a - c1 - c2
            pforeach(0, numel, [&](int64_t idx) {
              _m0[idx] = (_m0[idx] ^ (_b[idx][0] & 0x1) ^ (_b[idx][1] & 0x1)) *
                             _a[idx] -
                         _c1[idx] - _c2[idx];
            });

            pforeach(0, numel, [&](int64_t idx) {
              _m1[idx] = (_m1[idx] ^ (_b[idx][0] & 0x1) ^ (_b[idx][1] & 0x1)) *
                             _a[idx] -
                         _c1[idx] - _c2[idx];
            });

            return {c1, c2, m0, m1};
          } else if (self_rank == (sender + 1) % 3) {
            prg_state->genPrssPair(field, a.shape(),
                                   PrgState::GenPrssCtrl::None);
            auto c1 = prg_state
                          ->genPrssPair(field, a.shape(),
                                        PrgState::GenPrssCtrl::First)
                          .first;

            return {c1};
          } else {
            auto c2 = prg_state
                          ->genPrssPair(field, a.shape(),
                                        PrgState::GenPrssCtrl::Second)
                          .second;
            prg_state->genPrssPair(field, a.shape(),
                                   PrgState::GenPrssCtrl::None);

            return {c2};
          }
        });
  });
}

std::vector<uint8_t> ring_cast_boolean(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<PtTy>(), "expect PtTy type, got={}", x.eltype());
  const size_t numel = x.numel();
  std::vector<uint8_t> res(numel);

  DISPATCH_UINT_PT_TYPES(x.eltype().as<PtTy>()->pt_type(), "_", [&]() {
    NdArrayView<ScalarT> _x(x);
    pforeach(0, numel, [&](int64_t idx) {
      res[idx] = static_cast<uint8_t>(_x[idx] & 0x1);
    });
  });

  return res;
}

}  // namespace

NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  NdArrayRef out(makeType<AShrTy>(field), shape);

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using el_t = ring2k_t;

    std::vector<el_t> r0(shape.numel());
    std::vector<el_t> r1(shape.numel());
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);

    NdArrayView<std::array<el_t, 2>> _out(out);

    pforeach(0, out.numel(), [&](int64_t idx) {
      // Comparison only works for [-2^(k-2), 2^(k-2)).
      // TODO: Move this constraint to upper layer, saturate it here.
      _out[idx][0] = r0[idx] >> 2;
      _out[idx][1] = r1[idx] >> 2;
    });
  });

  return out;
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto numel = in.numel();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 2>;

    NdArrayRef out(makeType<Pub2kTy>(field), in.shape());
    NdArrayView<pshr_el_t> _out(out);
    NdArrayView<ashr_t> _in(in);

    std::vector<ashr_el_t> x2(numel);

    pforeach(0, numel, [&](int64_t idx) { x2[idx] = _in[idx][1]; });

    auto x3 = comm->rotate<ashr_el_t>(x2, "a2p");  // comm => 1, k

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
    });

    return out;
  });
}

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 2>;
    using pshr_el_t = ring2k_t;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);
    NdArrayView<pshr_el_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = rank == 0 ? _in[idx] : 0;
      _out[idx][1] = rank == 2 ? _in[idx] : 0;
    });

// for debug purpose, randomize the inputs to avoid corner cases.
#ifdef ENABLE_MASK_DURING_ABY3_P2A
    std::vector<ashr_el_t> r0(in.numel());
    std::vector<ashr_el_t> r1(in.numel());
    auto* prg_state = ctx->getState<PrgState>();
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      r0[idx] = r0[idx] - r1[idx];
    }
    r1 = comm->rotate<ashr_el_t>(r0, "p2a.zero");

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      _out[idx][0] += r0[idx];
      _out[idx][1] += r1[idx];
    }
#endif

    return out;
  });
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using vshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 2>;

    NdArrayView<ashr_t> _in(in);
    auto out_ty = makeType<Priv2kTy>(field, rank);

    if (comm->getRank() == rank) {
      auto x3 = comm->recv<ashr_el_t>(comm->nextRank(), "a2v");  // comm => 1, k
                                                                 //
      NdArrayRef out(out_ty, in.shape());
      NdArrayView<vshr_el_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
      });
      return out;

    } else if (comm->getRank() == (rank + 1) % 3) {
      std::vector<ashr_el_t> x2(in.numel());

      pforeach(0, in.numel(), [&](int64_t idx) { x2[idx] = _in[idx][1]; });

      comm->sendAsync<ashr_el_t>(comm->prevRank(), x2,
                                 "a2v");  // comm => 1, k
      return makeConstantArrayRef(out_ty, in.shape());
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  });
}

NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();

  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const auto field = in_ty->field();

  size_t owner_rank = in_ty->owner();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 2>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);

    if (comm->getRank() == owner_rank) {
      auto splits = ring_rand_additive_splits(in, 2);
      comm->sendAsync(comm->nextRank(), splits[1], "v2a");  // comm => 1, k
      comm->sendAsync(comm->prevRank(), splits[0], "v2a");  // comm => 1, k

      NdArrayView<ashr_el_t> _s0(splits[0]);
      NdArrayView<ashr_el_t> _s1(splits[1]);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = _s0[idx];
        _out[idx][1] = _s1[idx];
      });
    } else if (comm->getRank() == (owner_rank + 1) % 3) {
      auto x0 = comm->recv<ashr_el_t>(comm->prevRank(), "v2a");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = x0[idx];
        _out[idx][1] = 0;
      });
    } else {
      auto x1 = comm->recv<ashr_el_t>(comm->nextRank(), "v2a");  // comm => 1, k
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = x1[idx];
      });
    }

    return out;
  });
}

NdArrayRef NotA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using el_t = std::make_unsigned_t<ring2k_t>;
    using shr_t = std::array<el_t, 2>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

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
NdArrayRef AddAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      if (rank == 0) _out[idx][1] += _rhs[idx];
      if (rank == 1) _out[idx][0] += _rhs[idx];
    });
    return out;
  });
}

NdArrayRef AddAA::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using shr_t = std::array<ring2k_t, 2>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

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
NdArrayRef MulAP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] * _rhs[idx];
      _out[idx][1] = _lhs[idx][1] * _rhs[idx];
    });
    return out;
  });
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  return DISPATCH_ALL_FIELDS(field, "aby3.mulAA", [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    std::vector<el_t> r0(lhs.numel());
    std::vector<el_t> r1(lhs.numel());
    prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                            PrgState::GenPrssCtrl::Both);

    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

    // z1 = (x1 * y1) + (x1 * y2) + (x2 * y1) + (r0 - r1);
    pforeach(0, lhs.numel(), [&](int64_t idx) {
      r0[idx] = (_lhs[idx][0] * _rhs[idx][0]) + (_lhs[idx][0] * _rhs[idx][1]) +
                (_lhs[idx][1] * _rhs[idx][0]) + (r0[idx] - r1[idx]);
    });

    r1 = comm->rotate<el_t>(r0, "mulaa");  // comm => 1, k

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);

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
NdArrayRef MulA1B::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                        const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.shape() == rhs.shape());
  SPU_ENFORCE(lhs.eltype().isa<AShrTy>());
  SPU_ENFORCE(rhs.eltype().isa<BShrTy>() &&
              rhs.eltype().as<BShrTy>()->nbits() == 1);

  const auto field = lhs.eltype().as<AShrTy>()->field();
  const size_t in_nbits = rhs.eltype().as<BShrTy>()->nbits();

  SPU_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);

  const auto b_ty = *rhs.eltype().as<BShrTy>();

  NdArrayRef out(makeType<AShrTy>(field), lhs.shape());

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
    Ot3 ot(field, lhs.shape(),
           Ot3::RoleRanks{sender, (sender + 2) % 3, (sender + 1) % 3}, comm,
           prg_state, false);
    return ot;
  };

  // TODO: optimization for large input.
  // online part: tasks two rounds latency. do 3-parties OT.
  auto offline = [&](size_t sender, const NdArrayRef& a) {
    return a1b_offline(sender, a, field, self_rank, prg_state, rhs);
  };

  // parallel online: parallel two 3-parties OT.
  auto parallel_online =
      [&](size_t sender1, const std::vector<NdArrayRef>& data1, size_t sender2,
          const std::vector<NdArrayRef>& data2)
      -> std::pair<NdArrayRef, NdArrayRef> {
    auto ot1 = get_ot(sender1);
    auto ot2 = get_ot(sender2);

    std::pair<NdArrayRef, NdArrayRef> r1;
    std::pair<NdArrayRef, NdArrayRef> r2;

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
      r1 = {c1.reshape(data1[0].shape()), data1[0]};
    }
    if (self_rank == (sender2 + 2) % 3) {
      // 1 latency overlapping with "ABY3-MUL-R1C1"
      auto c1 = ot2.recv(ring_cast_boolean(b1));
      comm->sendAsync((sender2 + 1) % 3, c1, "ABY3-MUL-R2C1");  // 1k
      r2 = {c1.reshape(data2[0].shape()), data2[0]};
    }

    // // NOTE: here two sequential rounds are required
    if (self_rank == (sender1 + 1) % 3) {
      // 1 latency
      r1.second = comm->recv((sender1 + 2) % 3, a1.eltype(), "ABY3-MUL-R1C1")
                      .reshape(r1.first.shape());
    }
    if (self_rank == (sender2 + 1) % 3) {
      // 1 latency overlapping with "ABY3-MUL-R1C1"
      r2.second = comm->recv((sender2 + 2) % 3, a1.eltype(), "ABY3-MUL-R2C1")
                      .reshape(r2.first.shape());
    }

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      // using  = ring2k_t;
      NdArrayView<ring2k_t> r1_0(r1.first);
      NdArrayView<ring2k_t> r1_1(r1.second);

      NdArrayView<ring2k_t> r2_0(r2.first);
      NdArrayView<ring2k_t> r2_1(r2.second);
      // r1.first = r1.first + r2.first
      // r1.second = r1.second + r2.second
      pforeach(0, r1.first.numel(), [&](int64_t idx) {
        r1_0[idx] = r1_0[idx] + r2_0[idx];
        r1_1[idx] = r1_1[idx] + r2_1[idx];
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
NdArrayRef MatMulAP::proc(KernelEvalContext*, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  const auto field = x.eltype().as<Ring2k>()->field();

  NdArrayRef z(makeType<AShrTy>(field), {x.shape()[0], y.shape()[1]});

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto z1 = getFirstShare(z);
  auto z2 = getSecondShare(z);

  ring_mmul_(z1, x1, y);
  ring_mmul_(z2, x2, y);

  return z;
}

NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  auto r = std::async([&] {
    auto [r0, r1] = prg_state->genPrssPair(field, {x.shape()[0], y.shape()[1]},
                                           PrgState::GenPrssCtrl::Both);
    return ring_sub(r0, r1);
  });

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto y1 = getFirstShare(y);
  auto y2 = getSecondShare(y);

  // z1 := x1*y1 + x1*y2 + x2*y1 + k1
  // z2 := x2*y2 + x2*y3 + x3*y2 + k2
  // z3 := x3*y3 + x3*y1 + x1*y3 + k3
  NdArrayRef out(makeType<AShrTy>(field), {x.shape()[0], y.shape()[1]});
  auto o1 = getFirstShare(out);
  auto o2 = getSecondShare(out);

  auto t2 = std::async(ring_mmul, x2, y1);
  auto t0 = ring_mmul(x1, ring_add(y1, y2));  //
  auto z1 = ring_sum({t0, t2.get(), r.get()});

  auto f = std::async([&] { ring_assign(o1, z1); });
  ring_assign(o2, comm->rotate(z1, kBindName));  // comm => 1, k
  f.get();
  return out;
}

NdArrayRef LShiftA::proc(KernelEvalContext*, const NdArrayRef& in,
                         size_t bits) const {
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using shr_t = std::array<ring2k_t, 2>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

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
NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        size_t bits, SignType sign) const {
  (void)sign;  // TODO: optimize me.

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto r_future = std::async([&] {
    return prg_state->genPrssPair(field, in.shape(),
                                  PrgState::GenPrssCtrl::Both);
  });

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
NdArrayRef TruncAPr::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t bits, SignType sign) const {
  (void)sign;  // TODO, optimize me.

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

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_ALL_FIELDS(field, "aby3.truncpr", [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 2>;

    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    if (comm->getRank() == P0) {
      std::vector<el_t> r(numel);
      prg_state->fillPrssPair<el_t>(r.data(), nullptr, r.size(),
                                    PrgState::GenPrssCtrl::First);

      std::vector<el_t> x_plus_r(numel);
      pforeach(0, numel, [&](int64_t idx) {
        // convert to 2-outof-2 share.
        auto x = _in[idx][0] + _in[idx][1];

        // handle negative number.
        // assume secret x in [-2^(k-2), 2^(k-2)), by
        // adding 2^(k-2) x' = x + 2^(k-2) in [0, 2^(k-1)), with msb(x') == 0
        x += el_t(1) << (k - 2);

        // mask it with ra
        x_plus_r[idx] = x + r[idx];
      });

      // open c = <x> + <r>
      auto c = openWith<el_t>(comm, P1, x_plus_r);

      // get correlated randomness from P2
      // let rb = r{k-1},
      //     rc = sum(r{i-m}<<(i-m)) for i in range(m, k-2)
      auto cr = comm->recv<el_t>(P2, "cr0");
      auto rb = absl::MakeSpan(cr).subspan(0, numel);
      auto rc = absl::MakeSpan(cr).subspan(numel, numel);

      std::vector<el_t> y2(numel);  // the 2-out-of-2 truncation result
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
        y2[idx] = y - (el_t(1) << (k - 2 - bits));
      });

      //
      std::vector<el_t> y1(numel);
      prg_state->fillPrssPair<el_t>(y1.data(), nullptr, r.size(),
                                    PrgState::GenPrssCtrl::First);
      pforeach(0, numel, [&](int64_t idx) {  //
        y2[idx] -= y1[idx];
      });

      comm->sendAsync<el_t>(P1, y2, "2to3");
      auto tmp = comm->recv<el_t>(P1, "2to3");

      // rebuild the final result.
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = y1[idx];
        _out[idx][1] = y2[idx] + tmp[idx];
      });
    } else if (comm->getRank() == P1) {
      std::vector<el_t> r(numel);
      prg_state->fillPrssPair<el_t>(nullptr, r.data(), r.size(),
                                    PrgState::GenPrssCtrl::Second);

      std::vector<el_t> x_plus_r(numel);
      pforeach(0, numel, [&](int64_t idx) {
        // let t as 2-out-of-2 share, mask it with ra.
        x_plus_r[idx] = _in[idx][1] + r[idx];
      });

      // open c = <x> + <r>
      auto c = openWith<el_t>(comm, P0, x_plus_r);

      // get correlated randomness from P2
      // let rb = r{k-1},
      //     rc = sum(r{i-m}<<(i-m)) for i in range(m, k-2)
      auto cr = comm->recv<el_t>(P2, "cr1");
      auto rb = absl::MakeSpan(cr).subspan(0, numel);
      auto rc = absl::MakeSpan(cr).subspan(numel, numel);

      std::vector<el_t> y2(numel);  // the 2-out-of-2 truncation result
      pforeach(0, numel, [&](int64_t idx) {
        // <b> = <rb> ^ c{k-1} = <rb> + c{k-1} - 2*c{k-1}*<rb>
        // note: <rb> is a randbit (in r^2k)
        const auto ck_1 = c[idx] >> (k - 1);
        auto b = rb[idx] + 0 - 2 * ck_1 * rb[idx];

        // y = c_hat - <rc> + <b> * 2^(k-m-1)
        y2[idx] = 0 - rc[idx] + (b << (k - 1 - bits));
      });

      std::vector<el_t> y3(numel);
      prg_state->fillPrssPair<el_t>(nullptr, y3.data(), y3.size(),
                                    PrgState::GenPrssCtrl::Second);
      pforeach(0, numel, [&](int64_t idx) {  //
        y2[idx] -= y3[idx];
      });
      comm->sendAsync<el_t>(P0, y2, "2to3");
      auto tmp = comm->recv<el_t>(P0, "2to3");

      // rebuild the final result.
      pforeach(0, numel, [&](int64_t idx) {
        _out[idx][0] = y2[idx] + tmp[idx];
        _out[idx][1] = y3[idx];
      });
    } else if (comm->getRank() == P2) {
      std::vector<el_t> r0(numel);
      std::vector<el_t> r1(numel);
      prg_state->fillPrssPair(r0.data(), r1.data(), r0.size(),
                              PrgState::GenPrssCtrl::Both);

      std::vector<el_t> cr0(2 * numel);
      std::vector<el_t> cr1(2 * numel);
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

      comm->sendAsync<el_t>(P0, cr0, "cr0");
      comm->sendAsync<el_t>(P1, cr1, "cr1");

      std::vector<el_t> y3(numel);
      std::vector<el_t> y1(numel);
      prg_state->fillPrssPair(y3.data(), y1.data(), y1.size(),
                              PrgState::GenPrssCtrl::Both);
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
