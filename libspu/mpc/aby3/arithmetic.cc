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

#include <future>

#include "libspu/mpc/aby3/ot.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/utils.h"

#ifdef CUDA_ENABLED
#include "libspu/cuda_support/utils.h"
#include "libspu/mpc/aby3/arithmetic_gpu_ext.h"
#endif

namespace spu::mpc::aby3 {
namespace {

// [zx]: Adapt this to new semantics of boolean sharing
std::vector<MemRef> a1b_offline(size_t sender, const MemRef& a,
                                ArithShareTy ashr_type, size_t self_rank,
                                PrgState* prg_state, const MemRef& b) {
  SPU_ENFORCE(a.eltype().isa<RingTy>());
  SPU_ENFORCE(b.eltype().isa<BoolShareTy>());

  auto numel = a.numel();

  auto seman_type = ashr_type.semantic_type();
  auto storage_type = ashr_type.storage_type();
  auto field = SizeOf(storage_type) * 8;

  return DISPATCH_ALL_STORAGE_TYPES(storage_type, [&]() -> std::vector<MemRef> {
    using ashr_el_t = ScalarT;
    MemRef m0(makeType<RingTy>(seman_type, field), a.shape());
    MemRef m1(makeType<RingTy>(seman_type, field), a.shape());
    ring_set_value(m0, ashr_el_t(0));
    ring_set_value(m1, ashr_el_t(1));

    MemRefView<ashr_el_t> _m0(m0);
    MemRefView<ashr_el_t> _m1(m1);

    return DISPATCH_ALL_STORAGE_TYPES(
        b.eltype().storage_type(), [&]() -> std::vector<MemRef> {
          using bshr_t = std::array<ScalarT, 2>;

          Type type = makeType<RingTy>(seman_type, field);
          const Shape& shape = a.shape();
          size_t size = shape.numel() * type.size();

          MemRef c1(type, shape);
          MemRef c2(type, shape);

          if (self_rank == sender) {
            prg_state->fillPrssPair(c1.data(), nullptr, size);
            prg_state->fillPrssPair(nullptr, c2.data(), size);

            MemRefView<ashr_el_t> _a(a);
            MemRefView<bshr_t> _b(b);
            MemRefView<ashr_el_t> _c1(c1);
            MemRefView<ashr_el_t> _c2(c2);

            // (i \xor b1 \xor b2) * a - c1 - c2
            pforeach(0, numel, [&](int64_t idx) {
              _m0[idx] = (_m0[idx] ^ (_b[idx][0] & 0x1) ^ (_b[idx][1] & 0x1)) *
                             _a[idx] -
                         _c1[idx] - _c2[idx];
              _m1[idx] = (_m1[idx] ^ (_b[idx][0] & 0x1) ^ (_b[idx][1] & 0x1)) *
                             _a[idx] -
                         _c1[idx] - _c2[idx];
            });

            return {c1, c2, m0, m1};
          } else if (self_rank == (sender + 1) % 3) {
            prg_state->fillPrssPair(c1.data(), nullptr, size);
            return {c1};
          } else {
            prg_state->fillPrssPair(nullptr, c2.data(), size);
            return {c2};
          }
        });
  });
}

std::vector<uint8_t> ring_cast_boolean(const MemRef& x) {
  SPU_ENFORCE(x.eltype().isa<RingTy>(), "expect RingTy type, got={}",
              x.eltype());
  const size_t numel = x.numel();
  std::vector<uint8_t> res(numel);

  DISPATCH_ALL_STORAGE_TYPES(x.eltype().storage_type(), [&]() {
    MemRefView<ScalarT> _x(x);
    pforeach(0, numel, [&](int64_t idx) {
      res[idx] = static_cast<uint8_t>(_x[idx] & 0x1);
    });
  });

  return res;
}

}  // namespace

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
  SPU_ENFORCE(in.eltype().storage_type() == to_type.storage_type());

  return in.as(to_type);
}

MemRef RandA::proc(KernelEvalContext* ctx, SemanticType type,
                   const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();

  auto out_type = makeType<ArithShareTy>(
      type, ctx->getState<Z2kState>()->getDefaultField());
  MemRef out(out_type, shape);

  DISPATCH_ALL_STORAGE_TYPES(out_type.storage_type(), [&]() {
    using el_t = ScalarT;

    std::vector<el_t> r0(shape.numel());
    std::vector<el_t> r1(shape.numel());
    prg_state->fillPrssPair(r0.data(), r1.data(), GetVectorNumBytes(r0));

    MemRefView<std::array<el_t, 2>> _out(out);

    pforeach(0, out.numel(), [&](int64_t idx) {
      // Comparison only works for [-2^(k-2), 2^(k-2)).
      // TODO: Move this constraint to upper layer, saturate it here.
      _out[idx][0] = r0[idx] >> 2;
      _out[idx][1] = r1[idx] >> 2;
    });
  });

  return out;
}

MemRef A2P::proc(KernelEvalContext* ctx, const MemRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* ashr_type = in.eltype().as<ArithShareTy>();
  auto numel = in.numel();

  MemRef out(makeType<Pub2kTy>(ashr_type->semantic_type()), in.shape());

  DISPATCH_ALL_STORAGE_TYPES(ashr_type->storage_type(), [&]() {
    using ashr_el_t = ScalarT;
    using ashr_t = std::array<ashr_el_t, 2>;
    MemRefView<ashr_t> _in(in);

    DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
      using pshr_el_t = ScalarT;
      MemRefView<pshr_el_t> _out(out);
      std::vector<ashr_el_t> x2(numel);

      pforeach(0, numel, [&](int64_t idx) { x2[idx] = _in[idx][1]; });

      auto x3 = comm->rotate<ashr_el_t>(x2, "a2p");  // comm => 1, k

      pforeach(0, numel, [&](int64_t idx) {
        _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
      });
    });
  });

  return out;
}

MemRef P2A::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* pub2k_type = in.eltype().as<RingTy>();
  auto ashr_type =
      makeType<ArithShareTy>(pub2k_type->semantic_type(),
                             ctx->getState<Z2kState>()->getDefaultField());

  if (ashr_type.storage_type() != pub2k_type->storage_type()) {
    MemRef in_casted(makeType<RingTy>(in.eltype().semantic_type(),
                                      SizeOf(ashr_type.storage_type()) * 8),
                     in.shape());
    ring_assign(in_casted, in);
    return proc(ctx, in_casted);
  }

  auto* comm = ctx->getState<Communicator>();
  auto comm_rank = comm->getRank();

  MemRef out(ashr_type, in.shape());

  DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
    using pshr_el_t = ScalarT;
    using ashr_el_t = ScalarT;
    using ashr_t = std::array<ashr_el_t, 2>;
    MemRefView<ashr_t> _out(out);
    MemRefView<pshr_el_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = comm_rank == 0 ? _in[idx] : 0;
      _out[idx][1] = comm_rank == 2 ? _in[idx] : 0;
    });
  });

  // for debug purpose, randomize the inputs to avoid corner cases.
#ifdef ENABLE_MASK_DURING_ABY3_P2A
  DISPATCH_ALL_STORAGE_TYPES(out.eltype().storage_type(), [&]() {
    using ashr_el_t = ScalarT;
    std::vector<ashr_el_t> r0(in.numel());
    std::vector<ashr_el_t> r1(in.numel());
    auto* prg_state = ctx->getState<PrgState>();
    prg_state->fillPrssPair(r0.data(), r1.data(), GetVectorNumBytes(r0));

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      r0[idx] = r0[idx] - r1[idx];
    }
    r1 = comm->rotate<ashr_el_t>(r0, "p2a.zero");

    MemRefView<std::array<ashr_el_t, 2>> _out(out);
    for (int64_t idx = 0; idx < in.numel(); idx++) {
      _out[idx][0] += r0[idx];
      _out[idx][1] += r1[idx];
    }
  });
#endif

  return out;
}

MemRef A2V::proc(KernelEvalContext* ctx, const MemRef& in, size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  auto comm_rank = comm->getRank();

  const auto* ashr_type = in.eltype().as<ArithShareTy>();
  auto out_ty = makeType<Priv2kTy>(ashr_type->semantic_type(), rank);

  return DISPATCH_ALL_STORAGE_TYPES(ashr_type->storage_type(), [&]() {
    using ashr_el_t = ScalarT;
    using ashr_t = std::array<ashr_el_t, 2>;

    MemRefView<ashr_t> _in(in);

    if (comm_rank == rank) {
      auto x3 = comm->recv<ashr_el_t>(comm->nextRank(), "a2v");  // comm => 1, k
      MemRef out(out_ty, in.shape());

      DISPATCH_ALL_STORAGE_TYPES(out_ty.storage_type(), [&]() {
        using vshr_el_t = ScalarT;
        //
        MemRefView<vshr_el_t> _out(out);

        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
        });
      });

      return out;
    } else if (comm_rank == (rank + 1) % 3) {
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

MemRef V2A::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* priv2k_type = in.eltype().as<Priv2kTy>();
  auto ashr_type =
      makeType<ArithShareTy>(priv2k_type->semantic_type(),
                             ctx->getState<Z2kState>()->getDefaultField());

  auto* comm = ctx->getState<Communicator>();
  auto comm_rank = comm->getRank();
  size_t owner_rank = priv2k_type->owner();

  return DISPATCH_ALL_STORAGE_TYPES(ashr_type.storage_type(), [&]() {
    using ashr_el_t = ScalarT;
    using ashr_t = std::array<ashr_el_t, 2>;

    MemRef out(ashr_type, in.shape());
    MemRefView<ashr_t> _out(out);

    if (comm_rank == owner_rank) {
      std::vector<MemRef> splits;
      if (ashr_type.storage_type() != priv2k_type->storage_type()) {
        MemRef in_casted(makeType<RingTy>(in.eltype().semantic_type(),
                                          SizeOf(ashr_type.storage_type()) * 8),
                         in.shape());
        ring_assign(in_casted, in);
        splits = ring_rand_additive_splits(in_casted, 2);
      } else {
        splits = ring_rand_additive_splits(in, 2);
      }
      comm->sendAsync(comm->nextRank(), splits[1], "v2a");  // comm => 1, k
      comm->sendAsync(comm->prevRank(), splits[0], "v2a");  // comm => 1, k

      MemRefView<ashr_el_t> _s0(splits[0]);
      MemRefView<ashr_el_t> _s1(splits[1]);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = _s0[idx];
        _out[idx][1] = _s1[idx];
      });
    } else if (comm_rank == (owner_rank + 1) % 3) {
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

MemRef NegateA::proc(KernelEvalContext* ctx, const MemRef& in) const {
  const auto* ashr_type = in.eltype().as<ArithShareTy>();

  return DISPATCH_ALL_STORAGE_TYPES(ashr_type->storage_type(), [&]() {
    using el_t = std::make_unsigned_t<ScalarT>;
    using shr_t = std::array<el_t, 2>;

    MemRef out(
        makeType<ArithShareTy>(ashr_type->semantic_type(),
                               ctx->getState<Z2kState>()->getDefaultField()),
        in.shape());
    MemRefView<shr_t> _out(out);
    MemRefView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = -_in[idx][0];
      _out[idx][1] = -_in[idx][1];
    });

    return out;
  });
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
MemRef AddAP::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  if (rhs.eltype().storage_type() != lhs.eltype().storage_type()) {
    MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                     SizeOf(lhs.eltype().storage_type()) * 8),
                    rhs.shape());
    ring_assign(rhs_cast, rhs);
    return proc(ctx, lhs, rhs_cast);
  }

  auto comm_rank = ctx->getState<Communicator>()->getRank();

  const auto* lhs_ty = lhs.eltype().as<ArithShareTy>();

  MemRef out(lhs.eltype(), lhs.shape());

  DISPATCH_ALL_STORAGE_TYPES(lhs_ty->storage_type(), [&]() {
    using lhs_t = std::array<ScalarT, 2>;

    MemRefView<lhs_t> _lhs(lhs);
    MemRefView<lhs_t> _out(out);

    MemRefView<ScalarT> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      if (comm_rank == 0) _out[idx][1] += _rhs[idx];
      if (comm_rank == 1) _out[idx][0] += _rhs[idx];
    });
  });

  return out;
}

MemRef AddAA::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<ArithShareTy>();
  // const auto* rhs_ty = rhs.eltype().as<ArithShareTy>();

  MemRef out(
      makeType<ArithShareTy>(
          std::max(lhs.eltype().semantic_type(), rhs.eltype().semantic_type()),
          ctx->getState<Z2kState>()->getDefaultField()),
      lhs.shape());

  DISPATCH_ALL_STORAGE_TYPES(lhs_ty->storage_type(), [&]() {
    using lhs_t = std::array<ScalarT, 2>;

    MemRefView<lhs_t> _lhs(lhs);
    MemRefView<lhs_t> _out(out);

    using rhs_t = std::array<ScalarT, 2>;
    MemRefView<rhs_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] + _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] + _rhs[idx][1];
    });
  });

  return out;
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
MemRef MulAP::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  if (rhs.eltype().storage_type() != lhs.eltype().storage_type()) {
    MemRef rhs_cast(makeType<RingTy>(lhs.eltype().semantic_type(),
                                     SizeOf(lhs.eltype().storage_type()) * 8),
                    rhs.shape());
    ring_assign(rhs_cast, rhs);
    return proc(ctx, lhs, rhs_cast);
  }

  MemRef out(
      makeType<ArithShareTy>(lhs.eltype().semantic_type(),
                             ctx->getState<Z2kState>()->getDefaultField()),
      lhs.shape());
  DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
    MemRefView<std::array<ScalarT, 2>> _lhs(lhs);
    MemRefView<ScalarT> _rhs(rhs);
    MemRefView<std::array<ScalarT, 2>> _out(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] * _rhs[idx];
      _out[idx][1] = _lhs[idx][1] * _rhs[idx];
    });
  });

  return out;
}

MemRef MulAA::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  return DISPATCH_ALL_STORAGE_TYPES(lhs.eltype().storage_type(), [&]() {
    using el_t = ScalarT;
    using shr_t = std::array<el_t, 2>;

    std::vector<el_t> r0(lhs.numel());
    std::vector<el_t> r1(lhs.numel());
    prg_state->fillPrssPair(r0.data(), r1.data(), GetVectorNumBytes(r0));

    MemRefView<shr_t> _lhs(lhs);
    MemRefView<shr_t> _rhs(rhs);

    // z1 = (x1 * y1) + (x1 * y2) + (x2 * y1) + (r0 - r1);
    pforeach(0, lhs.numel(), [&](int64_t idx) {
      r0[idx] = (_lhs[idx][0] * _rhs[idx][0]) + (_lhs[idx][0] * _rhs[idx][1]) +
                (_lhs[idx][1] * _rhs[idx][0]) + (r0[idx] - r1[idx]);
    });

    r1 = comm->rotate<el_t>(r0, "mulaa");  // comm => 1, k

    MemRef out(
        makeType<ArithShareTy>(std::max(lhs.eltype().semantic_type(),
                                        rhs.eltype().semantic_type()),
                               ctx->getState<Z2kState>()->getDefaultField()),
        lhs.shape());
    MemRefView<shr_t> _out(out);

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
MemRef MulA1B::proc(KernelEvalContext* ctx, const MemRef& lhs,
                    const MemRef& rhs) const {
  SPU_ENFORCE(lhs.shape() == rhs.shape());
  SPU_ENFORCE(lhs.eltype().isa<ArithShareTy>());
  SPU_ENFORCE(rhs.eltype().isa<BoolShareTy>() &&
              rhs.eltype().as<BoolShareTy>()->valid_bits() == 1);

  const auto* ashr_type = lhs.eltype().as<ArithShareTy>();
  auto seman_type = ashr_type->semantic_type();

  const size_t in_nbits = rhs.eltype().as<BoolShareTy>()->valid_bits();
  SPU_ENFORCE(in_nbits <= ashr_type->size(), "invalid nbits={}", in_nbits);

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
    Ot3 ot(SizeOf(ashr_type->storage_type()) * 8, lhs.shape(),
           Ot3::RoleRanks{sender, (sender + 2) % 3, (sender + 1) % 3}, comm,
           prg_state, false);
    return ot;
  };

  // TODO: optimization for large input.
  // online part: tasks two rounds latency. do 3-parties OT.
  auto offline = [&](size_t sender, const MemRef& a) {
    return a1b_offline(sender, a, *ashr_type, self_rank, prg_state, rhs);
  };

  // parallel online: parallel two 3-parties OT.
  auto parallel_online =
      [&](size_t sender1, const std::vector<MemRef>& data1, size_t sender2,
          const std::vector<MemRef>& data2) -> std::pair<MemRef, MemRef> {
    auto ot1 = get_ot(sender1);
    auto ot2 = get_ot(sender2);

    std::pair<MemRef, MemRef> r1;
    std::pair<MemRef, MemRef> r2;

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
      auto c1 = ot1.recv(ring_cast_boolean(b1), seman_type);
      comm->sendAsync((sender1 + 1) % 3, c1, "ABY3-MUL-R1C1");  // 1k
      r1 = {c1.reshape(data1[0].shape()), data1[0]};
    }
    if (self_rank == (sender2 + 2) % 3) {
      // 1 latency overlapping with "ABY3-MUL-R1C1"
      auto c1 = ot2.recv(ring_cast_boolean(b1), seman_type);
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

    DISPATCH_ALL_STORAGE_TYPES(ashr_type->storage_type(), [&]() {
      MemRefView<ScalarT> r1_0(r1.first);
      MemRefView<ScalarT> r1_1(r1.second);

      MemRefView<ScalarT> r2_0(r2.first);
      MemRefView<ScalarT> r2_1(r2.second);
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

  return makeArithShare(ret.first, ret.second, seman_type,
                        ctx->getState<Z2kState>()->getDefaultField());
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
MemRef MatMulAP::proc(KernelEvalContext* ctx, const MemRef& x,
                      const MemRef& y) const {
  if (y.eltype().storage_type() != x.eltype().storage_type()) {
    MemRef y_casted(makeType<RingTy>(y.eltype().semantic_type(),
                                     SizeOf(x.eltype().storage_type()) * 8),
                    y.shape());
    ring_assign(y_casted, y);
    return proc(ctx, x, y_casted);
  }

  auto sem_type = x.eltype().semantic_type();
  MemRef z(makeType<ArithShareTy>(sem_type,
                                  ctx->getState<Z2kState>()->getDefaultField()),
           {x.shape()[0], y.shape()[1]});

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto z1 = getFirstShare(z);
  auto z2 = getSecondShare(z);

  ring_mmul_(z1, x1, y);
  ring_mmul_(z2, x2, y);

  return z;
}

MemRef MatMulAA::proc(KernelEvalContext* ctx, const MemRef& x,
                      const MemRef& y) const {
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  const auto* ring_type = x.eltype().as<RingTy>();
  auto seman_type =
      std::max(x.eltype().semantic_type(), y.eltype().semantic_type());

  auto M = x.shape()[0];
  auto N = y.shape()[1];

  auto r = std::async([&] {
    const Type type = makeType<RingTy>(seman_type, ring_type->valid_bits());
    const Shape shape = {M, N};

    MemRef r0(type, shape);
    MemRef r1(type, shape);

    prg_state->fillPrssPair(r0.data(), r1.data(), shape.numel() * type.size());

    return ring_sub(r0, r1);
  });

  MemRef out(makeType<ArithShareTy>(
                 seman_type, ctx->getState<Z2kState>()->getDefaultField()),
             {M, N});
  auto o1 = getFirstShare(out);
  auto o2 = getSecondShare(out);

#ifdef CUDA_ENABLED
  // FIXME: better heuristic?
  if (!spu::cuda::hasGPUDevice() || M * N <= 20000 || field != FM64) {
#endif
    auto x1 = getFirstShare(x);
    auto x2 = getSecondShare(x);

    auto y1 = getFirstShare(y);
    auto y2 = getSecondShare(y);
    // z1 := x1*y1 + x1*y2 + x2*y1 + k1
    // z2 := x2*y2 + x2*y3 + x3*y2 + k2
    // z3 := x3*y3 + x3*y1 + x1*y3 + k3

    // x1*(y1+y2) + x2*y1 + k1
    auto t2 = std::async(ring_mmul, x2, y1);
    auto t0 = ring_mmul(x1, ring_add(y1, y2));  //
    auto z1 = ring_sum({t0, t2.get(), r.get()});

    auto f = std::async([&] { ring_assign(o1, z1); });
    ring_assign(o2, comm->rotate(z1, kBindName()));  // comm => 1, k
    f.get();
#ifdef CUDA_ENABLED
  } else {
    matmul_aa_gpu(x, y, o1);
    ring_add_(o1, r.get());
    ring_assign(o2, comm->rotate(o1, kBindName()));  // comm => 1, k
  }
#endif

  return out;
}

MemRef LShiftA::proc(KernelEvalContext* ctx, const MemRef& in,
                     const Sizes& bits) const {
  const auto* ashr_type = in.eltype().as<ArithShareTy>();

  return DISPATCH_ALL_STORAGE_TYPES(ashr_type->storage_type(), [&]() {
    using shr_t = std::array<ScalarT, 2>;

    MemRef out(
        makeType<ArithShareTy>(ashr_type->semantic_type(),
                               ctx->getState<Z2kState>()->getDefaultField()),
        in.shape());
    MemRefView<shr_t> _out(out);
    MemRefView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      int64_t shift = bits.size() == 1 ? bits[0] : bits[idx];
      _out[idx][0] = _in[idx][0] << shift;
      _out[idx][1] = _in[idx][1] << shift;
    });

    return out;
  });
}

// Refer to:
// Share Truncation I, 5.1 Fixed-point Arithmetic, P13,
// ABY3: A Mixed Protocol Framework for Machine Learning
// - https://eprint.iacr.org/2018/403.pdf
MemRef TruncA::proc(KernelEvalContext* ctx, const MemRef& in, size_t bits,
                    SignType sign) const {
  (void)sign;  // TODO: optimize me.

  const auto* ring_type = in.eltype().as<RingTy>();
  auto seman_type = ring_type->semantic_type();

  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto r_future = std::async([&] {
    const Type type = makeType<RingTy>(seman_type, ring_type->valid_bits());
    const Shape shape = in.shape();

    MemRef r0(type, shape);
    MemRef r1(type, shape);

    prg_state->fillPrssPair(r0.data(), r1.data(), shape.numel() * type.size());

    return std::make_pair(r0, r1);
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
      const auto z1 = ring_arshift(x1, {static_cast<int64_t>(bits)});
      const auto z2 = comm->recv(1, x1.eltype(), kBindName());
      return makeArithShare(z1, z2, seman_type,
                            ctx->getState<Z2kState>()->getDefaultField());
    }

    case 1: {
      auto r1 = r_future.get().second;
      const auto z1 = ring_sub(
          ring_arshift(ring_add(x1, x2), {static_cast<int64_t>(bits)}), r1);
      comm->sendAsync(0, z1, kBindName());
      return makeArithShare(z1, r1, seman_type,
                            ctx->getState<Z2kState>()->getDefaultField());
    }

    case 2: {
      const auto z2 = ring_arshift(x2, {static_cast<int64_t>(bits)});
      return makeArithShare(r_future.get().first, z2, seman_type,
                            ctx->getState<Z2kState>()->getDefaultField());
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
MemRef TruncAPr::proc(KernelEvalContext* ctx, const MemRef& in, size_t bits,
                      SignType sign) const {
  (void)sign;  // TODO, optimize me.

  const auto* ashr_type = in.eltype().as<ArithShareTy>();
  const size_t k = SizeOf(ashr_type->storage_type()) * 8;
  const auto numel = in.numel();

  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  // TODO: cost model is asymmetric, but test framework requires the same.
  comm->addCommStatsManually(3, (k / 2) * numel);

  // 1. P0 & P1 samples r together.
  // 2. P2 knows r and compute correlated random r{k-1} & sum(r{m~(k-2)})

  size_t pivot;
  prg_state->fillPubl(&pivot, sizeof(pivot));
  size_t P0 = pivot % 3;
  size_t P1 = (pivot + 1) % 3;
  size_t P2 = (pivot + 2) % 3;

  MemRef out(in.eltype(), in.shape());
  DISPATCH_ALL_STORAGE_TYPES(ashr_type->storage_type(), [&]() {
    using el_t = ScalarT;
    using shr_t = std::array<el_t, 2>;

    MemRefView<shr_t> _out(out);
    MemRefView<shr_t> _in(in);

    if (comm->getRank() == P0) {
      std::vector<el_t> r(numel);
      prg_state->fillPrssPair(r.data(), nullptr, GetVectorNumBytes(r));
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
      prg_state->fillPrssPair(y1.data(), nullptr, GetVectorNumBytes(y1));
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
      prg_state->fillPrssPair(nullptr, r.data(), GetVectorNumBytes(r));

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
      prg_state->fillPrssPair(nullptr, y3.data(), GetVectorNumBytes(y3));
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
      prg_state->fillPrssPair(r0.data(), r1.data(), GetVectorNumBytes(r0));

      std::vector<el_t> cr0(2 * numel);
      std::vector<el_t> cr1(2 * numel);
      auto rb0 = absl::MakeSpan(cr0).subspan(0, numel);
      auto rc0 = absl::MakeSpan(cr0).subspan(numel, numel);
      auto rb1 = absl::MakeSpan(cr1).subspan(0, numel);
      auto rc1 = absl::MakeSpan(cr1).subspan(numel, numel);

      prg_state->fillPriv(cr0.data(), GetVectorNumBytes(cr0));
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
      prg_state->fillPrssPair(y3.data(), y1.data(), GetVectorNumBytes(y1));
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
