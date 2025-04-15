// Copyright 2025 Ant Group Co., Ltd.
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

#include "libspu/mpc/fantastic4/arithmetic.h"

#include <future>

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/fantastic4/jmp.h"
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/fantastic4/value.h"
#include "libspu/mpc/utils/ring_ops.h"

#ifdef OPTIMIZED_F4
#define OPTIMIZED_TRUNC
#endif

// We can alternatively check MAC in Trunc Kernel since it's widely used in
// applications
//    but it will output much times log infomation
// #define TRUNC_CHECKMAC

namespace spu::mpc::fantastic4 {

//  For Rep4 / Fantastic Four
//    Secret is split into 4 shares x_0, x_1, x_2, x_3
//    Differently from the paper, we let Party i (i in {0, 1, 2, 3}) holds x_i,
//    x_i+1, x_i+2 (mod 4) Similarly in prg_state.h, PRG keys are k_0, k_1, k_2,
//    k_3, we let Party i holds k_i--self, k_i+1 --next, k_i+2--next next Each
//    x_i, k_i is unknown to next party P_i+1

NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  NdArrayRef out(makeType<AShrTy>(field), shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;

    std::vector<el_t> r0(shape.numel());
    std::vector<el_t> r1(shape.numel());
    std::vector<el_t> r2(shape.numel());

    prg_state->fillPrssTuple<el_t>(r0.data(), nullptr, nullptr, r0.size(),
                                   PrgState::GenPrssCtrl::First);
    prg_state->fillPrssTuple<el_t>(nullptr, r1.data(), nullptr, r1.size(),
                                   PrgState::GenPrssCtrl::Second);
    prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r2.data(), r2.size(),
                                   PrgState::GenPrssCtrl::Third);

    NdArrayView<std::array<el_t, 3>> _out(out);
    pforeach(0, out.numel(), [&](int64_t idx) {
      // Comparison only works for [-2^(k-2), 2^(k-2)).
      // TODO: Move this constraint to upper layer, saturate it here.
      _out[idx][0] = r0[idx] >> 2;
      _out[idx][1] = r1[idx] >> 2;
      _out[idx][2] = r2[idx] >> 2;
    });
  });
  return out;
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<AShrTy>()->field();
  auto numel = in.numel();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<Pub2kTy>(field), in.shape());
    NdArrayView<pshr_el_t> _out(out);
    NdArrayView<ashr_t> _in(in);
    std::vector<ashr_el_t> x1(numel);
    std::vector<ashr_el_t> x2(numel);

    pforeach(0, numel, [&](int64_t idx) {
      x1[idx] = _in[idx][1];
      x2[idx] = _in[idx][2];
    });
    // Pass the third share to previous party
    auto x3 = JointMsgRotate<ashr_el_t>(ctx, x2, x1);

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + _in[idx][2] + x3[idx];
    });
    return out;
  });
}

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();
  auto rank = comm->getRank();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using pshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);
    NdArrayView<pshr_el_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = rank == 0 ? _in[idx] : 0;
      _out[idx][1] = rank == 3 ? _in[idx] : 0;
      _out[idx][2] = rank == 2 ? _in[idx] : 0;
    });
    return out;
  });
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using vshr_el_t = ring2k_t;
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayView<ashr_t> _in(in);
    auto out_ty = makeType<Priv2kTy>(field, rank);

    if (comm->getRank() == rank) {
      std::vector<ashr_el_t> x4(in.numel());
      JointMsgPass(ctx, x4, (rank + 1) % 4, (rank + 3) % 4, rank);

      NdArrayRef out(out_ty, in.shape());
      NdArrayView<vshr_el_t> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = _in[idx][0] + _in[idx][1] + _in[idx][2] + x4[idx];
      });
      return out;
    } else if (comm->getRank() == (rank + 1) % 4) {
      std::vector<ashr_el_t> x3(in.numel());

      pforeach(0, in.numel(), [&](int64_t idx) { x3[idx] = _in[idx][2]; });

      JointMsgPass(ctx, x3, (rank + 1) % 4, (rank + 3) % 4, rank);
      return makeConstantArrayRef(out_ty, in.shape());
    } else if (comm->getRank() == (rank + 3) % 4) {
      std::vector<ashr_el_t> x0(in.numel());

      pforeach(0, in.numel(), [&](int64_t idx) { x0[idx] = _in[idx][0]; });
      JointMsgPass(ctx, x0, (rank + 1) % 4, (rank + 3) % 4, rank);

      return makeConstantArrayRef(out_ty, in.shape());
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  });
}

// Note: Fantastic Four is for outsourced model, that is, the four parties are
// not the data owners In the 4pc.json, we use outsourced model where two
// parties holding private input distribute the shares This kernel is
// implemented for completeness, and in case for semi-honest security which
// ensures a party provide consistent input
NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const auto field = in_ty->field();
  size_t owner_rank = in_ty->owner();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using ashr_el_t = ring2k_t;
    using ashr_t = std::array<ashr_el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<ashr_t> _out(out);

    if (comm->getRank() == owner_rank) {
      auto splits = ring_rand_additive_splits(in, 3);
      comm->sendAsync(((owner_rank + 1) % 4), splits[1], "v2a 1");
      comm->sendAsync(((owner_rank + 1) % 4), splits[2], "v2a 2");
      comm->sendAsync(((owner_rank + 2) % 4), splits[2], "v2a 1");
      comm->sendAsync(((owner_rank + 2) % 4), splits[0], "v2a 2");
      comm->sendAsync(((owner_rank + 3) % 4), splits[0], "v2a 1");
      comm->sendAsync(((owner_rank + 3) % 4), splits[1], "v2a 2");

      NdArrayView<ashr_el_t> _s0(splits[0]);
      NdArrayView<ashr_el_t> _s1(splits[1]);
      NdArrayView<ashr_el_t> _s2(splits[2]);

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = _s0[idx];
        _out[idx][1] = _s1[idx];
        _out[idx][2] = _s2[idx];
      });
    } else if (comm->getRank() == (owner_rank + 1) % 4) {
      auto x1 = comm->recv<ashr_el_t>((comm->getRank() + 3) % 4, "v2a 1");
      auto x2 = comm->recv<ashr_el_t>((comm->getRank() + 3) % 4, "v2a 2");
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = x1[idx];
        _out[idx][1] = x2[idx];
        _out[idx][2] = 0;
      });
    } else if (comm->getRank() == (owner_rank + 2) % 4) {
      auto x2 = comm->recv<ashr_el_t>((comm->getRank() + 2) % 4, "v2a 1");
      auto x0 = comm->recv<ashr_el_t>((comm->getRank() + 2) % 4, "v2a 2");

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = x2[idx];
        _out[idx][1] = 0;
        _out[idx][2] = x0[idx];
      });
    } else {
      auto x0 = comm->recv<ashr_el_t>((comm->getRank() + 1) % 4, "v2a 1");
      auto x1 = comm->recv<ashr_el_t>((comm->getRank() + 1) % 4, "v2a 2");

      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = x0[idx];
        _out[idx][2] = x1[idx];
      });
    }

    return out;
  });
}

NdArrayRef NegateA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = std::make_unsigned_t<ring2k_t>;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = -_in[idx][0];
      _out[idx][1] = -_in[idx][1];
      _out[idx][2] = -_in[idx][2];
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

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      _out[idx][2] = _lhs[idx][2];
      if (rank == 0) {
        _out[idx][0] += _rhs[idx];
      }
      if (rank == 2) {
        _out[idx][2] += _rhs[idx];
      }
      if (rank == 3) {
        _out[idx][1] += _rhs[idx];
      }
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

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using shr_t = std::array<ring2k_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] + _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] + _rhs[idx][1];
      _out[idx][2] = _lhs[idx][2] + _rhs[idx][2];
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

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<el_t> _rhs(rhs);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] * _rhs[idx];
      _out[idx][1] = _lhs[idx][1] * _rhs[idx];
      _out[idx][2] = _lhs[idx][2] * _rhs[idx];
    });
    return out;
  });
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto next_rank = (rank + 1) % 4;
  auto out_ty = makeType<AShrTy>(field);
  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);

    auto out_buf =
        std::make_shared<yacl::Buffer>(lhs.shape().numel() * out_ty.size());
    memset(out_buf->data(), 0, lhs.shape().numel() * out_ty.size());
    NdArrayRef out(out_buf, out_ty, lhs.shape());
    NdArrayView<shr_t> _out(out);

    std::array<std::vector<el_t>, 5> a;

    for (auto& vec : a) {
      vec = std::vector<el_t>(lhs.numel());
    }
    pforeach(0, lhs.numel(), [&](int64_t idx) {
      for (auto i = 0; i < 5; i++) {
        a[i][idx] = 0;
      }
    });

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      a[rank][idx] = (_lhs[idx][0] + _lhs[idx][1]) * _rhs[idx][0] +
                     _lhs[idx][0] * _rhs[idx][1];  // xi*yi + xi*yj + xj*yi
      a[next_rank][idx] = (_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][1] +
                          _lhs[idx][1] * _rhs[idx][2];  // xj*yj + xj*yg + xg*yj
      a[4][idx] = _lhs[idx][0] * _rhs[idx][2] +
                  _lhs[idx][2] * _rhs[idx][0];  // xi*yg + xg*yi
    });

    JointInputArithSend<el_t>(ctx, a[0], out, 3, 0, 1, 2);
    JointInputArithSend<el_t>(ctx, a[3], out, 2, 3, 1, 0);
    JointInputArithSend<el_t>(ctx, a[1], out, 0, 1, 3, 2);
    JointInputArithSend<el_t>(ctx, a[4], out, 0, 2, 3, 1);
    JointInputArithSend<el_t>(ctx, a[2], out, 1, 2, 0, 3);
    JointInputArithSend<el_t>(ctx, a[4], out, 1, 3, 2, 0);

    JointInputArithRecv<el_t>(ctx, a[0], out, 3, 0, 1, 2);
    JointInputArithRecv<el_t>(ctx, a[3], out, 2, 3, 1, 0);
    JointInputArithRecv<el_t>(ctx, a[1], out, 0, 1, 3, 2);
    JointInputArithRecv<el_t>(ctx, a[4], out, 0, 2, 3, 1);
    JointInputArithRecv<el_t>(ctx, a[2], out, 1, 2, 0, 3);
    JointInputArithRecv<el_t>(ctx, a[4], out, 1, 3, 2, 0);

    return out;
  });
}

NdArrayRef MatMulAP::proc(KernelEvalContext*, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  const auto field = x.eltype().as<Ring2k>()->field();

  NdArrayRef z(makeType<AShrTy>(field), {x.shape()[0], y.shape()[1]});

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);
  auto x3 = getThirdShare(x);

  auto z1 = getFirstShare(z);
  auto z2 = getSecondShare(z);
  auto z3 = getThirdShare(z);

  ring_mmul_(z1, x1, y);
  ring_mmul_(z2, x2, y);
  ring_mmul_(z3, x3, y);

  return z;
}

NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto* prg_state = ctx->getState<PrgState>();
  auto* mac_state = ctx->getState<Fantastic4MacState>();
  const Type ty = makeType<RingTy>(field);
  auto M = x.shape()[0];
  // auto K = x.shape()[1];
  auto N = y.shape()[1];

  Shape mat_shape = {M, N};

  NdArrayRef out(makeType<AShrTy>(field), mat_shape);

  auto x0 = getFirstShare(x);
  auto x1 = getSecondShare(x);
  auto x2 = getThirdShare(x);

  auto y0 = getFirstShare(y);
  auto y1 = getSecondShare(y);
  auto y2 = getThirdShare(y);

  auto o0 = getFirstShare(out);
  auto o1 = getSecondShare(out);
  auto o2 = getThirdShare(out);

  NdArrayRef r_self(ty, mat_shape);
  NdArrayRef r_next(ty, mat_shape);
  NdArrayRef r_next_next(ty, mat_shape);

  prg_state->fillPrssTuple(
      r_self.data<std::uint8_t>(), r_next.data<std::uint8_t>(),
      r_next_next.data<std::uint8_t>(), mat_shape.numel() * ty.size(),
      PrgState::GenPrssCtrl::All);

  // a0 = (x0 + x1) * y0 + x0 * y1
  auto x0y1 = std::async(std::launch::async, ring_mmul, x0, y1);
  auto x0y0 = std::async(std::launch::async, ring_mmul, x0, y0);
  auto x1y0 = std::async(std::launch::async, ring_mmul, x1, y0);

  // a0 = (x1 + x2) * y1 + x1 * y2
  auto x1y2 = std::async(std::launch::async, ring_mmul, x1, y2);
  auto x1y1 = std::async(std::launch::async, ring_mmul, x1, y1);
  auto x2y1 = std::async(std::launch::async, ring_mmul, x2, y1);

  // c = x0 * y2 + x2 * y0
  auto x0y2 = std::async(std::launch::async, ring_mmul, x0, y2);
  auto x2y0 = ring_mmul(x2, y0);

  auto a0 = ring_sum({x0y1.get(), x0y0.get(), x1y0.get()});
  auto a1 = ring_sum({x1y2.get(), x1y1.get(), x2y1.get()});
  auto c = ring_add(x0y2.get(), x2y0);

  auto a0_sub_r1 = ring_sub(a0, r_next);
  auto a1_sub_r2 = ring_sub(a1, r_next_next);

  // Pi-1, Pi joint send Pi::share[0] to Pi-2
  comm->sendAsync((rank + 3) % 4, a1_sub_r2, "rotate");

  // As backup, rank = i, sender = prev rank = i-1, receiver = next next rank =
  // i+2
  mac_state->update_msg((rank + 3) % 4, rank, (rank + 2) % 4, a0_sub_r1);

  auto f0 = std::async([&] { ring_assign(o0, r_self); });
  auto f1 = std::async([&] { ring_assign(o1, r_next); });
  ring_assign(o2, r_next_next);
  f0.get();
  f1.get();

  ring_add_(o0, a0_sub_r1);
  ring_add_(o1, a1_sub_r2);

  NdArrayRef r_1(ty, mat_shape);
  NdArrayRef r_2(ty, mat_shape);
  if (rank == 0) {
    prg_state->fillPrssTuple(
        static_cast<uint8_t*>(nullptr), r_1.data<std::uint8_t>(),
        static_cast<uint8_t*>(nullptr), mat_shape.numel() * ty.size(),
        PrgState::GenPrssCtrl::Second);
    ring_add_(o1, r_1);

    prg_state->fillPrssTuple(
        static_cast<uint8_t*>(nullptr), static_cast<uint8_t*>(nullptr),
        r_2.data<std::uint8_t>(), mat_shape.numel() * ty.size(),
        PrgState::GenPrssCtrl::Third);
    auto c02_sub_r2 = ring_sub(c, r_2);
    comm->sendAsync(3, c02_sub_r2, "0-3");
    ring_add_(o0, c02_sub_r2);
    ring_add_(o2, r_2);

    auto a2_sub_r3 = comm->recv((rank + 1) % 4, c.eltype(), "rotate");
    a2_sub_r3 = a2_sub_r3.reshape(mat_shape);
    mac_state->update_msg((rank + 1) % 4, (rank + 2) % 4, rank, a2_sub_r3);
    ring_add_(o2, a2_sub_r3);
  } else if (rank == 1) {
    prg_state->fillPrssTuple(
        static_cast<uint8_t*>(nullptr), r_2.data<std::uint8_t>(),
        static_cast<uint8_t*>(nullptr), mat_shape.numel() * ty.size(),
        PrgState::GenPrssCtrl::Second);
    ring_add_(o1, r_2);

    prg_state->fillPrssTuple(
        r_1.data<std::uint8_t>(), static_cast<uint8_t*>(nullptr),
        static_cast<uint8_t*>(nullptr), mat_shape.numel() * ty.size(),
        PrgState::GenPrssCtrl::First);
    auto c13_sub_r1 = ring_sub(c, r_1);
    ring_add_(o0, r_1);
    ring_add_(o2, c13_sub_r1);
    mac_state->update_msg(3, 1, 2, c13_sub_r1);

    auto a2_sub_r3 = comm->recv((rank + 1) % 4, c.eltype(), "rotate");
    a2_sub_r3 = a2_sub_r3.reshape(mat_shape);
    mac_state->update_msg((rank + 1) % 4, (rank + 2) % 4, rank, a2_sub_r3);
    ring_add_(o2, a2_sub_r3);
  } else if (rank == 2) {
    prg_state->fillPrssTuple(
        r_2.data<std::uint8_t>(), static_cast<uint8_t*>(nullptr),
        static_cast<uint8_t*>(nullptr), mat_shape.numel() * ty.size(),
        PrgState::GenPrssCtrl::First);
    auto c02_sub_r2 = ring_sub(c, r_2);
    ring_add_(o0, r_2);
    ring_add_(o2, c02_sub_r2);
    mac_state->update_msg(0, 2, 3, c02_sub_r2);

    auto a2_sub_r3 = comm->recv((rank + 1) % 4, c.eltype(), "rotate");
    a2_sub_r3 = a2_sub_r3.reshape(mat_shape);
    mac_state->update_msg((rank + 1) % 4, (rank + 2) % 4, rank, a2_sub_r3);
    ring_add_(o2, a2_sub_r3);

    auto c13_sub_r1 = comm->recv(3, c.eltype(), "3-2");
    c13_sub_r1 = c13_sub_r1.reshape(mat_shape);
    ring_add_(o1, c13_sub_r1);
    mac_state->update_msg(3, 1, 2, c13_sub_r1);
  } else if (rank == 3) {
    prg_state->fillPrssTuple(
        static_cast<uint8_t*>(nullptr), static_cast<uint8_t*>(nullptr),
        r_1.data<std::uint8_t>(), mat_shape.numel() * ty.size(),
        PrgState::GenPrssCtrl::Third);
    auto c13_sub_r1 = ring_sub(c, r_1);
    comm->sendAsync(2, c13_sub_r1, "3-2");
    ring_add_(o0, c13_sub_r1);
    ring_add_(o2, r_1);

    auto a2_sub_r3 = comm->recv((rank + 1) % 4, c.eltype(), "rotate");
    a2_sub_r3 = a2_sub_r3.reshape(mat_shape);
    mac_state->update_msg((rank + 1) % 4, (rank + 2) % 4, rank, a2_sub_r3);
    ring_add_(o2, a2_sub_r3);

    auto c02_sub_r2 = comm->recv(0, c.eltype(), "0-3");
    c02_sub_r2 = c02_sub_r2.reshape(mat_shape);
    ring_add_(o1, c02_sub_r2);
    mac_state->update_msg(0, 2, 3, c02_sub_r2);
  }

  return out;
}

NdArrayRef LShiftA::proc(KernelEvalContext*, const NdArrayRef& in,
                         const Sizes& bits) const {
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();
  bool is_splat = bits.size() == 1;

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using shr_t = std::array<ring2k_t, 3>;

    NdArrayRef out(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);

    pforeach(0, in.numel(), [&](int64_t idx) {
      auto shift_bit = is_splat ? bits[0] : bits[idx];
      _out[idx][0] = _in[idx][0] << shift_bit;
      _out[idx][1] = _in[idx][1] << shift_bit;
      _out[idx][2] = _in[idx][2] << shift_bit;
    });

    return out;
  });
}

#ifndef OPTIMIZED_TRUNC
static NdArrayRef wrap_mul_aa(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
}
#else
// In truncA we let:
//    P0 sends P2 rb - r1
//    P2 sends P0 sb - r3
// As the result:
//    rb_shr = (0       ,- r1, rb - r1, 0)
//    sb_shr = (sb - r3 ,  0,    0    , r3)
// Recall that in 4PC MulAA:
//    pforeach(0, lhs.numel(), [&](int64_t idx) {
//      a[rank][idx] = (_lhs[idx][0] + _lhs[idx][1]) * _rhs[idx][0] +
//      _lhs[idx][0] * _rhs[idx][1]; // xi*yi + xi*yj + xj*yi a[next_rank][idx]
//      = (_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][1] + _lhs[idx][1] *
//      _rhs[idx][2];  // xj*yj + xj*yg + xg*yj a[4][idx] = _lhs[idx][0] *
//      _rhs[idx][2] + _lhs[idx][2] * _rhs[idx][0];                    // xi*yg
//      + xg*yi
//    });
// Here we have:
//    a[3] = 0
//    a[1] = 0
// For optimization, we do not send them in Opt_Mul.
NdArrayRef Opt_Mul(KernelEvalContext* ctx, const NdArrayRef& lhs,
                   const NdArrayRef& rhs) {
  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto next_rank = (rank + 1) % 4;

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;

    NdArrayView<shr_t> _lhs(lhs);
    NdArrayView<shr_t> _rhs(rhs);
    NdArrayRef out(makeType<AShrTy>(field), lhs.shape());
    NdArrayView<shr_t> _out(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = 0;
      _out[idx][1] = 0;
      _out[idx][2] = 0;
    });

    std::array<std::vector<el_t>, 5> a;

    for (auto& vec : a) {
      vec = std::vector<el_t>(lhs.numel(), el_t(0));
    }

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      a[rank][idx] = (_lhs[idx][0] + _lhs[idx][1]) * _rhs[idx][0] +
                     _lhs[idx][0] * _rhs[idx][1];  // xi*yi + xi*yj + xj*yi
      a[next_rank][idx] = (_lhs[idx][1] + _lhs[idx][2]) * _rhs[idx][1] +
                          _lhs[idx][1] * _rhs[idx][2];  // xj*yj + xj*yg + xg*yj
      a[4][idx] = _lhs[idx][0] * _rhs[idx][2] +
                  _lhs[idx][2] * _rhs[idx][0];  // xi*yg + xg*yi
      SPU_ENFORCE(a[3][idx] == 0);
      SPU_ENFORCE(a[1][idx] == 0);
    });
    // All parties send async their msgs
    JointInputArithSend<el_t>(ctx, a[0], out, 0, 3, 1, 2);
    JointInputArithSend<el_t>(ctx, a[2], out, 1, 2, 0, 3);
    JointInputArithSend<el_t>(ctx, a[4], out, 2, 0, 1, 3);
    JointInputArithSend<el_t>(ctx, a[4], out, 3, 1, 2, 0);

    // All parties recv
    JointInputArithRecv<el_t>(ctx, a[0], out, 0, 3, 1, 2);
    JointInputArithRecv<el_t>(ctx, a[2], out, 1, 2, 0, 3);
    JointInputArithRecv<el_t>(ctx, a[4], out, 2, 0, 1, 3);
    JointInputArithRecv<el_t>(ctx, a[4], out, 3, 1, 2, 0);

    return out;
  });
}
#endif

NdArrayRef TruncAPr::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t bits, SignType sign) const {
  (void)sign;

  const auto field = in.eltype().as<Ring2k>()->field();
  const size_t k = SizeOf(field) * 8;
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();
  auto rank = comm->getRank();
  auto out_ty = makeType<AShrTy>(field);

#ifdef TRUNC_CHECKMAC
  auto* mac_state = ctx->getState<Fantastic4MacState>();
#endif

  return DISPATCH_ALL_FIELDS(field, [&]() {
    using el_t = ring2k_t;
    using shr_t = std::array<el_t, 3>;
    auto buf_size = in.shape().numel() * out_ty.size();
    NdArrayRef out(out_ty, in.shape());

    // init zero buffer
    auto rb_buf = std::make_shared<yacl::Buffer>(buf_size);
    auto rc_buf = std::make_shared<yacl::Buffer>(buf_size);
    auto sb_buf = std::make_shared<yacl::Buffer>(buf_size);
    auto sc_buf = std::make_shared<yacl::Buffer>(buf_size);
    auto sb_mul_rb_buf = std::make_shared<yacl::Buffer>(buf_size);
    memset(rb_buf->data(), 0, buf_size);
    memset(rc_buf->data(), 0, buf_size);
    memset(sb_buf->data(), 0, buf_size);
    memset(sc_buf->data(), 0, buf_size);
    memset(sb_mul_rb_buf->data(), 0, buf_size);

    NdArrayRef rb_shr(rb_buf, out_ty, in.shape());
    NdArrayRef rc_shr(rc_buf, out_ty, in.shape());
    NdArrayRef sb_shr(sb_buf, out_ty, in.shape());
    NdArrayRef sc_shr(sc_buf, out_ty, in.shape());
    NdArrayRef sb_mul_rb(sb_mul_rb_buf, out_ty, in.shape());

    NdArrayView<shr_t> _out(out);
    NdArrayView<shr_t> _in(in);
    NdArrayView<shr_t> _rb_shr(rb_shr);
    NdArrayView<shr_t> _rc_shr(rc_shr);
    NdArrayView<shr_t> _sb_shr(sb_shr);
    NdArrayView<shr_t> _sc_shr(sc_shr);
    NdArrayView<shr_t> _sb_mul_rb(sb_mul_rb);

    NdArrayRef masked_input(makeType<AShrTy>(field), in.shape());
    NdArrayView<shr_t> _masked_input(masked_input);

    // Party 0
    if (rank == (size_t)0) {
      // Step 1: Generate r and rb, rc
      // locally compute PRG[1] (unknown to P2), PRG[2] (unknown to P3)
      // std::vector<el_t> r0(output.numel());
      std::vector<el_t> r1(out.numel());
      std::vector<el_t> r2(out.numel());

      auto fr = std::async([&] {
        prg_state->fillPrssTuple<el_t>(nullptr, r1.data(), nullptr, r1.size(),
                                       PrgState::GenPrssCtrl::Second);
      });
      prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r2.data(), r2.size(),
                                     PrgState::GenPrssCtrl::Third);
      fr.get();

      std::vector<el_t> rb(out.numel());
      std::vector<el_t> rc(out.numel());
      std::vector<el_t> masked_input_shr_1(out.numel());
      std::vector<el_t> masked_input_shr_2(out.numel());
      pforeach(0, out.numel(), [&](int64_t idx) {
        // r = r_{k-1}......r_{0}
        auto r = r1[idx] + r2[idx];
        // rb = r >> k-1
        rb[idx] = r >> (k - 1);
        // rc = r_{k-2}.....r_{m}
        rc[idx] = (r << 1) >> (bits + 1);

        // Step 2: compute [x] + [r]
        //  [r] = r0 + r1 + r2 + r3, only r1 and r2 are non-zero
        _masked_input[idx][0] = _in[idx][0] + (el_t(1) << (k - 2));
        _masked_input[idx][1] = _in[idx][1] + r1[idx];
        _masked_input[idx][2] = _in[idx][2] + r2[idx];

        masked_input_shr_1[idx] = _masked_input[idx][1];
        masked_input_shr_2[idx] = _masked_input[idx][2];
      });

      // Step 3: Generate the share of rb, rc
      JointInputArith(ctx, rb, rb_shr, 0, 1, 2, 3);
      JointInputArith(ctx, rc, rc_shr, 1, 0, 3, 2);

      // Step 4: Let P2 and P3 reconstruct s = x + r
      //         by P1 sends s1 to P2, P0 as backup
      //            P2 sends s2 to P3, P0 as backup
      JointMsgPass(ctx, masked_input_shr_2, 2, 0, 3);
      JointMsgPass(ctx, masked_input_shr_1, 1, 0, 2);

      // Step 5: compute sb = s{k-1} and sc = s{k-2}.....s{m}
      std::vector<el_t> sb(out.numel());
      std::vector<el_t> sc(out.numel());
      JointInputArith(ctx, sb, sb_shr, 2, 3, 0, 1);
      JointInputArith(ctx, sc, sc_shr, 3, 2, 1, 0);

#ifndef OPTIMIZED_TRUNC
      sb_mul_rb = wrap_mul_aa(ctx->sctx(), rb_shr, sb_shr);
#else
        sb_mul_rb = Opt_Mul(ctx, rb_shr, sb_shr);
#endif

      pforeach(0, out.numel(), [&](int64_t idx) {
        _out[idx][0] =
            _sc_shr[idx][0] - _rc_shr[idx][0] +
            ((_rb_shr[idx][0] + _sb_shr[idx][0] - 2 * _sb_mul_rb[idx][0])
             << (k - bits - 1)) -
            (el_t(1) << (k - bits - 2));
        _out[idx][1] =
            _sc_shr[idx][1] - _rc_shr[idx][1] +
            ((_rb_shr[idx][1] + _sb_shr[idx][1] - 2 * _sb_mul_rb[idx][1])
             << (k - bits - 1));
        _out[idx][2] =
            _sc_shr[idx][2] - _rc_shr[idx][2] +
            ((_rb_shr[idx][2] + _sb_shr[idx][2] - 2 * _sb_mul_rb[idx][2])
             << (k - bits - 1));
      });
    }
    // Party 1
    else if (rank == 1) {
      std::vector<el_t> r1(out.numel());
      std::vector<el_t> r2(out.numel());

      auto fr = std::async([&] {
        prg_state->fillPrssTuple<el_t>(r1.data(), nullptr, nullptr, r1.size(),
                                       PrgState::GenPrssCtrl::First);
      });
      prg_state->fillPrssTuple<el_t>(nullptr, r2.data(), nullptr, r2.size(),
                                     PrgState::GenPrssCtrl::Second);
      fr.get();

      std::vector<el_t> rb(out.numel());
      std::vector<el_t> rc(out.numel());
      std::vector<el_t> masked_input_shr_1(out.numel());

      pforeach(0, out.numel(), [&](int64_t idx) {
        // r = r_{k-1}......r_{0}
        auto r = r1[idx] + r2[idx];
        // rb = r >> k-1
        rb[idx] = r >> (k - 1);
        // rc = r_{k-2}.....r_{m}
        rc[idx] = (r << 1) >> (bits + 1);

        _masked_input[idx][0] = _in[idx][0] + r1[idx];
        _masked_input[idx][1] = _in[idx][1] + r2[idx];
        _masked_input[idx][2] = _in[idx][2];
        masked_input_shr_1[idx] = _masked_input[idx][0];
      });

      JointInputArith(ctx, rb, rb_shr, 0, 1, 2, 3);
      JointInputArith(ctx, rc, rc_shr, 1, 0, 3, 2);

      JointMsgPass(ctx, masked_input_shr_1, 1, 0, 2);

      std::vector<el_t> sb(out.numel());
      std::vector<el_t> sc(out.numel());
      JointInputArith(ctx, sb, sb_shr, 2, 3, 0, 1);
      JointInputArith(ctx, sc, sc_shr, 3, 2, 1, 0);
#ifndef OPTIMIZED_TRUNC
      sb_mul_rb = wrap_mul_aa(ctx->sctx(), rb_shr, sb_shr);
#else
        sb_mul_rb = Opt_Mul(ctx, rb_shr, sb_shr);
#endif

      pforeach(0, out.numel(), [&](int64_t idx) {
        _out[idx][0] =
            _sc_shr[idx][0] - _rc_shr[idx][0] +
            ((_rb_shr[idx][0] + _sb_shr[idx][0] - 2 * _sb_mul_rb[idx][0])
             << (k - bits - 1));
        _out[idx][1] =
            _sc_shr[idx][1] - _rc_shr[idx][1] +
            ((_rb_shr[idx][1] + _sb_shr[idx][1] - 2 * _sb_mul_rb[idx][1])
             << (k - bits - 1));
        _out[idx][2] =
            _sc_shr[idx][2] - _rc_shr[idx][2] +
            ((_rb_shr[idx][2] + _sb_shr[idx][2] - 2 * _sb_mul_rb[idx][2])
             << (k - bits - 1));
      });
    }
    // Party 2
    else if (rank == 2) {
      std::vector<el_t> r2(out.numel());
      std::vector<el_t> rb(out.numel());
      std::vector<el_t> rc(out.numel());
      std::vector<el_t> masked_input_shr_2(out.numel());
      std::vector<el_t> missing_shr(out.numel());

      prg_state->fillPrssTuple<el_t>(r2.data(), nullptr, nullptr, r2.size(),
                                     PrgState::GenPrssCtrl::First);

      pforeach(0, out.numel(), [&](int64_t idx) {
        _masked_input[idx][0] = _in[idx][0] + r2[idx];
        _masked_input[idx][1] = _in[idx][1];
        _masked_input[idx][2] = _in[idx][2] + (el_t(1) << (k - 2));
        masked_input_shr_2[idx] = _masked_input[idx][0];
      });

      JointMsgPass(ctx, masked_input_shr_2, 2, 0, 3);

      JointInputArith(ctx, rb, rb_shr, 0, 1, 2, 3);
      JointInputArith(ctx, rc, rc_shr, 1, 0, 3, 2);

      JointMsgPass(ctx, missing_shr, 1, 0, 2);

      std::vector<el_t> s(out.numel());
      std::vector<el_t> sb(out.numel());
      std::vector<el_t> sc(out.numel());
      pforeach(0, out.numel(), [&](int64_t idx) {
        s[idx] = _masked_input[idx][0] + _masked_input[idx][1] +
                 _masked_input[idx][2] + missing_shr[idx];
        sb[idx] = s[idx] >> (k - 1);
        sc[idx] = (s[idx] << 1) >> (bits + 1);
      });
      JointInputArith(ctx, sb, sb_shr, 2, 3, 0, 1);
      JointInputArith(ctx, sc, sc_shr, 3, 2, 1, 0);

#ifndef OPTIMIZED_TRUNC
      sb_mul_rb = wrap_mul_aa(ctx->sctx(), rb_shr, sb_shr);
#else
        sb_mul_rb = Opt_Mul(ctx, rb_shr, sb_shr);
#endif

      pforeach(0, out.numel(), [&](int64_t idx) {
        _out[idx][0] =
            _sc_shr[idx][0] - _rc_shr[idx][0] +
            ((_rb_shr[idx][0] + _sb_shr[idx][0] - 2 * _sb_mul_rb[idx][0])
             << (k - bits - 1));
        _out[idx][1] =
            _sc_shr[idx][1] - _rc_shr[idx][1] +
            ((_rb_shr[idx][1] + _sb_shr[idx][1] - 2 * _sb_mul_rb[idx][1])
             << (k - bits - 1));
        _out[idx][2] =
            _sc_shr[idx][2] - _rc_shr[idx][2] +
            ((_rb_shr[idx][2] + _sb_shr[idx][2] - 2 * _sb_mul_rb[idx][2])
             << (k - bits - 1)) -
            (el_t(1) << (k - bits - 2));
      });
    }
    // Party 3
    else if (rank == 3) {
      std::vector<el_t> r1(out.numel());
      std::vector<el_t> rb(out.numel());
      std::vector<el_t> rc(out.numel());
      std::vector<el_t> missing_shr(out.numel());

      prg_state->fillPrssTuple<el_t>(nullptr, nullptr, r1.data(), r1.size(),
                                     PrgState::GenPrssCtrl::Third);
      pforeach(0, out.numel(), [&](int64_t idx) {
        _masked_input[idx][0] = _in[idx][0] + (el_t(1) << (k - 2));
        _masked_input[idx][1] = _in[idx][1];
        _masked_input[idx][2] = _in[idx][2] + r1[idx];
      });

      JointInputArith(ctx, rb, rb_shr, 0, 1, 2, 3);
      JointInputArith(ctx, rc, rc_shr, 1, 0, 3, 2);

      JointMsgPass(ctx, missing_shr, 2, 0, 3);
      std::vector<el_t> s(out.numel());
      std::vector<el_t> sb(out.numel());
      std::vector<el_t> sc(out.numel());
      pforeach(0, out.numel(), [&](int64_t idx) {
        s[idx] = _masked_input[idx][0] + _masked_input[idx][1] +
                 _masked_input[idx][2] + missing_shr[idx];
        sb[idx] = s[idx] >> (k - 1);
        sc[idx] = (s[idx] << 1) >> (bits + 1);
      });
      JointInputArith(ctx, sb, sb_shr, 2, 3, 0, 1);
      JointInputArith(ctx, sc, sc_shr, 3, 2, 1, 0);

#ifndef OPTIMIZED_TRUNC
      sb_mul_rb = wrap_mul_aa(ctx->sctx(), rb_shr, sb_shr);
#else
        sb_mul_rb = Opt_Mul(ctx, rb_shr, sb_shr);
#endif

      pforeach(0, out.numel(), [&](int64_t idx) {
        _out[idx][0] =
            _sc_shr[idx][0] - _rc_shr[idx][0] +
            ((_rb_shr[idx][0] + _sb_shr[idx][0] - 2 * _sb_mul_rb[idx][0])
             << (k - bits - 1));
        _out[idx][1] =
            _sc_shr[idx][1] - _rc_shr[idx][1] +
            ((_rb_shr[idx][1] + _sb_shr[idx][1] - 2 * _sb_mul_rb[idx][1])
             << (k - bits - 1)) -
            (el_t(1) << (k - bits - 2));
        _out[idx][2] =
            _sc_shr[idx][2] - _rc_shr[idx][2] +
            ((_rb_shr[idx][2] + _sb_shr[idx][2] - 2 * _sb_mul_rb[idx][2])
             << (k - bits - 1));
      });
    }

#ifdef TRUNC_CHECKMAC
    mac_state->exchange_check(ctx);
#endif

    return out;
  });
}
}  // namespace spu::mpc::fantastic4