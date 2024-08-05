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

#include "libspu/mpc/spdz2k/conversion.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/core/xt_helper.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdz2k/arithmetic.h"
#include "libspu/mpc/spdz2k/boolean.h"
#include "libspu/mpc/spdz2k/state.h"
#include "libspu/mpc/spdz2k/type.h"
#include "libspu/mpc/spdz2k/value.h"
#include "libspu/mpc/utils/circuits.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {

namespace {

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape(), "x shape {}, y shape {}", x.shape(),
              y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_and_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(and_bb(ctx, WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_a2bit(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(dynDispatch(ctx, "a2bit", WrapValue(x)));
}

static NdArrayRef wrap_bit2a(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(dynDispatch(ctx, "bit2a", WrapValue(x)));
}

static NdArrayRef wrap_b2a(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(b2a(ctx, WrapValue(x)));
}

static NdArrayRef wrap_p2b(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(p2b(ctx, WrapValue(x)));
}

static NdArrayRef wrap_not_b(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(dynDispatch(ctx, "not_b", WrapValue(x)));
}

static NdArrayRef wrap_bitle_bb(SPUContext* ctx, const NdArrayRef& x,
                                const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(dynDispatch(ctx, "bitle_bb", WrapValue(x), WrapValue(y)));
}

static NdArrayRef wrap_carray_out(const CircuitBasicBlock<Value>& cbb,
                                  const NdArrayRef& x, const NdArrayRef& y,
                                  size_t nbits) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(carry_out(cbb, WrapValue(x), WrapValue(y), nbits));
}

static NdArrayRef wrap_kogge_stone(const CircuitBasicBlock<Value>& cbb,
                                   const NdArrayRef& x, const NdArrayRef& y,
                                   size_t nbits) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(kogge_stone(cbb, WrapValue(x), WrapValue(y), nbits));
}

inline bool _IsB(const Value& x) { return x.storage_type().isa<BShare>(); }
inline bool _IsP(const Value& x) { return x.storage_type().isa<Public>(); }

#define COMMUTATIVE_DISPATCH(FnPP, FnBP, FnBB)    \
  if (_IsP(x) && _IsP(y)) {                       \
    return FnPP(ctx, x, y);                       \
  } else if (_IsB(x) && _IsP(y)) {                \
    return FnBP(ctx, x, y);                       \
  } else if (_IsP(x) && _IsB(y)) {                \
    return FnBP(ctx, y, x);                       \
  } else if (_IsB(x) && _IsB(y)) {                \
    return FnBB(ctx, y, x);                       \
  } else {                                        \
    SPU_THROW("unsupported op x={}, y={}", x, y); \
  }

CircuitBasicBlock<Value> MakeSPDZBasicBlock(SPUContext* ctx, FieldType field) {
  using T = Value;
  CircuitBasicBlock<T> cbb;
  cbb._xor = [=](T const& x, T const& y) -> T {
    COMMUTATIVE_DISPATCH(xor_pp, xor_bp, xor_bb);
  };
  cbb._and = [=](T const& x, T const& y) -> T {
    COMMUTATIVE_DISPATCH(and_pp, and_bp, and_bb);
  };
  cbb.lshift = [=](T const& x, size_t bits) -> T {
    if (_IsP(x)) {
      return lshift_p(ctx, x, bits);
    } else if (_IsB(x)) {
      return lshift_b(ctx, x, bits);
    }
    SPU_THROW("unsupported op x={}", x);
  };
  cbb.rshift = [=](T const& x, size_t bits) -> T {
    if (_IsP(x)) {
      return rshift_p(ctx, x, bits);
    } else if (_IsB(x)) {
      return rshift_b(ctx, x, bits);
    }
    SPU_THROW("unsupported op x={}", x);
  };
  cbb.init_like = [=](T const& x, uint128_t init) -> T {
    return make_p(ctx, init, x.shape(), field);
  };
  cbb.set_nbits = [=](T& x, size_t nbits) {
    return x.storage_type().as<BShare>()->setNbits(nbits);
  };
  return cbb;
}

};  // namespace

NdArrayRef A2Bit::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto* state = ctx->getState<Spdz2kState>()->getStateImpl(field);
  const size_t s = state->s();
  const size_t nbits = 1;

  // 1. value mod 2^{s+1}
  //    mac   mod 2^{s+1}
  const auto& in_val = getValueShare(in);
  const auto& in_mac = GetMacShare(ctx, in);
  auto res_val = ring_bitmask(in_val, 0, s + 1);
  auto res_mac = ring_bitmask(in_mac, 0, s + 1);

  // 2. makeBShare
  return makeBShare(res_val, res_mac, field, nbits);
}

NdArrayRef Bit2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  const auto* state = ctx->getState<Spdz2kState>()->getStateImpl(field);
  auto* beaver = state->beaver();
  const auto key = state->key();
  const size_t k = state->k();
  const size_t s = state->s();

  // The protocol for Bit2A in SPDZ2k
  // reference: https://eprint.iacr.org/2019/599.pdf
  // Page 6, Figure 3.

  // 1. Reserve the least significant bit
  const auto [_in_val, _in_mac] = BShareSwitch2Nbits(in, 1);
  const auto _in = makeBShare(_in_val, _in_mac, field, 1);

  // 2. get random bit [r] in the form of A-share
  NdArrayRef r, r_mac;
  std::tie(r, r_mac) = beaver->AuthRandBit(field, _in.shape(), k, s);
  auto ar = makeAShare(r, r_mac, field);

  // 3. Convert [r] into B-share
  auto br = wrap_a2bit(ctx->sctx(), ar);

  // 4. c = open([x] + [r])
  auto bc = wrap_add_bb(ctx->sctx(), _in, br);
  // Notice we only reserve the least significant bit
  const auto [bc_val, bc_mac] = BShareSwitch2Nbits(bc, 1);

  NdArrayRef c, zero_mac;
  std::tie(c, zero_mac) = beaver->BatchOpen(bc_val, bc_mac, 1, s);
  SPU_ENFORCE(beaver->BatchMacCheck(c, zero_mac, 1, s));
  ring_bitmask_(c, 0, 1);

  // 5. [x] = c + [r] - 2 * c * [r]
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _c(c);
    NdArrayView<ring2k_t> _r(r);
    NdArrayView<ring2k_t> _r_mac(r_mac);

    NdArrayRef out(makeType<AShrTy>(field, true), _in.shape());
    NdArrayView<std::array<ring2k_t, 2>> _out(out);

    pforeach(0, out.numel(), [&](int64_t idx) {
      _out[idx][0] = _r[idx] - 2 * _c[idx] * _r[idx];
      if (comm->getRank() == 0) {
        _out[idx][0] += _c[idx];
      }

      _out[idx][1] = _c[idx] * key + _r_mac[idx] - 2 * _c[idx] * _r_mac[idx];
    });

    return out;
  });
}

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto* state = ctx->getState<Spdz2kState>()->getStateImpl(field);
  auto* beaver = state->beaver();
  const size_t k = state->k();
  const size_t s = state->s();

  // 1. get rand bit r
  Shape r_shape = in.shape();
  r_shape.back() *= k;
  auto [rbit, rbit_mac] = beaver->AuthRandBit(field, r_shape, k, s);
  auto arbit = makeAShare(rbit, rbit_mac, field);

  // 2. a2bit
  auto _br = wrap_a2bit(ctx->sctx(), arbit);
  auto br_val = getValueShare(_br);
  auto br_mac = getMacShare(_br);
  auto br = makeBShare(br_val, br_mac, field, k);
  auto ar = wrap_b2a(ctx->sctx(), br);

  // 3. open a - r
  const auto& in_val = getValueShare(in);
  const auto& r_val = getValueShare(ar);
  auto a_r_val = ring_sub(in_val, r_val);

  const auto& in_mac = GetMacShare(ctx, in);
  const auto& r_mac = getMacShare(ar);
  auto a_r_mac = ring_sub(in_mac, r_mac);

  auto [c, check_mac] = beaver->BatchOpen(a_r_val, a_r_mac, k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(c, check_mac, k, s));

  // 4. binary add
  auto ty = makeType<Pub2kTy>(field);
  ring_bitmask_(c, 0, k);
  auto bc = wrap_p2b(ctx->sctx(), c.as(ty));
  auto [bc_val, bc_mac] = BShareSwitch2Nbits(bc, k);

  bc = makeBShare(bc_val, bc_mac, field, k);
  auto res = wrap_add_bb(ctx->sctx(), br, bc);
  return res;
}

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<BShrTy>()->field();
  const auto nbits = in.eltype().as<BShrTy>()->nbits();
  auto* comm = ctx->getState<Communicator>();
  const auto* state = ctx->getState<Spdz2kState>()->getStateImpl(field);
  auto* beaver = state->beaver();
  const auto key = state->key();
  const size_t k = state->k();
  const size_t s = state->s();

  const auto _in_val = getValueShare(in);
  const auto _in_mac = getMacShare(in);
  const auto _in = makeBShare(_in_val, _in_mac, field, 1);
  // const size_t out_numel = _in.numel() / nbits;

  Shape out_shape = _in.shape();
  out_shape.back() /= nbits;

  // 1. get rand bit [r]
  NdArrayRef r;
  NdArrayRef r_mac;
  std::tie(r, r_mac) = beaver->AuthRandBit(field, _in.shape(), k, s);
  auto ar = makeAShare(r, r_mac, field);

  // 2. a2bit
  auto br = wrap_a2bit(ctx->sctx(), ar);

  // 3. c = open([x] + [r])
  auto bc = wrap_add_bb(ctx->sctx(), _in, br);
  // Notice we only reserve the least significant bit
  const auto [bc_val, bc_mac] = BShareSwitch2Nbits(bc, 1);

  NdArrayRef c;
  NdArrayRef zero_mac;
  std::tie(c, zero_mac) = beaver->BatchOpen(bc_val, bc_mac, 1, s);
  SPU_ENFORCE(beaver->BatchMacCheck(c, zero_mac, 1, s));
  ring_bitmask_(c, 0, 1);

  // 4. [x] = c + [r] - 2 * c * [r]
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayRef out(makeType<AShrTy>(field, true), out_shape);
    NdArrayRef expand_out(makeType<AShrTy>(field, true), _in.shape());

    NdArrayView<ring2k_t> _c(c);
    NdArrayView<ring2k_t> _r(r);
    NdArrayView<ring2k_t> _r_mac(r_mac);

    NdArrayView<std::array<ring2k_t, 2>> _out(out);
    NdArrayView<std::array<ring2k_t, 2>> _expand_out(expand_out);

    pforeach(0, _in.numel(), [&](int64_t idx) {
      _expand_out[idx][0] = (_r[idx] - 2 * _c[idx] * _r[idx]);
      if (comm->getRank() == 0) {
        _expand_out[idx][0] += _c[idx];
      }
      _expand_out[idx][1] =
          _c[idx] * key + _r_mac[idx] - 2 * _c[idx] * _r_mac[idx];
    });

    pforeach(0, out.numel(), [&](int64_t idx) {
      _out[idx][0] = 0;
      _out[idx][1] = 0;
      for (size_t jdx = 0; jdx < nbits; ++jdx) {
        _out[idx][0] += (_expand_out[idx * nbits + jdx][0]) << jdx;
        _out[idx][1] += (_expand_out[idx * nbits + jdx][1]) << jdx;
      }
    });

    return out;
  });
}

NdArrayRef MSB::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto* state = ctx->getState<Spdz2kState>()->getStateImpl(field);
  auto* beaver = state->beaver();
  const auto key = state->key();
  const int64_t k = state->k();
  const int64_t s = state->s();
  const auto* comm = ctx->getState<Communicator>();

  // The protocol for extracting MSB in SPDZ2k
  // reference: https://eprint.iacr.org/2019/599.pdf
  // Page7, Figure 6.

  auto _in = getValueShare(in);
  auto _in_mac = GetMacShare(ctx, in);

  auto [c_in, c_in_mac] = beaver->BatchOpen(_in, _in_mac, k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(c_in, c_in_mac, k, s));

  auto _r_val = ring_zeros(field, in.shape());
  auto _r_mac = ring_zeros(field, in.shape());
  std::vector<NdArrayRef> _r_vec;
  std::vector<NdArrayRef> _r_mac_vec;

  // 1. generate random bit b , r_0 , ... , r_{k-1}
  //    then set r = \sum r_i 2^{i}
  for (int64_t i = 0; i < k; ++i) {
    auto [_r_i, _r_i_mac] = beaver->AuthRandBit(field, in.shape(), k, s);
    ring_add_(_r_val, ring_lshift(_r_i, i));
    ring_add_(_r_mac, ring_lshift(_r_i_mac, i));
    // record r_i & r_i_mac
    _r_vec.emplace_back(std::move(_r_i));
    _r_mac_vec.emplace_back(std::move(_r_i_mac));
  }

  // 2. reveal a + r
  auto _c = ring_add(_in, _r_val);
  auto _c_mac = ring_add(_in_mac, _r_mac);
  auto [c_open, zero_mac] = beaver->BatchOpen(_c, _c_mac, k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(c_open, zero_mac, k, s));
  auto _c_open = ring_bitmask(c_open, 0, k - 1);

  // 3. convert r from A-share to B-share
  //    set r' be the B-share for sum_{i=0}^{k-2} r_i
  auto shape_k_1 = in.shape();
  shape_k_1.back() *= (k - 1);
  auto _bt_r = ring_zeros(field, shape_k_1);
  auto _bt_r_mac = ring_zeros(field, shape_k_1);
  auto _ar = ring_zeros(field, in.shape());
  auto _ar_mac = ring_zeros(field, in.shape());
  auto strides_k_1 = _bt_r.strides();
  strides_k_1.back() *= (k - 1);

  const auto ty = makeType<RingTy>(field);
  for (int64_t i = 0; i < k - 1; ++i) {
    ring_add_(_ar, ring_lshift(_r_vec[i], i));
    ring_add_(_ar_mac, ring_lshift(_r_mac_vec[i], i));

    auto at_r_i = makeAShare(_r_vec[i], _r_mac_vec[i], field);
    auto bt_r_i = wrap_a2bit(ctx->sctx(), at_r_i);
    const auto _bt_r_i = getValueShare(bt_r_i);
    const auto _bt_r_i_mac = getMacShare(bt_r_i);
    auto _sub_bt_r =
        NdArrayRef(_bt_r.buf(), ty, in.shape(), strides_k_1,
                   _bt_r.offset() + i * static_cast<int64_t>(ty.size()));
    auto _sub_bt_r_mac =
        NdArrayRef(_bt_r_mac.buf(), ty, in.shape(), strides_k_1,
                   _bt_r_mac.offset() + i * static_cast<int64_t>(ty.size()));
    ring_add_(_sub_bt_r, _bt_r_i.reshape(_sub_bt_r.shape()));
    ring_add_(_sub_bt_r_mac, _bt_r_i_mac.reshape(_sub_bt_r_mac.shape()));
  }
  auto br = makeBShare(_bt_r, _bt_r_mac, field, k - 1);

  // 4. u = BitLT( c , r' )
  // todo: Here should be ctx->caller()->call("bitlt_pb", _pc , br)
  //                   or ctx->caller()->call("bitle_bp", br , _pc)
  //                   or ctx->caller()->call("bitlt_bb", _bc, br)
  auto _pc = _c_open.as(makeType<Pub2kTy>(field));
  ring_bitmask_(_pc, 0, k);
  auto _bc = wrap_p2b(ctx->sctx(), _pc);
  auto not_u = wrap_bitle_bb(ctx->sctx(), br, _bc);
  auto bu = wrap_not_b(ctx->sctx(), not_u);

  // 5. convert u from B-share to A-share
  auto au = wrap_bit2a(ctx->sctx(), bu);
  // 6. Compute a' = c' - r' + 2^{k-1} u
  //            d  = a  - a'
  auto _au = getValueShare(au);
  auto _au_mac = getMacShare(au);
  auto _aa = ring_sub(ring_lshift(_au, k - 1), _ar);
  auto _aa_mac = ring_sub(ring_lshift(_au_mac, k - 1), _ar_mac);
  if (comm->getRank() == 0) {
    ring_add_(_aa, _c_open);
  }
  ring_add_(_aa_mac, ring_mul(_c_open, key));
  auto _d = ring_sub(_in, _aa);
  auto _d_mac = ring_sub(_in_mac, _aa_mac);

  // 7. let e = d + 2^{k-1} b, then open e
  auto [_b, _b_mac] = beaver->AuthRandBit(field, in.shape(), k, s);
  auto _e = ring_add(_d, ring_lshift(_b, k - 1));
  auto _e_mac = ring_add(_d_mac, ring_lshift(_b_mac, k - 1));

  auto [e_open, e_zero_mac] = beaver->BatchOpen(_e, _e_mac, k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(e_open, e_zero_mac, k, s));

  // 8. e' be the most significant bit of e
  auto _ee = ring_bitmask(ring_rshift(e_open, k - 1), 0, 1);

  // 9. output e_{k-1} + b - 2 e_{k-1} b
  auto _ret = ring_sub(_b, ring_lshift(ring_mul(_b, _ee), 1));
  auto _ret_mac = ring_sub(_b_mac, ring_lshift(ring_mul(_b_mac, _ee), 1));
  if (comm->getRank() == 0) {
    ring_add_(_ret, _ee);
  }
  ring_add_(_ret_mac, ring_mul(_ee, key));
  SPU_ENFORCE(_ret.shape() == in.shape());

  return makeBShare(_ret, _ret_mac, field, 1);
}

NdArrayRef AddBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
  const size_t nbits = maxNumBits(lhs, rhs);
  const auto field = lhs.eltype().as<Ring2k>()->field();
  const auto [x_val, x_mac] = BShareSwitch2Nbits(lhs, nbits);
  const auto x = makeBShare(x_val, x_mac, field, nbits);
  const auto [y_val, y_mac] = BShareSwitch2Nbits(rhs, nbits);
  const auto y = makeBShare(y_val, y_mac, field, nbits);

  auto cbb = MakeSPDZBasicBlock(ctx->sctx(), field);
  // sklansky has more local computation which leads to lower performance.
  auto res = wrap_kogge_stone(cbb, x, y, nbits);

  return res;
}

NdArrayRef AddBP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const size_t nbits = maxNumBits(lhs, rhs);
  const auto field = lhs.eltype().as<Ring2k>()->field();
  const auto [x_val, x_mac] = BShareSwitch2Nbits(lhs, nbits);
  const auto x = makeBShare(x_val, x_mac, field, nbits);
  const auto& y = rhs;

  auto d_field = Spdz2kState::getDataField(lhs.eltype().as<BShrTy>()->field());
  auto cbb = MakeSPDZBasicBlock(ctx->sctx(), d_field);
  // sklansky has more local computation which leads
  // to lower performance.
  return wrap_kogge_stone(cbb, x, y, nbits);
}

#if 0
ArrayRef BitLTBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                       const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
  const auto nbits = maxNumBits(lhs, rhs);
  const auto field = lhs.eltype().as<Ring2k>()->field();
  const auto numel = lhs.numel();
  auto rhs_not = ctx->caller()->call("not_b", rhs);
  auto cbb = MakeSPDZBasicBlock(ctx->sctx());

  // Full adder implementation using two half adders
  // TODO: design an efficient full adder in circuit.h
  auto sum = kogge_stone<ArrayRef>(cbb, lhs, rhs_not, nbits);
  auto carry1 = carry_out<ArrayRef>(cbb, lhs, rhs_not, nbits);

  const auto p_numel = numel;
  auto ones = pt_ones(field, p_numel);

  auto carry2 = carry_out<ArrayRef>(cbb, sum, ones, nbits);
  auto ret = ctx->caller()->call("xor_bb", carry1, carry2);

  auto res = ctx->caller()->call("not_b", ret);
  SPU_ENFORCE(res.numel() == lhs.numel(), "_ret numel {}, lhs numel is {}",
              res.numel(), lhs.numel());
  return res;
}
#else
NdArrayRef BitLTBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  auto res0 = wrap_bitle_bb(ctx->sctx(), lhs, rhs);
  auto res1 = wrap_bitle_bb(ctx->sctx(), rhs, lhs);
  auto eq = wrap_and_bb(ctx->sctx(), res0, res1);
  auto neq = wrap_not_b(ctx->sctx(), eq);
  auto res = wrap_and_bb(ctx->sctx(), neq, res0);

  return res;
}
#endif

NdArrayRef BitLEBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto nbits = maxNumBits(lhs, rhs);
  const auto field = lhs.eltype().as<Ring2k>()->field();

  const auto [x_val, x_mac] = BShareSwitch2Nbits(lhs, nbits);
  const auto x = makeBShare(x_val, x_mac, field, nbits);
  const auto [y_val, y_mac] = BShareSwitch2Nbits(rhs, nbits);
  const auto y = makeBShare(y_val, y_mac, field, nbits);

  auto y_not = wrap_not_b(ctx->sctx(), y);
  auto d_field = Spdz2kState::getDataField(lhs.eltype().as<BShrTy>()->field());
  auto cbb = MakeSPDZBasicBlock(ctx->sctx(), d_field);
  auto ret = wrap_carray_out(cbb, x, y_not, nbits);

  auto res = wrap_not_b(ctx->sctx(), ret);
  SPU_ENFORCE(res.shape() == lhs.shape());
  return res;
}
};  // namespace spu::mpc::spdz2k