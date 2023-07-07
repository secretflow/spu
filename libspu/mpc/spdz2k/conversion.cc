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
#include "libspu/mpc/spdz2k/type.h"
#include "libspu/mpc/spdz2k/value.h"
#include "libspu/mpc/utils/circuits.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {

namespace {

static ArrayRef wrap_add_bb(SPUContext* ctx, const ArrayRef& x,
                            const ArrayRef& y) {
  SPU_ENFORCE(x.numel() == y.numel());
  const Shape shape = {x.numel()};
  auto [res, _s, _t] =
      UnwrapValue(add_bb(ctx, WrapValue(x, shape), WrapValue(y, shape)));
  return res;
}

static ArrayRef wrap_and_bb(SPUContext* ctx, const ArrayRef& x,
                            const ArrayRef& y) {
  SPU_ENFORCE(x.numel() == y.numel());
  const Shape shape = {x.numel()};
  auto [res, _s, _t] =
      UnwrapValue(and_bb(ctx, WrapValue(x, shape), WrapValue(y, shape)));
  return res;
}

static ArrayRef wrap_a2bit(SPUContext* ctx, const ArrayRef& x) {
  const Shape shape = {x.numel()};
  auto [res, _s, _t] =
      UnwrapValue(dynDispatch(ctx, "a2bit", WrapValue(x, shape)));
  return res;
}

static ArrayRef wrap_bit2a(SPUContext* ctx, const ArrayRef& x) {
  const Shape shape = {x.numel()};
  auto [res, _s, _t] =
      UnwrapValue(dynDispatch(ctx, "bit2a", WrapValue(x, shape)));
  return res;
}

static ArrayRef wrap_b2a(SPUContext* ctx, const ArrayRef& x) {
  const Shape shape = {x.numel()};
  auto [res, _s, _t] = UnwrapValue(b2a(ctx, WrapValue(x, shape)));
  return res;
}

static ArrayRef wrap_p2b(SPUContext* ctx, const ArrayRef& x) {
  const Shape shape = {x.numel()};
  auto [res, _s, _t] = UnwrapValue(p2b(ctx, WrapValue(x, shape)));
  return res;
}

static ArrayRef wrap_not_b(SPUContext* ctx, const ArrayRef& x) {
  const Shape shape = {x.numel()};
  KernelEvalContext kctx(ctx);
  auto [res, _s, _t] =
      UnwrapValue(dynDispatch(ctx, "not_b", WrapValue(x, shape)));
  return res;
}

static ArrayRef wrap_bitle_bb(SPUContext* ctx, const ArrayRef& x,
                              const ArrayRef& y) {
  SPU_ENFORCE(x.numel() == y.numel());
  const Shape shape = {x.numel()};
  auto [res, _s, _t] = UnwrapValue(
      dynDispatch(ctx, "bitle_bb", WrapValue(x, shape), WrapValue(y, shape)));
  return res;
}

static ArrayRef wrap_carray_out(const CircuitBasicBlock<Value>& cbb,
                                const ArrayRef& x, const ArrayRef& y,
                                size_t nbits) {
  SPU_ENFORCE(x.numel() == y.numel());
  const Shape shape = {x.numel()};
  auto [res, _s, _t] = UnwrapValue(
      carry_out(cbb, WrapValue(x, shape), WrapValue(y, shape), nbits));
  return res;
}

[[maybe_unused]] auto pt_ones(FieldType field, size_t numel) {
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using PShrT = ring2k_t;
    ArrayRef out(makeType<Pub2kTy>(field), numel);
    auto _out = ArrayView<PShrT>(out);
    pforeach(0, numel, [&](int64_t idx) {
      PShrT t = 1;
      _out[idx] = t;
    });
    return out;
  });
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

CircuitBasicBlock<Value> MakeSPDZBasicBlock(SPUContext* ctx) {
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
    return make_p(ctx, init, {x.numel()});
  };
  cbb.set_nbits = [=](T& x, size_t nbits) {
    return x.storage_type().as<BShare>()->setNbits(nbits);
  };
  return cbb;
}

};  // namespace

ArrayRef A2Bit::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  // ArrayRef a2bit_impl(KernelEvalContext* ctx, const ArrayRef& in) {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const size_t s = ctx->getState<Spdz2kState>()->s();
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

ArrayRef Bit2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  // ArrayRef bit2a_impl(KernelEvalContext* ctx, const ArrayRef& in) {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const size_t s = ctx->getState<Spdz2kState>()->s();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const auto key = ctx->getState<Spdz2kState>()->key();

  // The protocol for Bit2A in SPDZ2k
  // reference: https://eprint.iacr.org/2019/599.pdf
  // Page 6, Figure 3.

  // 1. Reserve the least significant bit
  const auto [_in_val, _in_mac] = BShareSwitch2Nbits(in, 1);
  const auto _in = makeBShare(_in_val, _in_mac, field, 1);
  const size_t out_numel = _in.numel();

  // 2. get random bit [r] in the form of A-share
  auto [r, r_mac] = beaver->AuthRandBit(field, out_numel, k, s);
  auto ar = makeAShare(r, r_mac, field);

  // 3. Convert [r] into B-share
  auto br = wrap_a2bit(ctx->sctx(), ar);

  // 4. c = open([x] + [r])
  auto bc = wrap_add_bb(ctx->sctx(), _in, br);
  // Notice we only reserve the least significant bit
  const auto [bc_val, bc_mac] = BShareSwitch2Nbits(bc, 1);
  auto [c, zero_mac] = beaver->BatchOpen(bc_val, bc_mac, 1, s);
  SPU_ENFORCE(beaver->BatchMacCheck(c, zero_mac, 1, s));
  ring_bitmask_(c, 0, 1);

  // 5. [x] = c + [r] - 2 * c * [r]
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto _c = ArrayView<ring2k_t>(c);
    auto _r = ArrayView<ring2k_t>(r);
    auto _r_mac = ArrayView<ring2k_t>(r_mac);

    ArrayRef out(makeType<AShrTy>(field, true), out_numel);
    auto _out = ArrayView<std::array<ring2k_t, 2>>(out);

    pforeach(0, out_numel, [&](int64_t idx) {
      _out[idx][0] = (_r[idx] - 2 * _c[idx] * _r[idx]);
      if (comm->getRank() == 0) {
        _out[idx][0] += _c[idx];
      }
      _out[idx][1] = _c[idx] * key + _r_mac[idx] - 2 * _c[idx] * _r_mac[idx];
    });

    return out;
  });
}

ArrayRef A2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();

  // 1. get rand bit r
  auto [rbit, rbit_mac] = beaver->AuthRandBit(field, in.numel() * k, k, s);
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

ArrayRef B2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<BShrTy>()->field();
  const auto nbits = in.eltype().as<BShrTy>()->nbits();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();
  const auto key = ctx->getState<Spdz2kState>()->key();

  const auto _in_val = getValueShare(in);
  const auto _in_mac = getMacShare(in);
  const auto _in = makeBShare(_in_val, _in_mac, field, 1);
  const size_t out_numel = _in.numel() / nbits;

  // 1. get rand bit [r]
  auto [r, r_mac] = beaver->AuthRandBit(field, _in.numel(), k, s);
  auto ar = makeAShare(r, r_mac, field);

  // 2. a2bit
  auto br = wrap_a2bit(ctx->sctx(), ar);

  // 3. c = open([x] + [r])
  auto bc = wrap_add_bb(ctx->sctx(), _in, br);
  // Notice we only reserve the least significant bit
  const auto [bc_val, bc_mac] = BShareSwitch2Nbits(bc, 1);

  auto [c, zero_mac] = beaver->BatchOpen(bc_val, bc_mac, 1, s);
  SPU_ENFORCE(beaver->BatchMacCheck(c, zero_mac, 1, s));
  ring_bitmask_(c, 0, 1);

  // 4. [x] = c + [r] - 2 * c * [r]
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using PShrT = ring2k_t;
    auto _c = ArrayView<PShrT>(c);
    auto _r = ArrayView<ring2k_t>(r);
    auto _r_mac = ArrayView<ring2k_t>(r_mac);

    ArrayRef out(makeType<AShrTy>(field, true), out_numel);
    ArrayRef expand_out(makeType<AShrTy>(field, true), _in.numel());
    auto _out = ArrayView<std::array<ring2k_t, 2>>(out);
    auto _expand_out = ArrayView<std::array<ring2k_t, 2>>(expand_out);

    pforeach(0, _in.numel(), [&](int64_t idx) {
      _expand_out[idx][0] = (_r[idx] - 2 * _c[idx] * _r[idx]);
      if (comm->getRank() == 0) {
        _expand_out[idx][0] += _c[idx];
      }
      _expand_out[idx][1] =
          _c[idx] * key + _r_mac[idx] - 2 * _c[idx] * _r_mac[idx];
    });

    pforeach(0, out_numel, [&](int64_t idx) {
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

ArrayRef MSB::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);
  const auto field = in.eltype().as<Ring2k>()->field();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const auto* comm = ctx->getState<Communicator>();

  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto numel = in.numel();

  // The protocol for extracting MSB in SPDZ2k
  // reference: https://eprint.iacr.org/2019/599.pdf
  // Page7, Figure 6.

  auto _in = getValueShare(in);
  auto _in_mac = GetMacShare(ctx, in);

  auto [c_in, c_in_mac] = beaver->BatchOpen(_in, _in_mac, k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(c_in, c_in_mac, k, s));

  auto _r_val = ring_zeros(field, numel);
  auto _r_mac = ring_zeros(field, numel);
  std::vector<ArrayRef> _r_vec;
  std::vector<ArrayRef> _r_mac_vec;

  // 1. generate random bit b , r_0 , ... , r_{k-1}
  //    then set r = \sum r_i 2^{i}
  for (size_t i = 0; i < k; ++i) {
    auto [_r_i, _r_i_mac] = beaver->AuthRandBit(field, numel, k, s);
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
  ArrayRef _bt_r = ring_zeros(field, numel * (k - 1));
  ArrayRef _bt_r_mac = ring_zeros(field, numel * (k - 1));
  ArrayRef _ar = ring_zeros(field, numel);
  ArrayRef _ar_mac = ring_zeros(field, numel);

  const auto ty = makeType<RingTy>(field);
  for (size_t i = 0; i < k - 1; ++i) {
    ring_add_(_ar, ring_lshift(_r_vec[i], i));
    ring_add_(_ar_mac, ring_lshift(_r_mac_vec[i], i));

    auto at_r_i = makeAShare(_r_vec[i], _r_mac_vec[i], field);
    auto bt_r_i = wrap_a2bit(ctx->sctx(), at_r_i);
    const auto _bt_r_i = getValueShare(bt_r_i);
    const auto _bt_r_i_mac = getMacShare(bt_r_i);
    auto _sub_bt_r =
        ArrayRef(_bt_r.buf(), ty, numel, _bt_r.stride() * (k - 1),
                 _bt_r.offset() + i * static_cast<int64_t>(ty.size()));
    auto _sub_bt_r_mac =
        ArrayRef(_bt_r_mac.buf(), ty, numel, _bt_r_mac.stride() * (k - 1),
                 _bt_r_mac.offset() + i * static_cast<int64_t>(ty.size()));
    ring_add_(_sub_bt_r, _bt_r_i);
    ring_add_(_sub_bt_r_mac, _bt_r_i_mac);
  }
  auto br = makeBShare(_bt_r, _bt_r_mac, field, k - 1);

  // 4. u = BitLT( c , r' )
  // todo: Here should be ctx->caller()->call("bitlt_pb", _pc , br)
  //                   or ctx->caller()->call("bitle_bp", br , _pc)
  //                   or ctx->caller()->call("bitlt_bb", _bc, br)
  auto _pc = _c_open.as(makeType<Pub2kTy>(field));
  ring_bitmask_(_pc, 0, k);
  auto _bc = wrap_p2b(ctx->sctx(), _pc);

  // auto not_u = BitLEBB().proc(ctx, br, _bc);
  // auto bu = NotB().proc(ctx, not_u);
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
  auto [_b, _b_mac] = beaver->AuthRandBit(field, numel, k, s);
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
  SPU_ENFORCE(_ret.numel() == in.numel(), "_ret numel {}, in numel is {}",
              _ret.numel(), in.numel());

  return makeBShare(_ret, _ret_mac, field, 1);
}

static ArrayRef wrap_kogge_stone(const CircuitBasicBlock<Value>& cbb,
                                 const ArrayRef& x, const ArrayRef& y,
                                 size_t nbits) {
  SPU_ENFORCE(x.numel() == y.numel());
  const Shape shape = {x.numel()};
  auto [res, _s, _t] = UnwrapValue(
      kogge_stone(cbb, WrapValue(x, shape), WrapValue(y, shape), nbits));
  return res;
}

ArrayRef AddBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
  const size_t nbits = maxNumBits(lhs, rhs);
  auto cbb = MakeSPDZBasicBlock(ctx->sctx());
  // sklansky has more local computation which leads to lower performance.
  return wrap_kogge_stone(cbb, lhs, rhs, nbits);
}

ArrayRef AddBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
  const size_t nbits = maxNumBits(lhs, rhs);
  auto cbb = MakeSPDZBasicBlock(ctx->sctx());
  // sklansky has more local computation which leads to lower performance.
  return wrap_kogge_stone(cbb, lhs, rhs, nbits);
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
ArrayRef BitLTBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                       const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  auto res0 = wrap_bitle_bb(ctx->sctx(), lhs, rhs);
  auto res1 = wrap_bitle_bb(ctx->sctx(), rhs, lhs);
  auto eq = wrap_and_bb(ctx->sctx(), res0, res1);
  auto neq = wrap_not_b(ctx->sctx(), eq);
  auto res = wrap_and_bb(ctx->sctx(), neq, res0);

  return res;
}
#endif

ArrayRef BitLEBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                       const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto nbits = maxNumBits(lhs, rhs);

  auto rhs_not = wrap_not_b(ctx->sctx(), rhs);
  auto cbb = MakeSPDZBasicBlock(ctx->sctx());
  auto ret = wrap_carray_out(cbb, lhs, rhs_not, nbits);
  auto res = wrap_not_b(ctx->sctx(), ret);
  SPU_ENFORCE(res.numel() == lhs.numel(), "res numel {}, lhs numel {} ",
              res.numel(), lhs.numel());
  return res;
}

};  // namespace spu::mpc::spdz2k