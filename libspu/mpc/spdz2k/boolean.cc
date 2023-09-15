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

#include "libspu/mpc/spdz2k/boolean.h"

#include <algorithm>

#include "libspu/core/parallel_utils.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdz2k/state.h"
#include "libspu/mpc/spdz2k/type.h"
#include "libspu/mpc/spdz2k/value.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {

namespace {

// Input a plaintext
// Output the B-share without MAC
// LSB first, MSB last
NdArrayRef P2Value(FieldType out_field, const NdArrayRef& in, size_t k,
                   size_t new_nbits = 0) {
  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto in_field = in_ty->field();
  return DISPATCH_ALL_FIELDS(in_field, "_", [&]() {
    using PShrT = ring2k_t;

    size_t valid_nbits = k;
    if (new_nbits == 0) {
      valid_nbits = std::min(k, maxBitWidth<PShrT>(in));
    } else if (new_nbits < k) {
      valid_nbits = new_nbits;
    }

    Shape out_shape = in.shape();
    out_shape.back() *= valid_nbits;

    auto out = ring_zeros(out_field, out_shape);
    return DISPATCH_ALL_FIELDS(out_field, "_", [&]() {
      using BShrT = ring2k_t;
      NdArrayView<PShrT> _in(in);
      NdArrayView<BShrT> _out(out);

      pforeach(0, in.numel(), [&](int64_t idx) {
        pforeach(0, valid_nbits, [&](int64_t jdx) {
          size_t offset = idx * valid_nbits + jdx;
          _out[offset] = static_cast<BShrT>((_in[idx] >> jdx) & 1);
        });
      });

      return out;
    });
  });
}

// RShift implementation
std::pair<NdArrayRef, NdArrayRef> RShiftBImpl(const NdArrayRef& in,
                                              size_t bits) {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto old_nbits = in.eltype().as<BShrTy>()->nbits();
  int64_t new_nbits = old_nbits - bits;

  if (bits == 0) {
    return {getValueShare(in), getMacShare(in)};
  }

  if (new_nbits <= 0) {
    return {ring_zeros(field, in.shape()), ring_zeros(field, in.shape())};
  }

  const size_t p_num = in.numel();

  auto out_shape = in.shape();
  out_shape.back() *= new_nbits;

  auto out_val = ring_zeros(field, out_shape);
  auto out_mac = ring_zeros(field, out_shape);

  size_t out_offset = 0;
  size_t in_offset = bits;

  auto in_val = getValueShare(in).clone();
  auto in_mac = getMacShare(in).clone();

  for (size_t i = 0; i < p_num; ++i) {
    auto _in_val =
        NdArrayRef(in_val.buf(), makeType<RingTy>(field), {new_nbits}, {1},
                   (i * old_nbits + in_offset) * SizeOf(field));
    auto _in_mac =
        NdArrayRef(in_mac.buf(), makeType<RingTy>(field), {new_nbits}, {1},
                   (i * old_nbits + in_offset) * SizeOf(field));
    auto _out_val =
        NdArrayRef(out_val.buf(), makeType<RingTy>(field), {new_nbits}, {1},
                   (i * new_nbits + out_offset) * SizeOf(field));
    auto _out_mac =
        NdArrayRef(out_mac.buf(), makeType<RingTy>(field), {new_nbits}, {1},
                   (i * new_nbits + out_offset) * SizeOf(field));
    ring_add_(_out_val, _in_val);
    ring_add_(_out_mac, _in_mac);
  }

  return {out_val, out_mac};
}

// ARShift implementation
std::pair<NdArrayRef, NdArrayRef> ARShiftBImpl(const NdArrayRef& in,
                                               size_t bits, size_t k) {
  const auto old_nbits = in.eltype().as<BShrTy>()->nbits();
  // Only process negative number
  SPU_ENFORCE(old_nbits == k);
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto ty = makeType<RingTy>(field);

  if (bits == 0) {
    return {getValueShare(in), getMacShare(in)};
  }

  size_t p_num = in.numel();

  auto out_shape = in.shape();
  out_shape.back() *= old_nbits;

  NdArrayRef out_val(ty, out_shape);
  NdArrayRef out_mac(ty, out_shape);

  int64_t offset1 = bits < old_nbits ? bits : old_nbits;
  int64_t offset2 = bits < old_nbits ? old_nbits - bits : 0;

  auto in_val = getValueShare(in).clone();
  auto in_mac = getMacShare(in).clone();

  auto ones = ring_ones(field, {offset1});

  for (size_t i = 0; i < p_num; ++i) {
    auto _in_val1 = NdArrayRef(in_val.buf(), makeType<RingTy>(field), {offset2},
                               {1}, (i * old_nbits + offset1) * SizeOf(field));
    auto _in_mac1 = NdArrayRef(in_mac.buf(), makeType<RingTy>(field), {offset2},
                               {1}, (i * old_nbits + offset1) * SizeOf(field));
    auto _out_val1 =
        NdArrayRef(out_val.buf(), makeType<RingTy>(field), {offset2}, {1},
                   (i * old_nbits) * SizeOf(field));
    auto _out_mac1 =
        NdArrayRef(out_mac.buf(), makeType<RingTy>(field), {offset2}, {1},
                   (i * old_nbits) * SizeOf(field));
    ring_assign(_out_val1, _in_val1);
    ring_assign(_out_mac1, _in_mac1);

    auto _in_val_sign =
        NdArrayRef(in_val.buf(), makeType<RingTy>(field), {1}, {1},
                   ((i + 1) * old_nbits - 1) * SizeOf(field));
    auto _in_mac_sign =
        NdArrayRef(in_mac.buf(), makeType<RingTy>(field), {1}, {1},
                   ((i + 1) * old_nbits - 1) * SizeOf(field));

    // sign extension
    auto _in_val2 =
        ring_mmul(_in_val_sign.reshape({1, 1}), ones.reshape({1, offset1}));
    auto _in_mac2 =
        ring_mmul(_in_mac_sign.reshape({1, 1}), ones.reshape({1, offset1}));
    auto _out_val2 =
        NdArrayRef(out_val.buf(), makeType<RingTy>(field), {offset1}, {1},
                   (i * old_nbits + offset2) * SizeOf(field));
    auto _out_mac2 =
        NdArrayRef(out_mac.buf(), makeType<RingTy>(field), {offset1}, {1},
                   (i * old_nbits + offset2) * SizeOf(field));
    ring_assign(_out_val2, _in_val2.reshape({offset1}));
    ring_assign(_out_mac2, _in_mac2.reshape({offset1}));
  }
  return {out_val, out_mac};
}

// LShift implementation
std::pair<NdArrayRef, NdArrayRef> LShiftBImpl(const NdArrayRef& in, size_t bits,
                                              size_t k) {
  const auto field = in.eltype().as<Ring2k>()->field();

  if (bits == 0) {
    return {getValueShare(in), getMacShare(in)};
  }

  if (bits >= k) {
    return {ring_zeros(field, in.shape()), ring_zeros(field, in.shape())};
  }

  const auto old_nbits = in.eltype().as<BShrTy>()->nbits();
  size_t new_nbits = old_nbits + bits;

  if (new_nbits > k) {
    new_nbits = k;
  }
  int64_t min_nbits = new_nbits - bits;

  size_t p_num = in.numel();

  auto out_shape = in.shape();
  out_shape.back() *= new_nbits;

  auto out_val = ring_zeros(field, out_shape);
  auto out_mac = ring_zeros(field, out_shape);

  size_t out_offset = bits;
  size_t in_offset = 0;

  auto in_val = getValueShare(in).clone();
  auto in_mac = getMacShare(in).clone();

  for (size_t i = 0; i < p_num; ++i) {
    auto _in_val =
        NdArrayRef(in_val.buf(), makeType<RingTy>(field), {min_nbits}, {1},
                   (i * old_nbits + in_offset) * SizeOf(field));
    auto _in_mac =
        NdArrayRef(in_mac.buf(), makeType<RingTy>(field), {min_nbits}, {1},
                   (i * old_nbits + in_offset) * SizeOf(field));

    auto _out_val =
        NdArrayRef(out_val.buf(), makeType<RingTy>(field), {min_nbits}, {1},
                   (i * new_nbits + out_offset) * SizeOf(field));
    auto _out_mac =
        NdArrayRef(out_mac.buf(), makeType<RingTy>(field), {min_nbits}, {1},
                   (i * new_nbits + out_offset) * SizeOf(field));

    ring_add_(_out_val, _in_val);
    ring_add_(_out_mac, _in_mac);
  }

  return {out_val, out_mac};
}

};  // namespace

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

  SPU_ENFORCE(lhs == rhs, "spdz2k always use same bshare type, lhs={}, rhs={}",
              lhs, rhs);

  ctx->setOutput(lhs);
}

NdArrayRef CastTypeB::proc(KernelEvalContext*, const NdArrayRef& in,
                           const Type& to_type) const {
  SPU_ENFORCE(in.eltype() == to_type,
              "spdz2k always use same bshare type, lhs={}, rhs={}", in.eltype(),
              to_type);
  return in;
}

NdArrayRef B2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* beaver_ptr = ctx->getState<Spdz2kState>()->beaver();
  const auto s = ctx->getState<Spdz2kState>()->s();
  const auto field = in.eltype().as<BShrTy>()->field();
  const auto out_field = ctx->getState<Z2kState>()->getDefaultField();
  const auto nbits = in.eltype().as<BShrTy>()->nbits();

  // 1.  Open the least significant bit
  NdArrayRef pub, mac;
  std::tie(pub, mac) =
      beaver_ptr->BatchOpen(getValueShare(in), getMacShare(in), 1, s);

  // 2. Maccheck
  SPU_ENFORCE(beaver_ptr->BatchMacCheck(pub, mac, 1, s));

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using BShrT = ring2k_t;
    auto& value = pub;
    return DISPATCH_ALL_FIELDS(out_field, "_", [&]() {
      using PShrT = ring2k_t;

      NdArrayRef out(makeType<Pub2kTy>(out_field), in.shape());

      NdArrayView<PShrT> _out(out);
      NdArrayView<BShrT> _value(value);
      pforeach(0, in.numel(), [&](int64_t idx) {
        PShrT t = 0;
        for (size_t jdx = 0; jdx < nbits; ++jdx) {
          t |= static_cast<PShrT>((_value[idx * nbits + jdx] & 1) << jdx);
        }
        _out[idx] = t;
      });

      return out;
    });
  });
}

NdArrayRef P2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const auto out_field = ctx->getState<Spdz2kState>()->getDefaultField();
  auto* prg_state = ctx->getState<PrgState>();

  // 1. Convert plaintext into B-value
  auto p = P2Value(out_field, in, k);
  auto out = ring_zeros(out_field, p.shape());

  // 2. out = p
  if (comm->getRank() == 0) {
    ring_add_(out, p);
  }
  auto& out_mac = p;
  // 3. out_mac = p * key
  ring_mul_(p, key);
  // 4. add some random mask
  auto [r0, r1] = prg_state->genPrssPair(out_field, out.shape(),
                                         PrgState::GenPrssCtrl::Both);
  auto [r2, r3] = prg_state->genPrssPair(out_field, out.shape(),
                                         PrgState::GenPrssCtrl::Both);
  ring_add_(out, ring_sub(r0, r1));
  ring_add_(out_mac, ring_sub(r2, r3));
  // 5. makeBShare
  const auto nbits = out.numel() / in.numel();
  return makeBShare(out, out_mac, out_field, nbits);
}

NdArrayRef NotB::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = in.eltype().as<BShrTy>()->nbits();
  const auto key = ctx->getState<Spdz2kState>()->key();
  auto* comm = ctx->getState<Communicator>();

  // 1. convert B-share into x & x_mac
  auto [x, x_mac] = BShareSwitch2Nbits(in, nbits);

  // 2. create ones
  auto out_shape = in.shape();
  out_shape.back() *= nbits;
  auto ones = ring_ones(field, out_shape);

  // 3. ret = x + one
  auto ret = x.clone();
  if (comm->getRank() == 0) {
    ring_add_(ret, ones);
  }

  // 4. z_mac = x_mac + ones * key
  ring_mul_(ones, key);
  auto& ret_mac = ones;
  ring_add_(ret_mac, x_mac);

  return makeBShare(ret, ret_mac, field, nbits);
}

NdArrayRef BitrevB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         size_t start, size_t end) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = in.eltype().as<BShrTy>()->nbits();
  const auto numel = in.numel();

  SPU_ENFORCE(start <= end);
  SPU_ENFORCE(end <= nbits);

  auto x = getValueShare(in);
  auto x_mac = getMacShare(in);
  auto ret = x.clone();
  auto ret_mac = x_mac.clone();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _ret_mac(ret_mac);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _x_mac(x_mac);

    for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
      for (size_t j = start; j < end; ++j) {
        _ret[i * nbits + j] = _x[i * nbits + end + start - j - 1];
        _ret_mac[i * nbits + j] = _x_mac[i * nbits + end + start - j - 1];
      }
    }
  });

  return makeBShare(ret, ret_mac, field, nbits);
}

NdArrayRef XorBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
  const auto field = lhs.eltype().as<Ring2k>()->field();
  const auto nbits = maxNumBits(lhs, rhs);

  // lhs
  const auto [x, x_mac] = BShareSwitch2Nbits(lhs, nbits);

  // rhs
  const auto [y, y_mac] = BShareSwitch2Nbits(rhs, nbits);

  // ret
  const auto& z = ring_add(x, y);
  const auto& z_mac = ring_add(x_mac, y_mac);
  return makeBShare(z, z_mac, field, nbits);
}

NdArrayRef XorBP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  const auto nbits = maxNumBits(lhs, rhs);
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto key = ctx->getState<Spdz2kState>()->key();
  auto* comm = ctx->getState<Communicator>();

  // lhs
  auto [x, x_mac] = BShareSwitch2Nbits(lhs, nbits);

  // convert plaintext to B-value
  auto p = P2Value(field, rhs, k, nbits);

  // ret
  auto z = x.clone();
  if (comm->getRank() == 0) {
    // z += p
    ring_add_(z, p);
  }

  // z_mac = x_mac + p * key
  ring_mul_(p, key);
  auto& z_mac = p;
  ring_add_(z_mac, x_mac);

  return makeBShare(z, z_mac, field, nbits);
}

NdArrayRef AndBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver_ptr = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const auto s = ctx->getState<Spdz2kState>()->s();

  // 1. find the min nbits
  const auto nbits = minNumBits(lhs, rhs);

  // 2. convert B-share into value & mac
  auto [x, x_mac] = BShareSwitch2Nbits(lhs, nbits);
  auto [y, y_mac] = BShareSwitch2Nbits(rhs, nbits);

  SPU_ENFORCE(x.shape() == y.shape());
  // e = x - a, f = y - b
  auto [beaver_vec, beaver_mac] = beaver_ptr->AuthAnd(field, x.shape(), s);
  auto& [a, b, c] = beaver_vec;
  auto& [a_mac, b_mac, c_mac] = beaver_mac;

  auto e = ring_sub(x, a);
  auto e_mac = ring_sub(x_mac, a_mac);
  auto f = ring_sub(y, b);
  auto f_mac = ring_sub(y_mac, b_mac);

  // Open the least significant bit and Check them
  auto [p_e, pe_mac] = beaver_ptr->BatchOpen(e, e_mac, 1, s);
  auto [p_f, pf_mac] = beaver_ptr->BatchOpen(f, f_mac, 1, s);

  SPU_ENFORCE(beaver_ptr->BatchMacCheck(p_e, pe_mac, 1, s));
  SPU_ENFORCE(beaver_ptr->BatchMacCheck(p_f, pf_mac, 1, s));

  // Reserve the least significant bit only
  ring_bitmask_(p_e, 0, 1);
  ring_bitmask_(p_f, 0, 1);
  auto p_ef = ring_mul(p_e, p_f);

  // z = p_e * b + p_f * a + c;
  auto z = ring_add(ring_mul(p_e, b), ring_mul(p_f, a));
  ring_add_(z, c);
  if (comm->getRank() == 0) {
    // z += p_e * p_f;
    ring_add_(z, p_ef);
  }

  // z_mac = p_e * b_mac + p_f * a_mac + c_mac + p_e * p_f * key;
  auto z_mac = ring_add(ring_mul(p_e, b_mac), ring_mul(p_f, a_mac));
  ring_add_(z_mac, c_mac);
  ring_add_(z_mac, ring_mul(p_ef, key));

  return makeBShare(z, z_mac, field, nbits);
}

NdArrayRef AndBP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
  SPU_ENFORCE(lhs.shape() == rhs.shape(), "lhs shape {}, rhs shape {}",
              lhs.shape(), rhs.shape());

  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto field = lhs.eltype().as<Ring2k>()->field();
  const auto nbits = minNumBits(lhs, rhs);

  // lhs
  auto [x, x_mac] = BShareSwitch2Nbits(lhs, nbits);

  // convert rhs to B-share value
  const auto p = P2Value(field, rhs, k, nbits);
  SPU_ENFORCE(x.numel() == p.numel(), "x {} p {}", x.numel(), p.numel());
  // ret
  // z = x * p
  const auto z = ring_mul(x, p);
  // z = x_mac * p
  const auto z_mac = ring_mul(x_mac, p);
  return makeBShare(z, z_mac, field, nbits);
}

NdArrayRef LShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         size_t bits) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const size_t nbits = in.eltype().as<BShare>()->nbits();
  size_t res_nbits = nbits + bits;

  if (bits >= k) {
    res_nbits = 1;
  } else if (res_nbits > k) {
    res_nbits = k;
  }
  auto [ret, ret_mac] = LShiftBImpl(in, bits, k);
  return makeBShare(ret, ret_mac, field, res_nbits);
}

NdArrayRef RShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = in.eltype().as<BShare>()->nbits();
  size_t new_nbis = nbits > bits ? nbits - bits : 1;
  auto [ret, ret_mac] = RShiftBImpl(in, bits);
  return makeBShare(ret, ret_mac, field, new_nbis);
}

static NdArrayRef wrap_rshift_b(SPUContext* ctx, const NdArrayRef& x,
                                size_t bits) {
  return UnwrapValue(rshift_b(ctx, WrapValue(x), bits));
}

NdArrayRef ARShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto nbits = in.eltype().as<BShrTy>()->nbits();

  if (nbits != k) {
    return wrap_rshift_b(ctx->sctx(), in, bits);
  } else {
    auto [ret, ret_mac] = ARShiftBImpl(in, bits, k);
    return makeBShare(ret, ret_mac, field, k);
  }
}

// Only process k bits at now.
NdArrayRef BitIntlB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          size_t stride) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto k = ctx->getState<Spdz2kState>()->k();

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    if (in.eltype().isa<Pub2kTy>()) {
      NdArrayView<T> _out(out);
      NdArrayView<T> _in(in);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = BitIntl<T>(_in[idx], stride, k);
      });
    } else {
      NdArrayView<std::array<T, 2>> _in(in);
      NdArrayView<std::array<T, 2>> _out(out);
      size_t num_per_group = 1 << stride;
      size_t group_num = k / num_per_group + (k % num_per_group != 0);
      size_t half_group_num = (group_num + 1) / 2;

      pforeach(0, in.numel(), [&](size_t jdx) {
        size_t base_offset = jdx * k;
        pforeach(0, k, [&](size_t idx) {
          auto group = idx / num_per_group;
          auto offset = idx % num_per_group;
          size_t src_idx = base_offset;
          size_t dest_idx = base_offset;
          src_idx += idx;
          if (idx < (k + 1) / 2) {
            dest_idx += 2 * group * num_per_group + offset;
          } else {
            dest_idx +=
                (2 * (group - half_group_num) + 1) * num_per_group + offset;
          }
          _out[dest_idx][0] = _in[src_idx][0];
          _out[dest_idx][1] = _in[src_idx][1];
        });
      });
    }
  });

  return out;
}

// Only process k bits at now.
NdArrayRef BitDeintlB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                            size_t stride) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto k = ctx->getState<Spdz2kState>()->k();

  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using T = ring2k_t;

    if (in.eltype().isa<Pub2kTy>()) {
      NdArrayView<T> _out(out);
      NdArrayView<T> _in(in);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = BitDeintl<T>(_in[idx], stride, k);
      });
    } else {
      NdArrayView<std::array<T, 2>> _in(in);
      NdArrayView<std::array<T, 2>> _out(out);

      size_t num_per_group = 1 << stride;
      size_t group_num = k / num_per_group + (k % num_per_group != 0);
      size_t half_group_num = (group_num + 1) / 2;
      pforeach(0, in.numel(), [&](size_t jdx) {
        size_t base_offset = jdx * k;
        pforeach(0, k, [&](size_t idx) {
          auto group = idx / num_per_group;
          auto offset = idx % num_per_group;
          size_t src_idx = base_offset;
          size_t dest_idx = base_offset;
          dest_idx += idx;
          if (idx < (k + 1) / 2) {
            src_idx += 2 * group * num_per_group + offset;
          } else {
            src_idx +=
                (2 * (group - half_group_num) + 1) * num_per_group + offset;
          }
          _out[dest_idx][0] = _in[src_idx][0];
          _out[dest_idx][1] = _in[src_idx][1];
        });
      });
    }
  });

  return out;
}

}  // namespace spu::mpc::spdz2k
