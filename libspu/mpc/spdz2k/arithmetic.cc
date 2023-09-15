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

#include "libspu/mpc/spdz2k/arithmetic.h"

#include <random>

#include "yacl/crypto/base/hash/blake3.h"
#include "yacl/crypto/base/hash/hash_interface.h"
#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/link/link.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdz2k/commitment.h"
#include "libspu/mpc/spdz2k/state.h"
#include "libspu/mpc/spdz2k/type.h"
#include "libspu/mpc/spdz2k/value.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdz2k {
namespace {

// Input a plaintext
// Output the B-share without MAC
// LSB first, MSB last
// NdArrayRef CastToLargeRing(const NdArrayRef& in, FieldType out_field) {
NdArrayRef CastRing(const NdArrayRef& in, FieldType out_field) {
  const auto* in_ty = in.eltype().as<Ring2k>();
  const auto in_field = in_ty->field();
  auto out = ring_zeros(out_field, in.shape());

  return DISPATCH_ALL_FIELDS(in_field, "_", [&]() {
    NdArrayView<ring2k_t> _in(in);
    return DISPATCH_ALL_FIELDS(out_field, "_", [&]() {
      NdArrayView<ring2k_t> _out(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx] = static_cast<ring2k_t>(_in[idx]);
      });

      return out;
    });
  });
}

NdArrayRef zero_a_impl(KernelEvalContext* ctx, const Shape& shape) {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Spdz2kState>()->getDefaultField();

  auto [r0, r1] =
      prg_state->genPrssPair(field, shape, PrgState::GenPrssCtrl::Both);
  auto [r2, r3] =
      prg_state->genPrssPair(field, shape, PrgState::GenPrssCtrl::Both);

  auto x = ring_sub(r0, r1);
  auto x_mac = ring_sub(r2, r3);
  return makeAShare(x, x_mac, field);
}
}  // namespace

NdArrayRef GetMacShare(KernelEvalContext* ctx, const NdArrayRef& in) {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();

  const auto& x = getValueShare(in);
  NdArrayRef x_mac;
  if (in.eltype().as<AShrTy>()->hasMac()) {
    x_mac = getMacShare(in);
  } else {
    SPDLOG_DEBUG("generate mac share");
    x_mac = beaver->AuthArrayRef(x, field, k, s);
  }
  return x_mac;
}

NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  SPU_TRACE_MPC_LEAF(ctx, shape);

  const auto field = ctx->getState<Spdz2kState>()->getDefaultField();
  auto* prg_state = ctx->getState<PrgState>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto s = ctx->getState<Spdz2kState>()->s();

  // NOTES for ring_rshift to 2 bits.
  // Refer to:
  // New Primitives for Actively-Secure MPC over Rings with Applications to
  // Private Machine Learning
  // - https://eprint.iacr.org/2019/599.pdf
  // It's safer to keep the number within [-2**(k-2), 2**(k-2)) for comparison
  // operations.
  auto x = ring_rshift(prg_state->genPriv(field, shape), 2)
               .as(makeType<AShrTy>(field));
  auto x_mac = beaver->AuthArrayRef(x, field, k, s);
  return makeAShare(x, x_mac, field);
}

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = ctx->getState<Spdz2kState>()->getDefaultField();
  auto* comm = ctx->getState<Communicator>();
  const auto key = ctx->getState<Spdz2kState>()->key();

  auto res = zero_a_impl(ctx, in.shape());
  auto z = getValueShare(res);
  auto z_mac = getMacShare(res);

  auto t_in = CastRing(in, field);
  if (comm->getRank() == 0) {
    ring_add_(z, t_in);
  }

  ring_add_(z_mac, ring_mul(t_in, key));

  return res;
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto out_field = ctx->getState<Z2kState>()->getDefaultField();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto s = ctx->getState<Spdz2kState>()->s();

  // in
  const auto& x = getValueShare(in);
  const auto& x_mac = getMacShare(in);
  auto [t, check_mac] = beaver->BatchOpen(x, x_mac, k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(t, check_mac, k, s));

  // Notice that only the last sth bits is correct
  ring_bitmask_(t, 0, k);

  auto res = CastRing(t, out_field);
  return res.as(makeType<Pub2kTy>(out_field));
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = ctx->getState<Spdz2kState>()->getDefaultField();
  const auto out_field = ctx->getState<Z2kState>()->getDefaultField();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto s = ctx->getState<Spdz2kState>()->s();

  // generate mask
  auto zero = zero_a_impl(ctx, in.shape());
  auto z = getValueShare(zero);
  auto z_mac = getMacShare(zero);
  auto r = ring_rand(field, in.shape());
  if (comm->getRank() == rank) {
    ring_add_(z, r);
  }
  ring_add_(z_mac, beaver->AuthArrayRef(z, field, k, s));

  // add mask
  const auto& x = getValueShare(in);
  const auto& x_mac = getMacShare(in);
  auto mask_x = ring_add(x, z);
  auto mask_x_mac = ring_add(x_mac, z_mac);

  auto [t, check_mac] = beaver->BatchOpen(mask_x, mask_x_mac, k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(t, check_mac, k, s));

  // Notice that only the last s bits is correct
  if (comm->getRank() == rank) {
    auto t_r = ring_bitmask(ring_sub(t, r), 0, k);
    auto res = CastRing(t_r, out_field);
    return res.as(makeType<Priv2kTy>(out_field, rank));
  } else {
    auto out_ty = makeType<Priv2kTy>(out_field, rank);
    return makeConstantArrayRef(out_ty, in.shape());
  }
}

NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const size_t owner_rank = in_ty->owner();
  const auto field = ctx->getState<Spdz2kState>()->getDefaultField();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();

  auto res = zero_a_impl(ctx, in.shape());
  auto z = getValueShare(res);
  auto z_mac = getMacShare(res);

  auto t_in = CastRing(in, field);
  if (comm->getRank() == owner_rank) {
    ring_add_(z, t_in);
  }

  ring_add_(z_mac, beaver->AuthArrayRef(z, field, k, s));

  return res;
}

NdArrayRef NotA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto key = ctx->getState<Spdz2kState>()->key();
  auto* comm = ctx->getState<Communicator>();

  // in
  const auto& x = getValueShare(in);
  const auto& x_mac = GetMacShare(ctx, in);

  // compute neg_x, neg_x_mac
  auto neg_x = ring_neg(x);
  auto neg_x_mac = ring_neg(x_mac);

  // add public M-1
  const auto& neg_ones = ring_not(ring_zeros(field, in.shape()));
  if (comm->getRank() == 0) {
    ring_add_(neg_x, neg_ones);
  }
  const auto& ones = ring_ones(field, in.shape());
  ring_sub_(neg_x_mac, ring_mul(ones, key));

  return makeAShare(neg_x, neg_x_mac, field);
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
NdArrayRef AddAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  const auto key = ctx->getState<Spdz2kState>()->key();

  // lhs
  const auto& x = getValueShare(lhs);
  const auto& x_mac = GetMacShare(ctx, lhs);

  auto t_rhs = CastRing(rhs, field);

  // remember that rhs is public
  auto z = x.clone();
  if (comm->getRank() == 0) {
    ring_add_(z, t_rhs);
  }
  auto z_mac = ring_add(x_mac, ring_mul(t_rhs, key));

  return makeAShare(z, z_mac, field);
}

NdArrayRef AddAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x = getValueShare(lhs);
  const auto& x_mac = GetMacShare(ctx, lhs);
  // const auto& x_mac = getMacShare(lhs);

  // rhs
  const auto& y = getValueShare(rhs);
  const auto& y_mac = GetMacShare(ctx, rhs);
  // const auto& y_mac = getMacShare(rhs);

  // ret
  const auto& z = ring_add(x, y);
  const auto& z_mac = ring_add(x_mac, y_mac);
  return makeAShare(z, z_mac, field);
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////

// Refer to:
// Procedure SingleCheck, 3.1 Opening Values and Checking MACs,
// SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
bool SingleCheck(KernelEvalContext* ctx, const NdArrayRef& in) {
  static constexpr char kBindName[] = "single_check";

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();

  // 1. Generate a random, shared value [r]
  auto [r, r_mac] = beaver->AuthCoinTossing(field, in.shape(), k, s);

  // 2. Locally construct [y]
  const auto& x = getValueShare(in);
  const auto& x_mac = getMacShare(in);
  auto y = ring_add(x, ring_lshift(r, k));
  auto y_mac = ring_add(x_mac, ring_lshift(r_mac, k));

  // 3. Open the value
  auto plain_y = comm->allReduce(ReduceOp::ADD, y, kBindName);

  // 4. Check the consistency of y
  auto z = ring_sub(y_mac, ring_mul(plain_y, key));
  std::string z_str(z.data<char>(), z.numel() * z.elsize());
  std::vector<std::string> z_strs;
  SPU_ENFORCE(commit_and_open(comm->lctx(), z_str, &z_strs));
  SPU_ENFORCE(z_strs.size() == comm->getWorldSize());

  auto plain_z = ring_zeros(field, in.shape());
  for (size_t i = 0; i < comm->getWorldSize(); ++i) {
    const auto& _z_str = z_strs[i];
    auto mem = std::make_shared<yacl::Buffer>(_z_str.data(), _z_str.size());
    NdArrayRef a(mem, plain_z.eltype(),
                 {static_cast<int64_t>(_z_str.size() / SizeOf(field))}, {1}, 0);
    ring_add_(plain_z, a.reshape(plain_z.shape()));
  }

  auto ret = spu::mpc::ring_all_equal(plain_z, ring_zeros(field, in.shape()));
  SPU_ENFORCE(ret, "single check fail");
  return ret;
}

static NdArrayRef wrap_lshift_a(SPUContext* ctx, const NdArrayRef& x,
                                size_t k) {
  return UnwrapValue(lshift_a(ctx, WrapValue(x), k));
}

static NdArrayRef wrap_add_aa(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_aa(ctx, WrapValue(x), WrapValue(y)));
}

// Refer to:
// Procedure BatchCheck, 3.2 Batch MAC Checking with Random Linear
// Combinations, SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
//
// TODO: 1. maybe all shared values using one check is better
// 2. use DISPATCH_ALL_FIELDS to improve performance
bool BatchCheck(KernelEvalContext* ctx, const std::vector<NdArrayRef>& ins) {
  static constexpr char kBindName[] = "batch_check";

  SPU_ENFORCE(!ins.empty());
  const auto field = ins[0].eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  const auto& lctx = comm->lctx();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();

  const size_t size = ins.size();

  std::vector<NdArrayRef> x_hat_v;
  std::vector<NdArrayRef> mac_v;

  for (const auto& in : ins) {
    // 1. get random r and r_mac
    auto [r, r_mac] = beaver->AuthCoinTossing(field, ins[0].shape(), k, s);
    auto rmac = makeAShare(r, r_mac, field);

    // 2. [x_hat] = [x] + 2^k * [r]
    auto l_rmac = wrap_lshift_a(ctx->sctx(), rmac, k);
    auto _in = wrap_add_aa(ctx->sctx(), in, l_rmac);

    const auto& x_hat = getValueShare(_in);
    x_hat_v.emplace_back(x_hat);
    const auto& x_mac = getMacShare(_in);
    mac_v.emplace_back(x_mac);
  }

  // 3. broadcast x_hat && 4. open x_hat
  std::vector<NdArrayRef> plain_x_hat_v;
  vmap(x_hat_v.begin(), x_hat_v.end(), std::back_inserter(plain_x_hat_v),
       [&](const NdArrayRef& s) {
         return comm->allReduce(ReduceOp::ADD, s, kBindName);
       });

  // 5. get l public random values, compute plain y
  auto pub_r = beaver->genPublCoin(field, size);
  std::vector<uint128_t> rv;
  uint128_t mask = (static_cast<uint128_t>(1) << s) - 1;
  NdArrayView<uint128_t> _pub_r(pub_r);
  for (size_t i = 0; i < size; ++i) {
    rv.emplace_back(_pub_r[i] & mask);
  }

  auto plain_y = ring_zeros(field, ins[0].shape());
  for (size_t i = 0; i < size; ++i) {
    ring_add_(plain_y, ring_mul(plain_x_hat_v[i], rv[i]));
  }

  // 6. compute z, commit and open z
  auto m = ring_zeros(field, ins[0].shape());
  for (size_t i = 0; i < size; ++i) {
    ring_add_(m, ring_mul(mac_v[i], rv[i]));
  }

  auto plain_y_mac_share = ring_mul(plain_y, key);
  auto z = ring_sub(m, plain_y_mac_share);

  std::string z_str(z.data<char>(), z.numel() * z.elsize());
  std::vector<std::string> z_strs;
  YACL_ENFORCE(commit_and_open(lctx, z_str, &z_strs));
  YACL_ENFORCE(z_strs.size() == comm->getWorldSize());

  // since the commit size in commit_and_open is independent with numel, we
  // ignore it
  comm->addCommStatsManually(1, 0);
  // since the random string size in commit_and_open is independent with numel,
  // we ignore it
  comm->addCommStatsManually(1,
                             z_str.size() / size * (comm->getWorldSize() - 1));

  // 7. verify whether plain z is zero
  auto plain_z = ring_zeros(field, ins[0].shape());
  for (size_t i = 0; i < comm->getWorldSize(); ++i) {
    const auto& _z_str = z_strs[i];
    auto mem = std::make_shared<yacl::Buffer>(_z_str.data(), _z_str.size());
    NdArrayRef a(mem, plain_z.eltype(),
                 {static_cast<int64_t>(_z_str.size() / SizeOf(field))}, {1}, 0);
    ring_add_(plain_z, a.reshape(plain_z.shape()));
  }

  auto ret =
      spu::mpc::ring_all_equal(plain_z, ring_zeros(field, ins[0].shape()));
  return ret;
}

NdArrayRef MulAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x = getValueShare(lhs);
  const auto& x_mac = GetMacShare(ctx, lhs);

  // ret
  auto t_rhs = CastRing(rhs, field);
  const auto& z = ring_mul(x, t_rhs);
  const auto& z_mac = ring_mul(x_mac, t_rhs);

  return makeAShare(z, z_mac, field);
}

// Refer to:
// 3.3 Reducing the Number of Masks && 4 Online Phase
// SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
//
// TODO: use DISPATCH_ALL_FIELDS instead of ring ops to improve performance
NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto s = ctx->getState<Spdz2kState>()->s();

  // in
  const auto& x = getValueShare(lhs);
  const auto& x_mac = GetMacShare(ctx, lhs);
  const auto& y = getValueShare(rhs);
  const auto& y_mac = GetMacShare(ctx, rhs);

  // e = x - a, f = y - b
  auto [vec, mac_vec] = beaver->AuthMul(field, lhs.shape(), k, s);

  auto [a, b, c] = vec;
  auto [a_mac, b_mac, c_mac] = mac_vec;

  auto e = ring_sub(x, a);
  auto e_mac = ring_sub(x_mac, a_mac);
  auto f = ring_sub(y, b);
  auto f_mac = ring_sub(y_mac, b_mac);

  // open e, f
  auto res = vmap({e, f}, [&](const NdArrayRef& s) {
    return comm->allReduce(ReduceOp::ADD, s, kBindName);
  });
  auto p_e = std::move(res[0]);
  auto p_f = std::move(res[1]);

  // don't use BatchOpen to reduce the number of masks
  // auto [p_e, masked_e_mac] = beaver->BatchOpen(e, e_mac, k, s);
  // auto [p_f, masked_f_mac] = beaver->BatchOpen(f, f_mac, k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(p_e, e_mac, k, s));
  SPU_ENFORCE(beaver->BatchMacCheck(p_f, f_mac, k, s));

  auto p_ef = ring_mul(p_e, p_f);

  // z = p_e * b + p_f * a + c;
  auto z = ring_add(ring_mul(p_e, b), ring_mul(p_f, a));
  ring_add_(z, c);
  if (comm->getRank() == 0) {
    // z += p_e * p_f;
    ring_add_(z, p_ef);
  }

  // zmac = p_e * b_mac + p_f * a_mac + c_mac + p_e * p_f * key;
  auto zmac = ring_add(ring_mul(p_e, b_mac), ring_mul(p_f, a_mac));
  ring_add_(zmac, c_mac);
  ring_add_(zmac, ring_mul(p_ef, key));

  return makeAShare(z, zmac, field);
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
NdArrayRef MatMulAP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                          const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // in
  const auto& x = getValueShare(lhs);
  const auto& x_mac = GetMacShare(ctx, lhs);
  const auto& y = CastRing(rhs, field);

  // ret
  auto z = ring_mmul(x, y);
  auto z_mac = ring_mmul(x_mac, y);
  return makeAShare(z, z_mac, field);
}

NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                          const NdArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const auto k_bits = ctx->getState<Spdz2kState>()->k();
  const auto s_bits = ctx->getState<Spdz2kState>()->s();

  const auto& x = getValueShare(lhs);
  const auto& y = getValueShare(rhs);

  // generate beaver multiple triple.
  auto [vec, mac_vec] = beaver->AuthDot(field, lhs.shape()[0], rhs.shape()[1],
                                        lhs.shape()[1], k_bits, s_bits);
  auto [a, b, c] = vec;
  auto [a_mac, b_mac, c_mac] = mac_vec;

  // open x-a & y-b
  auto res = vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const NdArrayRef& s) {
    return comm->allReduce(ReduceOp::ADD, s, kBindName);
  });
  auto p_e = std::move(res[0]);
  auto p_f = std::move(res[1]);
  auto p_ef = ring_mmul(p_e, p_f);

  // z = p_e dot b + a dot p_f + c;
  auto z = ring_add(ring_add(ring_mmul(p_e, b), ring_mmul(a, p_f)), c);
  if (comm->getRank() == 0) {
    // z += p_e dot p_f;
    ring_add_(z, ring_mmul(p_e, p_f));
  }

  // zmac = p_e dot b_mac + a_mac dot p_f + c_mac + (p_e dot p_f) * key;
  auto zmac = ring_add(ring_mmul(p_e, b_mac), ring_mmul(a_mac, p_f));
  ring_add_(zmac, c_mac);
  ring_add_(zmac, ring_mul(p_ef, key));

  return makeAShare(z, zmac, field);
}

NdArrayRef LShiftA::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  // in
  const auto& x = getValueShare(in);
  const auto& x_mac = GetMacShare(ctx, in);

  // ret
  const auto& z = ring_lshift(x, bits);
  const auto& z_mac = ring_lshift(x_mac, bits);
  return makeAShare(z, z_mac, field);
}

// ABY3, truncation pair method.
// Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        size_t bits, SignType sign) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  (void)sign;  // TODO: optimize me.

  const auto key = ctx->getState<Spdz2kState>()->key();
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto k = ctx->getState<Spdz2kState>()->k();
  const auto s = ctx->getState<Spdz2kState>()->s();

  const auto& x = getValueShare(in);
  const auto& x_mac = getMacShare(in);
  const auto& [vec, mac_vec] = beaver->AuthTrunc(field, x.shape(), bits, k, s);
  const auto& [r, rb] = vec;
  const auto& [r_mac, rb_mac] = mac_vec;

  // open x - r
  auto [x_r, check_mac] =
      beaver->BatchOpen(ring_sub(x, r), ring_sub(x_mac, r_mac), k, s);
  SPU_ENFORCE(beaver->BatchMacCheck(x_r, check_mac, k, s));
  size_t bit_len = SizeOf(field) * 8;
  auto tr_x_r = ring_arshift(ring_lshift(x_r, bit_len - k), bit_len - k + bits);
  ring_bitmask_(tr_x_r, 0, k);

  // res = [x-r] + [r], which [*] is truncation operation.
  auto res = rb;
  if (comm->getRank() == 0) {
    ring_add_(res, tr_x_r);
  }

  // res_mac = [x-r] * key + [r_mac], which [*] is truncation operation.
  auto res_mac = rb_mac;
  ring_add_(res_mac, ring_mul(tr_x_r, key));

  return makeAShare(res, res_mac, field);
}

}  // namespace spu::mpc::spdz2k
