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

#include "libspu/core/array_ref.h"
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

ArrayRef zero_a_impl(KernelEvalContext* ctx, size_t size) {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  auto [r0, r1] = prg_state->genPrssPair(field, size);
  auto [r2, r3] = prg_state->genPrssPair(field, size);

  // NOTES for ring_rshift to 2 bits.
  // Refer to:
  // New Primitives for Actively-Secure MPC over Rings with Applications to
  // Private Machine Learning
  // - https://eprint.iacr.org/2019/599.pdf
  // It's safer to keep the number within [-2**(k-2), 2**(k-2)) for comparison
  // operations.
  auto x = ring_sub(r0, r1);
  auto x_mac = ring_sub(r2, r3);
  return makeAShare(x, x_mac, field);
}

}  // namespace

ArrayRef RandA::proc(KernelEvalContext* ctx, size_t size) const {
  SPU_TRACE_MPC_LEAF(ctx, size);
  SPU_THROW("NotImplemented");
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto key = ctx->getState<Spdz2kState>()->key();

  auto res = zero_a_impl(ctx, in.numel());
  auto z = getValueShare(res);
  auto z_mac = getMacShare(res);

  if (comm->getRank() == 0) {
    ring_add_(z, in);
  }

  ring_add_(z_mac, ring_mul(in, key));

  return res;
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* arr_ref_v = ctx->getState<Spdz2kState>()->arr_ref_v();
  arr_ref_v->emplace_back(in);

// #define SINGLE_CHECK
#ifdef SINGLE_CHECK
  for (auto& x : *arr_ref_v) {
    bool success = SingleCheck(ctx, x);
    SPU_ENFORCE(success, "single check fail");
  }
#else
  bool success = BatchCheck(ctx, *arr_ref_v);
  arr_ref_v->clear();
  SPU_ENFORCE(success, "batch check fail");
#endif

  // in
  const auto& x = getValueShare(in);
  auto t = comm->allReduce(ReduceOp::ADD, x, kBindName);
  auto ty = makeType<Pub2kTy>(field);
  return t.as(ty);
}

ArrayRef NotA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  const auto key = ctx->getState<Spdz2kState>()->key();
  auto* comm = ctx->getState<Communicator>();

  // in
  const auto& x = getValueShare(in);
  const auto& x_mac = getMacShare(in);

  // compute neg_x, neg_x_mac
  auto neg_x = ring_neg(x);
  auto neg_x_mac = ring_neg(x_mac);

  // add public M-1
  const auto& neg_ones = ring_not(ring_zeros(field, in.numel()));
  if (comm->getRank() == 0) {
    ring_add_(neg_x, neg_ones);
  }
  const auto& ones = ring_ones(field, in.numel());
  ring_sub_(neg_x_mac, ring_mul(ones, key));

  return makeAShare(neg_x, neg_x_mac, field);
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  const auto key = ctx->getState<Spdz2kState>()->key();

  // lhs
  const auto& x = getValueShare(lhs);
  const auto& x_mac = getMacShare(lhs);

  // remember that rhs is public
  auto z = x.clone();
  if (comm->getRank() == 0) {
    ring_add_(z, rhs);
  }
  auto z_mac = ring_add(x_mac, ring_mul(rhs, key));

  return makeAShare(z, z_mac, field);
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x = getValueShare(lhs);
  const auto& x_mac = getMacShare(lhs);

  // rhs
  const auto& y = getValueShare(rhs);
  const auto& y_mac = getMacShare(rhs);

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
bool SingleCheck(KernelEvalContext* ctx, const ArrayRef& in) {
  static constexpr char kBindName[] = "single_check";

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  const auto& lctx = comm->lctx();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();

  // 1. Generate a random, shared value [r]
  auto [r, r_mac] = beaver->AuthCoinTossing(field, in.numel(), s);

  // 2. Locally construct [y]
  const auto& x = getValueShare(in);
  const auto& x_mac = getMacShare(in);
  auto y = ring_add(x, ring_lshift(r, k));
  auto y_mac = ring_add(x_mac, ring_lshift(r_mac, k));

  // 3. Open the value
  auto plain_y = comm->allReduce(ReduceOp::ADD, y, kBindName);

  // 4. Check the consistency of y
  auto z = ring_sub(y_mac, ring_mul(plain_y, key));
  std::string z_str(reinterpret_cast<char*>(z.data()), z.numel() * z.elsize());
  std::vector<std::string> z_strs;
  YACL_ENFORCE(commit_and_open(lctx, z_str, &z_strs));
  YACL_ENFORCE(z_strs.size() == comm->getWorldSize());

  auto plain_z = ring_zeros(field, in.numel());
  for (size_t i = 0; i < comm->getWorldSize(); ++i) {
    const auto& _z_str = z_strs[i];
    auto mem = std::make_shared<yacl::Buffer>(_z_str.data(), _z_str.size());
    ArrayRef a(mem, plain_z.eltype(), _z_str.size() / SizeOf(field), 1, 0);
    ring_add_(plain_z, a);
  }

  auto ret = spu::mpc::ring_all_equal(plain_z, ring_zeros(field, in.numel()));
  return ret;
}

static ArrayRef wrap_lshift_a(SPUContext* ctx, const ArrayRef& x, size_t k) {
  const Shape shape = {x.numel()};
  auto [res, _s, _t] = UnwrapValue(lshift_a(ctx, WrapValue(x, shape), k));
  return res;
}

static ArrayRef wrap_add_aa(SPUContext* ctx, const ArrayRef& x,
                            const ArrayRef& y) {
  SPU_ENFORCE(x.numel() == y.numel());
  const Shape shape = {x.numel()};
  auto [res, _s, _t] =
      UnwrapValue(add_aa(ctx, WrapValue(x, shape), WrapValue(y, shape)));
  return res;
}

// Refer to:
// Procedure BatchCheck, 3.2 Batch MAC Checking with Random Linear
// Combinations, SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
//
// TODO: 1. maybe all shared values using one check is better
// 2. use DISPATCH_ALL_FIELDS to improve performance
bool BatchCheck(KernelEvalContext* ctx, const std::vector<ArrayRef>& ins) {
  static constexpr char kBindName[] = "batch_check";

  SPU_ENFORCE(!ins.empty());
  const auto field = ins[0].eltype().as<Ring2k>()->field();
  const auto numel = ins[0].numel();
  auto* comm = ctx->getState<Communicator>();
  const auto& lctx = comm->lctx();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();
  const size_t k = ctx->getState<Spdz2kState>()->k();
  const size_t s = ctx->getState<Spdz2kState>()->s();

  const size_t size = ins.size();

  std::vector<ArrayRef> x_hat_v;
  std::vector<ArrayRef> mac_v;

  for (const auto& in : ins) {
    // 1. get random r and r_mac
    auto [r, r_mac] = beaver->AuthCoinTossing(field, numel, s);
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
  std::vector<ArrayRef> plain_x_hat_v;
  vectorize(x_hat_v.begin(), x_hat_v.end(), std::back_inserter(plain_x_hat_v),
            [&](const ArrayRef& s) {
              return comm->allReduce(ReduceOp::ADD, s, kBindName);
            });

  // 5. get l public random values, compute plain y
  auto pub_r = ctx->getState<Spdz2kState>()->genPublCoin(field, size);
  std::vector<uint128_t> rv;
  uint128_t mask = (static_cast<uint128_t>(1) << s) - 1;
  for (size_t i = 0; i < size; ++i) {
    rv.emplace_back(pub_r.at<uint128_t>(i) & mask);
  }

  auto plain_y = ring_zeros(field, numel);
  for (size_t i = 0; i < size; ++i) {
    ring_add_(plain_y, ring_mul(plain_x_hat_v[i], rv[i]));
  }

  // 6. compute z, commit and open z
  auto m = ring_zeros(field, numel);
  for (size_t i = 0; i < size; ++i) {
    ring_add_(m, ring_mul(mac_v[i], rv[i]));
  }

  auto plain_y_mac_share = ring_mul(plain_y, key);
  auto z = ring_sub(m, plain_y_mac_share);

  std::string z_str(reinterpret_cast<char*>(z.data()), z.numel() * z.elsize());
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
  auto plain_z = ring_zeros(field, numel);
  for (size_t i = 0; i < comm->getWorldSize(); ++i) {
    const auto& _z_str = z_strs[i];
    auto mem = std::make_shared<yacl::Buffer>(_z_str.data(), _z_str.size());
    ArrayRef a(mem, plain_z.eltype(), _z_str.size() / SizeOf(field), 1, 0);
    ring_add_(plain_z, a);
  }

  auto ret = spu::mpc::ring_all_equal(plain_z, ring_zeros(field, numel));
  return ret;
}

ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // lhs
  const auto& x = getValueShare(lhs);
  const auto& x_mac = getMacShare(lhs);

  // ret
  const auto& z = ring_mul(x, rhs);
  const auto& z_mac = ring_mul(x_mac, rhs);

  return makeAShare(z, z_mac, field);
}

// Refer to:
// 4 Online Phase, SPDZ2k: Efficient MPC mod 2k for Dishonest Majority
// - https://eprint.iacr.org/2018/482.pdf
//
// TODO: use DISPATCH_ALL_FIELDS instead of ring ops to improve performance
ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  auto* arr_ref_v = ctx->getState<Spdz2kState>()->arr_ref_v();
  const auto key = ctx->getState<Spdz2kState>()->key();

  // in
  const auto& x = getValueShare(lhs);
  const auto& x_mac = getMacShare(lhs);
  const auto& y = getValueShare(rhs);
  const auto& y_mac = getMacShare(rhs);

  // e = x - a, f = y - b
  auto [vec, mac_vec] = beaver->AuthMul(field, lhs.numel());
  auto [a, b, c] = vec;
  auto [a_mac, b_mac, c_mac] = mac_vec;

  auto e = ring_sub(x, a);
  auto e_mac = ring_sub(x_mac, a_mac);
  auto f = ring_sub(y, b);
  auto f_mac = ring_sub(y_mac, b_mac);

  // add to check array
  arr_ref_v->emplace_back(makeAShare(e, e_mac, field));
  arr_ref_v->emplace_back(makeAShare(f, f_mac, field));

  // open e, f
  auto res = vectorize({e, f}, [&](const ArrayRef& s) {
    return comm->allReduce(ReduceOp::ADD, s, kBindName);
  });

  auto p_e = std::move(res[0]);
  auto p_f = std::move(res[1]);
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
ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                        const ArrayRef& rhs, size_t m, size_t n,
                        size_t k) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();

  // in
  const auto& x = getValueShare(lhs);
  const auto& x_mac = getMacShare(lhs);
  const auto& y = rhs;

  // ret
  auto z = ring_mmul(x, y, m, n, k);
  auto z_mac = ring_mmul(x_mac, y, m, n, k);
  return makeAShare(z, z_mac, field);
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                        const ArrayRef& rhs, size_t m, size_t n,
                        size_t k) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();
  const auto key = ctx->getState<Spdz2kState>()->key();

  const auto& x = getValueShare(lhs);
  const auto& y = getValueShare(rhs);
  // const auto& x_mac = getMacShare(lhs);
  // const auto& y_mac = getMacShare(rhs);

  // generate beaver multiple triple.
  auto [vec, mac_vec] = beaver->AuthDot(field, m, n, k);
  auto [a, b, c] = vec;
  auto [a_mac, b_mac, c_mac] = mac_vec;

  // open x-a & y-b
  auto res =
      vectorize({ring_sub(x, a), ring_sub(y, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::ADD, s, kBindName);
      });
  auto p_e = std::move(res[0]);
  auto p_f = std::move(res[1]);
  auto p_ef = ring_mmul(p_e, p_f, m, n, k);

  // z = p_e dot b + a dot p_f + c;
  auto z = ring_add(
      ring_add(ring_mmul(p_e, b, m, n, k), ring_mmul(a, p_f, m, n, k)), c);
  if (comm->getRank() == 0) {
    // z += p_e dot p_f;
    ring_add_(z, ring_mmul(p_e, p_f, m, n, k));
  }

  // zmac = p_e dot b_mac + a_mac dot p_f + c_mac + (p_e dot p_f) * key;
  auto zmac =
      ring_add(ring_mmul(p_e, b_mac, m, n, k), ring_mmul(a_mac, p_f, m, n, k));
  ring_add_(zmac, c_mac);
  ring_add_(zmac, ring_mul(p_ef, key));

  return makeAShare(z, zmac, field);
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  // in
  const auto& x = getValueShare(in);
  const auto& x_mac = getMacShare(in);

  // ret
  const auto& z = ring_lshift(x, bits);
  const auto& z_mac = ring_lshift(x_mac, bits);
  return makeAShare(z, z_mac, field);
}

// ABY3, truncation pair method.
// Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
//
// TODO: optimize for 2pc.
ArrayRef TruncA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                      size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto key = ctx->getState<Spdz2kState>()->key();
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Spdz2kState>()->beaver();

  const auto& x = getValueShare(in);
  const auto& [vec, mac_vec] = beaver->AuthTrunc(field, x.numel(), bits);
  const auto& [r, rb] = vec;
  const auto& [r_mac, rb_mac] = mac_vec;

  // open x - r
  auto x_r = comm->allReduce(ReduceOp::ADD, ring_sub(x, r), kBindName);
  auto tr_x_r = ring_arshift(x_r, bits);

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
