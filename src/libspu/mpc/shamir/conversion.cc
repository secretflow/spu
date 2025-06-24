// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/shamir/conversion.h"

#include <future>

#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/common/pv_gfmp.h"
#include "libspu/mpc/shamir/type.h"
#include "libspu/mpc/shamir/value.h"
#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::shamir {

namespace {

inline bool is_secret(const NdArrayRef& in) {
  return in.eltype().isa<Secret>();
}

inline bool is_public(const NdArrayRef& in) {
  return in.eltype().isa<Public>();
}

// this return value is splat while wrap_make_zeros  are
// tensors
NdArrayRef hack_make_p(SPUContext* ctx, uint128_t init, const Shape& shape) {
  return UnwrapValue(dynDispatch(ctx, "make_p", init, shape));
}

NdArrayRef wrap_make_zeros(SPUContext* ctx, const Shape& shape) {
  const auto field = ctx->getField();
  return ring_zeros(field, shape).as(makeType<PubGfmpTy>(field));
}

NdArrayRef wrap_arshift_p(SPUContext* ctx, const NdArrayRef& in,
                          const Sizes& bits) {
  return UnwrapValue(arshift_p(ctx, WrapValue(in), bits));
}

NdArrayRef wrap_a2p(SPUContext* ctx, const NdArrayRef& in) {
  return UnwrapValue(a2p(ctx, WrapValue(in)));
}

NdArrayRef wrap_mul_p(SPUContext* ctx, const NdArrayRef& x,
                      const NdArrayRef& y) {
  return UnwrapValue(mul_aa_p(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_mul_aaa(SPUContext* ctx, const NdArrayRef& x,
                        const NdArrayRef& y, const NdArrayRef& z) {
  return UnwrapValue(mul_aaa(ctx, WrapValue(x), WrapValue(y), WrapValue(z)));
}

NdArrayRef wrap_mul(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  if (is_public(x) && is_public(y)) {
    return UnwrapValue(mul_pp(ctx, WrapValue(x), WrapValue(y)));
  } else if (is_secret(x) && is_public(y)) {
    return UnwrapValue(mul_ap(ctx, WrapValue(x), WrapValue(y)));
  } else if (is_secret(y) && is_public(x)) {
    return UnwrapValue(mul_ap(ctx, WrapValue(y), WrapValue(x)));
  } else if (is_secret(x) && is_secret(y)) {
    return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
  }
  SPU_THROW("should not reach, x={}, y={}", x.eltype(), y.eltype());
}

NdArrayRef wrap_add(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  if (is_public(x) && is_public(y)) {
    return UnwrapValue(add_pp(ctx, WrapValue(x), WrapValue(y)));
  } else if (is_secret(x) && is_public(y)) {
    return UnwrapValue(add_ap(ctx, WrapValue(x), WrapValue(y)));
  } else if (is_secret(y) && is_public(x)) {
    return UnwrapValue(add_ap(ctx, WrapValue(y), WrapValue(x)));
  } else if (is_secret(x) && is_secret(y)) {
    return UnwrapValue(add_aa(ctx, WrapValue(x), WrapValue(y)));
  }
  SPU_THROW("should not reach");
}

NdArrayRef wrap_negate(SPUContext* ctx, const NdArrayRef& x) {
  if (is_public(x)) {
    return UnwrapValue(negate_p(ctx, WrapValue(x)));
  } else if (is_secret(x)) {
    return UnwrapValue(negate_a(ctx, WrapValue(x)));
  }
  SPU_THROW("should not reach");
}

NdArrayRef wrap_sub(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  return wrap_add(ctx, x, wrap_negate(ctx, y));
}

NdArrayRef wrap_rand_a(SPUContext* ctx, const Shape& shape) {
  return UnwrapValue(rand_a(ctx, shape));
}

// x ^ y = (x + y) - 2 * (x * y) for x,y in [0,1]
NdArrayRef wrap_xor(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  auto k2 = hack_make_p(ctx, 2, x.shape());
  auto x_mul_y = wrap_mul(ctx, x, y);
  auto x_add_y = wrap_add(ctx, x, y);
  auto k2xy = wrap_mul(ctx, x_mul_y, k2);
  auto out = wrap_sub(ctx, x_add_y, k2xy);
  return out;
}

// [Offline Phase]
std::pair<std::vector<NdArrayRef>, std::vector<NdArrayRef>>
gen_prefix_mult_share(SPUContext* ctx, const int64_t numel,
                      const int64_t num_prefix) {
  // let k denote num_prefix
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto ty = makeType<PubGfmpTy>(field);

  auto mul_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul(ctx, a, b);
  };
  auto mul_p_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul_p(ctx, a, b);
  };

  NdArrayRef rand_raw = wrap_rand_a(ctx, {(num_prefix << 1) * numel});

  // for each instance, generate r_1 ..., r_k and s_1 ..., s_k
  std::vector<NdArrayRef> rand_r;
  std::vector<NdArrayRef> rand_s;

  int64_t offset = num_prefix * numel;

  for (int64_t i = 0; i < num_prefix; ++i) {
    rand_r.push_back(
        rand_raw.slice({i * numel}, {(i + 1) * numel}, {}).reshape({numel}));
    rand_s.push_back(
        rand_raw.slice({offset + i * numel}, {offset + (i + 1) * numel}, {})
            .reshape({numel}));
  }

  std::vector<NdArrayRef> rand_prod;
  std::vector<NdArrayRef> rand_prod_offset;
  // TODO: The following two multiplications (mul and mul_p) can be run in
  // parallel. rand_prod is of length num_prefix storing B_i = r_i * s_i
  vmap(rand_r.cbegin(), rand_r.cend(), rand_s.cbegin(), rand_s.cend(),
       std::back_inserter(rand_prod), mul_p_lambda);

  // rand_prod_offset is of length num_prefix - 1 storing C_0 = s_0 and C_i =
  // r_{i-1} * s_{i} for i > 1
  rand_prod_offset.push_back(rand_s[0]);
  vmap(rand_r.cbegin(), rand_r.cend() - 1, rand_s.cbegin() + 1, rand_s.cend(),
       std::back_inserter(rand_prod_offset), mul_lambda);

  auto p_rand_prod = rand_prod[0];
  // rand_prod is of length num_prefix storing B_i^-1 = (r_i * s_i)^{-1}
  std::vector<NdArrayRef> rand_prod_inv;
  for (int64_t i = 0; i < num_prefix; ++i) {
    rand_prod_inv.push_back(gfmp_batch_inverse(rand_prod[i]));
  }

  // An unbounded multiplication instance
  // ( [r0]_t , [r0^-1]_t )
  // ( [r1]_t , [r0 * r1^-1]_t,)
  // ( [r2]_t , [r1 * r2^-1]_t)
  // ...
  // ( [ri]_t , [ri-1 * ri^-1]_t) for i = 2, ..., k
  std::vector<NdArrayRef> rand_r_aux;
  vmap(rand_prod_inv.cbegin(), rand_prod_inv.cend(), rand_prod_offset.cbegin(),
       rand_prod_offset.cend(), std::back_inserter(rand_r_aux), mul_lambda);

  std::pair<std::vector<NdArrayRef>, std::vector<NdArrayRef>> out;
  out.first = std::move(rand_r);
  out.second = std::move(rand_r_aux);
  return out;
}

// Generate zero sharings of degree = threshold
// [Offline Phase]
NdArrayRef gen_zero_shares(KernelEvalContext* ctx, int64_t numel,
                           int64_t threshold) {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  auto ty = makeType<PubGfmpTy>(field);
  auto coeffs = prg_state->genPublWithMersennePrime(field, {threshold * numel}).as(ty);
  NdArrayRef zeros = ring_zeros(field, {numel}).as(makeType<GfmpTy>(field));
  auto shares =
      gfmp_rand_shamir_shares(zeros, coeffs, comm->getWorldSize(), threshold);
  return shares[comm->getRank()].as(makeType<AShrTy>(field));
}
// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  Page 11: Protocol RAN2
// [Offline Phase]
NdArrayRef rand_bits(SPUContext* ctx, int64_t numel) {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  NdArrayRef out(makeType<AShrTy>(field), {numel});
  std::vector<int64_t> cur_failed_indices;
  std::vector<int64_t> pre_failed_indices;
  std::mutex idx_mtx;
  int64_t produced = 0;
  int64_t un_produced = numel;

  NdArrayRef rand_a(out.eltype(), out.shape());
  NdArrayRef rand_sqrt(out.eltype(), out.shape());
  while (un_produced > 0) {
    NdArrayRef tmp_rand_sqrt(out.eltype(), out.shape());
    auto tmp_rand_a = wrap_rand_a(ctx, {un_produced});
    auto tmp_rand_a_square_p = wrap_mul_p(ctx, tmp_rand_a, tmp_rand_a);
    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> _rand_a(rand_a);
      NdArrayView<ring2k_t> _tmp_rand_a(tmp_rand_a);
      NdArrayView<ring2k_t> _tmp_rand_a_square_p(tmp_rand_a_square_p);
      NdArrayView<ring2k_t> _rand_sqrt(rand_sqrt);
      NdArrayView<ring2k_t> _tmp_rand_sqrt(tmp_rand_sqrt);
      pforeach(0, un_produced, [&](int64_t idx) {
        if(_tmp_rand_a_square_p[idx] != 0) {
          _tmp_rand_sqrt[idx] = sqrt_mod(_tmp_rand_a_square_p[idx]);
        } else {
          std::unique_lock lock(idx_mtx);
          cur_failed_indices.push_back(idx);
        }
      });
      if(pre_failed_indices.empty()) {
        ring_assign(rand_a, tmp_rand_a);
        ring_assign(rand_sqrt, tmp_rand_sqrt);
      } else {
        pforeach(0, un_produced, [&](int64_t idx) {
          _rand_a[pre_failed_indices[idx]] = _tmp_rand_a[idx];
          _rand_sqrt[pre_failed_indices[idx]] = _tmp_rand_sqrt[idx];
        });
      }
      un_produced = cur_failed_indices.size();
      produced = numel - un_produced;
      pre_failed_indices = cur_failed_indices;
      cur_failed_indices.clear();
    });
  }

  NdArrayRef rand_sqrt_inverse = gfmp_batch_inverse(rand_sqrt);
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _rand_a(rand_a);
    NdArrayView<ring2k_t> _rand_sqrt_inverse(rand_sqrt_inverse);
    NdArrayView<ring2k_t> _out(out);
    const ring2k_t inverse_two = mul_inv(static_cast<ring2k_t>(2));
    pforeach(0, numel, [&](int64_t idx) {
      auto c = mul_mod(_rand_a[idx], _rand_sqrt_inverse[idx]);
      _out[idx] = mul_mod(inverse_two, add_mod(c, static_cast<ring2k_t>(1)));
    });
  });
  return out;
}

// Ref: https://dl.acm.org/doi/10.5555/3698900.3699009 Section 2.5
//  Unbounded Fan-In Prefix-products
std::vector<NdArrayRef> prefix_mul(SPUContext* ctx,
                                   const std::vector<NdArrayRef>& inputs) {
  SPU_ENFORCE(!inputs.empty());
  int64_t l = inputs.size();
  auto numel = inputs[0].numel();
  auto mul_p_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul_p(ctx, a, b);
  };
  auto mul_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul(ctx, a, b);
  };

  auto randomness = gen_prefix_mult_share(ctx, numel, l);
  auto r = randomness.first;
  auto r_aux = randomness.second;

  std::vector<NdArrayRef> out;
  vmap(inputs.cbegin(), inputs.cend(), r_aux.cbegin(), r_aux.cend(),
       std::back_inserter(out), mul_p_lambda);
  for (int64_t i = 1; i < l; ++i) {
    out[i] = wrap_mul(ctx, out[i], out[i - 1]);
  }
  vmap(out.cbegin(), out.cend(), r.cbegin(), r.cend(), out.begin(), mul_lambda);

  return out;
}

// Ref:
// https://www.usenix.org/system/files/sec24summer-prepub-278-liu-fengrun.pdf
//  Protocol 4.1: Prefix Or
// FIXME There is a security issue in this protocol. 
std::vector<NdArrayRef> prefix_or(SPUContext* ctx,
                                  const std::vector<NdArrayRef>& inputs) {
  SPU_ENFORCE(!inputs.empty());
  std::vector<NdArrayRef> b(inputs.size());
  const auto k1 = hack_make_p(ctx, 1, inputs[0].shape());
  for (size_t i = 0; i < inputs.size(); ++i) {
    b[i] = wrap_sub(ctx, k1, inputs[i]);
  }
  auto c = prefix_mul(ctx, b);
  for (size_t i = 0; i < inputs.size(); ++i) {
    c[i] = wrap_sub(ctx, k1, c[i]);
  }
  return c;
}

// Ref:
// https://dl.acm.org/doi/10.5555/3698900.3699009
//  Protocol 4.2: Optimized bitwise less-than for public a and secret b
NdArrayRef bit_lt_pa(SPUContext* ctx, const std::vector<NdArrayRef>& a,
                     const std::vector<NdArrayRef>& b) {
  SPU_ENFORCE(!a.empty());
  SPU_ENFORCE_EQ(a.size(), b.size());
  SPU_ENFORCE(std::all_of(a.begin(), a.end(),
                          [](const NdArrayRef& x) { return is_public(x); }));
  SPU_ENFORCE(std::all_of(b.begin(), b.end(),
                          [](const NdArrayRef& x) { return is_secret(x); }));
  std::vector<NdArrayRef> c;
  std::vector<NdArrayRef> inv_a(a.size());
  {
    const auto k1 = hack_make_p(ctx, 1, a[0].shape());
    std::vector<NdArrayRef> inv_b(a.size());
    auto xor_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
      return wrap_xor(ctx, a, b);
    };
    for (size_t i = 0; i < a.size(); ++i) {
      inv_a[i] = wrap_sub(ctx, k1, a[i]);
      inv_b[i] = wrap_sub(ctx, k1, b[i]);
    }
    vmap(inv_a.begin(), inv_a.end(), inv_b.begin(), inv_b.end(),
         std::back_inserter(c), xor_lambda);
  }
  std::vector<NdArrayRef> h;
  {
    std::reverse(c.begin(), c.end());
    auto d = prefix_or(ctx, c);
    std::reverse(d.begin(), d.end());

    std::vector<NdArrayRef> e(a.size());
    e.back() = d.back();
    for (size_t i = 0; i < e.size() - 1; ++i) {
      e[i] = wrap_sub(ctx, d[i], d[i + 1]);
    }

    auto mul_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
      return wrap_mul(ctx, a, b);
    };
    vmap(inv_a.begin(), inv_a.end(), e.begin(), e.end(), std::back_inserter(h),
         mul_lambda);
  }

  NdArrayRef out = wrap_make_zeros(ctx, a[0].shape());
  for (size_t i = 0; i < h.size(); ++i) {
    out = wrap_add(ctx, out, h[i]);
  }
  return out;
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  5.3: Generic bitwise less-than
NdArrayRef bit_lt(SPUContext* ctx, const std::vector<NdArrayRef>& a,
                  const std::vector<NdArrayRef>& b) {
  SPU_ENFORCE(!a.empty());
  SPU_ENFORCE_EQ(a.size(), b.size());
  std::vector<NdArrayRef> e;

  auto xor_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_xor(ctx, a, b);
  };

  vmap(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(e),
       xor_lambda);

  std::reverse(e.begin(), e.end());
  auto f = prefix_or(ctx, e);

  std::reverse(f.begin(), f.end());
  std::vector<NdArrayRef> g(f.size());
  g.back() = f.back();
  for (size_t i = 0; i < f.size() - 1; ++i) {
    g[i] = wrap_sub(ctx, f[i], f[i + 1]);
  }

  std::vector<NdArrayRef> h;
  auto mul_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul(ctx, a, b);
  };
  vmap(g.cbegin(), g.cend(), b.begin(), b.end(), std::back_inserter(h),
       mul_lambda);

  NdArrayRef out = wrap_make_zeros(ctx, a[0].shape());
  for (size_t i = 0; i < h.size(); ++i) {
    out = wrap_add(ctx, out, h[i]);
  }

  return out;
}

using Triple = std::tuple<NdArrayRef, NdArrayRef, NdArrayRef>;

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  6.4: Unbounded fan-in carry propagation
Triple unbounded_carries(SPUContext* ctx, const std::vector<NdArrayRef>& s,
                         const std::vector<NdArrayRef>& p,
                         const std::vector<NdArrayRef>& k) {
  SPU_ENFORCE(!s.empty());
  SPU_ENFORCE_EQ(s.size(), p.size());
  SPU_ENFORCE_EQ(k.size(), p.size());
  NdArrayRef a;

  auto p_copy = p;
  std::reverse(p_copy.begin(), p_copy.end());
  std::vector<NdArrayRef> q = prefix_mul(ctx, p_copy);
  NdArrayRef b = q[q.size() - 1];
  NdArrayRef c = wrap_make_zeros(ctx, s[0].shape());
  {
    std::reverse(q.begin(), q.end());
    auto mul_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
      return wrap_mul(ctx, a, b);
    };
    std::vector<NdArrayRef> c_vec;
    vmap(k.begin(), k.end() - 1, q.cbegin() + 1, q.cend(),
         std::back_inserter(c_vec), mul_lambda);
    c_vec.push_back(k.back());
    for (size_t i = 0; i < c_vec.size(); ++i) {
      c = wrap_add(ctx, c, c_vec[i]);
    }
    auto k1 = hack_make_p(ctx, 1, s[0].shape());
    a = wrap_sub(ctx, k1, wrap_add(ctx, b, c));
  }
  Triple out;
  std::get<0>(out) = std::move(a);
  std::get<1>(out) = std::move(b);
  std::get<2>(out) = std::move(c);
  return out;
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  6.1: Generic prefix computations for carry propagation
// Todo: Optimize this function using the CFL algorithm
//  Ref: https://dl.acm.org/doi/pdf/10.1145/800061.808732
std::vector<Triple> prefix_carries(SPUContext* ctx,
                                   const std::vector<NdArrayRef>& s,
                                   const std::vector<NdArrayRef>& p,
                                   const std::vector<NdArrayRef>& k) {
  SPU_ENFORCE(!s.empty());
  SPU_ENFORCE_EQ(s.size(), p.size());
  SPU_ENFORCE_EQ(k.size(), p.size());
  std::vector<Triple> out(s.size());
  out[0] = {s[0], p[0], k[0]};

  std::vector<std::future<Triple>> futures;
  std::vector<std::unique_ptr<SPUContext>> sub_ctxs;
  for (size_t i = 1; i < s.size(); ++i) {
    sub_ctxs.push_back(ctx->fork());
  }
  for (size_t i = 1; i < s.size(); ++i) {
    auto tmp_s = std::vector<NdArrayRef>(s.begin(), s.begin() + i + 1);
    auto tmp_p = std::vector<NdArrayRef>(p.begin(), p.begin() + i + 1);
    auto tmp_k = std::vector<NdArrayRef>(k.begin(), k.begin() + i + 1);
    auto async_res = std::async(unbounded_carries, sub_ctxs[i - 1].get(), tmp_s,
                                tmp_p, tmp_k);
    futures.push_back(std::move(async_res));
  }
  for (size_t i = 1; i < s.size(); ++i) {
    out[i] = futures[i - 1].get();
  }

  return out;
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  6.3: Carries
std::vector<NdArrayRef> carries(SPUContext* ctx,
                                const std::vector<NdArrayRef>& a,
                                const std::vector<NdArrayRef>& b) {
  SPU_ENFORCE(!a.empty());
  SPU_ENFORCE_EQ(a.size(), b.size());

  auto mul_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul(ctx, a, b);
  };

  std::vector<NdArrayRef> s;
  // S[i] = A[i] * B[i]
  vmap(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(s),
       mul_lambda);

  std::vector<NdArrayRef> p(a.size());
  std::vector<NdArrayRef> k(a.size());
  std::vector<NdArrayRef> c(a.size());
  const auto k2 = hack_make_p(ctx, 2, a[0].shape());
  const auto k1 = hack_make_p(ctx, 1, a[0].shape());
  for (size_t i = 0; i < a.size(); ++i) {
    // P[i] = A[i] + B[i] - 2 * S[i]
    p[i] = wrap_sub(ctx, wrap_add(ctx, a[i], b[i]), wrap_mul(ctx, k2, s[i]));
    // K[i] = 1 - S[i] - P[i]
    k[i] = wrap_sub(ctx, k1, wrap_add(ctx, s[i], p[i]));
  }
  auto f = prefix_carries(ctx, s, p, k);
  for (size_t i = 0; i < a.size(); ++i) {
    c[i] = std::move(std::get<0>(f[i]));
  }

  return c;
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  6.2: Bitwise sum
std::vector<NdArrayRef> bit_add(SPUContext* ctx,
                                const std::vector<NdArrayRef>& a,
                                const std::vector<NdArrayRef>& b) {
  SPU_ENFORCE(!a.empty());
  SPU_ENFORCE_EQ(a.size(), b.size());
  auto c = carries(ctx, a, b);

  std::vector<NdArrayRef> d(a.size() + 1);
  const auto k2 = hack_make_p(ctx, 2, a[0].shape());
  // D[0] = A[0] + B[0] - 2 * C[0]
  d[0] = wrap_sub(ctx, wrap_add(ctx, a[0], b[0]), wrap_mul(ctx, k2, c[0]));
  // D[l] = C[l-1]
  d[a.size()] = c[a.size() - 1];
  // D[i] = A[i] + B[i] + C[i-1] - 2 * C[i]
  for (size_t i = 1; i < a.size(); ++i) {
    d[i] = wrap_sub(ctx, wrap_add(ctx, wrap_add(ctx, a[i], b[i]), c[i - 1]),
                    wrap_mul(ctx, k2, c[i]));
  }

  return d;
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  Page 3.1: Solved bits
// [Offline Phase]
std::pair<std::vector<NdArrayRef>, NdArrayRef> solved_bits(SPUContext* ctx,
                                                           const Shape& shape) {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  std::vector<int64_t> cur_failed_indices;
  std::vector<int64_t> pre_failed_indices;
  std::mutex idx_mtx;
  int64_t numel = shape.numel();
  int64_t un_produced = numel;

  return DISPATCH_ALL_FIELDS(field, [&]() {
    size_t exp = ScalarTypeToPrime<ring2k_t>::exp;
    std::vector<NdArrayRef> out_bits(exp);
    NdArrayRef out = wrap_make_zeros(ctx, shape).as(makeType<AShrTy>(field));
    auto randbits = rand_bits(ctx, numel * exp);
    for (int64_t i = 0; i < static_cast<int64_t>(exp); ++i) {
      out_bits[i] =
          randbits.slice({i * numel}, {(i + 1) * numel}, {}).reshape(shape);
    }

    NdArrayView<ring2k_t> _out(out);
    pforeach(0, numel, [&](int64_t idx) {
      for (size_t i = 0; i < exp; ++i) {
        NdArrayView<ring2k_t> _out_i(out_bits[i]);
        _out[idx] = add_mod(
            _out[idx], mul_mod(static_cast<ring2k_t>(1) << i, _out_i[idx]));
      }
    });

    std::pair<std::vector<NdArrayRef>, NdArrayRef> ret;
    ret.first = std::move(out_bits);
    ret.second = std::move(out);
    return ret;
  });

  return DISPATCH_ALL_FIELDS(field, [&]() {
    size_t exp = ScalarTypeToPrime<ring2k_t>::exp;
    std::vector<NdArrayRef> out_bits(exp);
    NdArrayRef out = wrap_make_zeros(ctx, shape).as(makeType<AShrTy>(field));

    auto k1 = hack_make_p(ctx, 1, shape);
    std::vector<NdArrayRef> tmp_p_bits(exp, k1);
    while (un_produced > 0) {
      std::vector<NdArrayRef> tmp_out(exp);
      auto randbits = rand_bits(ctx, numel * exp);
      for (int64_t i = 0; i < static_cast<int64_t>(exp); ++i) {
        tmp_out[i] =
            randbits.slice({i * numel}, {(i + 1) * numel}, {}).reshape(shape);
      }
      auto cmp = bit_lt(ctx, tmp_out, tmp_p_bits);
      auto cmp_p = wrap_a2p(ctx, cmp);
      NdArrayView<ring2k_t> _cmp_p(cmp);
      pforeach(0, un_produced, [&](int64_t idx) {
        if (_cmp_p[idx] == 0) {
          std::unique_lock lock(idx_mtx);
          cur_failed_indices.push_back(idx);
        }
      });
      NdArrayView<ring2k_t> _out(out);
      if (pre_failed_indices.empty()) {
        for (size_t i = 0; i < exp; ++i) {
          out_bits[i] = tmp_out[i];
        }
        pforeach(0, un_produced, [&](int64_t idx) {
          for (size_t i = 0; i < exp; ++i) {
            NdArrayView<ring2k_t> _tmp_out_i(tmp_out[i]);
            _out[idx] = add_mod(
                _out[idx],
                mul_mod(static_cast<ring2k_t>(1) << i, _tmp_out_i[idx]));
          }
        });
      } else {
        pforeach(0, un_produced, [&](int64_t idx) {
          _out[pre_failed_indices[idx]] = 0;
          for (size_t i = 0; i < exp; ++i) {
            NdArrayView<ring2k_t> _tmp_out_i(tmp_out[i]);
            _out[pre_failed_indices[idx]] = add_mod(
                _out[pre_failed_indices[idx]],
                mul_mod(static_cast<ring2k_t>(1) << i, _tmp_out_i[idx]));
            ;
          }
        });
      }
      un_produced = cur_failed_indices.size();
      pre_failed_indices = cur_failed_indices;
      cur_failed_indices.clear();
    }
    std::pair<std::vector<NdArrayRef>, NdArrayRef> ret;
    ret.first = std::move(out_bits);
    ret.second = std::move(out);
    return ret;
  });
}

std::vector<NdArrayRef> bit_decompose(SPUContext* ctx, const NdArrayRef& in) {
  auto numel = in.numel();
  auto field = in.eltype().as<GfmpTy>()->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    size_t exp = ScalarTypeToPrime<ring2k_t>::exp;
    std::vector<NdArrayRef> out;
    for (size_t bit = 0; bit < exp; bit++) {
      out.push_back(
          wrap_make_zeros(ctx, in.shape()).as(makeType<PubGfmpTy>(field)));
    }
    NdArrayView<ring2k_t> _in(in);
    pforeach(0, numel, [&](int64_t idx) {
      const auto& v = _in[idx];
      for (size_t bit = 0; bit < exp; bit++) {
        NdArrayView<ring2k_t> _out(out[bit]);
        _out[idx] = (static_cast<ring2k_t>(v) >> bit) & 0x1;
      }
    });
    return out;
  });
}

}  // namespace

using Bits = std::vector<NdArrayRef>;
// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  Page 9: Protocol BITS
NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* sctx = ctx->sctx();
  std::vector<NdArrayRef> b_bits;
  NdArrayRef b;
  std::tie(b_bits, b) = solved_bits(sctx, x.shape());
  auto a_minus_b = wrap_sub(sctx, x, b);
  auto c = wrap_a2p(sctx, a_minus_b);
  auto c_bits = bit_decompose(sctx, c);
  const auto k1 = hack_make_p(sctx, 1, x.shape());
  auto c_plus_one = wrap_add(sctx, c, k1);
  auto c_plus_one_bits = bit_decompose(sctx, c_plus_one);

  std::unique_ptr<SPUContext> sub_ctxs = sctx->fork();

  auto futures = std::async(bit_add, sub_ctxs.get(), c_bits, b_bits);

  auto c2 = bit_add(sctx, c_plus_one_bits, b_bits);
  auto c1 = futures.get();

  NdArrayRef s = c1.back();

  auto sub_lambda = [sctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_sub(sctx, a, b);
  };
  auto add_lambda = [sctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_add(sctx, a, b);
  };
  auto mul_lambda = [sctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul(sctx, a, b);
  };
  std::vector<NdArrayRef> c_delta;
  vmap(c2.cbegin(), c2.cend() - 1, c1.cbegin(), c1.cend() - 1,
       std::back_inserter(c_delta), sub_lambda);

  std::vector<NdArrayRef> s_bits(c_delta.size(), s);
  std::vector<NdArrayRef> prod;
  vmap(s_bits.cbegin(), s_bits.cend(), c_delta.cbegin(), c_delta.cend(),
       std::back_inserter(prod), mul_lambda);

  std::vector<NdArrayRef> x_bits;
  vmap(c1.cbegin(), c1.cend() - 1, prod.cbegin(), prod.cend(),
       std::back_inserter(x_bits), add_lambda);

  NdArrayRef out(makeType<BShrTy>(field, prod.size()), x.shape());
  // vmap()
  for (size_t i = 0; i < x_bits.size(); ++i) {
    auto out_i = getBitShare(out, i);
    ring_assign(out_i, x_bits[i]);
  }
  return out;
}

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto* ty = x.eltype().as<BShrTy>();
  const auto field = ty->field();
  const auto nbits = ty->nbits();

  auto out = wrap_make_zeros(ctx->sctx(), x.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    for (size_t i = 0; i < nbits; ++i) {
      auto bit_i = getBitShare(x, i);
      NdArrayView<ring2k_t> _bit_i(bit_i);
      NdArrayView<ring2k_t> _out(out);
      pforeach(0, x.numel(), [&](int64_t idx) {
        _out[idx] = add_mod(
            _out[idx], mul_mod(static_cast<ring2k_t>(1) << i, _bit_i[idx]));
      });
    }
  });
  return out.as(makeType<AShrTy>(field));
}

// Ref:
// https://www.usenix.org/system/files/sec24summer-prepub-278-liu-fengrun.pdf
// Protocol 3.1: Truncation with 1-bit gap, i.e. x must be in the range of $[-2^{k-2}, 2^{k-2})$, where $k$ is the bits of the prime.
NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        size_t bits, SignType sign) const {
  (void)sign;  // TODO: optimize me.

  const auto field = x.eltype().as<GfmpTy>()->field();
  auto* sctx = ctx->sctx();
  std::vector<NdArrayRef> r_bits;
  NdArrayRef r;
  std::tie(r_bits, r) = solved_bits(sctx, x.shape());
  auto r_msb = r_bits.back();
  return DISPATCH_ALL_FIELDS(field, [&]() {
    auto l = ScalarTypeToPrime<ring2k_t>::exp;
    SPU_ENFORCE_LT(bits, l);
    NdArrayRef r_hat = wrap_make_zeros(sctx, x.shape());
    for (size_t i = bits; i < l; ++i) {
      auto k =
          hack_make_p(sctx, static_cast<uint128_t>(1) << (i - bits), x.shape());
      r_hat = wrap_add(sctx, r_hat, wrap_mul(sctx, k, r_bits[i]));
    }
    for (size_t i = l - bits; i < l; ++i) {
      auto k = hack_make_p(sctx, static_cast<uint128_t>(1) << i, x.shape());
      r_hat = wrap_add(sctx, r_hat, wrap_mul(sctx, k, r_msb));
    }
    // k2 = 2^(l-2)
    auto k2 =
        hack_make_p(sctx, static_cast<uint128_t>(1) << (l - 2), x.shape());
    auto b = wrap_add(sctx, x, k2);
    auto c = wrap_add(sctx, b, r);
    auto c_p = wrap_a2p(sctx, c);
    auto c_p_trunc = wrap_arshift_p(sctx, c_p, {static_cast<int64_t>(bits)});
    auto c_bits = bit_decompose(sctx, c_p);
    auto c_msb = c_bits.back();

    // k1 = 1
    auto k1 = hack_make_p(sctx, 1, x.shape());
    auto e = wrap_mul(sctx, wrap_sub(sctx, k1, r_msb), c_msb);
    // k3 = 2^(l-d) - 1
    auto k3 = hack_make_p(sctx, (static_cast<uint128_t>(1) << (l - bits)) - 1,
                          x.shape());
    auto f =
        wrap_add(sctx, wrap_sub(sctx, c_p_trunc, r_hat), wrap_mul(sctx, e, k3));

    // k4 = 2^(l-d-2)
    auto k4 = hack_make_p(sctx, static_cast<uint128_t>(1) << (l - bits - 2),
                          x.shape());
    auto ret = wrap_sub(sctx, f, k4);
    return ret;
  });
}

// Ref:
// https://www.usenix.org/system/files/sec24summer-prepub-278-liu-fengrun.pdf
// Protocol 3.2: Fixed-Mult
NdArrayRef MulAATrunc::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                            const NdArrayRef& y, size_t bits,
                            SignType sign) const {
  (void)sign;  // TODO: optimize me.
  SPU_ENFORCE(x.numel() == y.numel());
  SPU_ENFORCE_EQ(x.eltype(), y.eltype());

  // local mul
  auto tmp_2t = gfmp_mul_mod(x, y);

  const auto field = x.eltype().as<GfmpTy>()->field();
  auto* sctx = ctx->sctx();
  std::vector<NdArrayRef> r_bits;
  NdArrayRef r;
  std::tie(r_bits, r) = solved_bits(sctx, x.shape());
  auto zero_shares =
      gen_zero_shares(ctx, tmp_2t.numel(), sctx->config().sss_threshold() << 1)
          .reshape(tmp_2t.shape());

  auto r_msb = r_bits.back();
  return DISPATCH_ALL_FIELDS(field, [&]() {
    auto l = ScalarTypeToPrime<ring2k_t>::exp;
    SPU_ENFORCE_LT(bits, l);
    NdArrayRef r_hat = wrap_make_zeros(sctx, x.shape());
    for (size_t i = bits; i < l; ++i) {
      auto k =
          hack_make_p(sctx, static_cast<uint128_t>(1) << (i - bits), x.shape());
      r_hat = wrap_add(sctx, r_hat, wrap_mul(sctx, k, r_bits[i]));
    }
    for (size_t i = l - bits; i < l; ++i) {
      auto k = hack_make_p(sctx, static_cast<uint128_t>(1) << i, x.shape());
      r_hat = wrap_add(sctx, r_hat, wrap_mul(sctx, k, r_msb));
    }

    NdArrayRef r_2t = wrap_make_zeros(sctx, tmp_2t.shape());
    for (size_t i = 0; i < l; ++i) {
      auto k = hack_make_p(sctx, static_cast<uint128_t>(1) << i, x.shape());
      auto r_bit_square = gfmp_mul_mod(r_bits[i], r_bits[i]);
      r_2t = wrap_add(sctx, r_2t, wrap_mul(sctx, k, r_bit_square));
    }
    r_2t = wrap_add(sctx, r_2t, zero_shares);

    // k2 = 2^(l-2)
    auto k2 =
        hack_make_p(sctx, static_cast<uint128_t>(1) << (l - 2), x.shape());
    auto b = wrap_add(sctx, tmp_2t, k2);
    auto c = wrap_add(sctx, b, r_2t);
    auto c_p = wrap_a2p(sctx, c);
    auto c_p_trunc = wrap_arshift_p(sctx, c_p, {static_cast<int64_t>(bits)});
    auto c_bits = bit_decompose(sctx, c_p);
    auto c_msb = c_bits.back();

    // k1 = 1
    auto k1 = hack_make_p(sctx, 1, x.shape());
    auto e = wrap_mul(sctx, wrap_sub(sctx, k1, r_msb), c_msb);
    // k3 = 2^(l-d) - 1
    auto k3 = hack_make_p(sctx, (static_cast<uint128_t>(1) << (l - bits)) - 1,
                          x.shape());
    auto f =
        wrap_add(sctx, wrap_sub(sctx, c_p_trunc, r_hat), wrap_mul(sctx, e, k3));

    // k4 = 2^(l-d-2)
    auto k4 = hack_make_p(sctx, static_cast<uint128_t>(1) << (l - bits - 2),
                          x.shape());
    auto ret = wrap_sub(sctx, f, k4);
    return ret;
  });
}

// Ref: 
// https://dl.acm.org/doi/10.5555/3698900.3699009
// Protocol 5.1: DReLU
NdArrayRef MsbA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* sctx = ctx->sctx();
  std::vector<NdArrayRef> r_bits;
  NdArrayRef r;
  std::tie(r_bits, r) = solved_bits(sctx, in.shape());
  const auto k2 = hack_make_p(sctx, 2, in.shape());
  auto y = wrap_add(sctx, wrap_mul(sctx, in, k2), r);
  auto y_p = wrap_a2p(sctx, y);

  auto y_p_bits = bit_decompose(sctx, y_p);
  auto b = wrap_xor(sctx, y_p_bits[0], r_bits[0]);
  auto c = bit_lt_pa(sctx, y_p_bits, r_bits);
  auto msb = wrap_xor(sctx, b, c);
  return msb;
}

// Ref: 
// https://dl.acm.org/doi/10.5555/3698900.3699009
// Protocol 5.3: ReLU
NdArrayRef ReLU::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* sctx = ctx->sctx();
  std::vector<NdArrayRef> r_bits;
  NdArrayRef r;
  std::tie(r_bits, r) = solved_bits(sctx, in.shape());
  const auto k2 = hack_make_p(sctx, 2, in.shape());
  auto y = wrap_add(sctx, wrap_mul(sctx, in, k2), r);
  auto y_p = wrap_a2p(sctx, y);

  auto y_p_bits = bit_decompose(sctx, y_p);
  auto b = wrap_xor(sctx, y_p_bits[0], r_bits[0]);
  auto c = bit_lt_pa(sctx, y_p_bits, r_bits);
  NdArrayRef b_minus_c = wrap_sub(ctx->sctx(), b, c);
  NdArrayRef t = wrap_mul_aaa(ctx->sctx(), b_minus_c, b_minus_c, in);
  return wrap_sub(ctx->sctx(), in, t);
}

void CommonTypeV::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

  const auto* lhs_v = lhs.as<Priv2kTy>();
  const auto* rhs_v = rhs.as<Priv2kTy>();

  ctx->pushOutput(makeType<AShrTy>(std::max(lhs_v->field(), rhs_v->field())));
}

}  // namespace spu::mpc::shamir
