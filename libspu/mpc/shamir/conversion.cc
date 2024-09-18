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

NdArrayRef wrap_mul_p(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  return UnwrapValue(mul_aa_p(ctx, WrapValue(x), WrapValue(y)));
}

// void mult_reveal(SPUContext* ctx, const NdArrayRef& a, const NdArrayRef& b, std::string_view name) {
//   NdArrayRef x_open = wrap_mul_p(ctx, a, b);
//   if(ctx->getState<Communicator>()->getRank() == 0) {
//     ring_print(x_open, name);
//   }
// }

// void reveal(SPUContext* ctx, const NdArrayRef & x, std::string_view name) {
//   NdArrayRef x_open = wrap_a2p(ctx, x);
//   if(ctx->getState<Communicator>()->getRank() == 0) {
//     ring_print(x_open, name);
//   }
// }

// TODO: combine mul and mul_p
// NdArrayRef wrap_mul(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y, const NdArrayRef& z) {
//   if (is_public(x) && is_public(y)) {
//     return UnwrapValue(mul_pp(ctx, WrapValue(x), WrapValue(y)));
//   } else if (is_secret(x) && is_public(y)) {
//     return UnwrapValue(mul_ap(ctx, WrapValue(x), WrapValue(y)));
//   } else if (is_secret(y) && is_public(x)) {
//     return UnwrapValue(mul_ap(ctx, WrapValue(y), WrapValue(x)));
//   } else if (is_secret(x) && is_secret(y) && is_secret(z)) {
//     return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
//   } else if (is_secret(x) && is_secret(y) && is_public(z)) {
//     return UnwrapValue(mul_aa_p(ctx, WrapValue(x), WrapValue(y)));
//   }
//   SPU_THROW("should not reach, x={}, y={}", x.eltype(), y.eltype());
// }

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

std::pair<std::vector<NdArrayRef>, std::vector<NdArrayRef>> gen_prefix_mult_share(SPUContext* ctx, const int64_t numel, const int64_t num_prefix) {
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
    rand_r.push_back(rand_raw.slice({i * numel}, {(i + 1) * numel}, {}).reshape({numel}));
    rand_s.push_back(rand_raw.slice({offset + i * numel}, {offset + (i + 1) * numel}, {}).reshape({numel}));
  }

  std::vector<NdArrayRef> rand_prod; 
  std::vector<NdArrayRef> rand_prod_offset; 
  // TODO: The following two multiplications (mul and mul_p) can be run in parallel.
  // rand_prod is of length num_prefix storing B_i = r_i * s_i
  vmap(rand_r.cbegin(), rand_r.cend(), rand_s.cbegin(), rand_s.cend(), std::back_inserter(rand_prod), mul_p_lambda);

  // rand_prod_offset is of length num_prefix - 1 storing C_0 = s_0 and C_i = r_i * s_{i+1} for i > 1
  rand_prod_offset.push_back(rand_s[0]);
  vmap(rand_r.cbegin(), rand_r.cend() - 1, rand_s.cbegin() + 1, rand_s.cend(), std::back_inserter(rand_prod_offset), mul_lambda);


  auto p_rand_prod = rand_prod[0];
  // rand_prod is of length num_prefix storing B_i^-1 = (r_i * s_i)^{-1}
  std::vector<NdArrayRef> rand_prod_inv;
  for(int64_t i = 0; i < num_prefix; ++i) {
    rand_prod_inv.push_back(gfmp_batch_inverse(rand_prod[i]));
  }

  // An unbounded multiplication instance
  // ( [r0]_t , [r0^-1]_t )
  // ( [r1]_t , [r1 * r2^-1]_t,)
  // ( [r2]_t , [r2 * r3^-1]_t)
  // ...
  // ( [ri]_t , [ri-1 * ri^-1]_t) for i = 2, ..., k
  std::vector<NdArrayRef> rand_r_aux;
  vmap(rand_prod_inv.cbegin(), rand_prod_inv.cend(), rand_prod_offset.cbegin(), rand_prod_offset.cend(), std::back_inserter(rand_r_aux), mul_lambda);

  std::pair<std::vector<NdArrayRef>, std::vector<NdArrayRef>> out;
  out.first = std::move(rand_r);
  out.second = std::move(rand_r_aux);
  return out;
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  Page 11: Protocol RAN2
NdArrayRef rand_bits(SPUContext* ctx, int64_t numel) {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  NdArrayRef out(makeType<AShrTy>(field), {numel});
  std::vector<int64_t> cur_failed_indices;
  std::vector<int64_t> pre_failed_indices;
  std::mutex idx_mtx;
  int64_t produced = 0;
  int64_t un_produced = numel;

  while (un_produced > 0) {
    NdArrayRef tmp_out(out.eltype(), out.shape());
    auto rand_a = wrap_rand_a(ctx, {un_produced});
    auto rand_a_square = wrap_mul(ctx, rand_a, rand_a);
    auto rand_a_square_p = wrap_a2p(ctx, rand_a_square);
    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> _rand_a(rand_a);
      NdArrayView<ring2k_t> _rand_a_square_p(rand_a_square_p);
      NdArrayView<ring2k_t> _out(out);
      NdArrayView<ring2k_t> _tmp_out(tmp_out);
      pforeach(0, un_produced, [&](int64_t idx) {
        if (_rand_a_square_p[idx] != 0) {
          auto b = sqrt_mod(_rand_a_square_p[idx]);
          auto c = mul_mod(mul_inv(b), _rand_a[idx]);
          _tmp_out[idx] = mul_mod(mul_inv(static_cast<ring2k_t>(2)),
                                  add_mod(c, static_cast<ring2k_t>(1)));
        } else {  // abort
          std::unique_lock lock(idx_mtx);
          cur_failed_indices.push_back(idx);
        }
      });
      if (pre_failed_indices.empty()) {
        ring_assign(out, tmp_out);
      } else {
        pforeach(0, un_produced, [&](int64_t idx) {
          _out[pre_failed_indices[idx]] = _tmp_out[idx];
        });
      }
      un_produced = cur_failed_indices.size();
      produced = numel - un_produced;
      pre_failed_indices = cur_failed_indices;
      cur_failed_indices.clear();
    });
  }
  return out;
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  Generate random [r] and its multiplicative inverse [r]-1 on the prime field
//  Note: there is a typo in the original paper. The inverse of A equals the
//  inverse of revealed C multiplies B rather than A.
std::pair<NdArrayRef, NdArrayRef> rand_bits_pair(SPUContext* ctx,
                                                 int64_t numel) {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  NdArrayRef out(makeType<AShrTy>(field), {numel});
  NdArrayRef out_inv(makeType<AShrTy>(field), {numel});
  std::vector<int64_t> cur_failed_indices;
  std::vector<int64_t> pre_failed_indices;
  std::mutex idx_mtx;
  int64_t produced = 0;
  int64_t un_produced = numel;

  while (un_produced > 0) {
    NdArrayRef tmp_inv(out.eltype(), out.shape());
    auto rand_ab = wrap_rand_a(ctx, {un_produced * 2});
    auto rand_a = rand_ab.slice({0}, {un_produced}, {});
    auto rand_b = rand_ab.slice({un_produced}, {un_produced * 2}, {});
    auto rand_c = wrap_a2p(ctx, wrap_mul(ctx, rand_a, rand_b));

    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> _rand_a(rand_a);
      NdArrayView<ring2k_t> _rand_b(rand_b);
      NdArrayView<ring2k_t> _rand_c(rand_c);
      NdArrayView<ring2k_t> _out(out);
      NdArrayView<ring2k_t> _out_inv(out_inv);
      NdArrayView<ring2k_t> _tmp_inv(tmp_inv);
      pforeach(0, un_produced, [&](int64_t idx) {
        if (_rand_c[idx] != 0) {
          _tmp_inv[idx] = mul_mod(mul_inv(_rand_c[idx]), _rand_b[idx]);
        } else {  // abort
          std::unique_lock lock(idx_mtx);
          cur_failed_indices.push_back(idx);
        }
      });
      if (pre_failed_indices.empty()) {
        ring_assign(out, rand_a);
        ring_assign(out_inv, tmp_inv);
      } else {
        pforeach(0, un_produced, [&](int64_t idx) {
          _out[pre_failed_indices[idx]] = _rand_a[idx];
          _out_inv[pre_failed_indices[idx]] = _tmp_inv[idx];
        });
      }
      un_produced = cur_failed_indices.size();
      produced = numel - un_produced;
      pre_failed_indices = cur_failed_indices;
      cur_failed_indices.clear();
    });
  }
  return {std::move(out), std::move(out_inv)};
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  Unbounded Fan-In Multiplication
NdArrayRef unbounded_mul(SPUContext* ctx,
                         const std::vector<NdArrayRef>& inputs) {
  SPU_ENFORCE(!inputs.empty());
  int64_t l = inputs.size();
  auto numel = inputs[0].numel();
  NdArrayRef b;
  NdArrayRef b_inv;
  std::tie(b, b_inv) = rand_bits_pair(ctx, (l + 1) * numel);
  auto shape = inputs[0].shape();

  auto mul_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul(ctx, a, b);
  };
  auto open_lambda = [ctx](const NdArrayRef& a) { return wrap_a2p(ctx, a); };

  // D[i] = B[i-1] * A[i] * B_inv[i]
  std::vector<NdArrayRef> d_p;
  {
    std::vector<NdArrayRef> b_pre_vec;
    std::vector<NdArrayRef> b_inv_vec;
    for (int64_t i = 0; i < l; ++i) {
      b_pre_vec.push_back(
          b.slice({i * numel}, {(i + 1) * numel}, {}).reshape(shape));
      b_inv_vec.push_back(
          b_inv.slice({(i + 1) * numel}, {(i + 2) * numel}, {}).reshape(shape));
    }
    std::vector<NdArrayRef> tmp;
    std::vector<NdArrayRef> d;
    vmap(b_pre_vec.cbegin(), b_pre_vec.cend(), inputs.begin(), inputs.end(),
         std::back_inserter(tmp), mul_lambda);
    vmap(tmp.begin(), tmp.end(), b_inv_vec.begin(), b_inv_vec.end(),
         std::back_inserter(d), mul_lambda);
    vmap(d.begin(), d.end(), std::back_inserter(d_p), open_lambda);
  }

  // prefix public mul D_p
  for (int64_t i = 1; i < l; ++i) {
    d_p[i] = wrap_mul(ctx, d_p[i], d_p[i - 1]);
  }
  auto b_pre_inv = b_inv.slice({0}, {numel}, {}).reshape(shape);
  auto b_cur = b.slice({l * numel}, {(l + 1) * numel}, {}).reshape(shape);
  auto out = wrap_mul(ctx, d_p[l - 1], wrap_mul(ctx, b_pre_inv, b_cur));
  return out;
}

// Ref: https://iacr.org/archive/tcc2006/38760286/38760286.pdf
//  Unbounded Fan-In Multiplication
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
  vmap(inputs.cbegin(), inputs.cend(), r_aux.cbegin(), r_aux.cend(), std::back_inserter(out), mul_p_lambda);
  for(int64_t i = 1; i < l; ++i) {
    out[i] = wrap_mul(ctx, out[i], out[i-1]);
  }
  vmap(out.cbegin(), out.cend(), r.cbegin(), r.cend(), out.begin(), mul_lambda);

  return out;
}

// Ref:
// https://www.usenix.org/system/files/sec24summer-prepub-278-liu-fengrun.pdf
//  Protocol 4.1: Prefix Or
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
// https://www.usenix.org/system/files/sec24summer-prepub-278-liu-fengrun.pdf
//  Protocol 4.2: Optimized bitwise less-than for public a and secret b
NdArrayRef bit_lt_ap(SPUContext* ctx, const std::vector<NdArrayRef>& a,
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
  // TODO: remove this because this b is the partial result of q.
  NdArrayRef b = unbounded_mul(ctx, p);
  NdArrayRef c = wrap_make_zeros(ctx, s[0].shape());
  {
    auto p_copy = p;
    std::reverse(p_copy.begin(), p_copy.end());
    // equivalent to prefix_and
    // TODO: 
    auto q = prefix_mul(ctx, p_copy);
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
std::pair<std::vector<NdArrayRef>, NdArrayRef> solved_bits(SPUContext* ctx,
                                                           const Shape& shape) {
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  std::vector<int64_t> cur_failed_indices;
  std::vector<int64_t> pre_failed_indices;
  std::mutex idx_mtx;
  int64_t produced = 0;
  int64_t numel = shape.numel();
  int64_t un_produced = numel;

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
      produced = numel - un_produced;
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
  auto d_bits = bit_add(sctx, c_bits, b_bits);

  // For a Mersenne Prime p = 2^exp - 1. The length of d_bits is exp+1. So the
  // bits of p is all ones except the msb is zero.
  const auto k0 = hack_make_p(sctx, 0, x.shape());
  const auto k1 = hack_make_p(sctx, 1, x.shape());
  auto p_bits = std::vector<NdArrayRef>(d_bits.size(), k1);
  p_bits.back() = k0;
  auto q = bit_lt_ap(sctx, p_bits, d_bits);
  auto f_bits = std::vector<NdArrayRef>(d_bits.size() - 1, k0);
  f_bits[0] = k1;
  std::vector<NdArrayRef> g_bits(d_bits.size() - 1);
  for (size_t i = 0; i < f_bits.size(); ++i) {
    g_bits[i] = wrap_mul(sctx, f_bits[i], q);
  }
  auto h_bits = bit_add(sctx,
                        std::vector<NdArrayRef>(
                            d_bits.begin(), d_bits.begin() + d_bits.size() - 1),
                        g_bits);
  NdArrayRef out(makeType<BShrTy>(field, h_bits.size() - 1), x.shape());
  for (size_t i = 0; i < h_bits.size() - 1; ++i) {
    auto out_i = getBitShare(out, i);
    ring_assign(out_i, h_bits[i]);
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
// Protocol 3.1: Truncation
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
  auto c = bit_lt_ap(sctx, y_p_bits, r_bits);
  auto msb = wrap_xor(sctx, b, c);
  return msb;
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
