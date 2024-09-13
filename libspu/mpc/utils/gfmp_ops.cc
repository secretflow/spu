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

#define PFOR_GRAIN_SIZE 4096

#include "libspu/mpc/utils/gfmp_ops.h"

#include <cstring>

#include "absl/types/span.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/linalg.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {
namespace {

#define SPU_ENFORCE_RING(x)                                           \
  SPU_ENFORCE((x).eltype().isa<Ring2k>(), "expect ring type, got={}", \
              (x).eltype());

#define SPU_ENFORCE_GFMP(x)                                           \
  SPU_ENFORCE((x).eltype().isa<GfmpTy>(), "expect gfmp type, got={}", \
              (x).eltype());

#define ENFORCE_EQ_ELSIZE_AND_SHAPE(lhs, rhs)                      \
  SPU_ENFORCE((lhs).elsize() == (rhs).elsize(),                    \
              "type size mismatch lhs={}, rhs={}", (lhs).eltype(), \
              (rhs).eltype());                                     \
  SPU_ENFORCE((lhs).shape() == (rhs).shape(),                      \
              "numel mismatch, lhs={}, rhs={}", lhs, rhs);

// Fast mod operation for Mersenne prime
void gfmp_mod_impl(NdArrayRef& ret, const NdArrayRef& x) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  const auto* ty = ret.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = mod_p(_x[idx]); });
  });
}

// batch inversion
void gfmp_inverse_impl(NdArrayRef& ret, const NdArrayRef& x) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();

  NdArrayRef prefix_prod(ret.eltype(), ret.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _prefix_prod(prefix_prod);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _ret(ret);
    _prefix_prod[0] = _x[0];
    for (int64_t i = 1; i < numel; ++i) {
      _prefix_prod[i] = mul_mod(_prefix_prod[i - 1], _x[i]);
    }

    ring2k_t pprod_inverse = mul_inv(_prefix_prod[numel - 1]);
    _ret[numel - 1] = mul_inv(_prefix_prod[numel - 1]);
    for(int64_t i = numel - 1; i >= 1; i--) {
      _ret[i-1] = mul_mod(_ret[i], _x[i]);
    }

    for(int64_t i = 1; i < numel; ++i) {
      _ret[i] = mul_mod(_ret[i], _prefix_prod[i - 1]);
    }
  });
  
}

void gfmp_mul_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel,
             [&](int64_t idx) { _ret[idx] = mul_mod(_x[idx], _y[idx]); });
  });
}

void gfmp_add_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel,
             [&](int64_t idx) { _ret[idx] = add_mod(_x[idx], _y[idx]); });
  });
}

void gfmp_sub_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel, [&](int64_t idx) {
      _ret[idx] = add_mod(_x[idx], add_inv(_y[idx]));
    });
  });
}

void gfmp_div_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
                       const NdArrayRef& y) {
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, x);
  ENFORCE_EQ_ELSIZE_AND_SHAPE(ret, y);
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = x.numel();
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _y(y);
    pforeach(0, numel, [&](int64_t idx) {
      _ret[idx] = mul_mod(_x[idx], mul_inv(_y[idx]));
    });
  });
}

}  // namespace

NdArrayRef gfmp_rand(FieldType field, const Shape& shape) {
  uint64_t cnt = 0;
  return gfmp_rand(field, shape, yacl::crypto::SecureRandSeed(), &cnt);
}

// FIXME: this function is not strictly correct as the probability among the
// range [0, p-1] is not uniform.
NdArrayRef gfmp_rand(FieldType field, const Shape& shape, uint128_t prg_seed,
                     uint64_t* prg_counter) {
  constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;
  constexpr uint128_t kAesInitialVector = 0U;

  NdArrayRef res(makeType<GfmpTy>(field), shape);
  *prg_counter = yacl::crypto::FillPRand(
      kCryptoType, prg_seed, kAesInitialVector, *prg_counter,
      absl::MakeSpan(res.data<char>(), res.buf()->size()));
  gfmp_mod_(res);
  return res;
}

NdArrayRef gfmp_mod(const NdArrayRef& x) {
  SPU_ENFORCE_GFMP(x);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_mod_impl(ret, x);
  return ret;
}

void gfmp_mod_(NdArrayRef& x) {
  SPU_ENFORCE_GFMP(x);
  gfmp_mod_impl(x, x);
}

NdArrayRef gfmp_batch_inverse(const NdArrayRef& x) {
  SPU_ENFORCE_GFMP(x);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_inverse_impl(ret, x);
  return ret;
}

void gfmp_batch_inverse(NdArrayRef& x) {
  SPU_ENFORCE_GFMP(x);
  gfmp_inverse_impl(x, x);
}

NdArrayRef gfmp_mul_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_mul_mod_impl(ret, x, y);
  return ret;
}

void gfmp_mul_mod_(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  gfmp_mul_mod_impl(x, x, y);
}

NdArrayRef gfmp_div_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_div_mod_impl(ret, x, y);
  return ret;
}

void gfmp_div_mod_(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  gfmp_div_mod_impl(x, x, y);
}

NdArrayRef gfmp_add_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_add_mod_impl(ret, x, y);
  return ret;
}

void gfmp_add_mod_(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  gfmp_add_mod_impl(x, x, y);
}

NdArrayRef gfmp_sub_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_sub_mod_impl(ret, x, y);
  return ret;
}

void gfmp_sub_mod_(NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  gfmp_sub_mod_impl(x, x, y);
}

NdArrayRef gfmp_mmul_mod(const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE_GFMP(y);
  const auto field = x.eltype().as<GfmpTy>()->field();

  // Optimize me: remove copy
  return DISPATCH_ALL_FIELDS(field, [&]() {
    GfmpMatrix<ring2k_t> x_(x.shape()[0], x.shape()[1]);
    GfmpMatrix<ring2k_t> y_(y.shape()[0], y.shape()[1]);
    for (auto i = 0; i < x_.rows(); ++i) {
      for (auto j = 0; j < x_.cols(); ++j) {
        x_(i, j) = Gfmp(x.at<ring2k_t>({i, j}));
      }
    }
    for (auto i = 0; i < y_.rows(); ++i) {
      for (auto j = 0; j < y_.cols(); ++j) {
        y_(i, j) = Gfmp(y.at<ring2k_t>({i, j}));
      }
    }
    auto z_ = x_ * y_;
    NdArrayRef out(x.eltype(), {x.shape()[0], y.shape()[1]});
    NdArrayView<ring2k_t> _out(out);
    for (auto i = 0; i < z_.rows(); ++i) {
      for (auto j = 0; j < z_.cols(); ++j) {
        _out[i * z_.cols() + j] = z_(i, j).data();
      }
    }
    return out;
  });
}

NdArrayRef gfmp_arshift_mod(const NdArrayRef& in, const Sizes& bits) {
  SPU_ENFORCE_GFMP(in);
  const auto* ty = in.eltype().as<GfmpTy>();
  const auto field = ty->field();
  bool is_splat = bits.size() == 1;
  NdArrayRef out(in.eltype(), in.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    using ring_2k_s = std::make_signed_t<ring2k_t>;
    size_t exp = ScalarTypeToPrime<ring2k_t>::exp;
    ring2k_t prime = ScalarTypeToPrime<ring2k_t>::prime;
    auto mask = static_cast<ring2k_t>(0xFF) << (exp - 1);
    NdArrayView<ring2k_t> _in(in);
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, in.numel(), [&](int64_t idx) {
      auto msb = (_in[idx] >> (exp - 1)) & 1;
      auto mask_ = msb ? mask : 0;
      auto tmp = (_in[idx] & ~mask) | mask_;
      auto shift =
          static_cast<ring_2k_s>(tmp) >> (is_splat ? bits[0] : bits[idx]);
      _out[idx] = shift & prime;
    });
  });
  return out;
}

std::vector<NdArrayRef> gfmp_rand_shamir_shares(const NdArrayRef& x,
                                                size_t world_size,
                                                size_t threshold) {
  auto field = x.eltype().as<GfmpTy>()->field();
  auto coeffs = gfmp_rand(field, {static_cast<int64_t>(threshold) * x.numel()});
  return gfmp_rand_shamir_shares(x, coeffs, world_size, threshold);
}

std::vector<NdArrayRef> gfmp_rand_shamir_shares(const NdArrayRef& x,
                                                const NdArrayRef& coeffs,
                                                size_t world_size,
                                                size_t threshold) {
  SPU_ENFORCE_GFMP(x);
  SPU_ENFORCE(world_size > threshold && threshold >= 1,
              "invalid party numbers {} or threshold {}", world_size,
              threshold);
  SPU_ENFORCE_EQ(coeffs.numel(), static_cast<int64_t>(threshold) * x.numel());
  const auto* ty = x.eltype().as<GfmpTy>();
  const auto field = ty->field();
  // For each element, we need to generate `threshold` random coefficients
  std::vector<NdArrayRef> shares;
  shares.reserve(world_size);
  for (size_t i = 0; i < world_size; ++i) {
    shares.emplace_back(x.eltype(), x.shape());
  }

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _x(x);
    NdArrayView<ring2k_t> _coeffs(coeffs);
    pforeach(0, x.numel(), [&](int64_t idx) {
      for (size_t i = 1; i <= world_size; ++i) {
        ring2k_t share = _x[idx];
        size_t coeff_beg = 0 + idx * threshold;
        for (size_t j = 1; j < threshold + 1; ++j) {
          ring2k_t coeff = _coeffs[coeff_beg + j - 1];
          for (size_t k = 0; k < j; k++) {
            coeff = mul_mod(coeff, static_cast<ring2k_t>(i));
          }
          share = add_mod(share, coeff);
        }
        NdArrayView<ring2k_t> _share(shares[i - 1]);
        _share[idx] = share;
      }
    });
  });
  return shares;
}

NdArrayRef gfmp_reconstruct_shamir_shares(absl::Span<const NdArrayRef> shares,
                                          size_t world_size, size_t threshold) {
  SPU_ENFORCE(std::all_of(shares.begin(), shares.end(),
                          [&](const NdArrayRef& x) {
                            return x.eltype() == shares[0].eltype() &&
                                   x.shape() == shares[0].shape() &&
                                   x.eltype().isa<GfmpTy>();
                          }),
              "Share shape and type should be the same");
  SPU_ENFORCE_GE(shares.size(), threshold,
                 "Shares size and threshold are not matched");
  SPU_ENFORCE(world_size >= threshold * 2 + 1 && threshold >= 1,
              "invalid party numbers {} or threshold {}", world_size,
              threshold);
  const auto* ty = shares[0].eltype().as<GfmpTy>();
  const auto field = ty->field();
  const auto numel = shares[0].numel();
  NdArrayRef out(makeType<GfmpTy>(field), shares[0].shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, numel, [&](int64_t idx) {
      ring2k_t secret = 0;
      // TODO optimize me: the reconstruction vector for a fixed point can be pre-computed
      for (size_t i = 0; i < shares.size(); ++i) {
        NdArrayView<ring2k_t> _share(shares[i]);
        ring2k_t y = _share[idx];
        ring2k_t prod = 1;
        for (size_t j = 0; j < shares.size(); ++j) {
          if (i != j) {
            ring2k_t xi = i + 1;
            ring2k_t xj = j + 1;
            auto tmp = mul_mod(xj, mul_inv(add_mod(xj, add_inv(xi))));
            prod = mul_mod(prod, tmp);
          }
        }
        auto tmp = mul_mod(y, prod);
        secret = add_mod(secret, tmp);
      }
      _out[idx] = secret;
    });
  });
  return out;
}

}  // namespace spu::mpc
