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
NdArrayRef gfmp_zeros(FieldType field, const Shape& shape) {
  NdArrayRef ret(makeType<GfmpTy>(field), shape);
  auto numel = ret.numel();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    pforeach(0, numel, [&](int64_t idx) { _ret[idx] = 0; });
    return ret;
  });
}
NdArrayRef gfmp_rand(FieldType field, const Shape& shape) {
  uint64_t cnt = 0;
  return gfmp_rand(field, shape, yacl::crypto::SecureRandSeed(), &cnt);
}

NdArrayRef gfmp_rand(FieldType field, const Shape& shape, uint128_t prg_seed,
                     uint64_t* prg_counter) {
  constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
      yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;
  constexpr uint128_t kAesInitialVector = 0U;
  NdArrayRef res(makeType<GfmpTy>(field), shape);
  DISPATCH_ALL_FIELDS(field, [&]() {
    *prg_counter = yacl::crypto::FillPRandWithMersennePrime<ring2k_t>(
        kCryptoType, prg_seed, kAesInitialVector, *prg_counter,
        absl::MakeSpan(&res.at<ring2k_t>(0), res.numel()));
  });
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

// not requiring and not casting field.
void gfmp_exp_mod_impl(NdArrayRef& ret, const NdArrayRef& x,
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
             [&](int64_t idx) { _ret[idx] = exp_mod(_x[idx], _y[idx]); });
  });
}

NdArrayRef gfmp_exp_mod(const NdArrayRef& x, const NdArrayRef& y) {
  NdArrayRef ret(x.eltype(), x.shape());
  gfmp_exp_mod_impl(ret, x, y);
  return ret;
}

void gfmp_exp_mod_(NdArrayRef& x, const NdArrayRef& y) {
  gfmp_exp_mod_impl(x, x, y);
}

}  // namespace spu::mpc
