// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/psi/core/polynomial/polynomial.h"

#include <algorithm>
#include <memory>

#include "openssl/bn.h"

#include "libspu/core/prelude.h"

namespace spu::psi {

namespace {
class BNDeleter {
 public:
  void operator()(BIGNUM *bn) { BN_free(bn); }
};
using BigNumPtr = std::unique_ptr<BIGNUM, BNDeleter>;

BigNumPtr GetBigNumPtr(int v) {
  BIGNUM *bn = BN_new();
  BN_set_word(bn, v);
  return BigNumPtr(bn);
}

BigNumPtr GetBigNumPtr(std::string_view v) {
  BIGNUM *bn = BN_bin2bn(reinterpret_cast<const uint8_t *>(v.data()),
                         v.length(), nullptr);
  SPU_ENFORCE(bn != nullptr);
  return BigNumPtr(bn);
}

std::vector<BigNumPtr> GetBigNumPtrVector(size_t data_size, int value = 0) {
  std::vector<BigNumPtr> res(data_size);

  for (size_t idx = 0; idx < data_size; idx++) {
    BIGNUM *bn = BN_new();
    BN_set_word(bn, value);
    res[idx] = BigNumPtr(bn);
  }

  return res;
}

std::vector<BigNumPtr> GetBigNumPtrVector(
    const std::vector<absl::string_view> &data_vec) {
  std::vector<BigNumPtr> res(data_vec.size());

  for (size_t idx = 0; idx < data_vec.size(); idx++) {
    BIGNUM *bn =
        BN_bin2bn(reinterpret_cast<const uint8_t *>(data_vec[idx].data()),
                  data_vec[idx].length(), nullptr);
    SPU_ENFORCE(bn != nullptr);
    res[idx] = BigNumPtr(bn);
  }

  return res;
}
}  // namespace

std::string EvalPolynomial(const std::vector<absl::string_view> &coeff,
                           absl::string_view poly_x, std::string_view p_str) {
  BigNumPtr acc = GetBigNumPtr(0);
  BigNumPtr bn_x = GetBigNumPtr(poly_x);
  BigNumPtr bn_p = GetBigNumPtr(p_str);

  BN_CTX *bn_ctx = BN_CTX_new();
  for (int64_t i = coeff.size() - 1; i >= 0; i--) {
    // acc = acc * X;         // mul(acc, acc, a);
    // acc = acc + coeff[i];  // add(acc, acc, f.rep[i]);
    BigNumPtr bn_coeff = GetBigNumPtr(coeff[i]);
    BN_mod_mul(acc.get(), acc.get(), bn_x.get(), bn_p.get(), bn_ctx);
    BN_mod_add(acc.get(), acc.get(), bn_coeff.get(), bn_p.get(), bn_ctx);
  }

  BN_CTX_free(bn_ctx);

  std::string res;
  int len =
      std::max(BN_num_bytes(acc.get()), static_cast<int>(poly_x.length()));
  res.resize(len);

  BN_bn2binpad(acc.get(), reinterpret_cast<unsigned char *>(res.data()), len);

  return res;
}

std::string EvalPolynomial(const std::vector<std::string> &coeff,
                           absl::string_view poly_x, std::string_view p_str) {
  std::vector<absl::string_view> coeff2(coeff.size());

  for (size_t idx = 0; idx < coeff.size(); idx++) {
    coeff2[idx] = absl::string_view(coeff[idx]);
  }

  return EvalPolynomial(coeff2, poly_x, p_str);
}

std::vector<std::string> EvalPolynomial(
    const std::vector<absl::string_view> &coeff,
    const std::vector<absl::string_view> &poly_x, std::string_view p_str) {
  std::vector<std::string> res(poly_x.size());

  for (size_t idx = 0; idx < poly_x.size(); idx++) {
    res[idx] = EvalPolynomial(coeff, poly_x[idx], p_str);
  }
  return res;
}

std::vector<std::string> InterpolatePolynomial(
    const std::vector<absl::string_view> &poly_x,
    const std::vector<absl::string_view> &poly_y, std::string_view p_str) {
  int64_t m = poly_x.size();

  SPU_ENFORCE(poly_y.size() == poly_x.size());

  BigNumPtr bn_p = GetBigNumPtr(p_str);

  // std::vector<MersennePrime> prod(X);
  std::vector<BigNumPtr> prod = GetBigNumPtrVector(poly_x);

  BigNumPtr t1 = GetBigNumPtr(0);
  BigNumPtr t2 = GetBigNumPtr(0);

  int64_t k;
  int64_t i;

  std::vector<BigNumPtr> bn_x = GetBigNumPtrVector(poly_x);
  std::vector<BigNumPtr> bn_y = GetBigNumPtrVector(poly_y);
  std::vector<BigNumPtr> res_bn = GetBigNumPtrVector(m);

  BN_CTX *bn_ctx = BN_CTX_new();

  for (k = 0; k < m; k++) {
    // const MersennePrime &aa = X[k];
    const BigNumPtr aa = GetBigNumPtr(poly_x[k]);

    // t1 = (uint64_t)1;
    BN_one(t1.get());
    for (i = k - 1; i >= 0; i--) {
      // t1 = t1 * aa;       // mul(t1, t1, aa);
      // t1 = t1 + prod[i];  // add(t1, t1, prod[i]);
      BN_mod_mul(t1.get(), t1.get(), aa.get(), bn_p.get(), bn_ctx);

      BN_mod_add(t1.get(), t1.get(), prod[i].get(), bn_p.get(), bn_ctx);
    }

    // t2 = (uint64_t)0;  // clear(t2);
    BN_zero(t2.get());
    for (i = k - 1; i >= 0; i--) {
      // t2 = t2 * aa;         // mul(t2, t2, aa);
      // t2 = t2 + res_bn[i];  // add(t2, t2, res[i]);
      BN_mod_mul(t2.get(), t2.get(), aa.get(), bn_p.get(), bn_ctx);

      BN_mod_add(t2.get(), t2.get(), res_bn[i].get(), bn_p.get(), bn_ctx);
    }

    // t1 = one / t1;   // inv(t1, t1);
    // t2 = Y[k] - t2;  // sub(t2, b[k], t2);
    // t1 = t1 * t2;    // mul(t1, t1, t2);

    BN_mod_inverse(t1.get(), t1.get(), bn_p.get(), bn_ctx);

    BN_mod_sub(t2.get(), bn_y[k].get(), t2.get(), bn_p.get(), bn_ctx);

    BN_mod_mul(t1.get(), t1.get(), t2.get(), bn_p.get(), bn_ctx);

    for (i = 0; i < k; i++) {
      // t2 = prod[i] * t1;           // mul(t2, prod[i], t1);
      // res_bn[i] = res_bn[i] + t2;  // add(res[i], res[i], t2);
      BN_mod_mul(t2.get(), prod[i].get(), t1.get(), bn_p.get(), bn_ctx);

      BN_mod_add(res_bn[i].get(), res_bn[i].get(), t2.get(), bn_p.get(),
                 bn_ctx);
    }

    // res_bn[k] = t1;
    BN_copy(res_bn[k].get(), t1.get());

    if (k < m - 1) {
      if (k == 0) {
        // prod[0] = p - prod[0];
        BN_mod_sub(prod[0].get(), bn_p.get(), prod[0].get(), bn_p.get(),
                   bn_ctx);

      } else {
        // t1 = p - X[k];
        BN_mod_sub(t1.get(), bn_p.get(), bn_x[k].get(), bn_p.get(), bn_ctx);

        // prod[k] = t1 + prod[k - 1];  // add(prod[k], t1, prod[k-1]);
        BN_mod_add(prod[k].get(), t1.get(), prod[k - 1].get(), bn_p.get(),
                   bn_ctx);

        for (i = k - 1; i >= 1; i--) {
          // t2 = prod[i] * t1;           // mul(t2, prod[i], t1);
          // prod[i] = t2 + prod[i - 1];  // add(prod[i], t2, prod[i-1]);
          BN_mod_mul(t2.get(), prod[i].get(), t1.get(), bn_p.get(), bn_ctx);

          BN_mod_add(prod[i].get(), t2.get(), prod[i - 1].get(), bn_p.get(),
                     bn_ctx);
        }
        // prod[0] = prod[0] * t1;  // mul(prod[0], prod[0], t1);
        BN_mod_mul(prod[0].get(), prod[0].get(), t1.get(), bn_p.get(), bn_ctx);
      }
    }
  }

  BN_CTX_free(bn_ctx);
  // while (m > 0 && !(res_bn[m - 1] != zero)) m--;
  while ((m > 0) && (BN_is_zero(res_bn[m - 1].get()) != 0)) {
    m--;
  }

  res_bn.resize(m);

  std::vector<std::string> res(m);
  for (int64_t idx = 0; idx < m; idx++) {
    int len = std::max(BN_num_bytes(res_bn[idx].get()),
                       static_cast<int>(poly_y[0].length()));
    res[idx].resize(len);
    BN_bn2binpad(res_bn[idx].get(),
                 reinterpret_cast<unsigned char *>(res[idx].data()), len);
  }

  return res;
}

}  // namespace spu::psi
