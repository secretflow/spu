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

#include "spu/mpc/util/seal_help.h"

#include "seal/ciphertext.h"
#include "yasl/base/exception.h"

namespace spu::mpc {

// Truncate ciphertexts for a smaller communication.
// NOTE: after truncation, further homomorphic operation is meaningless.
void TruncateBFVForDecryption(seal::Ciphertext &ct,
                              const seal::SEALContext &context) {
  auto make_bits_mask = [](int n_low_zeros) {
    n_low_zeros = std::max(0, n_low_zeros);
    n_low_zeros = std::min(63, n_low_zeros);
    return (static_cast<uint64_t>(-1) >> n_low_zeros) << n_low_zeros;
  };

  auto context_data = context.last_context_data();

  const auto &parms = context_data->parms();
  YASL_ENFORCE(parms.scheme() == seal::scheme_type::bfv,
               "TruncateSEALCtInplace: scheme_type not supported");
  YASL_ENFORCE(ct.size() == 2, "TruncateSEALCtInplace: invalid ct.size");
  YASL_ENFORCE(ct.coeff_modulus_size() == 1,
               "TruncateSEALCtInplace: invalid ct.coeff_modulus_size");
  YASL_ENFORCE(!ct.is_ntt_form(),
               "TruncateSEALCtInplace: invalid ct.is_ntt_form");
  // Hack on BFV decryption formula: c0 + c1*s mod p0 = m' = Delta*m + e ->
  // round(m'/Delta) = m The low-end bits of c0, c1 are useless for decryption,
  // and thus we can trucate those bits
  const size_t poly_n = ct.poly_modulus_degree();
  // Delta := round(p0/t), Then floor(log2(Delta)) = ceil(log2(p0)) -
  // ceil(log2(t)) - 1
  const int n_delta_bits =
      parms.coeff_modulus()[0].bit_count() - parms.plain_modulus().bit_count();
  const uint64_t mask0 = make_bits_mask(n_delta_bits - 2);
  std::transform(ct.data(0), ct.data(0) + poly_n, ct.data(0),
                 [mask0](uint64_t u) { return u & mask0; });

  // Norm |c1 * s|_infty < |c1|_infty * |s|_infty.
  // The value of |c1|_infty * |s|_infty is heuristically bounded by 12. *
  // Std(|c1|_infty) * Std(|s|_infty) Assume |c1| < B is B-bounded uniform. Then
  // the variance Var(|c1|_infty) = B^2*N/12. We need to make sure the value |c1
  // * s|_infty is NOT overflow Delta.
  constexpr double heuristic_bound = 12.;  // P(|x| > Delta) < 2^−{40}
  int n_var_bits{0};
  // The variance Var(|s|_infty) = 2/3*N since the secret key s is uniform
  // from {-1, 0, 1}.
  n_var_bits = std::log2(heuristic_bound * poly_n * std::sqrt(1 / 18.));
  const uint64_t mask1 = make_bits_mask(n_delta_bits - n_var_bits);
  std::transform(ct.data(1), ct.data(1) + poly_n, ct.data(1),
                 [mask1](uint64_t u) { return u & mask1; });
}

}  // namespace spu::mpc
