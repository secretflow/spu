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

#include "libspu/mpc/cheetah/rlwe/lwe_decryptor.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"

namespace spu::mpc::cheetah {

LWEDecryptor::LWEDecryptor(const LWESecretKey &sk,
                           const seal::SEALContext &context,
                           const ModulusSwitchHelper &ms_helper)
    : sk_(sk), context_(context), ms_helper_(ms_helper) {
  SPU_ENFORCE(context.parameters_set());
}

LWEDecryptor::~LWEDecryptor() = default;

template <typename T>
void LWEDecryptor::DoDecrypt(const LWECt &ciphertext, T *out) const {
  SPU_ENFORCE(ciphertext.lazy_counter_ == 0, "call LWECt::Reduce() first");
  SPU_ENFORCE(out != nullptr, "nullptr out");
  SPU_ENFORCE(ciphertext.IsValid(), "invalid LWECt");

  size_t num_coeff = ciphertext.poly_modulus_degree();
  size_t num_modulus = ciphertext.coeff_modulus_size();

  auto cntxt_dat = context_.get_context_data(ciphertext.parms_id());
  SPU_ENFORCE(cntxt_dat != nullptr, "parms_id");
  const auto &modulus = cntxt_dat->parms().coeff_modulus();
  SPU_ENFORCE(modulus.size() >= num_modulus);
  SPU_ENFORCE(sk_.secret_non_ntt_.data().coeff_count() >=
              num_modulus * num_coeff);

  std::vector<uint64_t> dot(num_modulus);
  const auto *op0 = sk_.secret_non_ntt_.data().data();
  const auto *op1 = ciphertext.vec_.data();
  yacl::CheckNotNull(op0);
  yacl::CheckNotNull(op1);
  for (size_t l = 0; l < num_modulus; ++l, op0 += num_coeff, op1 += num_coeff) {
    using namespace seal::util;
    dot[l] = dot_product_mod(op0, op1, num_coeff, modulus[l]);
    dot[l] = add_uint_mod(dot[l], ciphertext.cnst_term_[l], modulus[l]);
  }

  absl::Span<uint64_t> in_span(dot.data(), dot.size());
  absl::Span<T> out_span(out, 1);
  ms_helper_.ModulusDownRNS(in_span, out_span);
}

void LWEDecryptor::Decrypt(const LWECt &ciphertext, uint32_t *out) const {
  // 2, ..., 32 -> uint32_t
  uint32_t bitlen = absl::bit_ceil(ms_helper_.base_mod_bitlen());
  SPU_ENFORCE_EQ(bitlen, 32U);
  return DoDecrypt<uint32_t>(ciphertext, out);
}

void LWEDecryptor::Decrypt(const LWECt &ciphertext, uint64_t *out) const {
  // 33, ..., 64 -> uint64_t
  uint32_t bitlen = absl::bit_ceil(ms_helper_.base_mod_bitlen());
  SPU_ENFORCE_EQ(bitlen, 64U);
  return DoDecrypt<uint64_t>(ciphertext, out);
}

void LWEDecryptor::Decrypt(const LWECt &ciphertext, uint128_t *out) const {
  // 65, ..., 128 -> uint128_t
  uint32_t bitlen = absl::bit_ceil(ms_helper_.base_mod_bitlen());
  SPU_ENFORCE_EQ(bitlen, 128U);
  return DoDecrypt<uint128_t>(ciphertext, out);
}

}  // namespace spu::mpc::cheetah
