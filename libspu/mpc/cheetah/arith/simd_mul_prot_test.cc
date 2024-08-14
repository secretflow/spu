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

#include "libspu/mpc/cheetah/arith/simd_mul_prot.h"

#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/util/rns.h"
#include "seal/util/scalingvariant.h"

#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class SIMDMulTest : public ::testing::Test, public EnableCPRNG {
 public:
  size_t poly_N = 8192;
  int plain_bits = 44;
  std::shared_ptr<seal::SEALContext> context_;

  std::shared_ptr<RLWESecretKey> rlwe_sk_;
  std::shared_ptr<RLWEPublicKey> rlwe_pk_;
  std::shared_ptr<SIMDMulProt> simd_mul_prot_;

  void SetUp() override {
    std::vector<int> modulus_bits;

    modulus_bits = {60, 30, 52, plain_bits};

    auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_use_special_prime(false);
    parms.set_poly_modulus_degree(poly_N);
    parms.set_plain_modulus(modulus.back());
    modulus.pop_back();
    parms.set_coeff_modulus(modulus);

    context_ = std::make_shared<seal::SEALContext>(parms, true,
                                                   seal::sec_level_type::none);

    seal::KeyGenerator keygen(*context_);
    rlwe_sk_ = std::make_shared<RLWESecretKey>(keygen.secret_key());
    rlwe_pk_ = std::make_shared<RLWEPublicKey>();

    keygen.create_public_key(*rlwe_pk_);

    simd_mul_prot_ =
        std::make_shared<SIMDMulProt>(poly_N, parms.plain_modulus().value());
  }

  void RandomPlain(absl::Span<uint64_t> out) const {
    auto cntxt = context_->key_context_data();
    std::uniform_int_distribution<uint64_t> uniform(
        0, cntxt->parms().plain_modulus().value() - 1);

    std::default_random_engine rdv(std::time(0));
    std::generate_n(out.data(), out.size(), [&]() { return uniform(rdv); });
  }
};

TEST_F(SIMDMulTest, NoiseFlooding) {
  int64_t n = 16384;

  size_t num_pt =
      (n + simd_mul_prot_->SIMDLane() - 1) / simd_mul_prot_->SIMDLane();
  std::vector<uint64_t> ole_a(n);
  std::vector<uint64_t> ole_b(n);
  std::vector<uint64_t> out_a(n);
  std::vector<uint64_t> out_b(n);
  std::vector<RLWEPt> encode_a(num_pt);
  std::vector<RLWEPt> encode_b(num_pt);
  std::vector<RLWECt> encrypt_b(num_pt);
  seal::Decryptor decryptor(*context_, *rlwe_sk_);

  // total n = 2^21
  for (size_t rep = 0; rep < 128; ++rep) {
    RandomPlain(absl::MakeSpan(ole_a));
    RandomPlain(absl::MakeSpan(ole_b));

    simd_mul_prot_->EncodeBatch(ole_a, absl::MakeSpan(encode_a));
    simd_mul_prot_->EncodeBatch(ole_b, absl::MakeSpan(encode_b));
    simd_mul_prot_->SymEncrypt(encode_b, *rlwe_sk_, *context_, false,
                               absl::MakeSpan(encrypt_b));

    RandomPlain(absl::MakeSpan(out_a));
    simd_mul_prot_->MulThenReshareInplace(absl::MakeSpan(encrypt_b), encode_a,
                                          absl::MakeConstSpan(out_a), *rlwe_pk_,
                                          *context_);

    auto _out_b = absl::MakeSpan(out_b);
    for (size_t i = 0; i < num_pt; ++i) {
      seal::Plaintext pt;
      decryptor.decrypt(encrypt_b[i], pt);
      simd_mul_prot_->DecodeSingle(pt, _out_b.subspan(i * poly_N, poly_N));
    }

    auto plain = context_->key_context_data()->parms().plain_modulus();
    for (int64_t i = 0; i < n; ++i) {
      uint64_t expected =
          seal::util::multiply_uint_mod(ole_a[i], ole_b[i], plain);
      uint64_t got = seal::util::add_uint_mod(out_a[i], out_b[i], plain);
      ASSERT_EQ(expected, got);
    }
  }
}

}  // namespace spu::mpc::cheetah::test
