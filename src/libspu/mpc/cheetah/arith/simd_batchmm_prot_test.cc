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

#include "libspu/mpc/cheetah/arith/simd_batchmm_prot.h"

#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "seal/decryptor.h"
#include "seal/keygenerator.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

namespace spu::mpc::cheetah::test {

class SIMDBatchMMTest : public ::testing::TestWithParam<bool> {
 public:
  size_t poly_N = 8192;
  int plain_bits = 60;
  std::shared_ptr<seal::SEALContext> context_;

  std::shared_ptr<RLWESecretKey> rlwe_sk_;
  std::shared_ptr<RLWEPublicKey> rlwe_pk_;
  std::shared_ptr<GaloisKeys> rlwe_gk_;
  std::shared_ptr<SIMDBatchMMProt> simd_batchmm_prot_;

  void SetUp() override {
    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_use_special_prime(true);
    parms.set_poly_modulus_degree(poly_N);
    parms.set_plain_modulus(seal::PlainModulus::Batching(poly_N, plain_bits));
    parms.set_coeff_modulus((seal::CoeffModulus::BFVDefault(poly_N)));

    context_ = std::make_shared<seal::SEALContext>(parms, true,
                                                   seal::sec_level_type::none);

    seal::KeyGenerator keygen(*context_);
    rlwe_sk_ = std::make_shared<RLWESecretKey>(keygen.secret_key());
    rlwe_pk_ = std::make_shared<RLWEPublicKey>();
    keygen.create_public_key(*rlwe_pk_);

    rlwe_gk_ = std::make_shared<GaloisKeys>();
    keygen.create_galois_keys(*rlwe_gk_);

    simd_batchmm_prot_ = std::make_shared<SIMDBatchMMProt>(
        poly_N, parms.plain_modulus().value());
  }

  void RandomPlain(absl::Span<uint64_t> out) const {
    auto cntxt = context_->key_context_data();
    std::uniform_int_distribution<uint64_t> uniform(
        0, cntxt->parms().plain_modulus().value() - 1);

    std::default_random_engine rdv(std::time(0));
    std::generate_n(out.data(), out.size(), [&]() { return uniform(rdv); });
  }
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, SIMDBatchMMTest, testing::Values(true),
    [](const testing::TestParamInfo<SIMDBatchMMTest::ParamType> &p) {
      return fmt::format("{}", p.param ? "NoiseFlood" : "Approx");
    });

TEST_P(SIMDBatchMMTest, Correctness) {
  size_t batch = 64;
  int64_t d0 = 24;
  int64_t d1 = 2048;
  int64_t d2 = 1408;

  SIMDBatchMMProt::Meta meta;
  meta.batch = batch;
  meta.dims = {d0, d1, d2};

  std::vector<uint64_t> input(batch * d0 * d1);
  std::vector<uint64_t> weight(batch * d1 * d2);
  RandomPlain(absl::MakeSpan(input));
  RandomPlain(absl::MakeSpan(weight));

  seal::BatchEncoder batch_encoder(*context_);

  // Prepare basic numbers
  Shape2D in_shape = simd_batchmm_prot_->ComputeInShape(meta);
  size_t simd_lane = simd_batchmm_prot_->SIMDLane();

  size_t input_ct_num = simd_batchmm_prot_->ComputeInputCtNum(meta, in_shape);
  size_t weight_pt_num = simd_batchmm_prot_->ComputeWeightPtNum(meta, in_shape);
  size_t output_ct_num = simd_batchmm_prot_->ComputeOutputCtNum(meta, in_shape);

  // Encode and encrypt input
  std::vector<uint64_t> input_vec(input_ct_num * simd_lane);
  std::vector<RLWEPt> encode_input(input_ct_num);
  std::vector<RLWECt> enc_input(input_ct_num);
  simd_batchmm_prot_->PrepareInputVector(meta, in_shape, absl::MakeSpan(input),
                                         absl::MakeSpan(input_vec));
  simd_batchmm_prot_->EncodeBatch(absl::MakeSpan(input_vec),
                                  absl::MakeSpan(encode_input), batch_encoder);
  simd_batchmm_prot_->SymEncrypt(absl::MakeSpan(encode_input), *rlwe_sk_,
                                 *context_, false, absl::MakeSpan(enc_input));

  // Encode weight
  std::vector<uint64_t> weight_vec(weight_pt_num * simd_lane);
  std::vector<RLWEPt> encode_weight(weight_pt_num);
  simd_batchmm_prot_->PrepareWeightVector(
      meta, in_shape, absl::MakeSpan(weight), absl::MakeSpan(weight_vec));
  simd_batchmm_prot_->EncodeBatch(absl::MakeSpan(weight_vec),
                                  absl::MakeSpan(encode_weight), batch_encoder);

  // Compute MatMatMul
  std::vector<RLWECt> enc_output(output_ct_num);
  simd_batchmm_prot_->BatchMatMatMul(
      meta, in_shape, absl::MakeSpan(enc_input), absl::MakeSpan(encode_weight),
      *rlwe_pk_, *rlwe_gk_, *context_, absl::MakeSpan(enc_output));

  // Decrypt and decode output
  seal::Decryptor decryptor(*context_, *rlwe_sk_);
  std::vector<RLWEPt> dec_output(output_ct_num);
  std::vector<uint64_t> output_vec(output_ct_num * simd_lane);
  std::vector<uint64_t> res(batch * d0 * d2);

  for (size_t i = 0; i < enc_output.size(); ++i) {
    seal::Plaintext pt;
    decryptor.decrypt(enc_output[i], pt);
    dec_output[i] = pt;
  }
  simd_batchmm_prot_->DecodeBatch(absl::MakeSpan(dec_output),
                                  absl::MakeSpan(output_vec), batch_encoder);
  simd_batchmm_prot_->ParseResult(meta, in_shape, absl::MakeSpan(output_vec),
                                  absl::MakeSpan(res));

  // Compute expected and verify
  auto plain = context_->key_context_data()->parms().plain_modulus();
  for (size_t b = 0; b < batch; ++b) {
    for (size_t i = 0; i < static_cast<size_t>(d0); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(d2); ++j) {
        uint64_t acc = 0;
        for (size_t k = 0; k < static_cast<size_t>(d1); ++k) {
          uint64_t lhs = input[b * d0 * d1 + i * d1 + k];
          uint64_t rhs = weight[b * d1 * d2 + k * d2 + j];
          acc = seal::util::add_uint_mod(
              acc, seal::util::multiply_uint_mod(lhs, rhs, plain), plain);
        }
        uint64_t got = res[b * d0 * d2 + i * d2 + j];
        ASSERT_EQ(acc, got);
      }
    }
  }
}

}  // namespace spu::mpc::cheetah::test
