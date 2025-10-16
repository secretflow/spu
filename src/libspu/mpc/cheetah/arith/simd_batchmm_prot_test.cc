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
#include <chrono>

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

class SIMDBatchMMTest : public ::testing::TestWithParam<bool>, public EnableCPRNG {
 public:
  size_t poly_N = 8192;
  int plain_bits = 60;
  std::shared_ptr<seal::SEALContext> context_;

  std::shared_ptr<RLWESecretKey> rlwe_sk_;
  std::shared_ptr<RLWEPublicKey> rlwe_pk_;
  std::shared_ptr<GaloisKeys> rlwe_gk_;
  std::shared_ptr<SIMDBatchMMProt> simd_batchmm_prot_;

  void SetUp() override {
    std::cout << "setup" << std::endl;
    std::vector<int> modulus_bits;
    if (GetParam()) {
      // modulus_bits = {60, 30, 52, plain_bits};
    } else {
      // modulus_bits = {60, 45, 45, 58, plain_bits};
    }

    auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_use_special_prime(true);
    parms.set_poly_modulus_degree(poly_N);
    // parms.set_plain_modulus(modulus.back());
    parms.set_plain_modulus(seal::PlainModulus::Batching(poly_N, plain_bits));
    parms.set_coeff_modulus((seal::CoeffModulus::BFVDefault(poly_N)));
    // modulus.pop_back();
    // parms.set_coeff_modulus(modulus);


    context_ = std::make_shared<seal::SEALContext>(parms, true,
                                                   seal::sec_level_type::none);

    seal::KeyGenerator keygen(*context_);
    rlwe_sk_ = std::make_shared<RLWESecretKey>(keygen.secret_key());
    rlwe_pk_ = std::make_shared<RLWEPublicKey>();
    keygen.create_public_key(*rlwe_pk_);

    rlwe_gk_ = std::make_shared<GaloisKeys>();
    keygen.create_galois_keys(*rlwe_gk_);


    simd_batchmm_prot_ =
        std::make_shared<SIMDBatchMMProt>(poly_N, parms.plain_modulus().value());
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

TEST_P(SIMDBatchMMTest, ) {
  size_t batch = 64;
  int64_t d0 = 24;
  int64_t d1 = 2048;
  int64_t d2 = 1408;
  std::cout << "Testing Batch MatMat " << batch << "x" << d0 << "x" << d1 << " * " << d1 << "x" << d2 << std::endl;

  SIMDBatchMMProt::Meta meta;
  meta.batch = batch;
  meta.dims = {d0, d1, d2};

  std::cout << simd_batchmm_prot_->SIMDLane() << std::endl;
  
  std::vector<uint64_t> input(batch * d0 * d1);
  std::vector<uint64_t> weight(batch * d1 * d2);  
  RandomPlain(absl::MakeSpan(input));
  RandomPlain(absl::MakeSpan(weight));

  seal::BatchEncoder batch_encoder(*context_);

  auto start = std::chrono::high_resolution_clock::now();
  
  // Prepare basic numbers
  
  Shape2D in_shape = simd_batchmm_prot_->ComputeInShape(meta);
  std::cout << "in_shape: " << in_shape[0] << " x " << in_shape[1] << std::endl;
  size_t block_size = in_shape[1];
  size_t num_row_blocks = CeilDiv(static_cast<uint64_t>(d1), block_size);
  size_t num_col_blocks = CeilDiv(static_cast<uint64_t>(d2), block_size);
  size_t simd_lane = simd_batchmm_prot_->SIMDLane();
  // size_t row_size = simd_batchmm_prot_->SIMDLane() / 2;
  std::cout << "num_input_blocks: " << num_row_blocks << std::endl;
  std::cout << "num_row_blocks: " << num_row_blocks << std::endl;
  std::cout << "num_col_blocks: " << num_col_blocks << std::endl;

  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane);
  std::cout << "input_groups: " << input_groups << std::endl;

  size_t input_ct_num = simd_batchmm_prot_->ComputeInputCtNum(meta, in_shape);
  size_t weight_pt_num = simd_batchmm_prot_->ComputeWeightPtNum(meta, in_shape);
  size_t output_ct_num = simd_batchmm_prot_->ComputeOutputCtNum(meta, in_shape);
  std::cout << "input ct num: " << input_ct_num << std::endl;
  std::cout << "weight pt num: " << weight_pt_num << std::endl;
  std::cout << "output ct num: " << output_ct_num << std::endl;

  // Encode and encrypt input
  std::vector<uint64_t> input_vec(input_ct_num * simd_lane);
  std::vector<RLWEPt> encode_input(input_ct_num);
  std::vector<RLWECt> enc_input(input_ct_num);
  simd_batchmm_prot_->PrepareInputVector(meta, in_shape, absl::MakeSpan(input), absl::MakeSpan(input_vec));
  simd_batchmm_prot_->EncodeBatch(absl::MakeSpan(input_vec), absl::MakeSpan(encode_input), batch_encoder);  
  simd_batchmm_prot_->SymEncrypt(absl::MakeSpan(encode_input), *rlwe_sk_, *context_, false, absl::MakeSpan(enc_input));


  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Preprocess input time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  // Encode weight
  std::vector<uint64_t> weight_vec(weight_pt_num * simd_lane);
  std::vector<RLWEPt> encode_weight(weight_pt_num);
  simd_batchmm_prot_->PrepareWeightVector(meta, in_shape, absl::MakeSpan(weight), absl::MakeSpan(weight_vec));

  simd_batchmm_prot_->EncodeBatch(absl::MakeSpan(weight_vec), absl::MakeSpan(encode_weight), batch_encoder);

  auto end2 = std::chrono::high_resolution_clock::now();
  std::cout << "Preprocess weight time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count() << " ms" << std::endl;

  // Compute MatMatMul
  std::vector<RLWECt> enc_output(output_ct_num);
  simd_batchmm_prot_->BatchMatMatMul(meta, in_shape, absl::MakeSpan(enc_input), absl::MakeSpan(encode_weight),
                                  *rlwe_pk_, *rlwe_gk_, *context_, absl::MakeSpan(enc_output));

  auto end3 = std::chrono::high_resolution_clock::now();
  std::cout << "Batch MatMatMul time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - end2).count() << " ms" << std::endl;

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
  simd_batchmm_prot_->DecodeBatch(absl::MakeSpan(dec_output), absl::MakeSpan(output_vec), batch_encoder);  
  simd_batchmm_prot_->ParseResult(meta, in_shape, absl::MakeSpan(output_vec), absl::MakeSpan(res));

  auto end4 = std::chrono::high_resolution_clock::now();
  std::cout << "Postprocess output time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end4 - end3).count() << " ms" << std::endl;

  // compute expected
  auto plain = context_->key_context_data()->parms().plain_modulus();
  for (size_t b = 0; b < batch; ++b) {
    for (size_t i = 0; i < (size_t)d0; ++i) {
      for (size_t j = 0; j < (size_t)d2; ++j) {
        // compute res[i, j]
        uint64_t acc = 0;
        for (size_t k = 0; k < (size_t)d1; ++k) {
          uint64_t lhs = input[b * d0 * d1 + i * d1 + k];
          uint64_t rhs = weight[b * d1 * d2 + k * d2 + j];
          acc = seal::util::add_uint_mod(
              acc, seal::util::multiply_uint_mod(lhs, rhs, plain), plain);
        }
        uint64_t got = res[b * d0 * d2 + i * d2 + j];
        // std::cout << "res[" << i << "," << j << "] = " << acc << " ?= " << got << std::endl;
        ASSERT_EQ(acc, got);
      }
    }
  }
  
}

}  // namespace spu::mpc::cheetah::test
