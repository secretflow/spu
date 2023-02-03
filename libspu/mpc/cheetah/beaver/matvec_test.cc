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

#include "libspu/mpc/cheetah/beaver/matvec.h"

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/beaver/lwe_decryptor.h"
#include "libspu/mpc/cheetah/beaver/poly_encoder.h"
#include "libspu/mpc/cheetah/beaver/types.h"
#include "libspu/mpc/cheetah/beaver/util.h"
#include "libspu/mpc/common/prg_tensor.h"
#include "libspu/mpc/util/ring_ops.h"
#include "libspu/mpc/util/seal_help.h"

namespace spu::mpc::test {

template <typename T>
T MakeMask(size_t bw) {
  size_t n = sizeof(T) * 8;
  if (n == bw) {
    return static_cast<T>(-1);
  } else {
    return (static_cast<T>(1) << bw) - 1;
  }
}

class MatVecTest : public ::testing::TestWithParam<
                       std::tuple<FieldType, std::tuple<size_t, size_t>>> {
 protected:
  static constexpr size_t poly_deg = 4096;

  FieldType field_;
  std::shared_ptr<ModulusSwitchHelper> ms_helper_;
  std::shared_ptr<seal::SEALContext> context_;

  std::shared_ptr<RLWESecretKey> rlwe_sk_;
  std::shared_ptr<LWESecretKey> lwe_sk_;

  PrgSeed seed_;
  PrgCounter prng_counter_;

  inline uint32_t FieldBitLen(FieldType f) const { return 8 * SizeOf(f); }

  ArrayRef CPRNG(FieldType field, size_t size) {
    PrgArrayDesc prg_desc;
    return prgCreateArray(field, size, seed_, &prng_counter_, &prg_desc);
  }

  void SetUp() override {
    field_ = std::get<0>(GetParam());

    // NOTE(juhou) same parameters in beaver_cheetah.cc
    std::vector<int> modulus_bits;
    switch (field_) {
      case FieldType::FM32:
        modulus_bits = {55, 39};
        break;
      case FieldType::FM64:
        modulus_bits = {55, 55, 48};
        break;
      case FieldType::FM128:
        modulus_bits = {59, 59, 59, 59, 50};
        break;
      default:
        YACL_THROW("Not support field type {}", field_);
    }

    auto scheme_type = seal::scheme_type::ckks;
    auto parms = seal::EncryptionParameters(scheme_type);
    parms.set_poly_modulus_degree(poly_deg);
    auto modulus = seal::CoeffModulus::Create(poly_deg, modulus_bits);
    parms.set_coeff_modulus(modulus);
    parms.set_use_special_prime(false);

    context_ = std::make_shared<seal::SEALContext>(parms, true,
                                                   seal::sec_level_type::none);
    seal::SEALContext ms_context(parms, false, seal::sec_level_type::none);

    uint32_t bitlen = FieldBitLen(field_);
    ms_helper_ = std::make_shared<ModulusSwitchHelper>(ms_context, bitlen);
    seal::KeyGenerator keygen(*context_);
    rlwe_sk_ = std::make_shared<RLWESecretKey>(keygen.secret_key());
    lwe_sk_ = std::make_shared<LWESecretKey>(*rlwe_sk_, *context_);

    seed_ = yacl::crypto::RandSeed();
    prng_counter_ = 0;
  }
};

INSTANTIATE_TEST_SUITE_P(
    NormalCase, MatVecTest,
    testing::Combine(
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(
            std::make_tuple<size_t>(8, 128),   // fit into one poly
            std::make_tuple<size_t>(8, 1024),  // multi-rows, no margin
            std::make_tuple<size_t>(9, 1024),  // multi-rows, with margin
            std::make_tuple<size_t>(17, 255),  // non-two power rows
            std::make_tuple<size_t>(5, 4096),  // single row
            std::make_tuple<size_t>(5, 8000)   // split columns
            )),
    [](const testing::TestParamInfo<MatVecTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         std::get<0>(std::get<1>(p.param)),
                         std::get<1>(std::get<1>(p.param)));
    });

TEST_P(MatVecTest, RLWE) {
  MatVecProtocol::Meta meta;
  meta.nrows = std::get<0>(std::get<1>(GetParam()));
  meta.ncols = std::get<1>(std::get<1>(GetParam()));
  {
    auto mat = CPRNG(field_, meta.nrows * meta.ncols);
    auto vec = CPRNG(field_, meta.ncols);

    MatVecProtocol matvec_prot(*context_, *ms_helper_);
    seal::Encryptor encryptor(*context_, *rlwe_sk_);
    seal::Evaluator evaluator(*context_);

    std::vector<RLWEPt> ecd_mat;
    matvec_prot.EncodeMatrix(meta, mat, &ecd_mat);

    std::vector<RLWEPt> ecd_vec;
    matvec_prot.EncodeVector(meta, vec, &ecd_vec);
    std::vector<RLWECt> vec_cipher(ecd_vec.size());
    for (size_t i = 0; i < ecd_vec.size(); ++i) {
      NttInplace(ecd_vec[i], *context_);
      encryptor.encrypt_symmetric(ecd_vec[i], vec_cipher[i]);
    }

    std::vector<RLWECt> matvec_prod;
    matvec_prot.MatVecNoExtract(meta, ecd_mat, vec_cipher, &matvec_prod);

    std::vector<RLWEPt> matvec_pt(matvec_prod.size());
    seal::Decryptor decryptor(*context_, *rlwe_sk_);
    for (size_t i = 0; i < matvec_prod.size(); ++i) {
      evaluator.transform_to_ntt_inplace(matvec_prod[i]);
      decryptor.decrypt(matvec_prod[i], matvec_pt[i]);
      InvNttInplace(matvec_pt[i], *context_);
    }

    auto computed = matvec_prot.ParseMatVecResult(field_, meta, matvec_pt);
    auto ground = ring_mmul(mat, vec, meta.nrows, 1, meta.ncols);
    EXPECT_EQ(computed, ground);
  }

  {
    auto mat = CPRNG(field_, meta.nrows * meta.ncols);
    auto vec = CPRNG(field_, meta.ncols);

    MatVecProtocol matvec_prot(*context_, *ms_helper_);
    seal::Encryptor encryptor(*context_, *rlwe_sk_);
    seal::Evaluator evaluator(*context_);

    std::vector<RLWEPt> ecd_mat;
    matvec_prot.EncodeMatrix(meta, mat, &ecd_mat);

    std::vector<RLWEPt> ecd_vec;
    matvec_prot.EncodeVector(meta, vec, &ecd_vec);
    std::vector<RLWECt> vec_cipher(ecd_vec.size());
    for (size_t i = 0; i < ecd_vec.size(); ++i) {
      NttInplace(ecd_vec[i], *context_);
      encryptor.encrypt_symmetric(ecd_vec[i], vec_cipher[i]);
    }

    std::vector<RLWECt> matvec_prod;
    matvec_prot.MatVecNoExtract(meta, ecd_mat, vec_cipher, &matvec_prod);
    matvec_prot.ExtractLWEsInplace(meta, matvec_prod);

    std::vector<RLWEPt> matvec_pt(matvec_prod.size());
    seal::Decryptor decryptor(*context_, *rlwe_sk_);
    for (size_t i = 0; i < matvec_prod.size(); ++i) {
      evaluator.transform_to_ntt_inplace(matvec_prod[i]);
      decryptor.decrypt(matvec_prod[i], matvec_pt[i]);
      InvNttInplace(matvec_pt[i], *context_);
    }

    auto computed = matvec_prot.ParseMatVecResult(field_, meta, matvec_pt);
    auto ground = ring_mmul(mat, vec, meta.nrows, 1, meta.ncols);
    EXPECT_EQ(computed, ground);
  }
}

TEST_P(MatVecTest, LWE) {
  MatVecProtocol::Meta meta;
  meta.nrows = std::get<0>(std::get<1>(GetParam()));
  meta.ncols = std::get<1>(std::get<1>(GetParam()));

  auto mat = CPRNG(field_, meta.nrows * meta.ncols);
  auto vec = CPRNG(field_, meta.ncols);
  auto ground = ring_mmul(mat, vec, meta.nrows, 1, meta.ncols);

  MatVecProtocol matvec_prot(*context_, *ms_helper_);
  seal::Encryptor encryptor(*context_, *rlwe_sk_);
  seal::Evaluator evaluator(*context_);

  std::vector<RLWEPt> ecd_mat;
  matvec_prot.EncodeMatrix(meta, mat, &ecd_mat);

  std::vector<RLWEPt> ecd_vec;
  matvec_prot.EncodeVector(meta, vec, &ecd_vec);
  std::vector<RLWECt> vec_cipher(ecd_vec.size());
  for (size_t i = 0; i < ecd_vec.size(); ++i) {
    NttInplace(ecd_vec[i], *context_);
    encryptor.encrypt_symmetric(ecd_vec[i], vec_cipher[i]);
  }

  std::vector<LWECt> matvec_prod;
  matvec_prot.MatVec(meta, ecd_mat, vec_cipher, &matvec_prod);
  EXPECT_EQ(matvec_prod.size(), meta.nrows);

  LWEDecryptor decryptor(*lwe_sk_, *context_, *ms_helper_);

  ArrayRef computed = ring_zeros(field_, meta.nrows);
  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto xc = xt_mutable_adapt<ring2k_t>(computed);
    for (size_t i = 0; i < meta.nrows; ++i) {
      decryptor.Decrypt(matvec_prod[i], xc.data() + i);
    }
  });
  EXPECT_EQ(computed, ground);
}

}  // namespace spu::mpc::test
