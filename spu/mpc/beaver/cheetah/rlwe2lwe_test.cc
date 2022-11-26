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

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"

#include "spu/core/xt_helper.h"
#include "spu/mpc/beaver/cheetah/lwe_decryptor.h"
#include "spu/mpc/beaver/cheetah/modswitch_helper.h"
#include "spu/mpc/beaver/cheetah/poly_encoder.h"
#include "spu/mpc/beaver/cheetah/types.h"
#include "spu/mpc/beaver/cheetah/util.h"
#include "spu/mpc/beaver/prg_tensor.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/seal_help.h"

namespace spu::mpc::test {

static PrgSeed GetHardwareRandom128() {
  // NOTE(juhou) can we use thr rdseed instruction ?
  std::random_device rd;
  // call random_device four times, make sure uint128 is random in 2^128 set.
  uint64_t lhs = static_cast<uint64_t>(rd()) << 32 | rd();
  uint64_t rhs = static_cast<uint64_t>(rd()) << 32 | rd();
  return yacl::MakeUint128(lhs, rhs);
}

class RLWE2LWETest : public testing::TestWithParam<FieldType> {
 protected:
  static constexpr size_t poly_deg = 4096;
  FieldType field_;
  PrgSeed seed_;
  PrgCounter prng_counter_;

  std::shared_ptr<seal::SEALContext> context_;
  std::shared_ptr<seal::SEALContext> ms_context_;
  std::shared_ptr<RLWESecretKey> rlwe_sk_;
  std::shared_ptr<LWESecretKey> lwe_sk_;
  std::shared_ptr<ModulusSwitchHelper> ms_helper_;

  inline uint32_t FieldBitLen(FieldType f) const { return 8 * SizeOf(f); }

  ArrayRef CPRNG(FieldType field, size_t size) {
    PrgArrayDesc prg_desc;
    return prgCreateArray(field, size, seed_, &prng_counter_, &prg_desc);
  }

  void SetUp() override {
    field_ = GetParam();
    std::vector<int> modulus_bits;
    switch (field_) {
      case FieldType::FM32:
        modulus_bits = {40, 24 + 5};
        break;
      case FieldType::FM64:
        modulus_bits = {45, 45, 38 + 5};
        break;
      case FieldType::FM128:
        modulus_bits = {55, 55, 55, 55, 36 + 5};
        break;
      default:
        YACL_THROW("Not support field type {}", field_);
    }

    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_deg);
    auto modulus = seal::CoeffModulus::Create(poly_deg, modulus_bits);
    parms.set_coeff_modulus(modulus);
    parms.set_use_special_prime(false);

    context_ = std::make_shared<seal::SEALContext>(parms, true,
                                                   seal::sec_level_type::none);
    ms_context_ = std::make_shared<seal::SEALContext>(
        parms, false, seal::sec_level_type::none);
    uint32_t bitlen = FieldBitLen(field_);
    ms_helper_ = std::make_shared<ModulusSwitchHelper>(*ms_context_, bitlen);

    seal::KeyGenerator keygen(*context_);
    rlwe_sk_ = std::make_shared<RLWESecretKey>(keygen.secret_key());
    lwe_sk_ = std::make_shared<LWESecretKey>(*rlwe_sk_, *context_);

    seed_ = GetHardwareRandom128();
    prng_counter_ = 0;
  }
};

bool dyadic_product(RLWEPt &pt, const RLWEPt &oth,
                    const seal::SEALContext &context) {
  using namespace seal::util;
  auto cntxt_data = context.get_context_data(pt.parms_id());
  if (!cntxt_data) {
    return false;
  }

  auto L = cntxt_data->parms().coeff_modulus().size();
  if (pt.coeff_count() % L != 0) {
    return false;
  }

  auto ntt_tables = cntxt_data->small_ntt_tables();
  size_t n = pt.coeff_count() / L;
  auto pt_ptr = pt.data();
  auto oth_ptr = oth.data();
  for (size_t l = 0; l < L; ++l) {
    dyadic_product_coeffmod(pt_ptr, oth_ptr, n, ntt_tables[l].modulus(),
                            pt_ptr);
    pt_ptr += n;
    oth_ptr += n;
  }
  return true;
}

INSTANTIATE_TEST_SUITE_P(
    NormalCase, RLWE2LWETest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<RLWE2LWETest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(RLWE2LWETest, Extract) {
  seal::Encryptor encryptor(*context_, *rlwe_sk_);
  seal::Evaluator evaluator(*context_);
  LWEDecryptor decryptor(*lwe_sk_, *context_, *ms_helper_);

  auto vec = CPRNG(field_, poly_deg);
  PolyEncoder encoder(*context_, *ms_helper_);

  RLWEPt pt;
  encoder.Forward(vec, &pt, true);
  NttInplace(pt, *context_);

  RLWECt ct;
  encryptor.encrypt_symmetric(pt, ct);
  evaluator.transform_from_ntt_inplace(ct);

  LWECt accum, sub;
  LWECt lazy_accum, lazy_sub;
  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto xvec = xt_adapt<ring2k_t>(vec);
    for (size_t i = 0; i < xvec.size(); ++i) {
      lazy_accum.AddLazyInplace(ct, i, *context_);
      lazy_sub.SubLazyInplace(ct, i, *context_);

      LWECt lwe(ct, i, *context_);
      accum.AddInplace(lwe, *context_);
      sub.SubInplace(lwe, *context_);

      ring2k_t dec;
      decryptor.Decrypt(lwe, &dec);
      EXPECT_EQ(dec, xvec[i]);

      lwe.NegateInplace(*context_);
      decryptor.Decrypt(lwe, &dec);
      EXPECT_EQ(dec, -xvec[i]);
    }

    ring2k_t ground_sum = xt::sum(xvec)[0];

    ring2k_t sum{0};
    decryptor.Decrypt(accum, &sum);
    EXPECT_EQ(ground_sum, sum);

    lazy_accum.Reduce(*context_);
    decryptor.Decrypt(lazy_accum, &sum);
    EXPECT_EQ(ground_sum, sum);

    lazy_sub.Reduce(*context_);
    sub.AddInplace(lazy_sub, *context_);
    decryptor.Decrypt(sub, &sum);
    EXPECT_EQ(-(ground_sum + ground_sum), sum);
  });
}

TEST_P(RLWE2LWETest, ForwardBackward) {
  auto vec = CPRNG(field_, poly_deg);
  PolyEncoder encoder(*context_, *ms_helper_);

  for (bool fb : {true, false}) {
    RLWEPt pt0, pt1;
    encoder.Forward(vec, &pt0, /*scale*/ fb);
    encoder.Backward(vec, &pt1, /*scale*/ !fb);

    NttInplace(pt0, *context_);
    NttInplace(pt1, *context_);
    dyadic_product(pt0, pt1, *context_);
    InvNttInplace(pt0, *context_);

    size_t num_modulus = pt0.coeff_count() / poly_deg;
    std::vector<uint64_t> cnst(num_modulus);
    for (size_t l = 0; l < num_modulus; ++l) {
      cnst[l] = pt0.data()[l * poly_deg];
    }

    absl::Span<const uint64_t> _wrap(cnst.data(), num_modulus);
    auto computed = ms_helper_->ModulusDownRNS(field_, _wrap);

    DISPATCH_ALL_FIELDS(field_, "", [&]() {
      auto xvec = xt_adapt<ring2k_t>(vec);
      auto xcomputed = xt_adapt<ring2k_t>(computed);
      ring2k_t ground{0};
      for (size_t i = 0; i < xvec.size(); ++i) {
        ground += xvec[i] * xvec[i];
      }
      EXPECT_EQ(ground, xcomputed[0]);
    });
  }
}

TEST_P(RLWE2LWETest, CompressViaExtract) {
  seal::Encryptor encryptor(*context_, *rlwe_sk_);
  seal::Evaluator evaluator(*context_);

  auto vec = CPRNG(field_, poly_deg);
  PolyEncoder encoder(*context_, *ms_helper_);

  RLWEPt pt;
  encoder.Forward(vec, &pt, true);
  NttInplace(pt, *context_);

  RLWECt ct;
  encryptor.encrypt_symmetric(pt, ct);
  evaluator.transform_from_ntt_inplace(ct);

  double kb_before_compress = EncodeSEALObject(ct).size() / 1024.;

  // Keep only 3 coefficients
  std::set<size_t> coeff_to_keep{1, 2, 3};
  auto copy{ct};
  KeepCoefficientsInplace(copy, coeff_to_keep);
  double kb_after_compress = EncodeSEALObject(copy).size() / 1024.;
  EXPECT_NEAR(0.5 * kb_before_compress, kb_after_compress, 1.0);

  // Remove half coefficients
  std::set<size_t> coeff_to_remove;
  for (size_t idx = 0; idx < poly_deg / 2; ++idx) {
    coeff_to_remove.insert(idx);
  }
  RemoveCoefficientsInplace(ct, coeff_to_remove);
  kb_after_compress = EncodeSEALObject(copy).size() / 1024.;
  EXPECT_NEAR(0.5 * kb_before_compress, kb_after_compress, 1.0);
}

TEST_P(RLWE2LWETest, IO) {
  yacl::Buffer lwe_sk_str;
  lwe_sk_str = EncodeSEALObject(*lwe_sk_);

  seal::Encryptor encryptor(*context_, *rlwe_sk_);
  seal::Evaluator evaluator(*context_);
  LWEDecryptor decryptor(*lwe_sk_, *context_, *ms_helper_);
  PolyEncoder encoder(*context_, *ms_helper_);

  RLWEPt pt;
  auto vec = CPRNG(field_, poly_deg);
  encoder.Forward(vec, &pt, true);
  NttInplace(pt, *context_);

  RLWECt ct;
  encryptor.encrypt_symmetric(pt, ct);
  evaluator.transform_from_ntt_inplace(ct);

  LWECt lwe(ct, 1, *context_);

  yacl::Buffer lwe_ct_str;
  lwe_ct_str = EncodeSEALObject(lwe);

  LWECt lwe2;
  LWESecretKey sk;
  EXPECT_THROW(DecodeSEALObject(lwe_sk_str, *context_, &lwe2), std::exception);
  EXPECT_THROW(DecodeSEALObject(lwe_ct_str, *context_, &sk), std::exception);

  DecodeSEALObject(lwe_sk_str, *context_, &sk);
  DecodeSEALObject(lwe_ct_str, *context_, &lwe2);

  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto xvec = xt_adapt<ring2k_t>(vec);
    ring2k_t out{0};
    LWEDecryptor decryptor2(sk, *context_, *ms_helper_);
    decryptor2.Decrypt(lwe2, &out);
    EXPECT_EQ(out, xvec[1]);
  });
}

}  // namespace spu::mpc::test
