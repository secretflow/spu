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
#include "libspu/mpc/cheetah/rlwe/packlwes.h"

#include <random>

#include "gtest/gtest.h"
#include "seal/seal.h"

#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/rlwe/lwe_ct.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah::test {
class VectorEncoder {
 public:
  explicit VectorEncoder(const seal::SEALContext &context,
                         const ModulusSwitchHelper &msh);
  void Forward(const NdArrayRef &vec, RLWEPt *out,
               bool scale_delta = true) const;

  void Backward(const NdArrayRef &vec, RLWEPt *out,
                bool scale_delta = false) const;

  const ModulusSwitchHelper &ms_helper() const { return *msh_; }

  size_t poly_degree() const { return poly_deg_; }

 private:
  size_t poly_deg_{0};
  std::shared_ptr<ModulusSwitchHelper> msh_;
};

class PackLWEsTest
    : public testing::TestWithParam<std::tuple<FieldType, size_t>> {
 protected:
  static constexpr size_t poly_N = 8192;
  static constexpr size_t ring_len = 64;

  std::shared_ptr<seal::SEALContext> N_context_;

  std::shared_ptr<RLWESecretKey> N_rlwe_sk_;

  std::shared_ptr<GaloisKeys> galois_;
  std::shared_ptr<VectorEncoder> N_encoder_;
  std::shared_ptr<ModulusSwitchHelper> N_ms_helper_;

  void SetUp() override {
    std::vector<int> modulus_bits;
    modulus_bits = {55, 54, 50};

    auto modulus = seal::CoeffModulus::Create(poly_N, modulus_bits);

    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_use_special_prime(true);
    parms.set_poly_modulus_degree(poly_N);
    parms.set_coeff_modulus(modulus);
    N_context_ = std::make_shared<seal::SEALContext>(
        parms, true, seal::sec_level_type::none);

    auto m = modulus;
    auto p = parms;
    m.pop_back();
    p.set_coeff_modulus(m);
    p.set_use_special_prime(false);
    seal::SEALContext N_ms_cntxt(p, false, seal::sec_level_type::none);
    N_ms_helper_ = std::make_shared<ModulusSwitchHelper>(N_ms_cntxt, ring_len);

    N_encoder_ = std::make_shared<VectorEncoder>(*N_context_, *N_ms_helper_);

    seal::KeyGenerator keygen(*N_context_);
    N_rlwe_sk_ = std::make_shared<RLWESecretKey>(keygen.secret_key());
    galois_ = std::make_shared<seal::GaloisKeys>();
    GenerateGaloisKeyForPacking(*N_context_, *N_rlwe_sk_, /*seed*/ false,
                                galois_.get());
  }
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, PackLWEsTest,
    testing::Combine(testing::Values(FieldType::FM64),
                     testing::Values(1024UL, 2048UL, 4096UL, 8192UL)),
    [](const testing::TestParamInfo<PackLWEsTest::ParamType> &p) {
      return fmt::format("{}n{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(PackLWEsTest, PackRLWEs) {
  auto field = std::get<0>(GetParam());  // FM64
  size_t num_rlwes = std::get<1>(GetParam()) / 128;
  SPU_ENFORCE(num_rlwes <= poly_N);

  seal::Encryptor encryptor(*N_context_, *N_rlwe_sk_);
  seal::Decryptor decryptor(*N_context_, *N_rlwe_sk_);

  using scalar_t = uint64_t;
  std::vector<NdArrayRef> arrays(num_rlwes);
  std::vector<RLWECt> rlwes(num_rlwes);
  for (size_t i = 0; i < num_rlwes; ++i) {
    arrays[i] = ring_rand(field, {poly_N});
    RLWEPt pt;
    N_encoder_->Forward(arrays[i], &pt, true);
    NttInplace(pt, *N_context_);
    CATCH_SEAL_ERROR(encryptor.encrypt_symmetric(pt, rlwes[i]));
    InvNttInplace(rlwes[i], *N_context_);
  }

  PackingHelper ph(num_rlwes, N_ms_helper_->coeff_modulus_size(), *galois_,
                   *N_context_);

  RLWECt packed;
  ph.PackingWithModulusDrop(absl::MakeSpan(rlwes), packed);

  EXPECT_EQ(packed.size(), 2);
  EXPECT_FALSE(packed.is_ntt_form());
  EXPECT_EQ(packed.poly_modulus_degree(), poly_N);
  EXPECT_EQ(packed.coeff_modulus_size(), N_ms_helper_->coeff_modulus_size());

  NttInplace(packed, *N_context_);
  RLWEPt dec;
  decryptor.decrypt(packed, dec);
  InvNttInplace(dec, *N_context_);

  std::vector<scalar_t> coefficients(poly_N);
  N_ms_helper_->ModulusDownRNS(absl::MakeSpan(dec.data(), dec.coeff_count()),
                               absl::MakeSpan(coefficients));

  for (size_t i = 0; i < poly_N; i += num_rlwes) {
    size_t offset = i / num_rlwes;
    for (size_t j = 0; j < num_rlwes; ++j) {
      NdArrayView<scalar_t> expected(arrays[j]);
      EXPECT_EQ(expected[offset * num_rlwes], coefficients[i + j]);
    }
  }
}

TEST_P(PackLWEsTest, Basic) {
  auto field = std::get<0>(GetParam());  // FM64
  size_t num_lwes = std::get<1>(GetParam()) / 128;
  SPU_ENFORCE(num_lwes <= poly_N);

  seal::Encryptor encryptor(*N_context_, *N_rlwe_sk_);
  seal::Evaluator N_evaluator(*N_context_);
  seal::Decryptor decryptor(*N_context_, *N_rlwe_sk_);

  using scalar_t = uint64_t;
  NdArrayRef array = ring_rand(field, {poly_N});
  NdArrayView<scalar_t> _array(array);

  RLWEPt pt;
  N_encoder_->Forward(array, &pt, true);
  NttInplace(pt, *N_context_);

  RLWECt ct;
  CATCH_SEAL_ERROR(encryptor.encrypt_symmetric(pt, ct));
  if (ct.is_ntt_form()) {
    CATCH_SEAL_ERROR(N_evaluator.transform_from_ntt_inplace(ct));
  }

  // Perform some computation on LWEs and results at `num_lwes' LWEs.
  size_t n_stride = poly_N / num_lwes;
  std::vector<LWECt> n_lwes(num_lwes);
  std::vector<scalar_t> expects(num_lwes, 0);

  for (size_t i = 0, j = 0; i < poly_N; i += n_stride, ++j) {
    for (size_t k = 0; k < n_stride; ++k) {
      n_lwes[j].AddLazyInplace(ct, i + k, *N_context_);
      expects[j] += _array[i + k];
    }
    n_lwes[j].Reduce(*N_context_);
  }

  // re-randomize to avoid transparent error
  RLWECt zero;
  encryptor.encrypt_zero_symmetric(zero);
  N_evaluator.transform_from_ntt_inplace(zero);
  for (size_t j = 0; j < num_lwes; ++j) {
    n_lwes[j].AddLazyInplace(zero, j, *N_context_);
    n_lwes[j].Reduce(*N_context_);
  }

  RLWECt packed;
  PackLWEs(absl::MakeSpan(n_lwes), *galois_, *N_context_, {&packed, 1});

  EXPECT_EQ(packed.size(), 2);
  EXPECT_FALSE(packed.is_ntt_form());
  EXPECT_EQ(packed.poly_modulus_degree(), poly_N);
  EXPECT_EQ(packed.coeff_modulus_size(), N_ms_helper_->coeff_modulus_size());

  N_evaluator.transform_to_ntt_inplace(packed);
  RLWEPt dec;
  decryptor.decrypt(packed, dec);
  InvNttInplace(dec, *N_context_);

  std::vector<scalar_t> coefficients(poly_N);
  N_ms_helper_->ModulusDownRNS(absl::MakeSpan(dec.data(), dec.coeff_count()),
                               absl::MakeSpan(coefficients));
  size_t N_stride = poly_N / num_lwes;
  for (size_t i = 0, j = 0; i < poly_N; i += N_stride, ++j) {
    EXPECT_EQ(expects[j], coefficients[i]);
    for (size_t k = 1; k < N_stride; ++k) {
      ASSERT_EQ(coefficients[i + k], 0UL);
    }
  }
}

TEST_P(PackLWEsTest, Phantom) {
  seal::Encryptor encryptor(*N_context_, *N_rlwe_sk_);
  seal::Evaluator N_evaluator(*N_context_);
  seal::Decryptor decryptor(*N_context_, *N_rlwe_sk_);

  auto field = std::get<0>(GetParam());       // FM64
  size_t num_lwes = std::get<1>(GetParam());  // n
  SPU_ENFORCE(num_lwes <= poly_N);

  using scalar_t = uint64_t;
  NdArrayRef array = ring_rand(field, {poly_N});
  NdArrayView<scalar_t> _array(array);

  RLWEPt pt;
  N_encoder_->Forward(array, &pt, true);
  NttInplace(pt, *N_context_);

  RLWECt rlwe0, rlwe1;
  CATCH_SEAL_ERROR(encryptor.encrypt_symmetric(pt, rlwe0));
  CATCH_SEAL_ERROR(encryptor.encrypt_symmetric(pt, rlwe1));
  if (rlwe0.is_ntt_form()) {
    CATCH_SEAL_ERROR(N_evaluator.transform_from_ntt_inplace(rlwe0));
    CATCH_SEAL_ERROR(N_evaluator.transform_from_ntt_inplace(rlwe1));
  }

  // Perform some computation on LWEs and results at `num_lwes' LWEs.
  size_t n_stride = poly_N / num_lwes;
  std::vector<PhantomLWECt> n_lwes(num_lwes);
  std::vector<scalar_t> expects(num_lwes, 0);

  for (size_t i = 0, j = 0; i < poly_N; i += n_stride, ++j) {
    expects[j] = _array[i];
    if (1 == (j & 1)) {
      n_lwes[j].WrapIt(rlwe0, i);
    } else {
      n_lwes[j].WrapIt(rlwe1, i);
    }
  }

  RLWECt packed;
  PackLWEs(absl::MakeSpan(n_lwes), *galois_, *N_context_, {&packed, 1});

  EXPECT_EQ(packed.size(), 2);
  EXPECT_FALSE(packed.is_ntt_form());
  EXPECT_EQ(packed.poly_modulus_degree(), poly_N);
  EXPECT_EQ(packed.coeff_modulus_size(), N_ms_helper_->coeff_modulus_size());

  N_evaluator.transform_to_ntt_inplace(packed);
  RLWEPt dec;
  decryptor.decrypt(packed, dec);
  InvNttInplace(dec, *N_context_);

  std::vector<scalar_t> coefficients(poly_N);
  N_ms_helper_->ModulusDownRNS(absl::MakeSpan(dec.data(), dec.coeff_count()),
                               absl::MakeSpan(coefficients));
  size_t N_stride = poly_N / num_lwes;
  for (size_t i = 0, j = 0; i < poly_N; i += N_stride, ++j) {
    ASSERT_EQ(expects[j], coefficients[i]);
    for (size_t k = 1; k < N_stride; ++k) {
      ASSERT_EQ(coefficients[i + k], 0UL);
    }
  }
}

VectorEncoder::VectorEncoder(const seal::SEALContext &context,
                             const ModulusSwitchHelper &msh) {
  SPU_ENFORCE(context.parameters_set());
  auto pid0 = context.first_parms_id();
  auto pid1 = msh.parms_id();
  SPU_ENFORCE_EQ(0, std::memcmp(&pid0, &pid1, sizeof(seal::parms_id_type)),
                 fmt::format("parameter set mismatch"));
  msh_ = std::make_shared<ModulusSwitchHelper>(msh);
  poly_deg_ = context.first_context_data()->parms().poly_modulus_degree();
}

void VectorEncoder::Forward(const NdArrayRef &vec, RLWEPt *out,
                            bool scale_delta) const {
  // Place the vector elements as polynomial coefficients forwardly.
  // a0, a1, ..., an -> \sum_i ai*X^i
  yacl::CheckNotNull(out);

  size_t num_coeffs = vec.numel();
  size_t num_modulus = msh_->coeff_modulus_size();
  SPU_ENFORCE(vec.shape().size() == 1, "need 1D array");
  SPU_ENFORCE_GT(num_coeffs, 0UL);
  SPU_ENFORCE(num_coeffs <= poly_deg_);

  out->parms_id() = seal::parms_id_zero;
  out->resize(seal::util::mul_safe(poly_deg_, num_modulus));

  uint64_t *dst = out->data();
  for (size_t mod_idx = 0; mod_idx < num_modulus; ++mod_idx) {
    std::fill_n(dst, poly_deg_, 0);
    absl::Span<uint64_t> dst_wrap(dst, num_coeffs);

    if (scale_delta) {
      msh_->ModulusUpAt(vec, mod_idx, dst_wrap);
    } else {
      msh_->CenteralizeAt(vec, mod_idx, dst_wrap);
    }
    dst += poly_deg_;
  }

  out->parms_id() = msh_->parms_id();
  out->scale() = 1.;
}

void VectorEncoder::Backward(const NdArrayRef &vec, RLWEPt *out,
                             bool scale_delta) const {
  // Place the vector elements as polynomial coefficients in backward.
  // a0, a1, ..., an -> a0 - \sum_{i>0} ai*X^{N-i}
  // where N defines the base ring X^N + 1.
  yacl::CheckNotNull(out);
  SPU_ENFORCE(vec.shape().size() == 1, "need 1D array");

  size_t num_coeffs = vec.numel();
  size_t num_modulus = msh_->coeff_modulus_size();
  SPU_ENFORCE_GT(num_coeffs, 0UL);
  SPU_ENFORCE(num_coeffs <= poly_deg_);

  const Type &eltype = vec.eltype();
  SPU_ENFORCE(eltype.isa<RingTy>(), "must be ring_type, got={}", eltype);
  out->parms_id() = seal::parms_id_zero;
  out->resize(seal::util::mul_safe(poly_deg_, num_modulus));

  const auto field = eltype.as<Ring2k>()->field();

  DISPATCH_ALL_FIELDS(field, "Backward", [&]() {
    auto tmp_buff = ring_zeros(field, {(int64_t)poly_deg_});
    auto xvec = NdArrayView<const ring2k_t>(vec);
    auto xtmp = NdArrayView<ring2k_t>(tmp_buff);

    xtmp[0] = xvec[0];
    // reverse and sign flip
    for (size_t i = 1; i < num_coeffs; ++i) {
      xtmp[num_coeffs - 1 - i] = -xvec[i];
    }

    uint64_t *dst = out->data();
    for (size_t mod_idx = 0; mod_idx < num_modulus; ++mod_idx) {
      std::fill_n(dst, poly_deg_, 0);
      absl::Span<uint64_t> dst_wrap(dst, poly_deg_);

      if (scale_delta) {
        msh_->ModulusUpAt(tmp_buff, mod_idx, dst_wrap);
      } else {
        msh_->CenteralizeAt(tmp_buff, mod_idx, dst_wrap);
      }
      dst += poly_deg_;
    }

    // clean up sensitive data
    seal::util::seal_memzero(&tmp_buff.at<ring2k_t>(0),
                             sizeof(ring2k_t) * poly_deg_);
  });

  out->parms_id() = msh_->parms_id();
  out->scale() = 1.;
}
}  // namespace spu::mpc::cheetah::test
