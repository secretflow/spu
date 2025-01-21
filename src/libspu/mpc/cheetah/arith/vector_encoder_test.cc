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

#include "libspu/mpc/cheetah/arith/vector_encoder.h"

#include "gtest/gtest.h"
#include "seal/util/polyarithsmallmod.h"

#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah::test {

class VectorEncoderTest : public ::testing::TestWithParam<FieldType> {
 protected:
  static constexpr size_t poly_deg = 4096;

  FieldType field_;
  std::shared_ptr<ModulusSwitchHelper> ms_helper_;
  std::shared_ptr<seal::SEALContext> context_;

  inline uint32_t FieldBitLen(FieldType f) const { return 8 * SizeOf(f); }

  void SetUp() override {
    field_ = GetParam();
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
        SPU_THROW("Not support field type {}", field_);
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
  }
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, VectorEncoderTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<VectorEncoderTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

bool DyadicProduct(RLWEPt &pt, const RLWEPt &oth,
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

TEST_P(VectorEncoderTest, ForwardBackward) {
  VectorEncoder vencoder(*context_, *ms_helper_);
  for (bool scale_forward : {true, false}) {
    for (int64_t n : {100L, (long)poly_deg}) {
      auto vec0 = ring_rand(field_, {n});
      auto vec1 = ring_rand(field_, {n});

      RLWEPt poly0;
      RLWEPt poly1;
      vencoder.Forward(vec0, &poly0, scale_forward);
      vencoder.Backward(vec1, &poly1, !scale_forward);

      NttInplace(poly0, *context_);
      NttInplace(poly1, *context_);
      DyadicProduct(poly0, poly1, *context_);
      InvNttInplace(poly0, *context_);

      size_t num_modulus = poly0.coeff_count() / poly_deg;
      std::vector<uint64_t> cnst(num_modulus);
      for (size_t l = 0; l < num_modulus; ++l) {
        cnst[l] = poly0.data()[l * poly_deg];
      }

      auto computed =
          ms_helper_->ModulusDownRNS(field_, {1L}, absl::MakeSpan(cnst));

      DISPATCH_ALL_FIELDS(field_, [&]() {
        NdArrayView<ring2k_t> got(computed);
        NdArrayView<ring2k_t> v0(vec0);
        NdArrayView<ring2k_t> v1(vec1);
        ring2k_t expected = 0;
        for (int64_t i = 0; i < v0.numel(); ++i) {
          expected += v0[i] * v1[i];
        }
        ASSERT_EQ(expected, got[0]);
      });
    }
  }
}

}  // namespace spu::mpc::cheetah::test
