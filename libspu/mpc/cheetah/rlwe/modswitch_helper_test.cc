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

#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/util/rns.h"
#include "yacl/crypto/utils/rand.h"  // RandSeed

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah::test {

class RLWE2LWETest : public testing::TestWithParam<FieldType> {
 protected:
  static constexpr size_t poly_deg = 4096;
  FieldType field_;
  uint128_t seed_;
  uint64_t prng_counter_;

  std::shared_ptr<seal::SEALContext> context_;
  std::shared_ptr<seal::SEALContext> ms_context_;
  std::shared_ptr<RLWESecretKey> rlwe_sk_;
  std::shared_ptr<ModulusSwitchHelper> ms_helper_;

  inline uint32_t FieldBitLen(FieldType f) const { return 8 * SizeOf(f); }

  ArrayRef CPRNG(FieldType field, size_t size) {
    return ring_rand(field, size, seed_, &prng_counter_);
  }

  void UniformPoly(const seal::SEALContext &context, RLWEPt *pt) {
    SPU_ENFORCE(pt != nullptr);
    auto &parms = context.first_context_data()->parms();
    size_t N = parms.poly_modulus_degree();
    size_t L = parms.coeff_modulus().size();
    pt->resize(N * L);
    auto prng = parms.random_generator()->create();
    seal::util::sample_poly_uniform(prng, parms, pt->data());
  }

  void SetUp() override {
    field_ = GetParam();
    std::vector<int> modulus_bits;
    switch (field_) {
      case FieldType::FM32:
        modulus_bits = {33};
        break;
      case FieldType::FM64:
        modulus_bits = {33, 33};
        break;
      case FieldType::FM128:
        modulus_bits = {44, 44, 44};
        break;
      default:
        SPU_THROW("Not support field type {}", field_);
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

    seed_ = yacl::crypto::RandSeed();
    prng_counter_ = 0;
  }
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, RLWE2LWETest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<RLWE2LWETest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(RLWE2LWETest, ModulusSwitch_UpDown) {
  // r <- R_t
  for (size_t stride : {1, 2, 3}) {
    auto vec = CPRNG(field_, poly_deg);
    auto _vec = vec.slice(3, vec.numel(), stride);

    // r' = Delta*r \in R_q
    RLWEPt pt;
    pt.resize(poly_deg * ms_helper_->coeff_modulus_size());
    for (size_t j = 0; j < ms_helper_->coeff_modulus_size(); ++j) {
      size_t nn = _vec.numel();
      auto out = absl::MakeSpan(pt.data() + j * poly_deg, nn);
      ms_helper_->ModulusUpAt(_vec, j, out);
      std::fill_n(pt.data() + j * poly_deg + nn, poly_deg - nn, 0);
    }
    // e = round(r'/Delta) mod t \in R_t
    auto src = absl::MakeSpan(pt.data(), pt.coeff_count());
    auto cmp = ring_zeros(field_, poly_deg);
    ms_helper_->ModulusDownRNS(src, cmp);
    // check r =? e
    DISPATCH_ALL_FIELDS(field_, "", [&]() {
      auto expected = xt_adapt<ring2k_t>(_vec);
      auto computed = xt_adapt<ring2k_t>(cmp);
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], computed[i]);
      }
    });
  }
}

TEST_P(RLWE2LWETest, ModulusSwitch_DownUp) {
  // a <- R_t
  auto vec_a = CPRNG(field_, poly_deg);
  RLWEPt poly0;
  poly0.resize(poly_deg * ms_helper_->coeff_modulus_size());
  // a' = round(Delta*a) in R_q
  for (size_t j = 0; j < ms_helper_->coeff_modulus_size(); ++j) {
    auto out = absl::MakeSpan(poly0.data() + j * poly_deg, poly_deg);
    ms_helper_->ModulusUpAt(vec_a, j, out);
  }
  // r <- R_q
  RLWEPt rnd;
  UniformPoly(*context_, &rnd);
  auto &modulus = context_->first_context_data()->parms().coeff_modulus();
  // b = a' - r
  RLWEPt poly1;
  {
    poly1.resize(poly0.coeff_count());
    seal::util::ConstRNSIter op0(poly0.data(), poly_deg);
    seal::util::ConstRNSIter op1(rnd.data(), poly_deg);
    seal::util::RNSIter dest(poly1.data(), poly_deg);
    seal::util::sub_poly_coeffmod(op0, op1, modulus.size(), modulus, dest);
  }

  // r' = round(r/Delta) mod t \in R_t
  // b' = round(b/Delta) mod t \in R_t
  auto shr0 = ring_zeros(field_, poly_deg);
  auto shr1 = ring_zeros(field_, poly_deg);
  {
    auto src = absl::MakeSpan(rnd.data(), rnd.coeff_count());
    ms_helper_->ModulusDownRNS(src, shr0);

    src = absl::MakeSpan(poly1.data(), poly1.coeff_count());
    ms_helper_->ModulusDownRNS(src, shr1);
  }

  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto expected = xt_adapt<ring2k_t>(vec_a);
    auto computed0 = xt_adapt<ring2k_t>(shr0);
    auto computed1 = xt_adapt<ring2k_t>(shr1);

    for (size_t i = 0; i < poly_deg; ++i) {
      auto cmp = computed0[i] + computed1[i];
      ring2k_t diff = expected[i] - cmp;
      if (expected[i] < cmp) {
        diff = -diff;
      }
      EXPECT_LE(diff, static_cast<ring2k_t>(1));
    }
  });
}

}  // namespace spu::mpc::cheetah::test
