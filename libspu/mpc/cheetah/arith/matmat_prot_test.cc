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

#include "libspu/mpc/cheetah/arith/matmat_prot.h"

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah::test {

class MatMatProtTest
    : public ::testing::TestWithParam<std::tuple<FieldType, Shape3D>> {
 protected:
  static constexpr size_t poly_deg = 4096;

  FieldType field_;
  std::shared_ptr<ModulusSwitchHelper> ms_helper_;
  std::shared_ptr<seal::SEALContext> context_;

  std::shared_ptr<RLWESecretKey> rlwe_sk_;

  uint128_t seed_;
  uint64_t prng_counter_;

  inline uint32_t FieldBitLen(FieldType f) const { return 8 * SizeOf(f); }

  ArrayRef CPRNG(FieldType field, size_t size) {
    return ring_rand(field, size, seed_, &prng_counter_);
  }

  void SetUp() override {
    field_ = std::get<0>(GetParam());
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
    seal::KeyGenerator keygen(*context_);
    rlwe_sk_ = std::make_shared<RLWESecretKey>(keygen.secret_key());

    seed_ = yacl::crypto::RandSeed();
    prng_counter_ = 0;
  }
};

std::ostream& operator<<(std::ostream& out, const Shape3D& f) {
  out << fmt::format("{}x{}x{}", f[0], f[1], f[2]);
  return out;
}

INSTANTIATE_TEST_SUITE_P(
    Cheetah, MatMatProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(Shape3D{181, 899, 1},  // matvec
                                     Shape3D{1, 181, 89},   // matvec
                                     Shape3D{18, 8, 41},    // dim-out > dim-in
                                     Shape3D{52, 81, 18},   // dim-in > dim-out
                                     Shape3D{8, 7, 5},      // small case
                                     Shape3D{254, 253, 126})  // large case
                     ),
    [](const testing::TestParamInfo<MatMatProtTest::ParamType>& p) {
      return fmt::format("{}_{}x{}x{}", std::get<0>(p.param),
                         std::get<0>(std::get<1>(p.param)),
                         std::get<1>(std::get<1>(p.param)),
                         std::get<2>(std::get<1>(p.param)));
    });

TEST_P(MatMatProtTest, Plain) {
  MatMatProtocol::Meta meta;
  meta.dims = std::get<1>(GetParam());

  // NOTE(juhou): Cheetah now supports strided ArrayRef
  for (size_t stride : {1, 2, 3}) {
    auto _lhs = CPRNG(field_, meta.dims[0] * meta.dims[1] * stride);
    auto _rhs = CPRNG(field_, meta.dims[1] * meta.dims[2] * stride + 1);

    auto lhs = _lhs.slice(0, _lhs.numel(), stride);
    auto rhs = _rhs.slice(1, _rhs.numel(), stride);

    SPU_ENFORCE_EQ(lhs.numel(), meta.dims[0] * meta.dims[1]);
    SPU_ENFORCE_EQ(rhs.numel(), meta.dims[1] * meta.dims[2]);

    MatMatProtocol matmat_prot(*context_, *ms_helper_);

    size_t lhs_n = matmat_prot.GetLeftSize(meta);
    size_t rhs_n = matmat_prot.GetRightSize(meta);
    size_t out_n = matmat_prot.GetOutSize(meta);
    std::vector<RLWEPt> lhs_poly(lhs_n);
    std::vector<RLWEPt> rhs_poly(rhs_n);
    std::vector<RLWEPt> out_poly(out_n);
    matmat_prot.EncodeLHS(lhs, meta, true, absl::MakeSpan(lhs_poly));
    matmat_prot.EncodeRHS(rhs, meta, false, absl::MakeSpan(rhs_poly));

    for (auto& p : lhs_poly) {
      NttInplace(p, *context_);
    }
    for (auto& p : rhs_poly) {
      NttInplace(p, *context_);
    }

    matmat_prot.Compute(absl::MakeSpan(lhs_poly), absl::MakeSpan(rhs_poly),
                        meta, absl::MakeSpan(out_poly));

    for (auto& p : out_poly) {
      SPU_ENFORCE(p.coeff_count() > 0);
      InvNttInplace(p, *context_);
    }

    auto expected =
        ring_mmul(lhs, rhs, meta.dims[0], meta.dims[2], meta.dims[1]);
    auto computed =
        matmat_prot.ParseResult(field_, meta, absl::MakeSpan(out_poly));

    EXPECT_EQ(expected.numel(), computed.numel());

    DISPATCH_ALL_FIELDS(field_, "", [&]() {
      auto xe = xt_adapt<ring2k_t>(expected);
      auto xc = xt_adapt<ring2k_t>(computed);
      for (size_t i = 0; i < xc.size(); ++i) {
        EXPECT_EQ(xe[i], xc[i]);
      }
    });
  }
}

TEST_P(MatMatProtTest, EncLHS) {
  MatMatProtocol::Meta meta;
  meta.dims = std::get<1>(GetParam());

  auto lhs = CPRNG(field_, meta.dims[0] * meta.dims[1]);
  auto rhs = CPRNG(field_, meta.dims[1] * meta.dims[2]);
  MatMatProtocol matmat_prot(*context_, *ms_helper_, /*mont*/ true);

  size_t lhs_n = matmat_prot.GetLeftSize(meta);
  size_t rhs_n = matmat_prot.GetRightSize(meta);
  size_t out_n = matmat_prot.GetOutSize(meta);
  std::vector<RLWEPt> lhs_poly(lhs_n);
  std::vector<RLWEPt> rhs_poly(rhs_n);
  matmat_prot.EncodeLHS(lhs, meta, true, absl::MakeSpan(lhs_poly));
  matmat_prot.EncodeRHS(rhs, meta, false, absl::MakeSpan(rhs_poly));

  seal::Encryptor encryptor(*context_, *rlwe_sk_);
  std::vector<RLWECt> enc_poly(lhs_n);
  for (size_t i = 0; i < lhs_n; ++i) {
    NttInplace(lhs_poly[i], *context_);
    encryptor.encrypt_symmetric(lhs_poly[i], enc_poly[i]);
  }
  for (auto& p : rhs_poly) {
    NttInplace(p, *context_);
  }
  matmat_prot.Montgomerize(absl::MakeSpan(rhs_poly));

  std::vector<RLWECt> out_ct(out_n);
  matmat_prot.Compute(absl::MakeSpan(enc_poly), absl::MakeSpan(rhs_poly), meta,
                      absl::MakeSpan(out_ct));
  matmat_prot.ExtractLWEsInplace(meta, absl::MakeSpan(out_ct));

  seal::Evaluator evaluator(*context_);
  seal::Decryptor decryptor(*context_, *rlwe_sk_);
  std::vector<RLWEPt> out_poly(out_n);
  for (size_t i = 0; i < out_n; ++i) {
    if (!out_ct[i].is_ntt_form()) evaluator.transform_to_ntt_inplace(out_ct[i]);
    decryptor.decrypt(out_ct[i], out_poly[i]);
  }

  for (auto& p : out_poly) {
    SPU_ENFORCE(p.coeff_count() > 0);
    InvNttInplace(p, *context_);
  }

  auto expected = ring_mmul(lhs, rhs, meta.dims[0], meta.dims[2], meta.dims[1]);
  auto computed =
      matmat_prot.ParseResult(field_, meta, absl::MakeSpan(out_poly));

  EXPECT_EQ(expected.numel(), computed.numel());

  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto xe = xt_adapt<ring2k_t>(expected);
    auto xc = xt_adapt<ring2k_t>(computed);
    for (size_t i = 0; i < xc.size(); ++i) {
      EXPECT_EQ(xe[i], xc[i]);
    }
  });
}

TEST_P(MatMatProtTest, EncRHS) {
  MatMatProtocol::Meta meta;
  meta.dims = std::get<1>(GetParam());

  auto lhs = CPRNG(field_, meta.dims[0] * meta.dims[1]);
  auto rhs = CPRNG(field_, meta.dims[1] * meta.dims[2]);
  MatMatProtocol matmat_prot(*context_, *ms_helper_);

  size_t lhs_n = matmat_prot.GetLeftSize(meta);
  size_t rhs_n = matmat_prot.GetRightSize(meta);
  size_t out_n = matmat_prot.GetOutSize(meta);
  std::vector<RLWEPt> lhs_poly(lhs_n);
  std::vector<RLWEPt> rhs_poly(rhs_n);
  matmat_prot.EncodeLHS(lhs, meta, false, absl::MakeSpan(lhs_poly));
  matmat_prot.EncodeRHS(rhs, meta, true, absl::MakeSpan(rhs_poly));

  seal::Encryptor encryptor(*context_, *rlwe_sk_);
  std::vector<RLWECt> enc_poly(rhs_n);
  for (size_t i = 0; i < rhs_n; ++i) {
    NttInplace(rhs_poly[i], *context_);
    encryptor.encrypt_symmetric(rhs_poly[i], enc_poly[i]);
  }
  for (auto& p : lhs_poly) {
    NttInplace(p, *context_);
  }

  std::vector<RLWECt> out_ct(out_n);
  matmat_prot.Compute(absl::MakeSpan(lhs_poly), absl::MakeSpan(enc_poly), meta,
                      absl::MakeSpan(out_ct));
  matmat_prot.ExtractLWEsInplace(meta, absl::MakeSpan(out_ct));

  seal::Evaluator evaluator(*context_);
  seal::Decryptor decryptor(*context_, *rlwe_sk_);
  std::vector<RLWEPt> out_poly(out_n);
  for (size_t i = 0; i < out_n; ++i) {
    if (!out_ct[i].is_ntt_form()) evaluator.transform_to_ntt_inplace(out_ct[i]);
    decryptor.decrypt(out_ct[i], out_poly[i]);
  }

  for (auto& p : out_poly) {
    SPU_ENFORCE(p.coeff_count() > 0);
    InvNttInplace(p, *context_);
  }

  auto expected = ring_mmul(lhs, rhs, meta.dims[0], meta.dims[2], meta.dims[1]);
  auto computed =
      matmat_prot.ParseResult(field_, meta, absl::MakeSpan(out_poly));

  EXPECT_EQ(expected.numel(), computed.numel());

  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto xe = xt_adapt<ring2k_t>(expected);
    auto xc = xt_adapt<ring2k_t>(computed);
    for (size_t i = 0; i < xc.size(); ++i) {
      EXPECT_EQ(xe[i], xc[i]);
    }
  });
}
}  // namespace spu::mpc::cheetah::test
