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

#include "libspu/mpc/cheetah/arith/conv2d_prot.h"

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/mpc/cheetah/arith/conv2d_helper.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah::test {

class Conv2DProtTest : public ::testing::TestWithParam<
                           std::tuple<FieldType, Shape3D, Shape3D, Shape2D>> {
 protected:
  static constexpr size_t poly_deg = 4096;

  FieldType field_;
  std::shared_ptr<ModulusSwitchHelper> ms_helper_;
  std::shared_ptr<seal::SEALContext> context_;

  inline uint32_t FieldBitLen(FieldType f) const { return 8 * SizeOf(f); }

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
  }
};

std::string to_string(const Shape3D& dim3) {
  return fmt::format("H{}W{}C{}", dim3[0], dim3[1], dim3[2]);
}

INSTANTIATE_TEST_SUITE_P(
    Cheetah, Conv2DProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     // HxWxC
                     testing::Values(Shape3D{8, 8, 4}, Shape3D{18, 18, 3}),
                     // hxwxO
                     testing::Values(Shape3D{1, 1, 3}, Shape3D{2, 2, 3}),
                     // window_strides
                     testing::Values(Shape2D{3, 3}, Shape2D{1, 1})),
    [](const testing::TestParamInfo<Conv2DProtTest::ParamType>& p) {
      return fmt::format("{}{}h{}O{}s{}", std::get<0>(p.param),
                         to_string(std::get<1>(p.param)),
                         std::get<2>(p.param)[0], std::get<2>(p.param)[2],
                         std::get<3>(p.param)[0]);
    });

TEST_P(Conv2DProtTest, Plain) {
  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    Conv2DProtocol conv2d_prot(*context_, *ms_helper_);
    Conv2DProtocol::Meta meta;
    meta.input_batch = 1;
    meta.input_shape = std::get<1>(GetParam());
    meta.kernel_shape = std::get<2>(GetParam());
    meta.num_kernels = meta.kernel_shape[2];
    meta.kernel_shape[2] = meta.input_shape[2];
    meta.window_strides = std::get<3>(GetParam());

    auto tensor =
        ring_rand(field_, calcNumel(meta.input_shape) * meta.input_batch);
    auto filter =
        ring_rand(field_, calcNumel(meta.kernel_shape) * meta.num_kernels);
    auto expected =
        ring_conv2d(tensor, filter, meta.input_batch, meta.input_shape,
                    meta.num_kernels, meta.kernel_shape, meta.window_strides);

    std::vector<RLWEPt> ecd_tensor(conv2d_prot.GetInputSize(meta));
    std::vector<RLWEPt> ecd_kernels(conv2d_prot.GetKernelSize(meta));
    std::vector<RLWEPt> out_poly(conv2d_prot.GetOutSize(meta));

    conv2d_prot.EncodeInput(tensor, meta, true, absl::MakeSpan(ecd_tensor));
    conv2d_prot.EncodeKernels(filter, meta, false, absl::MakeSpan(ecd_kernels));

    for (auto& poly : ecd_tensor) {
      NttInplace(poly, *context_);
    }
    for (auto& poly : ecd_kernels) {
      NttInplace(poly, *context_);
    }
    conv2d_prot.Compute(absl::MakeSpan(ecd_tensor), absl::MakeSpan(ecd_kernels),
                        meta, absl::MakeSpan(out_poly));

    for (auto& poly : out_poly) {
      InvNttInplace(poly, *context_);
    }

    ArrayRef computed =
        conv2d_prot.ParseResult(field_, meta, absl::MakeSpan(out_poly));
    EXPECT_TRUE(ring_all_equal(expected, computed));
  });
}

}  // namespace spu::mpc::cheetah::test
