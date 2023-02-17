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

#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/kernel/hal/integer.h"
#include "libspu/kernel/hal/random.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/test_util.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/convolution.h"
#include "libspu/kernel/hlo/rand.h"
#include "libspu/mpc/cheetah/arith/conv2d_helper.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah::test {

static ArrayRef ring_conv2d(const ArrayRef& tensor, const ArrayRef& filter,
                            int64_t num_tensors, Shape3D tensor_shape,
                            int64_t num_filters, Shape3D filter_shape,
                            Shape2D window_strides) {
  auto field = tensor.eltype().as<Ring2k>()->field();
  Shape4D result_shape;
  result_shape[0] = num_tensors;
  for (int s : {0, 1}) {
    result_shape[s + 1] =
        (tensor_shape[s] - filter_shape[s] + window_strides[s]) /
        window_strides[s];
  }
  result_shape[3] = num_filters;

  std::vector<int64_t> ts = {num_tensors, tensor_shape[0], tensor_shape[1],
                             tensor_shape[2]};
  std::vector<int64_t> fs = {filter_shape[0], filter_shape[1], filter_shape[2],
                             num_filters};

  NdArrayRef _tensor = unflatten(tensor, ts);
  NdArrayRef _filter = unflatten(filter, fs);
  NdArrayRef _ret =
      unflatten(ring_zeros(field, calcNumel(result_shape)), result_shape);

  DISPATCH_ALL_FIELDS(field, "ring_conv2d", [&]() {
    // NOTE(juhou): valid padding so offset are always 0.
    constexpr int64_t padh = 0;
    constexpr int64_t padw = 0;

    for (int64_t ib = 0; ib < ts[0]; ++ib) {
      for (int64_t oc = 0; oc < fs[3]; ++oc) {
        for (int64_t ih = -padh, oh = 0; oh < result_shape[1];
             ih += window_strides[0], ++oh) {
          for (int64_t iw = -padw, ow = 0; ow < result_shape[2];
               iw += window_strides[1], ++ow) {
            ring2k_t sum{0};

            for (int64_t ic = 0; ic < filter_shape[2]; ++ic) {
              for (int64_t fh = 0; fh < filter_shape[0]; ++fh) {
                for (int64_t fw = 0; fw < filter_shape[1]; ++fw) {
                  auto f = _filter.at<ring2k_t>({fh, fw, ic, oc});
                  auto t = _tensor.at<ring2k_t>({ib, ih + fh, iw + fw, ic});
                  sum += f * t;
                }
              }
            }
            _ret.at<ring2k_t>({ib, oh, ow, oc}) = sum;
          }
        }
      }
    }
  });

  return flatten(_ret);
}

class Conv2DProtTest
    : public ::testing::TestWithParam<
          std::tuple<FieldType, int64_t, Shape3D, Shape3D, Shape2D>> {
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

    seed_ = yacl::crypto::RandSeed();
    prng_counter_ = 0;
  }
};

std::string to_string(const Shape3D& dim3) {
  return fmt::format("H{}W{}C{}", dim3[0], dim3[1], dim3[2]);
}

INSTANTIATE_TEST_SUITE_P(
    Cheetah, Conv2DProtTest,
    testing::Combine(
        testing::Values(FieldType::FM32, FieldType::FM64),
        testing::Values(1LL, 2LL, 3LL),                           // O
        testing::Values(Shape3D{8, 8, 4}, Shape3D{128, 128, 3}),  // input_shape
        testing::Values(Shape3D{1, 1, 3}, Shape3D{2, 2, 3}),  // kernel_shape
        testing::Values(Shape2D{3, 3}, Shape2D{1, 1})),       // window_strides
    [](const testing::TestParamInfo<Conv2DProtTest::ParamType>& p) {
      return fmt::format("{}{}h{}O{}s{}", std::get<0>(p.param),
                         to_string(std::get<2>(p.param)),
                         std::get<3>(p.param)[0], std::get<1>(p.param),
                         std::get<4>(p.param)[0]);
    });

TEST_P(Conv2DProtTest, Plain) {
  using namespace spu::kernel;
  RuntimeConfig run_config;
  run_config.set_protocol(ProtocolKind::REF2K);
  run_config.set_field(field_);
  run_config.set_sigmoid_mode(RuntimeConfig::SIGMOID_REAL);
  HalContext hctx = hal::test::makeRefHalContext(run_config);

  auto _m = std::get<1>(GetParam());
  auto _t = std::get<2>(GetParam());
  auto _k = std::get<3>(GetParam());
  // NxHxWxC
  Shape4D tensor_shape = {1, _t[0], _t[1], _t[2]};
  // HxWxCxO
  Shape4D kernel_shape = {_k[0], _k[1], _t[2], _m};
  Shape2D window_strides = std::get<4>(GetParam());

  hlo::ConvolutionConfig config;
  Shape4D result_shape;
  result_shape[0] = tensor_shape[0];
  for (int s : {1, 2}) {
    result_shape[s] =
        (tensor_shape[s] - kernel_shape[s - 1] + window_strides[s - 1]) /
        window_strides[s - 1];
  }
  result_shape[3] = kernel_shape[3];

  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto tensor =
        hlo::Iota<ring2k_t>(&hctx, calcNumel(tensor_shape), VIS_PUBLIC);
    auto filter =
        hlo::Iota<ring2k_t>(&hctx, calcNumel(kernel_shape), VIS_PUBLIC);

    tensor = hal::reshape(&hctx, tensor, tensor_shape);
    filter = hal::reshape(&hctx, filter, kernel_shape);
    config.window_strides = absl::MakeSpan(window_strides);
    spu::Value expected =
        hlo::Convolution2D(&hctx, tensor, filter, config, result_shape);
    expected = hal::reshape(
        &hctx, expected, {result_shape[1], result_shape[2], result_shape[3]});

    Conv2DProtocol conv2d_prot(*context_, *ms_helper_);
    Conv2DProtocol::Meta meta;
    meta.num_kernels = kernel_shape[3];
    meta.input_shape = _t;
    meta.kernel_shape = {_k[0], _k[1], _t[2]};
    meta.window_strides = window_strides;

    std::vector<int64_t> start_indices = {0, 0, 0, 0};
    std::vector<int64_t> end_indices = {1, _t[0], _t[1], _t[2]};
    // NxHxWxC
    std::vector<int64_t> strides = {1, 1, 1, 1};
    for (int d : {0, 1}) {
      if (meta.kernel_shape[0] == 1) {
        strides[1 + d] = meta.window_strides[d];
        meta.input_shape[d] =
            (meta.input_shape[d] + meta.window_strides[d] - 1) /
            meta.window_strides[d];
        meta.window_strides[d] = 1;
      }
    }
    if (std::any_of(strides.begin(), strides.end(),
                    [](int64_t s) { return s > 1; })) {
      tensor = hal::slice(&hctx, tensor, start_indices, end_indices, strides);
    }

    std::vector<RLWEPt> ecd_tensor(conv2d_prot.GetInputSize(meta));
    std::vector<RLWEPt> ecd_kernels(conv2d_prot.GetKernelSize(meta));
    std::vector<RLWEPt> out_poly(conv2d_prot.GetOutSize(meta));

    tensor = hal::reshape(&hctx, tensor, meta.input_shape);

    auto _tensor = flatten(tensor.data());
    auto _filter = flatten(filter.data());

    ArrayRef _expected =
        ring_conv2d(_tensor, _filter, tensor_shape[0], meta.input_shape,
                    meta.num_kernels, meta.kernel_shape, meta.window_strides);

    conv2d_prot.EncodeInput(_tensor, meta, true, absl::MakeSpan(ecd_tensor));
    conv2d_prot.EncodeKernels(_filter, meta, false,
                              absl::MakeSpan(ecd_kernels));

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
    EXPECT_TRUE(ring_all_equal(_expected, computed));

    auto fe = flatten(expected.data());
    // NOTE(juhou): we haved used VIS_PUBLIC to create spu::Value
    // So the eltype will mismatch.
    auto fc = computed.as(fe.eltype());
    EXPECT_TRUE(fc == fe);
  });
}

TEST_P(Conv2DProtTest, EncTensor) {
  using namespace spu::kernel;
  RuntimeConfig run_config;
  run_config.set_protocol(ProtocolKind::REF2K);
  run_config.set_field(field_);
  run_config.set_sigmoid_mode(RuntimeConfig::SIGMOID_REAL);
  HalContext hctx = hal::test::makeRefHalContext(run_config);

  auto _m = std::get<1>(GetParam());
  auto _t = std::get<2>(GetParam());
  auto _k = std::get<3>(GetParam());
  // NxHxWxC
  Shape4D tensor_shape = {1, _t[0], _t[1], _t[2]};
  // HxWxCxO
  Shape4D kernel_shape = {_k[0], _k[1], _t[2], _m};
  Shape2D window_strides = std::get<4>(GetParam());

  hlo::ConvolutionConfig config;
  Shape4D result_shape;
  result_shape[0] = tensor_shape[0];
  for (int s : {1, 2}) {
    result_shape[s] =
        (tensor_shape[s] - kernel_shape[s - 1] + window_strides[s - 1]) /
        window_strides[s - 1];
  }
  result_shape[3] = kernel_shape[3];

  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto tensor =
        hlo::Iota<ring2k_t>(&hctx, calcNumel(tensor_shape), VIS_PUBLIC);
    auto filter =
        hlo::Iota<ring2k_t>(&hctx, calcNumel(kernel_shape), VIS_PUBLIC);

    tensor = hal::reshape(&hctx, tensor, tensor_shape);
    filter = hal::reshape(&hctx, filter, kernel_shape);
    config.window_strides = absl::MakeSpan(window_strides);
    spu::Value expected =
        hlo::Convolution2D(&hctx, tensor, filter, config, result_shape);
    expected = hal::reshape(
        &hctx, expected, {result_shape[1], result_shape[2], result_shape[3]});

    Conv2DProtocol conv2d_prot(*context_, *ms_helper_);
    Conv2DProtocol::Meta meta;
    meta.num_kernels = kernel_shape[3];
    meta.input_shape = _t;
    meta.kernel_shape = {_k[0], _k[1], _t[2]};
    meta.window_strides = window_strides;

    std::vector<int64_t> start_indices = {0, 0, 0, 0};
    std::vector<int64_t> end_indices = {1, _t[0], _t[1], _t[2]};
    // NxHxWxC
    std::vector<int64_t> strides = {1, 1, 1, 1};
    for (int d : {0, 1}) {
      if (meta.kernel_shape[0] == 1) {
        strides[1 + d] = meta.window_strides[d];
        meta.input_shape[d] =
            (meta.input_shape[d] + meta.window_strides[d] - 1) /
            meta.window_strides[d];
        meta.window_strides[d] = 1;
      }
    }
    if (std::any_of(strides.begin(), strides.end(),
                    [](int64_t s) { return s > 1; })) {
      tensor = hal::slice(&hctx, tensor, start_indices, end_indices, strides);
    }

    std::vector<RLWEPt> ecd_tensor(conv2d_prot.GetInputSize(meta));
    std::vector<RLWEPt> ecd_kernels(conv2d_prot.GetKernelSize(meta));

    tensor = hal::reshape(&hctx, tensor, meta.input_shape);

    auto _tensor = flatten(tensor.data());
    auto _filter = flatten(filter.data());
    conv2d_prot.EncodeInput(_tensor, meta, true, absl::MakeSpan(ecd_tensor));

    conv2d_prot.EncodeKernels(_filter, meta, false,
                              absl::MakeSpan(ecd_kernels));

    for (auto& poly : ecd_tensor) {
      NttInplace(poly, *context_);
    }
    for (auto& poly : ecd_kernels) {
      NttInplace(poly, *context_);
    }

    seal::Encryptor encryptor(*context_, *rlwe_sk_);
    seal::Decryptor decryptor(*context_, *rlwe_sk_);
    std::vector<RLWECt> enc_tensor(ecd_tensor.size());
    for (size_t i = 0; i < ecd_tensor.size(); ++i) {
      encryptor.encrypt_symmetric(ecd_tensor[i], enc_tensor[i]);
    }

    std::vector<RLWECt> enc_out(conv2d_prot.GetOutSize(meta));
    conv2d_prot.Compute(absl::MakeSpan(enc_tensor), absl::MakeSpan(ecd_kernels),
                        meta, absl::MakeSpan(enc_out));

    conv2d_prot.ExtractLWEsInplace(meta, absl::MakeSpan(enc_out));

    std::vector<RLWEPt> out_poly(enc_out.size());
    seal::Evaluator evoluator(*context_);
    for (size_t i = 0; i < out_poly.size(); ++i) {
      if (!enc_out[i].is_ntt_form()) {
        evoluator.transform_to_ntt_inplace(enc_out[i]);
      }
      decryptor.decrypt(enc_out[i], out_poly[i]);
      InvNttInplace(out_poly[i], *context_);
    }

    ArrayRef computed =
        conv2d_prot.ParseResult(field_, meta, absl::MakeSpan(out_poly));

    auto fe = flatten(expected.data());
    // NOTE(juhou): we haved used VIS_PUBLIC to create spu::Value
    // So the eltype will mismatch.
    auto fc = computed.as(fe.eltype());
    EXPECT_TRUE(fc == fe);
  });
}

TEST_P(Conv2DProtTest, EncKernels) {
  using namespace spu::kernel;
  RuntimeConfig run_config;
  run_config.set_protocol(ProtocolKind::REF2K);
  run_config.set_field(field_);
  run_config.set_sigmoid_mode(RuntimeConfig::SIGMOID_REAL);
  HalContext hctx = hal::test::makeRefHalContext(run_config);

  auto _m = std::get<1>(GetParam());
  auto _t = std::get<2>(GetParam());
  auto _k = std::get<3>(GetParam());
  // NxHxWxC
  Shape4D tensor_shape = {1, _t[0], _t[1], _t[2]};
  // HxWxCxO
  Shape4D kernel_shape = {_k[0], _k[1], _t[2], _m};
  Shape2D window_strides = std::get<4>(GetParam());

  hlo::ConvolutionConfig config;
  Shape4D result_shape;
  result_shape[0] = tensor_shape[0];
  for (int s : {1, 2}) {
    result_shape[s] =
        (tensor_shape[s] - kernel_shape[s - 1] + window_strides[s - 1]) /
        window_strides[s - 1];
  }
  result_shape[3] = kernel_shape[3];

  DISPATCH_ALL_FIELDS(field_, "", [&]() {
    auto tensor =
        hlo::Iota<ring2k_t>(&hctx, calcNumel(tensor_shape), VIS_PUBLIC);
    auto filter =
        hlo::Iota<ring2k_t>(&hctx, calcNumel(kernel_shape), VIS_PUBLIC);

    tensor = hal::reshape(&hctx, tensor, tensor_shape);
    filter = hal::reshape(&hctx, filter, kernel_shape);
    config.window_strides = absl::MakeSpan(window_strides);
    spu::Value expected =
        hlo::Convolution2D(&hctx, tensor, filter, config, result_shape);
    expected = hal::reshape(
        &hctx, expected, {result_shape[1], result_shape[2], result_shape[3]});

    Conv2DProtocol conv2d_prot(*context_, *ms_helper_);
    Conv2DProtocol::Meta meta;
    meta.num_kernels = kernel_shape[3];
    meta.input_shape = _t;
    meta.kernel_shape = {_k[0], _k[1], _t[2]};
    meta.window_strides = window_strides;

    std::vector<int64_t> start_indices = {0, 0, 0, 0};
    std::vector<int64_t> end_indices = {1, _t[0], _t[1], _t[2]};
    // NxHxWxC
    std::vector<int64_t> strides = {1, 1, 1, 1};
    for (int d : {0, 1}) {
      if (meta.kernel_shape[0] == 1) {
        strides[1 + d] = meta.window_strides[d];
        meta.input_shape[d] =
            (meta.input_shape[d] + meta.window_strides[d] - 1) /
            meta.window_strides[d];
        meta.window_strides[d] = 1;
      }
    }
    if (std::any_of(strides.begin(), strides.end(),
                    [](int64_t s) { return s > 1; })) {
      tensor = hal::slice(&hctx, tensor, start_indices, end_indices, strides);
    }

    std::vector<RLWEPt> ecd_tensor(conv2d_prot.GetInputSize(meta));
    std::vector<RLWEPt> ecd_kernels(conv2d_prot.GetKernelSize(meta));

    tensor = hal::reshape(&hctx, tensor, meta.input_shape);

    auto _tensor = flatten(tensor.data());
    auto _filter = flatten(filter.data());
    conv2d_prot.EncodeInput(_tensor, meta, false, absl::MakeSpan(ecd_tensor));
    conv2d_prot.EncodeKernels(_filter, meta, true, absl::MakeSpan(ecd_kernels));

    for (auto& poly : ecd_tensor) {
      NttInplace(poly, *context_);
    }
    for (auto& poly : ecd_kernels) {
      NttInplace(poly, *context_);
    }

    seal::Encryptor encryptor(*context_, *rlwe_sk_);
    seal::Decryptor decryptor(*context_, *rlwe_sk_);
    std::vector<RLWECt> enc_kernels(ecd_kernels.size());
    for (size_t i = 0; i < ecd_kernels.size(); ++i) {
      encryptor.encrypt_symmetric(ecd_kernels[i], enc_kernels[i]);
    }

    std::vector<RLWECt> enc_out(conv2d_prot.GetOutSize(meta));
    conv2d_prot.Compute(absl::MakeSpan(ecd_tensor), absl::MakeSpan(enc_kernels),
                        meta, absl::MakeSpan(enc_out));

    conv2d_prot.ExtractLWEsInplace(meta, absl::MakeSpan(enc_out));

    std::vector<RLWEPt> out_poly(enc_out.size());
    seal::Evaluator evoluator(*context_);
    for (size_t i = 0; i < out_poly.size(); ++i) {
      if (!enc_out[i].is_ntt_form()) {
        evoluator.transform_to_ntt_inplace(enc_out[i]);
      }
      decryptor.decrypt(enc_out[i], out_poly[i]);
      InvNttInplace(out_poly[i], *context_);
    }

    ArrayRef computed =
        conv2d_prot.ParseResult(field_, meta, absl::MakeSpan(out_poly));

    auto fe = flatten(expected.data());
    // NOTE(juhou): we haved used VIS_PUBLIC to create spu::Value
    // So the eltype will mismatch.
    auto fc = computed.as(fe.eltype());
    EXPECT_TRUE(fc == fe);
  });
}

}  // namespace spu::mpc::cheetah::test
