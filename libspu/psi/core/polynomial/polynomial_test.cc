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

#include "libspu/psi/core/polynomial/polynomial.h"

#include <future>
#include <iostream>
#include <random>

#include "absl/strings/escaping.h"
#include "gtest/gtest.h"
#include "yacl/crypto/tools/prg.h"

namespace {
struct TestParams {
  uint64_t polynomial_order;
};

// first prime over 2^256, used as module for polynoimal interpolate
std::string kPrimeOver256bHexStr =
    "010000000000000000000000000000000000000000000000000000000000000129";
constexpr uint32_t kBnByteSize = 32;

}  // namespace

namespace spu::psi {
//  test 256b big num polynomial interpolate and eval
class PolynomialBnTest : public testing::TestWithParam<TestParams> {};

TEST_P(PolynomialBnTest, Works) {
  auto params = GetParam();
  uint64_t polynomial_order = params.polynomial_order;

  std::vector<std::string> poly_x(polynomial_order);
  std::vector<std::string> poly_y(polynomial_order);
  std::vector<absl::string_view> poly_x_sv(polynomial_order);
  std::vector<absl::string_view> poly_y_sv(polynomial_order);
  std::vector<std::string> coeff;

  std::random_device rd;
  yacl::crypto::Prg<uint64_t> prg1(rd());
  yacl::crypto::Prg<uint64_t> prg2(rd());

  std::string prime_data = absl::HexStringToBytes(kPrimeOver256bHexStr);

  for (uint64_t i = 0; i < polynomial_order; ++i) {
    poly_x[i].resize(kBnByteSize);
    poly_y[i].resize(kBnByteSize);
    prg1.Fill(absl::MakeSpan(poly_x[i].data(), kBnByteSize));
    prg2.Fill(absl::MakeSpan(poly_y[i].data(), kBnByteSize));

    poly_x_sv[i] = poly_x[i];
    poly_y_sv[i] = poly_y[i];
  }

  coeff = InterpolatePolynomial(poly_x_sv, poly_y_sv, prime_data);

  for (uint64_t i = 0; i < polynomial_order; ++i) {
    std::string eval_y = EvalPolynomial(coeff, poly_x[i], prime_data);
    EXPECT_EQ(eval_y, poly_y[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, PolynomialBnTest,
                         testing::Values(TestParams{0},     //
                                         TestParams{1},     //
                                         TestParams{10},    //
                                         TestParams{32},    //
                                         TestParams{128},   //
                                         TestParams{256},   //
                                         TestParams{1024},  //
                                         TestParams{1025}   //
                                         ));

}  // namespace spu::psi
