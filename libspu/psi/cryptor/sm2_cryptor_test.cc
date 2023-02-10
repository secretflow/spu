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

#include "libspu/psi/cryptor/sm2_cryptor.h"

#include <future>
#include <iostream>

#include "absl/strings/escaping.h"
#include "gtest/gtest.h"
#include "yacl/crypto/base/hash/hash_utils.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/core/prelude.h"

namespace spu::psi {

struct TestParams {
  size_t items_size;
  CurveType type = CurveType::CURVE_SM2;
};

class Sm2CryptorTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(Sm2CryptorTest, Works) {
  auto params = GetParam();
  std::random_device rd;
  yacl::crypto::Prg<uint64_t> prg(rd());

  std::shared_ptr<Sm2Cryptor> sm2_cryptor_a =
      std::make_shared<Sm2Cryptor>(params.type);
  std::shared_ptr<Sm2Cryptor> sm2_cryptor_b =
      std::make_shared<Sm2Cryptor>(params.type);

  std::string items_a(params.items_size * kEccKeySize, '\0');
  std::string items_b(params.items_size * kEccKeySize, '\0');
  std::string masked_a(params.items_size * (kEccKeySize + 1), '\0');
  std::string masked_b(params.items_size * (kEccKeySize + 1), '\0');

  std::string masked_ab(params.items_size * (kEccKeySize + 1), '\0');
  std::string masked_ba(params.items_size * (kEccKeySize + 1), '\0');

  prg.Fill(absl::MakeSpan(items_a.data(), items_a.length()));

  items_b = items_a;

  std::string items_a_point(params.items_size * (kEccKeySize + 1), '\0');
  std::string items_b_point(params.items_size * (kEccKeySize + 1), '\0');

  for (size_t idx = 0; idx < params.items_size; ++idx) {
    absl::Span<const char> items_span =
        absl::MakeSpan(&items_a[idx * kEccKeySize], kEccKeySize);
    std::vector<uint8_t> point_data = sm2_cryptor_a->HashToCurve(items_span);
    std::memcpy(&items_a_point[idx * (kEccKeySize + 1)], point_data.data(),
                point_data.size());

    items_span = absl::MakeSpan(&items_b[idx * kEccKeySize], kEccKeySize);
    point_data = sm2_cryptor_b->HashToCurve(items_span);
    std::memcpy(&items_b_point[idx * (kEccKeySize + 1)], point_data.data(),
                point_data.size());
  }

  // g^a
  sm2_cryptor_a->EccMask(
      absl::MakeSpan(items_a_point.data(), items_a_point.length()),
      absl::MakeSpan(masked_a.data(), masked_a.length()));

  // (g^a)^b
  sm2_cryptor_b->EccMask(absl::MakeSpan(masked_a.data(), masked_a.length()),
                         absl::MakeSpan(masked_ab.data(), masked_ab.length()));

  // g^b
  sm2_cryptor_b->EccMask(
      absl::MakeSpan(items_b_point.data(), items_b_point.length()),
      absl::MakeSpan(masked_b.data(), masked_b.length()));

  // (g^b)^a
  sm2_cryptor_a->EccMask(absl::MakeSpan(masked_b.data(), masked_b.length()),
                         absl::MakeSpan(masked_ba.data(), masked_ba.length()));

  EXPECT_EQ(masked_ab, masked_ba);
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, Sm2CryptorTest,
    testing::Values(TestParams{1}, TestParams{10}, TestParams{50},
                    TestParams{100},
                    // CURVE_SECP256K1
                    TestParams{1, CurveType::CURVE_SECP256K1},
                    TestParams{10, CurveType::CURVE_SECP256K1},
                    TestParams{50, CurveType::CURVE_SECP256K1},
                    TestParams{100, CurveType::CURVE_SECP256K1}));

}  // namespace spu::psi