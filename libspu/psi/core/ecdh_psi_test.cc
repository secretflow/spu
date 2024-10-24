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

#include "libspu/psi/core/ecdh_psi.h"

#include <future>
#include <iostream>

#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/utils/test_utils.h"

struct TestParams {
  std::vector<std::string> items_a;
  std::vector<std::string> items_b;
  size_t target_rank;
  spu::psi::CurveType curve_type = spu::psi::CurveType::CURVE_25519;
};

namespace std {

std::ostream& operator<<(std::ostream& out, const TestParams& params) {
  out << "target_rank=" << params.target_rank;
  return out;
}

}  // namespace std

namespace spu::psi {

TEST(EcdhPsiTestFailed, TargetRankMismatched) {
  for (std::pair<size_t, size_t> ranks : std::vector<std::pair<size_t, size_t>>{
           {0, 1}, {0, yacl::link::kAllRank}, {1, yacl::link::kAllRank}}) {
    auto ctxs = yacl::link::test::SetupWorld(2);
    auto proc = [&](const std::shared_ptr<yacl::link::Context>& ctx,
                    const std::vector<std::string>& items,
                    size_t target_rank) -> std::vector<std::string> {
      return RunEcdhPsi(ctx, items, target_rank);
    };

    std::future<std::vector<std::string>> fa =
        std::async(proc, ctxs[0], std::vector<std::string>{}, ranks.first);
    std::future<std::vector<std::string>> fb =
        std::async(proc, ctxs[1], std::vector<std::string>{}, ranks.second);

    ASSERT_THROW(fa.get(), ::yacl::EnforceNotMet);
    ASSERT_THROW(fb.get(), ::yacl::EnforceNotMet);
  }
}

TEST(EcdhPsiTestFailed, CurveTypeMismatched) {
  std::pair<CurveType, CurveType> curves = {CurveType::CURVE_FOURQ,
                                            CurveType::CURVE_25519};

  auto ctxs = yacl::link::test::SetupWorld(2);
  auto proc = [&](const std::shared_ptr<yacl::link::Context>& ctx,
                  const std::vector<std::string>& items,
                  CurveType type) -> std::vector<std::string> {
    return RunEcdhPsi(ctx, items, yacl::link::kAllRank, type);
  };

  std::future<std::vector<std::string>> fa =
      std::async(proc, ctxs[0], std::vector<std::string>{}, curves.first);
  std::future<std::vector<std::string>> fb =
      std::async(proc, ctxs[1], std::vector<std::string>{}, curves.second);

  ASSERT_THROW(fa.get(), ::yacl::EnforceNotMet);
  ASSERT_THROW(fb.get(), ::yacl::EnforceNotMet);
}

class EcdhPsiTest : public testing::TestWithParam<TestParams> {};

TEST_P(EcdhPsiTest, Works) {
  auto params = GetParam();
  auto ctxs = yacl::link::test::SetupWorld(2);
  auto proc =
      [&](const std::shared_ptr<yacl::link::Context>& ctx,
          const std::vector<std::string>& items) -> std::vector<std::string> {
    return RunEcdhPsi(ctx, items, params.target_rank, params.curve_type);
  };

  std::future<std::vector<std::string>> fa =
      std::async(proc, ctxs[0], params.items_a);
  std::future<std::vector<std::string>> fb =
      std::async(proc, ctxs[1], params.items_b);

  auto results_a = fa.get();
  auto results_b = fb.get();

  auto intersection = test::GetIntersection(params.items_a, params.items_b);
  if (params.target_rank == yacl::link::kAllRank || params.target_rank == 0) {
    EXPECT_EQ(results_a, intersection);
  } else {
    EXPECT_TRUE(results_a.empty());
  }
  if (params.target_rank == yacl::link::kAllRank || params.target_rank == 1) {
    EXPECT_EQ(results_b, intersection);
  } else {
    EXPECT_TRUE(results_b.empty());
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, EcdhPsiTest,
    testing::Values(
        TestParams{{"a", "b"}, {"b", "c"}, yacl::link::kAllRank},  //
        TestParams{{"a", "b"}, {"b", "c"}, 0},                     //
        TestParams{{"a", "b"}, {"b", "c"}, 1},                     //
        //
        TestParams{{"a", "b"}, {"c", "d"}, yacl::link::kAllRank},  //
        TestParams{{"a", "b"}, {"c", "d"}, 0},                     //
        TestParams{{"a", "b"}, {"c", "d"}, 1},                     //
        //
        TestParams{{}, {"a"}, yacl::link::kAllRank},  //
        TestParams{{}, {"a"}, 0},                     //
        TestParams{{}, {"a"}, 1},                     //
        //
        TestParams{{"a"}, {}, yacl::link::kAllRank},  //
        TestParams{{"a"}, {}, 0},                     //
        TestParams{{"a"}, {}, 1},                     //
        // less than one batch
        TestParams{test::CreateRangeItems(0, 4095),
                   test::CreateRangeItems(1, 4095), yacl::link::kAllRank},  //
        TestParams{test::CreateRangeItems(0, 4095),
                   test::CreateRangeItems(1, 4095), 0},  //
        TestParams{test::CreateRangeItems(0, 4095),
                   test::CreateRangeItems(1, 4095), 1},  //
        // exactly one batch
        TestParams{test::CreateRangeItems(0, 4096),
                   test::CreateRangeItems(1, 4096), yacl::link::kAllRank},  //
        TestParams{test::CreateRangeItems(0, 4096),
                   test::CreateRangeItems(1, 4096), 0},  //
        TestParams{test::CreateRangeItems(0, 4096),
                   test::CreateRangeItems(1, 4096), 1},  //
        // more than one batch
        TestParams{test::CreateRangeItems(0, 40961),
                   test::CreateRangeItems(5, 40961), yacl::link::kAllRank},  //
        TestParams{test::CreateRangeItems(0, 40961),
                   test::CreateRangeItems(5, 40961), 0},  //
        TestParams{test::CreateRangeItems(0, 40961),
                   test::CreateRangeItems(5, 40961), 1},  //
        //
        TestParams{{}, {}, yacl::link::kAllRank},  //
        TestParams{{}, {}, 0},                     //
        TestParams{{}, {}, 1},                     //
        // test sm2
        TestParams{test::CreateRangeItems(0, 4096),
                   test::CreateRangeItems(1, 4095), yacl::link::kAllRank,
                   spu::psi::CurveType::CURVE_SM2},  //
        // exactly one batch
        TestParams{test::CreateRangeItems(0, 4096),
                   test::CreateRangeItems(1, 4096), yacl::link::kAllRank,
                   spu::psi::CurveType::CURVE_SECP256K1},  //
        // more than one batch
        TestParams{test::CreateRangeItems(0, 4096),
                   test::CreateRangeItems(1, 4096), yacl::link::kAllRank,
                   spu::psi::CurveType::CURVE_FOURQ}  //
        ));

}  // namespace spu::psi
