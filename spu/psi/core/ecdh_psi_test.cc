// Copyright 2021 Ant Group Co., Ltd.
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

#include "spu/psi/core/ecdh_psi.h"

#include <future>
#include <iostream>

#include "gtest/gtest.h"
#include "yasl/base/exception.h"
#include "yasl/link/test_util.h"

struct TestParams {
  std::vector<std::string> items_a;
  std::vector<std::string> items_b;
  size_t target_rank;
  spu::CurveType curve_type = spu::CurveType::Curve25519;
};

namespace std {

std::ostream& operator<<(std::ostream& out, const TestParams& params) {
  out << "target_rank=" << params.target_rank;
  return out;
}

}  // namespace std

namespace spu::psi {

std::vector<std::string> GetIntersection(const TestParams& params) {
  std::set<std::string> set(params.items_a.begin(), params.items_a.end());
  std::vector<std::string> ret;
  for (const auto& s : params.items_b) {
    if (set.count(s) != 0) {
      ret.push_back(s);
    }
  }
  return ret;
}

TEST(EcdhPsiTestFailed, TargetRankMismatched) {
  for (std::pair<size_t, size_t> ranks : std::vector<std::pair<size_t, size_t>>{
           {0, 1}, {0, yasl::link::kAllRank}, {1, yasl::link::kAllRank}}) {
    auto ctxs = yasl::link::test::SetupWorld(2);
    auto proc = [&](std::shared_ptr<yasl::link::Context> ctx,
                    const std::vector<std::string>& items,
                    size_t target_rank) -> std::vector<std::string> {
      return RunEcdhPsi(ctx, items, target_rank);
    };

    std::future<std::vector<std::string>> fa =
        std::async(proc, ctxs[0], std::vector<std::string>{}, ranks.first);
    std::future<std::vector<std::string>> fb =
        std::async(proc, ctxs[1], std::vector<std::string>{}, ranks.second);

    ASSERT_THROW(fa.get(), ::yasl::EnforceNotMet);
    ASSERT_THROW(fb.get(), ::yasl::EnforceNotMet);
  }
}

TEST(EcdhPsiTestFailed, CurveTypeMismatched) {
  std::pair<CurveType, CurveType> curves = {CurveType::CurveFourQ,
                                            CurveType::Curve25519};

  auto ctxs = yasl::link::test::SetupWorld(2);
  auto proc = [&](std::shared_ptr<yasl::link::Context> ctx,
                  const std::vector<std::string>& items,
                  CurveType type) -> std::vector<std::string> {
    return RunEcdhPsi(ctx, items, yasl::link::kAllRank, type);
  };

  std::future<std::vector<std::string>> fa =
      std::async(proc, ctxs[0], std::vector<std::string>{}, curves.first);
  std::future<std::vector<std::string>> fb =
      std::async(proc, ctxs[1], std::vector<std::string>{}, curves.second);

  ASSERT_THROW(fa.get(), ::yasl::EnforceNotMet);
  ASSERT_THROW(fb.get(), ::yasl::EnforceNotMet);
}

class EcdhPsiTest : public testing::TestWithParam<TestParams> {};

TEST_P(EcdhPsiTest, Works) {
  auto params = GetParam();
  auto ctxs = yasl::link::test::SetupWorld(2);
  auto proc =
      [&](std::shared_ptr<yasl::link::Context> ctx,
          const std::vector<std::string>& items) -> std::vector<std::string> {
    return RunEcdhPsi(ctx, items, params.target_rank, params.curve_type);
  };

  std::future<std::vector<std::string>> fa =
      std::async(proc, ctxs[0], params.items_a);
  std::future<std::vector<std::string>> fb =
      std::async(proc, ctxs[1], params.items_b);

  auto results_a = fa.get();
  auto results_b = fb.get();

  auto intersection = GetIntersection(params);
  if (params.target_rank == yasl::link::kAllRank || params.target_rank == 0) {
    EXPECT_EQ(results_a, intersection);
  } else {
    EXPECT_TRUE(results_a.empty());
  }
  if (params.target_rank == yasl::link::kAllRank || params.target_rank == 1) {
    EXPECT_EQ(results_b, intersection);
  } else {
    EXPECT_TRUE(results_b.empty());
  }
}

std::vector<std::string> CreateRangeItems(size_t begin, size_t size) {
  std::vector<std::string> ret;
  for (size_t i = 0; i < size; i++) {
    ret.push_back(std::to_string(begin + i));
  }
  return ret;
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, EcdhPsiTest,
    testing::Values(
        TestParams{{"a", "b"}, {"b", "c"}, yasl::link::kAllRank},  //
        TestParams{{"a", "b"}, {"b", "c"}, 0},                     //
        TestParams{{"a", "b"}, {"b", "c"}, 1},                     //
        //
        TestParams{{"a", "b"}, {"c", "d"}, yasl::link::kAllRank},  //
        TestParams{{"a", "b"}, {"c", "d"}, 0},                     //
        TestParams{{"a", "b"}, {"c", "d"}, 1},                     //
        //
        TestParams{{}, {"a"}, yasl::link::kAllRank},  //
        TestParams{{}, {"a"}, 0},                     //
        TestParams{{}, {"a"}, 1},                     //
        //
        TestParams{{"a"}, {}, yasl::link::kAllRank},  //
        TestParams{{"a"}, {}, 0},                     //
        TestParams{{"a"}, {}, 1},                     //
        // less than one batch
        TestParams{CreateRangeItems(0, 4095), CreateRangeItems(1, 4095),
                   yasl::link::kAllRank},                                     //
        TestParams{CreateRangeItems(0, 4095), CreateRangeItems(1, 4095), 0},  //
        TestParams{CreateRangeItems(0, 4095), CreateRangeItems(1, 4095), 1},  //
        // exactly one batch
        TestParams{CreateRangeItems(0, 4096), CreateRangeItems(1, 4096),
                   yasl::link::kAllRank},                                     //
        TestParams{CreateRangeItems(0, 4096), CreateRangeItems(1, 4096), 0},  //
        TestParams{CreateRangeItems(0, 4096), CreateRangeItems(1, 4096), 1},  //
        // more than one batch
        TestParams{CreateRangeItems(0, 40961), CreateRangeItems(5, 40961),
                   yasl::link::kAllRank},  //
        TestParams{CreateRangeItems(0, 40961), CreateRangeItems(5, 40961),
                   0},  //
        TestParams{CreateRangeItems(0, 40961), CreateRangeItems(5, 40961),
                   1},  //
        //
        TestParams{{}, {}, yasl::link::kAllRank},  //
        TestParams{{}, {}, 0},                     //
        TestParams{{}, {}, 1},                     //
        // test sm2
        TestParams{CreateRangeItems(0, 4096), CreateRangeItems(1, 4095),
                   yasl::link::kAllRank, spu::CurveType::CurveSm2},  //
        // exactly one batch
        TestParams{CreateRangeItems(0, 4096), CreateRangeItems(1, 4096),
                   yasl::link::kAllRank, spu::CurveType::CurveSecp256k1},  //
        // more than one batch
        TestParams{CreateRangeItems(0, 4096), CreateRangeItems(1, 4096),
                   yasl::link::kAllRank, spu::CurveType::CurveFourQ}  //
        ));

}  // namespace spu::psi
