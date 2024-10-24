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

#include "libspu/psi/memory_psi.h"

#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/utils/test_utils.h"

namespace spu::psi {

namespace {
struct MemoryTaskTestParams {
  std::vector<size_t> item_size;
  size_t intersection_size;
  spu::psi::PsiType psi_protocol;
};

std::vector<std::vector<std::string>> CreateMemoryTaskItems(
    const MemoryTaskTestParams& params) {
  std::vector<std::vector<std::string>> ret(params.item_size.size() + 1);
  ret[params.item_size.size()] =
      test::CreateRangeItems(1, params.intersection_size);

  for (size_t idx = 0; idx < params.item_size.size(); ++idx) {
    ret[idx] =
        test::CreateRangeItems((idx + 1) * 1000000, params.item_size[idx]);
  }

  for (size_t idx = 0; idx < params.item_size.size(); ++idx) {
    std::set<size_t> idx_set;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, params.item_size[idx] - 1);

    while (idx_set.size() < params.intersection_size) {
      idx_set.insert(dis(gen));
    }
    size_t j = 0;
    for (const auto& iter : idx_set) {
      ret[idx][iter] = ret[params.item_size.size()][j++];
    }
  }
  return ret;
}

}  // namespace

class MemoryTaskPsiTest : public testing::TestWithParam<MemoryTaskTestParams> {
};

TEST_P(MemoryTaskPsiTest, Works) {
  std::vector<std::vector<std::string>> items;

  auto params = GetParam();
  items = CreateMemoryTaskItems(params);

  auto lctxs = yacl::link::test::SetupWorld(params.item_size.size());

  auto proc = [&](int idx) -> std::vector<std::string> {
    spu::psi::MemoryPsiConfig config;
    config.set_psi_type(params.psi_protocol);
    config.set_broadcast_result(true);

    MemoryPsi ctx(config, lctxs[idx]);

    return ctx.Run(items[idx]);
  };

  size_t world_size = lctxs.size();
  std::vector<std::future<std::vector<std::string>>> f_links(world_size);
  for (size_t i = 0; i < world_size; i++) {
    f_links[i] = std::async(proc, i);
  }

  std::vector<std::string> intersection =
      test::GetIntersection(items[params.item_size.size()], items[0]);
  std::sort(intersection.begin(), intersection.end());

  std::vector<std::vector<std::string>> results(world_size);
  for (size_t i = 0; i < world_size; i++) {
    results[i] = f_links[i].get();

    EXPECT_EQ(results[i].size(), intersection.size());
  }
}

TEST_P(MemoryTaskPsiTest, BroadcastFalse) {
  std::vector<std::vector<std::string>> items;

  auto params = GetParam();
  items = CreateMemoryTaskItems(params);
  size_t receiver_rank = 0;

  auto lctxs = yacl::link::test::SetupWorld(params.item_size.size());

  auto proc = [&](int idx) -> std::vector<std::string> {
    spu::psi::MemoryPsiConfig config;
    config.set_psi_type(params.psi_protocol);
    config.set_receiver_rank(receiver_rank);
    config.set_broadcast_result(false);

    MemoryPsi ctx(config, lctxs[idx]);

    return ctx.Run(items[idx]);
  };

  size_t world_size = lctxs.size();
  std::vector<std::future<std::vector<std::string>>> f_links(world_size);
  for (size_t i = 0; i < world_size; i++) {
    f_links[i] = std::async(proc, i);
  }

  std::vector<std::string> intersection =
      test::GetIntersection(items[params.item_size.size()], items[0]);
  std::sort(intersection.begin(), intersection.end());

  std::vector<std::vector<std::string>> results(world_size);
  for (size_t i = 0; i < world_size; i++) {
    results[i] = f_links[i].get();

    if (i == receiver_rank) {
      EXPECT_EQ(results[i].size(), intersection.size());
    } else {
      EXPECT_EQ(results[i].size(), 0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, MemoryTaskPsiTest,
    testing::Values(
        MemoryTaskTestParams{{0, 3}, 0, spu::psi::PsiType::ECDH_PSI_2PC},     //
        MemoryTaskTestParams{{3, 0}, 0, spu::psi::PsiType::KKRT_PSI_2PC},     //
        MemoryTaskTestParams{{0, 0}, 0, spu::psi::PsiType::KKRT_PSI_2PC},     //
        MemoryTaskTestParams{{3, 0}, 0, spu::psi::PsiType::BC22_PSI_2PC},     //
        MemoryTaskTestParams{{0, 0}, 0, spu::psi::PsiType::BC22_PSI_2PC},     //
        MemoryTaskTestParams{{4, 3, 0}, 0, spu::psi::PsiType::ECDH_PSI_3PC},  //
        MemoryTaskTestParams{
            {4, 3, 0, 6}, 0, spu::psi::PsiType::ECDH_PSI_NPC},  //

        //
        MemoryTaskTestParams{{20, 20}, 10, spu::psi::PsiType::KKRT_PSI_2PC},  //
        MemoryTaskTestParams{{20, 17}, 10, spu::psi::PsiType::KKRT_PSI_2PC},  //
        MemoryTaskTestParams{{17, 20}, 10, spu::psi::PsiType::KKRT_PSI_2PC},  //
        MemoryTaskTestParams{{33, 45}, 20, spu::psi::PsiType::ECDH_PSI_2PC},  //
        MemoryTaskTestParams{
            {100, 100}, 30, spu::psi::PsiType::BC22_PSI_2PC},  //
        MemoryTaskTestParams{
            {200, 100}, 60, spu::psi::PsiType::BC22_PSI_2PC},  //
        MemoryTaskTestParams{
            {100, 200}, 50, spu::psi::PsiType::BC22_PSI_2PC},  //

        MemoryTaskTestParams{
            {20, 17, 14}, 10, spu::psi::PsiType::ECDH_PSI_3PC},  //
        MemoryTaskTestParams{
            {20, 17, 14, 30}, 10, spu::psi::PsiType::ECDH_PSI_NPC},  //
        MemoryTaskTestParams{
            {20, 17, 14, 30, 35}, 11, spu::psi::PsiType::KKRT_PSI_NPC}));

struct FailedTestParams {
  size_t party_num;
  size_t receiver_rank;
  spu::psi::PsiType psi_protocol;
};

class MemoryTaskPsiTestFailedTest
    : public testing::TestWithParam<FailedTestParams> {};

TEST_P(MemoryTaskPsiTestFailedTest, FailedWorks) {
  auto params = GetParam();

  auto lctxs = yacl::link::test::SetupWorld(params.party_num);

  spu::psi::MemoryPsiConfig config;
  config.set_psi_type(params.psi_protocol);
  config.set_receiver_rank(params.receiver_rank);
  config.set_broadcast_result(true);

  ASSERT_ANY_THROW(MemoryPsi ctx(config, lctxs[0]));
}

INSTANTIATE_TEST_SUITE_P(FailedWorks_Instances, MemoryTaskPsiTestFailedTest,
                         testing::Values(
                             // invalid link world size
                             FailedTestParams{3, 0, PsiType::KKRT_PSI_2PC},
                             FailedTestParams{4, 0, PsiType::ECDH_PSI_2PC},
                             FailedTestParams{1, 0, PsiType::BC22_PSI_2PC},
                             FailedTestParams{2, 0, PsiType::ECDH_PSI_3PC},
                             // invalid receiver_rank
                             FailedTestParams{3, 4, PsiType::ECDH_PSI_3PC},
                             FailedTestParams{2, 5, PsiType::KKRT_PSI_2PC},
                             // invalid psi_type
                             FailedTestParams{3, 4,
                                              PsiType::INVALID_PSI_TYPE}));

}  // namespace spu::psi