// Copyright 2023 zhangwfjh
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

#include "libspu/psi/core/fnp04_mp_psi/fnp04_mp_psi.h"

#include <random>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/utils/test_utils.h"

namespace spu::psi {

namespace {

struct FnpMpTestParams {
  std::vector<size_t> item_size;
  size_t intersection_size;
};

std::vector<std::vector<std::string>> CreateNPartyItems(
    const FnpMpTestParams& params) {
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

class FnpMpPsiTest : public testing::TestWithParam<FnpMpTestParams> {};

TEST_P(FnpMpPsiTest, Works) {
  std::vector<std::vector<std::string>> items;

  auto params = GetParam();
  items = CreateNPartyItems(params);
  size_t leader_rank = 0;

  auto ctxs = yacl::link::test::SetupWorld(params.item_size.size());

  auto proc = [&](int idx) -> std::vector<std::string> {
    Party::Options opts;
    opts.link_ctx = ctxs[idx];
    opts.leader_rank = leader_rank;
    Party op(opts);
    return op.Run(items[idx]);
  };

  size_t world_size = ctxs.size();
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
    if (i == leader_rank) {
      std::sort(results[i].begin(), results[i].end());
      EXPECT_EQ(results[i].size(), intersection.size());
      EXPECT_EQ(results[i], intersection);
    } else {
      EXPECT_EQ(results[i].size(), 0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, FnpMpPsiTest,
    testing::Values(FnpMpTestParams{{0, 3}, 0},                 //
                    FnpMpTestParams{{3, 0}, 0},                 //
                    FnpMpTestParams{{0, 0}, 0},                 //
                    FnpMpTestParams{{4, 3}, 2},                 //
                    FnpMpTestParams{{20, 17, 14}, 10},          //
                    FnpMpTestParams{{20, 17, 14, 30}, 10},      //
                    FnpMpTestParams{{20, 17, 14, 30, 35}, 11},  //
                    FnpMpTestParams{{20, 17, 14, 30, 35}, 0}));

}  // namespace spu::psi