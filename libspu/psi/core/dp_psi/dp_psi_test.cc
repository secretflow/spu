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

#include "libspu/psi/core/dp_psi/dp_psi.h"

#include <random>
#include <set>

#include "absl/container/flat_hash_set.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

namespace spu::psi {

namespace {

std::vector<std::string> CreateRangeItems(size_t begin, size_t size) {
  std::vector<std::string> ret;
  for (size_t i = 0; i < size; i++) {
    ret.push_back(std::to_string(begin + i));
  }
  return ret;
}

std::vector<size_t> GetIntersectionIdx(
    const std::vector<std::string>& items_a,
    const std::vector<std::string>& items_b) {
  absl::flat_hash_set<std::string> set(items_a.begin(), items_a.end());
  std::vector<size_t> ret;
  for (size_t idx = 0; idx < items_b.size(); ++idx) {
    if (set.count(items_b[idx]) != 0) {
      ret.push_back(idx);
    }
  }
  return ret;
}

std::vector<std::string> GetIntersection(
    const std::vector<std::string>& items_a,
    const std::vector<std::string>& items_b) {
  absl::flat_hash_set<std::string> set(items_a.begin(), items_a.end());
  std::vector<std::string> ret;
  for (const auto& s : items_b) {
    if (set.count(s) != 0) {
      ret.push_back(s);
    }
  }
  return ret;
}

struct TestParams {
  size_t items_size;
  DpPsiOptions options;
};

constexpr double kIntersectionRatio = 0.7;

}  // namespace

class DpPsiTest : public testing::TestWithParam<TestParams> {};

TEST_P(DpPsiTest, Works) {
  const auto& param = GetParam();

  SPDLOG_INFO("param.items_size: {}", param.items_size);

  size_t items_size = param.items_size;

  auto link_ctxs = yacl::link::test::SetupWorld(2);

  std::vector<std::string> items_a = CreateRangeItems(0, items_size);
  std::vector<std::string> items_b =
      CreateRangeItems(items_size * (1 - kIntersectionRatio), items_size);

  size_t alice_rank = 0;
  size_t bob_rank = 1;

  size_t alice_sub_sample_size = 0;
  size_t alice_up_sample_size = 0;
  size_t bob_sub_sample_size = 0;

  std::future<size_t> f_dp_psi_a = std::async([&] {
    return RunDpEcdhPsiAlice(param.options, link_ctxs[alice_rank], items_a,
                             &alice_sub_sample_size, &alice_up_sample_size);
  });

  std::future<std::vector<size_t>> f_dp_psi_b = std::async([&] {
    return RunDpEcdhPsiBob(param.options, link_ctxs[bob_rank], items_b,
                           &bob_sub_sample_size);
  });

  size_t alice_intersection_size = f_dp_psi_a.get();
  std::vector<size_t> dp_psi_result = f_dp_psi_b.get();

  EXPECT_EQ(alice_intersection_size, dp_psi_result.size());

  SPDLOG_INFO(
      "alice_intersection_size:{} "
      "alice_sub_sample_size:{},alice_up_sample_size:{}",
      alice_intersection_size, alice_sub_sample_size, alice_up_sample_size);

  SPDLOG_INFO("bob intersection size:{}, bob_sub_sample_size:{}",
              dp_psi_result.size(), bob_sub_sample_size);

  std::vector<std::string> real_intersection =
      GetIntersection(items_a, items_b);
  std::vector<size_t> real_intersection_idx =
      GetIntersectionIdx(items_a, items_b);

  SPDLOG_INFO("items_size: {} ,real_intersection.size(): {}", items_size,
              real_intersection.size());

  auto stats0 = link_ctxs[alice_rank]->GetStats();
  auto stats1 = link_ctxs[bob_rank]->GetStats();

  double total_comm_bytes = stats0->sent_bytes + stats0->recv_bytes;
  SPDLOG_INFO("bob: sent_bytes:{} recv_bytes:{}", stats1->sent_bytes,
              stats1->recv_bytes);

  total_comm_bytes /= 1024 * 1024;

  SPDLOG_INFO("total_comm_bytes: {} MB", total_comm_bytes);
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, DpPsiTest,
                         testing::Values(                       //
                             TestParams{20, DpPsiOptions(0.8)}  // dummy
                             )                                  //
);

}  // namespace spu::psi
