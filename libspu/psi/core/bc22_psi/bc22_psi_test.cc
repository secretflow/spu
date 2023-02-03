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

#include "libspu/psi/core/bc22_psi/bc22_psi.h"

#include <algorithm>
#include <future>
#include <random>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"
#include "yacl/utils/parallel.h"

namespace spu::psi {

namespace {

std::vector<std::string> CreateRangeItems(size_t start_pos, size_t size) {
  std::vector<std::string> ret(size);

  auto gen_items_proc = [&](size_t begin, size_t end) -> void {
    for (size_t i = begin; i < end; ++i) {
      ret[i] = std::to_string(start_pos + i);
    }
  };

  std::future<void> f_gen = std::async(gen_items_proc, size / 2, size);

  gen_items_proc(0, size / 2);

  f_gen.get();

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

// Special test data
// [20000000, ] [20000001, ]
// 5560969 和 21906809 2-hash simple-hash conflict
// constexpr size_t kTestItemsSize = 9000000;
//
// constexpr size_t kTestItemsSize = 10000000;

}  // namespace

class PcgPsiTest : public testing::TestWithParam<size_t> {};

TEST_P(PcgPsiTest, Works) {
  auto params = GetParam();
  size_t items_size = params;
  auto ctxs = yacl::link::test::SetupWorld(2);

  std::vector<std::string> alice_data = CreateRangeItems(20000000, items_size);
  std::vector<std::string> bob_data = CreateRangeItems(20000001, items_size);

  std::vector<std::string> intersection_std =
      GetIntersection(alice_data, bob_data);

  std::sort(intersection_std.begin(), intersection_std.end());

  Bc22PcgPsi pcg_psi_send(ctxs[0], PsiRoleType::Sender);
  Bc22PcgPsi pcg_psi_recv(ctxs[1], PsiRoleType::Receiver);

  std::future<void> send_thread =
      std::async([&] { pcg_psi_send.RunPsi(alice_data); });

  std::future<void> recv_thread =
      std::async([&] { return pcg_psi_recv.RunPsi(bob_data); });

  send_thread.get();
  recv_thread.get();
  std::vector<std::string> intersection = pcg_psi_recv.GetIntersection();

  std::sort(intersection.begin(), intersection.end());

  size_t std_size = intersection_std.size();
  size_t size2 = intersection.size();
  SPDLOG_INFO("intersection_std size:{} intersection_size:{} diff: {}",
              std_size, size2, (std_size - size2));

  EXPECT_EQ(intersection, intersection_std);

  auto stats0 = ctxs[0]->GetStats();
  auto stats1 = ctxs[1]->GetStats();
  SPDLOG_INFO("sender ctx0 sent_bytes:{} recv_bytes:{}", stats0->sent_bytes,
              stats0->recv_bytes);
  SPDLOG_INFO("receiver ctx1 sent_bytes:{} recv_bytes:{}", stats1->sent_bytes,
              stats1->recv_bytes);
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, PcgPsiTest,
                         testing::Values(10000, 100000, 1000000));

}  // namespace spu::psi
