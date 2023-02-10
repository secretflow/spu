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

#include "libspu/psi/core/mini_psi.h"

#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/utils/test_utils.h"

struct TestParams {
  std::vector<std::string> items_a;
  std::vector<std::string> items_b;
  bool batch = false;
};

namespace spu::psi {
class MiniPsiTest : public testing::TestWithParam<TestParams> {};

TEST_P(MiniPsiTest, Works) {
  auto params = GetParam();

  auto link_ab = yacl::link::test::SetupWorld("mini", 2);

  auto intersection_std_ab =
      test::GetIntersection(params.items_a, params.items_b);

  std::future<void> f_send;
  std::future<std::vector<std::string>> f_recv;
  if (!params.batch) {
    f_send = std::async(
        [&] { return spu::psi::MiniPsiSend(link_ab[0], params.items_a); });

    f_recv = std::async(
        [&] { return spu::psi::MiniPsiRecv(link_ab[1], params.items_b); });
  } else {
    f_send = std::async(
        [&] { return spu::psi::MiniPsiSendBatch(link_ab[0], params.items_a); });

    f_recv = std::async(
        [&] { return spu::psi::MiniPsiRecvBatch(link_ab[1], params.items_b); });
  }
  f_send.get();
  auto intersection = f_recv.get();

  SPDLOG_INFO("{}:{}, intersection.size():{}", __func__, __LINE__,
              intersection.size());

  EXPECT_EQ(intersection, intersection_std_ab);
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, MiniPsiTest,
    testing::Values(TestParams{{"a", "b"}, {"b", "c"}},  //
                    TestParams{{"a", "b"}, {"c", "d"}},  //
                    //
                    TestParams{{}, {"a"}},  //
                    //
                    TestParams{{"a"}, {}},  //
                    // less than one batch
                    TestParams{test::CreateRangeItems(0, 1800),
                               test::CreateRangeItems(1, 1800), false},
                    // exactly one batch
                    TestParams{test::CreateRangeItems(0, 2000),
                               test::CreateRangeItems(1, 2000), true},  //
                    //
                    TestParams{{}, {}}  //
                    ));
}  // namespace spu::psi
