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

#include "libspu/psi/core/ecdh_3pc_psi.h"

#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/utils/test_utils.h"

struct TestParams {
  std::vector<std::string> items_a;
  std::vector<std::string> items_b;
  std::vector<std::string> items_c;
};

namespace spu::psi::test {

class Ecdh3PcPsiTest : public testing::TestWithParam<TestParams> {};

TEST_P(Ecdh3PcPsiTest, MaskMaster) {
  auto params = GetParam();

  auto link_abc = yacl::link::test::SetupWorld("abc", 3);

  size_t alice_rank = 1;
  size_t bob_rank = 2;
  size_t candy_rank = 0;

  size_t master_rank = alice_rank;

  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master;
  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master_next;
  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master_prev;

  ShuffleEcdh3PcPsi::Options opts;
  opts.link_ctx = link_abc[alice_rank];
  opts.master_rank = master_rank;
  ecdh_3pc_psi_master = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  opts.link_ctx = link_abc[bob_rank];
  ecdh_3pc_psi_master_next = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  opts.link_ctx = link_abc[candy_rank];
  ecdh_3pc_psi_master_prev = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  std::vector<std::string> master_res;
  std::vector<std::string> master_next_res;
  std::vector<std::string> master_prev_res;

  auto master_runner = std::async([&] {
    return ecdh_3pc_psi_master->MaskMaster(params.items_a, &master_res);
  });
  auto master_next_runner = std::async([&] {
    return ecdh_3pc_psi_master_next->MaskMaster(params.items_b,
                                                &master_next_res);
  });
  auto master_prev_runner = std::async([&] {
    return ecdh_3pc_psi_master_prev->MaskMaster(params.items_c,
                                                &master_prev_res);
  });

  master_next_runner.get();
  master_prev_runner.get();
  master_runner.get();

  // check correctness
  EXPECT_EQ(master_res.size(), params.items_a.size());
  EXPECT_EQ(master_next_res.size(), 0);
  EXPECT_EQ(master_prev_res.size(), 0);
}

TEST_P(Ecdh3PcPsiTest, PartnersPsi) {
  auto params = GetParam();

  auto link_abc = yacl::link::test::SetupWorld("abc", 3);

  size_t alice_rank = 1;
  size_t bob_rank = 2;
  size_t candy_rank = 0;

  size_t master_rank = alice_rank;

  auto intersection_std_bc = GetIntersection(params.items_b, params.items_c);

  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master;
  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master_next;
  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master_prev;

  ShuffleEcdh3PcPsi::Options opts;
  opts.link_ctx = link_abc[alice_rank];
  opts.master_rank = master_rank;
  ecdh_3pc_psi_master = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  opts.link_ctx = link_abc[bob_rank];
  ecdh_3pc_psi_master_next = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  opts.link_ctx = link_abc[candy_rank];
  ecdh_3pc_psi_master_prev = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  std::vector<std::string> master_res;
  std::vector<std::string> master_next_res;
  std::vector<std::string> master_prev_res;

  auto master_runner = std::async([&] {
    return ecdh_3pc_psi_master->PartnersPsi(params.items_a, &master_res);
  });
  auto master_next_runner = std::async([&] {
    return ecdh_3pc_psi_master_next->PartnersPsi(params.items_b,
                                                 &master_next_res);
  });
  auto master_prev_runner = std::async([&] {
    return ecdh_3pc_psi_master_prev->PartnersPsi(params.items_c,
                                                 &master_prev_res);
  });

  master_next_runner.get();
  master_prev_runner.get();
  master_runner.get();

  // check correctness
  EXPECT_EQ(master_res.size(), intersection_std_bc.size());
  EXPECT_EQ(master_next_res.size(), 0);
  EXPECT_EQ(master_prev_res.size(), 0);
}

TEST_P(Ecdh3PcPsiTest, Works) {
  auto params = GetParam();

  auto link_abc = yacl::link::test::SetupWorld("abc", 3);

  size_t alice_rank = 1;
  size_t bob_rank = 2;
  size_t candy_rank = 0;

  size_t master_rank = alice_rank;

  auto intersection_std_ab = GetIntersection(params.items_a, params.items_b);
  auto intersection_std_bc = GetIntersection(params.items_b, params.items_c);
  auto intersection_std_abc =
      GetIntersection(intersection_std_ab, params.items_c);

  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master;
  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master_next;
  std::shared_ptr<ShuffleEcdh3PcPsi> ecdh_3pc_psi_master_prev;

  ShuffleEcdh3PcPsi::Options opts;
  opts.link_ctx = link_abc[alice_rank];
  opts.master_rank = master_rank;
  ecdh_3pc_psi_master = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  opts.link_ctx = link_abc[bob_rank];
  ecdh_3pc_psi_master_next = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  opts.link_ctx = link_abc[candy_rank];
  ecdh_3pc_psi_master_prev = std::make_shared<ShuffleEcdh3PcPsi>(opts);

  // simple runner
  auto psi_func = [&](const std::shared_ptr<ShuffleEcdh3PcPsi>& handler,
                      const std::vector<std::string>& items,
                      std::vector<std::string>* results) {
    std::vector<std::string> masked_master_items;
    std::vector<std::string> partner_psi_items;

    auto mask_master = std::async(
        [&] { return handler->MaskMaster(items, &masked_master_items); });
    auto partner_psi = std::async(
        [&] { return handler->PartnersPsi(items, &partner_psi_items); });

    mask_master.get();
    partner_psi.get();

    handler->FinalPsi(items, masked_master_items, partner_psi_items, results);
  };

  std::vector<std::string> master_res;
  std::vector<std::string> master_next_res;
  std::vector<std::string> master_prev_res;

  auto master_runner = std::async(
      [&] { psi_func(ecdh_3pc_psi_master, params.items_a, &master_res); });
  auto master_next_runner = std::async([&] {
    psi_func(ecdh_3pc_psi_master_next, params.items_b, &master_next_res);
  });
  auto master_prev_runner = std::async([&] {
    psi_func(ecdh_3pc_psi_master_prev, params.items_c, &master_prev_res);
  });

  master_runner.get();
  master_next_runner.get();
  master_prev_runner.get();

  // check correctness
  EXPECT_EQ(master_res.size(), intersection_std_abc.size());
  EXPECT_EQ(master_next_res.size(), 0);
  EXPECT_EQ(master_prev_res.size(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, Ecdh3PcPsiTest,
    testing::Values(
        TestParams{{"a", "b"}, {"b", "c"}, {"b", "d"}},  //
        TestParams{{"a", "b"}, {"b", "c"}, {"b", "d"}},  //

        TestParams{{"a", "b"}, {"b", "c"}, {"c", "d"}},  //
        //
        TestParams{{"a", "b"}, {"c", "d"}, {"d", "e"}},  //
        TestParams{{"a", "b"}, {"c", "d"}, {"e", "f"}},  //

        //
        TestParams{{}, {"a"}, {}},  //
        TestParams{{"a"}, {}, {}},  //
        TestParams{{}, {}, {"a"}},  //
        //
        // less than one batch
        TestParams{CreateRangeItems(0, 4095), CreateRangeItems(1, 4095),
                   CreateRangeItems(2, 4095)},  //

        // exactly one batch
        TestParams{CreateRangeItems(0, 4096), CreateRangeItems(1, 4096),
                   CreateRangeItems(2, 4096)},  //
        // more than one batch
        TestParams{CreateRangeItems(0, 8193), CreateRangeItems(5, 8193),
                   CreateRangeItems(10, 8193)},  //
        //
        TestParams{{}, {}, {}}  //
        ));

}  // namespace spu::psi::test