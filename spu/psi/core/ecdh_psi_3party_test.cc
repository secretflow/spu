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

#include "spu/psi/core/ecdh_psi_3party.h"

#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "yasl/base/exception.h"
#include "yasl/link/test_util.h"

#include "spu/psi/provider/batch_provider_impl.h"
#include "spu/psi/store/cipher_store_impl.h"

struct TestParams {
  std::vector<std::string> items_a;
  std::vector<std::string> items_b;
  std::vector<std::string> items_c;
};

namespace {

std::vector<std::string> CreateRangeItems(size_t begin, size_t size) {
  std::vector<std::string> ret;
  for (size_t i = 0; i < size; i++) {
    ret.push_back(std::to_string(begin + i));
  }
  return ret;
}

std::vector<std::string> GetIntersection(
    const std::vector<std::string> &items_a,
    const std::vector<std::string> &items_b) {
  std::set<std::string> set(items_a.begin(), items_a.end());
  std::vector<std::string> ret;
  for (const auto &s : items_b) {
    if (set.count(s) != 0) {
      ret.push_back(s);
    }
  }
  return ret;
}

}  // namespace

namespace spu::psi {
class EcdhPsi3PartyTest : public testing::TestWithParam<TestParams> {};

TEST_P(EcdhPsi3PartyTest, Works) {
  auto params = GetParam();

  auto link_abc = yasl::link::test::SetupWorld("abc", 3);

  auto memory_store_a = std::make_shared<MemoryCipherStore>();
  auto memory_store_b = std::make_shared<MemoryCipherStore>();
  auto memory_store_c = std::make_shared<MemoryCipherStore>();

  size_t alice_rank = 1;
  size_t bob_rank = 2;
  size_t candy_rank = 0;

  size_t master_rank = alice_rank;

  auto intersection_std_ab = GetIntersection(params.items_a, params.items_b);
  auto intersection_std_bc = GetIntersection(params.items_b, params.items_c);

  auto intersection_std_abc =
      GetIntersection(intersection_std_ab, params.items_c);

  std::vector<std::shared_ptr<MemoryCipherStore>> cipher_stores{
      memory_store_a, memory_store_b, memory_store_c};

  std::shared_ptr<psi::ShuffleEcdhPSI3Party> ecdh_3party_psi_master,
      ecdh_3party_psi_master_next, ecdh_3party_psi_master_prev;

  ecdh_3party_psi_master = std::make_shared<psi::ShuffleEcdhPSI3Party>(
      link_abc[alice_rank], master_rank, params.items_a,
      cipher_stores[alice_rank]);

  ecdh_3party_psi_master_next = std::make_shared<psi::ShuffleEcdhPSI3Party>(
      link_abc[bob_rank], master_rank, params.items_b, cipher_stores[bob_rank]);

  ecdh_3party_psi_master_prev = std::make_shared<psi::ShuffleEcdhPSI3Party>(
      link_abc[candy_rank], master_rank, params.items_c,
      cipher_stores[candy_rank]);

  std::future<void> f_dhpsi_3party_master_step1 =
      std::async([&] { return ecdh_3party_psi_master->RunEcdhPsiStep1(); });

  std::future<void> f_dhpsi_3party_slave1_step1 = std::async(
      [&] { return ecdh_3party_psi_master_next->RunEcdhPsiStep1(); });

  std::future<void> f_dhpsi_3party_slave2_step1 = std::async(
      [&] { return ecdh_3party_psi_master_prev->RunEcdhPsiStep1(); });

  f_dhpsi_3party_master_step1.get();
  f_dhpsi_3party_slave1_step1.get();
  f_dhpsi_3party_slave2_step1.get();

  std::vector<std::string> intersection_abc;

  //

  std::vector<std::string> &self_result_master =
      cipher_stores[master_rank]->self_results();
  std::vector<std::string> &peer_result_master =
      cipher_stores[master_rank]->peer_results();

  std::vector<std::string> &self_result_master_next =
      cipher_stores[link_abc[master_rank]->NextRank()]->self_results();
  std::vector<std::string> &peer_result_master_next =
      cipher_stores[link_abc[master_rank]->NextRank()]->peer_results();

  std::vector<std::string> &self_result_master_prev =
      cipher_stores[link_abc[master_rank]->PrevRank()]->self_results();
  std::vector<std::string> &peer_result_master_prev =
      cipher_stores[link_abc[master_rank]->PrevRank()]->peer_results();

  EXPECT_EQ(self_result_master.size(), params.items_a.size());
  EXPECT_EQ(peer_result_master.size(), 0);

  EXPECT_EQ(self_result_master_next.size(), 0);
  EXPECT_EQ(peer_result_master_next.size(), params.items_c.size());

  EXPECT_EQ(self_result_master_prev.size(), 0);
  EXPECT_EQ(peer_result_master_prev.size(), params.items_b.size());

  // step2 shuffle
  std::sort(peer_result_master_prev.begin(), peer_result_master_prev.end());
  std::shared_ptr<IBatchProvider> shuffle_batch_provider =
      std::make_shared<MemoryBatchProvider>(peer_result_master_prev);

  // send shuffled set
  std::future<void> f_dhpsi_3party_master_step2 =
      std::async([&] { return ecdh_3party_psi_master->RunEcdhPsiStep2(); });

  std::future<void> f_dhpsi_3party_slave1_step2 = std::async(
      [&] { return ecdh_3party_psi_master_next->RunEcdhPsiStep2(); });

  std::future<void> f_dhpsi_3party_slave2_step2 = std::async([&] {
    return ecdh_3party_psi_master_prev->RunEcdhPsiStep2(shuffle_batch_provider);
  });

  f_dhpsi_3party_master_step2.get();
  f_dhpsi_3party_slave1_step2.get();
  f_dhpsi_3party_slave2_step2.get();

  EXPECT_EQ(self_result_master_next.size(), params.items_b.size());
  EXPECT_EQ(peer_result_master_next.size(), params.items_c.size());

  // step3 get intersection and send to master
  std::sort(self_result_master_next.begin(), self_result_master_next.end());
  std::sort(peer_result_master_next.begin(), peer_result_master_next.end());

  std::vector<std::string> intersection_ab;
  std::set_intersection(
      self_result_master_next.begin(), self_result_master_next.end(),
      peer_result_master_next.begin(), peer_result_master_next.end(),
      std::back_inserter(intersection_ab));

  //
  std::shared_ptr<IBatchProvider> ab_batch_provider =
      std::make_shared<MemoryBatchProvider>(intersection_ab);

  std::future<void> f_dhpsi_3party_master_step3 =
      std::async([&] { return ecdh_3party_psi_master->RunEcdhPsiStep3(); });

  std::future<void> f_dhpsi_3party_slave1_step3 = std::async([&] {
    return ecdh_3party_psi_master_next->RunEcdhPsiStep3(ab_batch_provider);
  });

  std::future<void> f_dhpsi_3party_slave2_step3 = std::async(
      [&] { return ecdh_3party_psi_master_prev->RunEcdhPsiStep3(); });

  f_dhpsi_3party_master_step3.get();
  f_dhpsi_3party_slave1_step3.get();
  f_dhpsi_3party_slave2_step3.get();

  EXPECT_EQ(intersection_std_bc.size(), peer_result_master.size());

  std::sort(peer_result_master.begin(), peer_result_master.end());

  for (uint32_t index = 0; index < self_result_master.size(); index++) {
    if (std::binary_search(peer_result_master.begin(), peer_result_master.end(),
                           self_result_master[index])) {
      EXPECT_TRUE(index < params.items_a.size());
      intersection_abc.push_back(params.items_a[index]);
    }
  }

  // check correctness
  EXPECT_EQ(intersection_std_abc.size(), intersection_abc.size());
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, EcdhPsi3PartyTest,
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

// EcdhPsi3Party in memory case
class EcdhPsi3PartyMemoryTest : public testing::TestWithParam<TestParams> {};

TEST_P(EcdhPsi3PartyMemoryTest, Works) {
  auto params = GetParam();

  auto link_abc = yasl::link::test::SetupWorld("abc", 3);

  size_t alice_rank = 1;
  size_t bob_rank = 2;
  size_t candy_rank = 0;

  size_t master_rank = alice_rank;

  auto intersection_std_ab = GetIntersection(params.items_a, params.items_b);
  auto intersection_std_abc =
      GetIntersection(intersection_std_ab, params.items_c);

  std::future<std::vector<std::string>> f_dhpsi_3party_master = std::async([&] {
    return psi::RunShuffleEcdhPsi3Party(link_abc[alice_rank], master_rank,
                                        params.items_a);
  });

  std::future<std::vector<std::string>> f_dhpsi_3party_slave1 = std::async([&] {
    return psi::RunShuffleEcdhPsi3Party(link_abc[bob_rank], master_rank,
                                        params.items_b);
  });

  std::future<std::vector<std::string>> f_dhpsi_3party_slave2 = std::async([&] {
    return psi::RunShuffleEcdhPsi3Party(link_abc[candy_rank], master_rank,
                                        params.items_c);
  });

  std::vector<std::string> intersection_master = f_dhpsi_3party_master.get();
  std::vector<std::string> intersection_slave1 = f_dhpsi_3party_slave1.get();
  std::vector<std::string> intersection_slave2 = f_dhpsi_3party_slave2.get();

  EXPECT_EQ(intersection_master.size(), intersection_std_abc.size());
  EXPECT_EQ(intersection_slave1.size(), 0);
  EXPECT_EQ(intersection_slave1.size(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, EcdhPsi3PartyMemoryTest,
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

}  // namespace spu::psi