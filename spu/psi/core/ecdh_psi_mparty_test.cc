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

#include "spu/psi/core/ecdh_psi_mparty.h"

#include <future>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/link/test_util.h"

#include "spu/psi/provider/batch_provider_impl.h"
#include "spu/psi/store/cipher_store_impl.h"

struct TestParams {
  std::vector<std::string> items_a;
  std::vector<std::string> items_b;
  std::vector<std::string> items_c;
  std::vector<std::string> items_d;
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
    const std::vector<std::string>& items_a,
    const std::vector<std::string>& items_b) {
  std::set<std::string> set(items_a.begin(), items_a.end());
  std::vector<std::string> ret;
  for (const auto& s : items_b) {
    if (set.count(s) != 0) {
      ret.push_back(s);
    }
  }
  return ret;
}

}  // namespace

namespace spu::psi {

//
//  test case for 2 party
//     alice           bob
//  x^a  |      x^a     |
//       |   -------->  | x^a^b
//       |      x^a^b   |
//       |   <--------  |
// ===========================
//       |              |
//       |      y^b     | y^b
//       |   <--------  |
// y^b^a |              |
//       |              |
// alice calc x^a^b y^b^a intersection
//

class EcdhPsi2PartyTest : public testing::TestWithParam<TestParams> {};

TEST_P(EcdhPsi2PartyTest, Works) {
  auto params = GetParam();

  auto link_ab = yasl::link::test::SetupWorld("ab", 2);

  std::vector<std::shared_ptr<yasl::link::Context>> link_ab2(2);
  link_ab2[0] = link_ab[0]->Spawn();
  link_ab2[1] = link_ab[1]->Spawn();

  size_t alice_rank = 0;
  size_t bob_rank = 1;

  auto memory_store_a = std::make_shared<MemoryCipherStore>();
  auto memory_store_b = std::make_shared<MemoryCipherStore>();

  EcdhPsiMParty role_a(params.items_a, memory_store_a);
  EcdhPsiMParty role_b(params.items_b, memory_store_b);

  // a->b->a  x^a->x^a^b

  std::future<void> f_mask_self_a = std::async(
      [&] { return role_a.RunMaskSelfAndSend(link_ab[alice_rank], bob_rank); });
  std::future<void> f_dual_mask_a = std::async([&] {
    return role_b.RunMaskRecvAndForward(link_ab[bob_rank], alice_rank,
                                        alice_rank, kFinalCompareBytes);
  });
  std::future<void> f_dual_mask_a_recv = std::async([&] {
    return role_a.RunRecvAndStore(link_ab[alice_rank], bob_rank,
                                  kFinalCompareBytes);
  });

  // b->a  y^b
  std::future<void> f_mask_self_b = std::async([&] {
    return role_b.RunMaskSelfAndSend(link_ab2[bob_rank], alice_rank);
  });
  std::future<void> f_dual_mask_b = std::async([&] {
    return role_a.RunMaskRecvAndStore(link_ab2[alice_rank], bob_rank,
                                      kFinalCompareBytes);
  });

  f_mask_self_a.get();
  f_dual_mask_a.get();
  f_dual_mask_a_recv.get();

  f_mask_self_b.get();
  f_dual_mask_b.get();

  std::vector<std::string>& self_result_a = memory_store_a->self_results();
  std::vector<std::string>& peer_result_a = memory_store_a->peer_results();
  std::vector<std::string>& self_result_b = memory_store_b->self_results();
  std::vector<std::string>& peer_result_b = memory_store_b->peer_results();

  EXPECT_EQ(self_result_a.size(), params.items_a.size());
  EXPECT_EQ(peer_result_a.size(), params.items_b.size());
  EXPECT_EQ(self_result_b.size(), 0);
  EXPECT_EQ(peer_result_b.size(), 0);

  std::vector<std::string> intersection_ab;

  std::sort(peer_result_a.begin(), peer_result_a.end());

  for (uint32_t index = 0; index < self_result_a.size(); index++) {
    if (std::binary_search(peer_result_a.begin(), peer_result_a.end(),
                           self_result_a[index])) {
      EXPECT_TRUE(index < params.items_a.size());
      intersection_ab.push_back(params.items_a[index]);
    }
  }

  auto intersection_std_ab = GetIntersection(params.items_a, params.items_b);

  EXPECT_EQ(intersection_std_ab.size(), intersection_ab.size());
}

//
//  test case for 3 party
//     alice           bob                    candy
// step 1 =============================================
//       |              | shuffle b items       |
//  x^a  |      x^a     |                       |
//       |   -------->  | x^a^b                 |
//       |              |                       |
//       |      y^b     | y^b                   |
//       |   <--------  |                       |
// y^b^a |              |                       |
// step 2 ==============================================
//       |              | shuffle x^a^b         |
//       |      x^a^b   |                       |
//       |   <--------  |                       |
// calc intersection_ab |                       |
//       |            intersection_ab           |
//       |   ---------------------------------> | {intersection_ab}^c
// step 3 ==============================================
//       |                 z^c                  |
//       |   <--------------------------------- |
// z^c^a |              |                       |
//       |    z^c^a     |                       |
//       |  -------->   |                       |
//       |              | calc {z^c^a}^b        |
//       |              |  send z^c^a^b         |
//       |              |  ------------------>  |
//       |              |                       |
//                                      calc intersection_abc
class EcdhPsi3PartyTest : public testing::TestWithParam<TestParams> {};

TEST_P(EcdhPsi3PartyTest, Works) {
  auto params = GetParam();

  auto link_ab = yasl::link::test::SetupWorld("ab", 2);
  auto link_abc = yasl::link::test::SetupWorld("abc", 3);

  size_t alice_rank = 1;
  size_t bob_rank = 2;
  size_t candy_rank = 0;

  std::random_device rd;
  std::mt19937 g(rd());
  // shuffle { y }
  std::shuffle(params.items_b.begin(), params.items_b.end(), g);

  auto memory_store_a = std::make_shared<MemoryCipherStore>();
  auto memory_store_b = std::make_shared<MemoryCipherStore>();
  auto memory_store_c = std::make_shared<MemoryCipherStore>();

  EcdhPsiMParty role_a(params.items_a, memory_store_a);
  EcdhPsiMParty role_b(params.items_b, memory_store_b);
  EcdhPsiMParty role_c(params.items_c, memory_store_c);

  // a->b  x^a
  // b->a  y^b
  std::future<void> f_mask_self_a = std::async([&] {
    return role_a.RunMaskSelfAndSend(link_ab[0], link_ab[0]->NextRank());
  });
  std::future<void> f_mask_self_b = std::async([&] {
    return role_b.RunMaskSelfAndSend(link_ab[1], link_ab[1]->NextRank());
  });
  std::future<void> f_dual_mask_a = std::async([&] {
    return role_a.RunMaskRecvAndStore(link_ab[0], link_ab[0]->NextRank(),
                                      kHashSize);
  });
  std::future<void> f_dual_mask_b = std::async([&] {
    return role_b.RunMaskRecvAndStore(link_ab[1], link_ab[1]->NextRank(),
                                      kHashSize);
  });

  // c->a->b->c
  std::future<void> f_abc_c_send = std::async([&] {
    return role_c.RunMaskSelfAndSend(link_abc[candy_rank], alice_rank);
  });

  std::future<void> f_abc_a = std::async([&] {
    return role_a.RunMaskRecvAndForward(link_abc[alice_rank], candy_rank,
                                        bob_rank, kHashSize);
  });
  std::future<void> f_abc_b = std::async([&] {
    return role_b.RunMaskRecvAndForward(link_abc[bob_rank], alice_rank,
                                        candy_rank, kFinalCompareBytes);
  });
  std::future<void> f_abc_c_recv = std::async([&] {
    return role_c.RunRecvAndStore(link_abc[candy_rank], bob_rank,
                                  kFinalCompareBytes);
  });

  f_mask_self_a.get();
  f_mask_self_b.get();
  f_dual_mask_a.get();
  f_dual_mask_b.get();

  f_abc_c_send.get();
  f_abc_a.get();
  f_abc_b.get();
  f_abc_c_recv.get();

  std::vector<std::string>& self_result_a = memory_store_a->self_results();
  std::vector<std::string>& peer_result_a = memory_store_a->peer_results();
  std::vector<std::string>& self_result_b = memory_store_b->self_results();
  std::vector<std::string>& peer_result_b = memory_store_b->peer_results();
  std::vector<std::string>& self_result_c = memory_store_c->self_results();
  std::vector<std::string>& peer_result_c = memory_store_c->peer_results();

  EXPECT_EQ(self_result_a.size(), 0);
  EXPECT_EQ(self_result_b.size(), 0);
  EXPECT_EQ(self_result_c.size(), params.items_c.size());

  EXPECT_EQ(peer_result_a.size(), params.items_b.size());
  EXPECT_EQ(peer_result_b.size(), params.items_a.size());
  EXPECT_EQ(peer_result_c.size(), 0);

  // shuffle x^a^b, use sort repalce shuffle
  std::sort(peer_result_b.begin(), peer_result_b.end());

  std::shared_ptr<IBatchProvider> shuffle_batch_provider =
      std::make_shared<MemoryBatchProvider>(peer_result_b);

  // send x^a^b to alice
  std::future<void> f_send_shuffle_dual_mask_b = std::async([&] {
    return role_b.RunSendBatch(link_ab[1], link_ab[1]->NextRank(),
                               shuffle_batch_provider);
  });
  // recv x^a^b from bob
  std::future<void> f_recv_shuffle_dual_mask_a = std::async([&] {
    return role_a.RunRecvAndStore(link_ab[0], link_ab[0]->NextRank(),
                                  kHashSize);
  });

  f_send_shuffle_dual_mask_b.get();
  f_recv_shuffle_dual_mask_a.get();

  // std::set_intersection requires sorted inputs.
  std::sort(self_result_a.begin(), self_result_a.end());
  std::sort(peer_result_a.begin(), peer_result_a.end());

  std::vector<std::string> intersection_ab;
  std::set_intersection(self_result_a.begin(), self_result_a.end(),
                        peer_result_a.begin(), peer_result_a.end(),
                        std::back_inserter(intersection_ab));

  // shuffle x y intersection, use sort replace shuffle
  std::sort(intersection_ab.begin(), intersection_ab.end());
  std::shared_ptr<IBatchProvider> ab_batch_provider =
      std::make_shared<MemoryBatchProvider>(intersection_ab);

  // alice send alice and bob intersecton to candy, shuffle{x^a^b}
  // candy receive and mask, get {x^a^b^c}
  std::future<void> f_ac_a = std::async([&] {
    return role_a.RunSendBatch(link_abc[alice_rank], candy_rank,
                               ab_batch_provider);
  });
  std::future<void> f_ac_c = std::async([&] {
    return role_c.RunMaskRecvAndStore(link_abc[candy_rank], alice_rank,
                                      kFinalCompareBytes);
  });
  f_ac_a.get();
  f_ac_c.get();

  std::vector<std::string> intersection_abc;

  std::sort(peer_result_c.begin(), peer_result_c.end());

  for (uint32_t index = 0; index < self_result_c.size(); index++) {
    if (std::binary_search(peer_result_c.begin(), peer_result_c.end(),
                           self_result_c[index])) {
      EXPECT_TRUE(index < params.items_c.size());
      intersection_abc.push_back(params.items_c[index]);
    }
  }

  auto intersection_std_ab = GetIntersection(params.items_a, params.items_b);

  auto intersection_std_abc =
      GetIntersection(intersection_std_ab, params.items_c);

  EXPECT_EQ(intersection_std_ab.size(), intersection_ab.size());

  EXPECT_EQ(intersection_std_abc.size(), intersection_abc.size());
  if (intersection_std_abc.size() > 0) {
    EXPECT_EQ(intersection_std_abc, intersection_abc);
  }
}

//
//  test case for 4 party
//     alice           bob                    candy                  david
// step 1 ==============================================================
//       |              | shuffle b items       |                      |
//  x^a  |      x^a     |                       |                      |
//       |   -------->  | x^a^b                 |                      |
//       |              |                       |                      |
//       |      y^b     | y^b                   |                      |
//       |   <--------  |                       |                      |
// y^b^a |              |                       |                      |
// step 2 =============================================================|
//       |              | shuffle x^a^b         |                      |
//       |      x^a^b   |                       |                      |
//       |   <--------  |                       |                      |
// calc intersection_ab |                       |                      |
//       |            intersection_ab           |                      |
//       |   ---------------------------------> | {intersection_ab}^c  |
// step 3 =============================================================|
//       |                 z^c                  |                      |
//       |   <--------------------------------- |                      |
// z^c^a |              |                       |                      |
//       |    z^c^a     |                       |                      |
//       |  -------->   |                       |                      |
//       |              | calc {z^c^a}^b        |                      |
//       |              | **shuffle{z^c^a^b}    |                      |
//       |              |  ------------------>  |                      |
//       |              |                       |                      |
//                                      calc intersection_abc          |
//       |              |                       |   intersection_abc   |
//       |              |                       |  ------------------> |
//       |              |                  d calc {intersection_abc}^d |
// step 4 =============================================================|
//       |                         u^d                                 |
//       |  <--------------------------------------------------------- |
//       |    u^d^a     |                       |                      |
//       |  ----------> |      u^d^a^b          |                      |
//       |              |---------------------> |                      |
//       |              |                       | calc {u^d^a^b}^c     |
//       |              |                       |      u^d^a^b^c       |
//       |              |                       | ------------------>  |
//       |              |                   d  calc intersection_abcd  |
//
class EcdhPsi4PartyTest : public testing::TestWithParam<TestParams> {};

TEST_P(EcdhPsi4PartyTest, Works) {
  auto params = GetParam();

  auto link_ab = yasl::link::test::SetupWorld("ab", 2);
  auto link_abc = yasl::link::test::SetupWorld("abc", 3);
  auto link_abcd = yasl::link::test::SetupWorld("abcd", 4);

  size_t alice_rank = 1;
  size_t bob_rank = 2;
  size_t candy_rank = 0;
  size_t david_rank = 3;

  std::random_device rd;
  std::mt19937 g(rd());
  // shuffle { y }
  std::shuffle(params.items_b.begin(), params.items_b.end(), g);

  auto memory_store_a = std::make_shared<MemoryCipherStore>();
  auto memory_store_b = std::make_shared<MemoryCipherStore>();
  auto memory_store_c = std::make_shared<MemoryCipherStore>();
  auto memory_store_d = std::make_shared<MemoryCipherStore>();

  EcdhPsiMParty role_a(params.items_a, memory_store_a);
  EcdhPsiMParty role_b(params.items_b, memory_store_b);
  EcdhPsiMParty role_c(params.items_c, memory_store_c);
  EcdhPsiMParty role_d(params.items_d, memory_store_d);

  // a->b  x^a
  // b->a  y^b
  std::future<void> f_mask_self_a = std::async([&] {
    return role_a.RunMaskSelfAndSend(link_ab[0], link_ab[0]->NextRank());
  });
  std::future<void> f_mask_self_b = std::async([&] {
    return role_b.RunMaskSelfAndSend(link_ab[1], link_ab[1]->NextRank());
  });
  std::future<void> f_dual_mask_a = std::async([&] {
    return role_a.RunMaskRecvAndStore(link_ab[0], link_ab[0]->NextRank(),
                                      kHashSize);
  });
  std::future<void> f_dual_mask_b = std::async([&] {
    return role_b.RunMaskRecvAndStore(link_ab[1], link_ab[1]->NextRank(),
                                      kHashSize);
  });

  f_mask_self_a.get();
  f_mask_self_b.get();
  f_dual_mask_a.get();
  f_dual_mask_b.get();

  std::vector<std::string>& self_result_a = memory_store_a->self_results();
  std::vector<std::string>& peer_result_a = memory_store_a->peer_results();
  std::vector<std::string>& self_result_b = memory_store_b->self_results();
  std::vector<std::string>& peer_result_b = memory_store_b->peer_results();
  std::vector<std::string>& self_result_c = memory_store_c->self_results();
  std::vector<std::string>& peer_result_c = memory_store_c->peer_results();
  std::vector<std::string>& self_result_d = memory_store_d->self_results();
  std::vector<std::string>& peer_result_d = memory_store_d->peer_results();

  // shuffle x^a^b, use sort repalce shuffle
  std::sort(peer_result_b.begin(), peer_result_b.end());

  std::shared_ptr<IBatchProvider> batch_provider =
      std::make_shared<MemoryBatchProvider>(peer_result_b);

  // send x^a^b to alice
  std::future<void> f_send_shuffle_dual_mask_b = std::async([&] {
    return role_b.RunSendBatch(link_ab[1], link_ab[1]->NextRank(),
                               batch_provider);
  });
  // recv x^a^b from bob
  std::future<void> f_recv_shuffle_dual_mask_a = std::async([&] {
    return role_a.RunRecvAndStore(link_ab[0], link_ab[0]->NextRank(),
                                  kHashSize);
  });

  f_send_shuffle_dual_mask_b.get();
  f_recv_shuffle_dual_mask_a.get();

  // std::set_intersection requires sorted inputs.
  std::sort(self_result_a.begin(), self_result_a.end());
  std::sort(peer_result_a.begin(), peer_result_a.end());

  std::vector<std::string> intersection_ab;
  std::set_intersection(self_result_a.begin(), self_result_a.end(),
                        peer_result_a.begin(), peer_result_a.end(),
                        std::back_inserter(intersection_ab));

  // shuffle x y intersection, use sort replace shuffle
  std::sort(intersection_ab.begin(), intersection_ab.end());
  std::shared_ptr<IBatchProvider> xy_batch_provider =
      std::make_shared<MemoryBatchProvider>(intersection_ab);

  // alice send alice and bob intersecton to candy, shuffle{x^a^b}
  // candy receive and mask, get {x^a^b^c}
  std::future<void> f_ac_a = std::async([&] {
    return role_a.RunSendBatch(link_abc[alice_rank], candy_rank,
                               xy_batch_provider);
  });
  std::future<void> f_ac_c = std::async([&] {
    return role_c.RunMaskRecvAndStore(link_abc[candy_rank], alice_rank,
                                      kHashSize);
  });
  f_ac_a.get();
  f_ac_c.get();

  // clear peer b items(x^ab), used for store z^cab
  peer_result_b.clear();
  // c->a->b
  std::future<void> f_abc_c_send = std::async([&] {
    return role_c.RunMaskSelfAndSend(link_abc[candy_rank], alice_rank);
  });

  std::future<void> f_abc_a = std::async([&] {
    return role_a.RunMaskRecvAndForward(link_abc[alice_rank], candy_rank,
                                        bob_rank, kHashSize);
  });
  std::future<void> f_abc_b = std::async([&] {
    return role_b.RunMaskRecvAndStore(link_abc[bob_rank], alice_rank,
                                      kHashSize);
  });
  f_abc_c_send.get();
  f_abc_a.get();
  f_abc_b.get();

  // shuffle z^c^a^b, use sort repalce shuffle
  std::sort(peer_result_b.begin(), peer_result_b.end());

  std::shared_ptr<IBatchProvider> candy_batch_provider =
      std::make_shared<MemoryBatchProvider>(peer_result_b);

  // send z^c^a^b to candy
  std::future<void> f_send_shuffle_mask_abc_b = std::async([&] {
    return role_b.RunSendBatch(link_abc[bob_rank], candy_rank,
                               candy_batch_provider);
  });
  // recv z^c^a^b from bob
  std::future<void> f_recv_shuffle_mask_abc_c = std::async([&] {
    return role_c.RunRecvAndStore(link_abc[candy_rank], bob_rank, kHashSize);
  });

  f_send_shuffle_mask_abc_b.get();
  f_recv_shuffle_mask_abc_c.get();

  // get a b c intersection
  std::vector<std::string> intersection_abc;

  std::sort(self_result_c.begin(), self_result_c.end());
  std::sort(peer_result_c.begin(), peer_result_c.end());

  std::set_intersection(self_result_c.begin(), self_result_c.end(),
                        peer_result_c.begin(), peer_result_c.end(),
                        std::back_inserter(intersection_abc));

  // candy send intersection_abc to david
  // david receive and mask, get {x^a^b^c}^d
  std::shared_ptr<IBatchProvider> abc_batch_provider =
      std::make_shared<MemoryBatchProvider>(intersection_abc);

  std::future<void> f_abc_c = std::async([&] {
    return role_c.RunSendBatch(link_abcd[candy_rank], david_rank,
                               abc_batch_provider);
  });
  std::future<void> f_abc_d = std::async([&] {
    return role_d.RunMaskRecvAndStore(link_abcd[david_rank], candy_rank,
                                      kFinalCompareBytes);
  });
  f_abc_c.get();
  f_abc_d.get();

  // d->a->b->c->d
  std::future<void> f_abcd_d_send = std::async([&] {
    return role_d.RunMaskSelfAndSend(link_abcd[david_rank], alice_rank);
  });

  std::future<void> f_abcd_a = std::async([&] {
    return role_a.RunMaskRecvAndForward(link_abcd[alice_rank], david_rank,
                                        bob_rank, kHashSize);
  });
  std::future<void> f_abcd_b = std::async([&] {
    return role_b.RunMaskRecvAndForward(link_abcd[bob_rank], alice_rank,
                                        candy_rank, kHashSize);
  });
  std::future<void> f_abcd_c = std::async([&] {
    return role_c.RunMaskRecvAndForward(link_abcd[candy_rank], bob_rank,
                                        david_rank, kFinalCompareBytes);
  });
  std::future<void> f_abcd_d_recv = std::async([&] {
    return role_d.RunRecvAndStore(link_abcd[david_rank], candy_rank,
                                  kFinalCompareBytes);
  });

  f_abcd_d_send.get();
  f_abcd_a.get();
  f_abcd_b.get();
  f_abcd_c.get();
  f_abcd_d_recv.get();

  std::vector<std::string> intersection_abcd;

  std::sort(peer_result_d.begin(), peer_result_d.end());

  for (uint32_t index = 0; index < self_result_d.size(); index++) {
    if (std::binary_search(peer_result_d.begin(), peer_result_d.end(),
                           self_result_d[index])) {
      EXPECT_TRUE(index < params.items_d.size());
      intersection_abcd.push_back(params.items_d[index]);
    }
  }

  auto intersection_std_ab = GetIntersection(params.items_a, params.items_b);

  auto intersection_std_abc =
      GetIntersection(intersection_std_ab, params.items_c);
  auto intersection_std_abcd =
      GetIntersection(intersection_std_abc, params.items_d);

  EXPECT_EQ(self_result_a.size(), params.items_a.size());
  EXPECT_EQ(self_result_b.size(), 0);
  EXPECT_EQ(self_result_c.size(), params.items_c.size());
  EXPECT_EQ(self_result_d.size(), params.items_d.size());

  EXPECT_EQ(peer_result_a.size(), params.items_b.size());
  EXPECT_EQ(peer_result_b.size(), params.items_c.size());
  EXPECT_EQ(peer_result_c.size(), intersection_std_ab.size());
  EXPECT_EQ(peer_result_d.size(), intersection_std_abc.size());

  EXPECT_EQ(intersection_std_ab.size(), intersection_ab.size());
  EXPECT_EQ(intersection_std_abc.size(), intersection_abc.size());
  EXPECT_EQ(intersection_std_abcd.size(), intersection_abcd.size());

  if (intersection_std_abcd.size() > 0) {
    EXPECT_EQ(intersection_std_abcd, intersection_abcd);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, EcdhPsi2PartyTest,
    testing::Values(
        TestParams{{"a", "b"}, {"b", "c"}, {}, {}},  //

        TestParams{{"a", "b"}, {"b", "c"}, {}, {}},  //
        //
        TestParams{{"a", "b"}, {"c", "d"}, {}, {}},  //
        TestParams{{"a", "b"}, {"c", "d"}, {}, {}},  //

        //
        TestParams{{}, {"a"}, {}, {}},  //
        TestParams{{"a"}, {}, {}, {}},  //
        //
        // less than one batch
        TestParams{
            CreateRangeItems(0, 4095), CreateRangeItems(1, 4095), {}, {}},  //

        // exactly one batch
        TestParams{
            CreateRangeItems(0, 4096), CreateRangeItems(1, 4096), {}, {}},  //

        // more than one batch
        TestParams{
            CreateRangeItems(0, 8193), CreateRangeItems(5, 8193), {}, {}},  //
        //
        TestParams{{}, {}, {}, {}}  //
        ));

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, EcdhPsi3PartyTest,
    testing::Values(TestParams{{"a", "b"}, {"b", "c"}, {"b", "d"}, {}},  //

                    TestParams{{"a", "b"}, {"b", "c"}, {"c", "d"}, {}},  //
                    //
                    TestParams{{"a", "b"}, {"c", "d"}, {"d", "e"}, {}},  //
                    TestParams{{"a", "b"}, {"c", "d"}, {"e", "f"}, {}},  //

                    //
                    TestParams{{}, {"a"}, {}, {}},  //
                    TestParams{{"a"}, {}, {}, {}},  //
                    TestParams{{}, {}, {"a"}, {}},  //
                    //
                    // less than one batch
                    TestParams{CreateRangeItems(0, 4095),
                               CreateRangeItems(1, 4095),
                               CreateRangeItems(2, 4095),
                               {}},  //

                    // exactly one batch
                    TestParams{CreateRangeItems(0, 4096),
                               CreateRangeItems(1, 4096),
                               CreateRangeItems(2, 4096),
                               {}},  //

                    // more than one batch
                    TestParams{CreateRangeItems(0, 8193),
                               CreateRangeItems(5, 8193),
                               CreateRangeItems(10, 8193),
                               {}},  //
                    //
                    TestParams{{}, {}, {}, {}}  //
                    ));

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, EcdhPsi4PartyTest,
    testing::Values(
        TestParams{{"a", "b"}, {"b", "c"}, {"b", "d"}, {"b", "e"}},  //

        TestParams{{"a", "b"}, {"b", "c"}, {"c", "d"}, {"d", "e"}},  //
        //
        TestParams{{"a", "b"}, {"c", "d"}, {"d", "e"}, {"e", "f"}},  //
        TestParams{{"a", "b"}, {"c", "d"}, {"e", "f"}, {"g", "h"}},  //

        //
        TestParams{{}, {"a"}, {}, {}},  //
        TestParams{{"a"}, {}, {}, {}},  //
        TestParams{{}, {}, {"a"}, {}},  //
        TestParams{{}, {}, {}, {"a"}},  //
        //
        // less than one batch
        TestParams{CreateRangeItems(0, 4095), CreateRangeItems(1, 4095),
                   CreateRangeItems(2, 4095), CreateRangeItems(3, 4095)},  //
        // exactly one batch
        TestParams{CreateRangeItems(0, 4096), CreateRangeItems(1, 4096),
                   CreateRangeItems(2, 4096), CreateRangeItems(3, 4096)},  //

        // more than one batch
        TestParams{CreateRangeItems(0, 4097), CreateRangeItems(5, 4097),
                   CreateRangeItems(10, 4097), CreateRangeItems(15, 4097)},  //
        //
        TestParams{{}, {}, {}, {}}  //
        ));

}  // namespace spu::psi
