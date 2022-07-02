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

#include "spu/psi/core/ecdh_oprf_psi.h"

#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/pseudo_random_generator.h"
#include "yasl/link/test_util.h"

#include "spu/psi/cryptor/ecdh_oprf/ecdh_oprf_selector.h"
#include "spu/psi/provider/batch_provider_impl.h"
#include "spu/psi/store/cipher_store_impl.h"

namespace spu {

namespace {
std::vector<std::string> GetIntersection(
    absl::Span<const std::string> items_a,
    absl::Span<const std::string> items_b) {
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
struct TestParams {
  size_t items_size;
  CurveType curve_type = CurveType::CurveFourQ;
};

class BasicEcdhOprfTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(BasicEcdhOprfTest, Works) {
  auto params = GetParam();
  auto ctxs = yasl::link::test::SetupWorld(2);
  std::random_device rd;
  yasl::PseudoRandomGenerator<uint64_t> prg(rd());

  std::vector<std::string> items_a_vec(params.items_size);
  std::vector<std::string> items_b_vec(params.items_size);

  for (size_t idx = 0; idx < params.items_size; ++idx) {
    items_a_vec[idx].resize(kEccKeySize);
    items_b_vec[idx].resize(kEccKeySize);
    prg.Fill(absl::MakeSpan(items_a_vec[idx]));
    prg.Fill(absl::MakeSpan(items_b_vec[idx]));
  }

  // items_a and items_b get 1/3 intersection
  for (size_t idx = params.items_size / 3; idx < params.items_size * 2 / 3;
       ++idx) {
    items_b_vec[idx] = items_a_vec[idx];
  }

  spu::psi::EcdhOprfPsiOptions server_options;
  spu::psi::EcdhOprfPsiOptions client_options;

  server_options.link0 = ctxs[0];
  server_options.link1 = ctxs[0]->Spawn();
  server_options.curve_type = params.curve_type;

  client_options.link0 = ctxs[1];
  client_options.link1 = ctxs[1]->Spawn();
  client_options.curve_type = params.curve_type;

  // todo spu not support now
  // server_options.link0->SetThrottleWindowSize(server_options.window_size);
  // client_options.link0->SetThrottleWindowSize(server_options.window_size);

  std::shared_ptr<psi::EcdhOprfPsiServer> dh_oprf_psi_server_offline =
      std::make_shared<psi::EcdhOprfPsiServer>(server_options);
  std::shared_ptr<psi::EcdhOprfPsiClient> dh_oprf_psi_client_offline =
      std::make_shared<psi::EcdhOprfPsiClient>(client_options);

  //
  // save server side private key for online use
  //
  std::array<uint8_t, kEccKeySize> server_private_key =
      dh_oprf_psi_server_offline->GetPrivateKey();

  // server input
  std::shared_ptr<spu::psi::IBatchProvider> batch_provider_server =
      std::make_shared<spu::psi::MemoryBatchProvider>(items_a_vec);

  // server output
  auto memory_store_server = std::make_shared<spu::psi::MemoryCipherStore>();
  // client output
  auto memory_store_client = std::make_shared<spu::psi::MemoryCipherStore>();

  //
  // offline phase:  FullEvaluate server's data and store
  //
  dh_oprf_psi_server_offline->FullEvaluate(batch_provider_server,
                                           memory_store_server);

  std::vector<std::string> &server_evaluate_items =
      memory_store_server->self_results();
  //
  // shuffle server side FullEvaluated data
  //
  std::mt19937 rng(rd());
  std::shuffle(server_evaluate_items.begin(), server_evaluate_items.end(), rng);

  //
  // offline phase:  server send FullEvaluated data to client
  //
  std::future<void> f_sever_send_fullevaluate = std::async([&] {
    std::vector<std::string> &server_stored_items =
        memory_store_server->self_results();

    std::shared_ptr<spu::psi::IBatchProvider> batch_server_evaluate_items =
        std::make_shared<spu::psi::MemoryBatchProvider>(server_stored_items);

    return dh_oprf_psi_server_offline->SendFinalEvaluatedItems(
        batch_server_evaluate_items);
  });

  std::future<void> f_client_recv_full_evaluate = std::async([&] {
    dh_oprf_psi_client_offline->RecvFinalEvaluatedItems(memory_store_client);
  });

  f_sever_send_fullevaluate.get();
  f_client_recv_full_evaluate.get();

  // online phase
  /*
  auto ctxs_online = yasl::link::test::SetupWorld(2);

  server_options.link0 = ctxs_online[0];
  server_options.link1 = ctxs_online[0]->Spawn();
  server_options.curve_type = params.curve_type;

  client_options.link0 = ctxs_online[1];
  client_options.link1 = ctxs_online[1]->Spawn();
  client_options.curve_type = params.curve_type;
  */

  // online server, load private key saved by offline phase
  std::shared_ptr<psi::EcdhOprfPsiServer> dh_oprf_psi_server_online =
      std::make_shared<psi::EcdhOprfPsiServer>(server_options,
                                               server_private_key);

  std::shared_ptr<psi::EcdhOprfPsiClient> dh_oprf_psi_client_online =
      std::make_shared<psi::EcdhOprfPsiClient>(client_options);

  std::future<void> f_sever_recv_blind = std::async(
      [&] { dh_oprf_psi_server_online->RecvBlindAndSendEvaluate(); });

  std::future<void> f_client_send_blind = std::async([&] {
    std::shared_ptr<spu::psi::IBatchProvider> batch_provider_client =
        std::make_shared<spu::psi::MemoryBatchProvider>(items_b_vec);

    dh_oprf_psi_client_online->SendBlindedItems(batch_provider_client);
  });

  std::future<void> f_client_recv_evaluate = std::async([&] {
    std::shared_ptr<spu::psi::IBatchProvider> batch_provider_client =
        std::make_shared<spu::psi::MemoryBatchProvider>(items_b_vec);

    dh_oprf_psi_client_online->RecvEvaluatedItems(batch_provider_client,
                                                  memory_store_client);
  });

  f_sever_recv_blind.get();
  f_client_send_blind.get();
  f_client_recv_evaluate.get();

  std::vector<std::string> &client_peer_evaluate_items =
      memory_store_client->peer_results();

  std::vector<std::string> &client_self_evaluate_items =
      memory_store_client->self_results();

  std::sort(client_peer_evaluate_items.begin(),
            client_peer_evaluate_items.end());

  // intersection
  std::vector<std::string> intersection;
  for (size_t index = 0; index < client_self_evaluate_items.size(); ++index) {
    if (std::binary_search(client_peer_evaluate_items.begin(),
                           client_peer_evaluate_items.end(),
                           client_self_evaluate_items[index])) {
      YASL_ENFORCE(index < items_b_vec.size());
      intersection.push_back(items_b_vec[index]);
    }
  }

  std::vector<std::string> intersection_std =
      GetIntersection(items_a_vec, items_b_vec);

  EXPECT_EQ(intersection_std, intersection);
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, BasicEcdhOprfTest,
    testing::Values(
        // CurveFourQ
        TestParams{1},      //
        TestParams{10},     //
        TestParams{50},     //
        TestParams{4095},   // less than one batch
        TestParams{4096},   // exactly one batch
        TestParams{10000},  // more than one batch
        // CurveSm2
        TestParams{1000, CurveType::CurveSm2},  // more than one batch
        // Curve256k1
        TestParams{1000, CurveType::CurveSecp256k1}  // more than one batch
        ));
}  // namespace spu
