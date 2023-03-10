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

#include "libspu/psi/core/ecdh_oprf_psi.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/link/test_util.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/io/io.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/cipher_store.h"
#include "libspu/psi/utils/test_utils.h"

namespace spu::psi {
namespace {

void WriteCsvFile(const std::string &file_name,
                  const std::vector<std::string> &items) {
  auto out = io::BuildOutputStream(io::FileIoOptions(file_name));
  out->Write("id\n");
  for (const auto &data : items) {
    out->Write(fmt::format("{}\n", data));
  }
  out->Close();
}

}  // namespace

struct TestParams {
  size_t items_size;
  CurveType curve_type = CurveType::CURVE_FOURQ;
  bool shuffle_online = false;
};

class BasicEcdhOprfTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(BasicEcdhOprfTest, Works) {
  auto params = GetParam();
  auto ctxs = yacl::link::test::SetupWorld(2);

  std::vector<std::string> items_a =
      test::CreateRangeItems(0, params.items_size);
  std::vector<std::string> items_b = test::CreateRangeItems(
      params.items_size / 4,
      std::max(static_cast<size_t>(1), params.items_size / 2));

  auto timestamp_str = std::to_string(absl::ToUnixNanos(absl::Now()));
  // server input
  auto server_input_path =
      std::filesystem::path(fmt::format("server-input-{}", timestamp_str));
  // client input
  auto client_input_path =
      std::filesystem::path(fmt::format("client-input-{}", timestamp_str));

  // server output
  auto server_output_path =
      std::filesystem::path(fmt::format("server-output-{}", timestamp_str));
  // client output
  auto client_output_path =
      std::filesystem::path(fmt::format("client-output-{}", timestamp_str));
  // server output
  auto server_tmp_cache_path =
      std::filesystem::path(fmt::format("tmp-cache-{}", timestamp_str));

  // register remove of temp file.
  ON_SCOPE_EXIT([&] {
    std::error_code ec;
    std::filesystem::remove(server_input_path, ec);
    if (ec.value() != 0) {
      SPDLOG_WARN("can not remove tmp file: {}, msg: {}",
                  server_input_path.c_str(), ec.message());
    }
    std::filesystem::remove(client_input_path, ec);
    if (ec.value() != 0) {
      SPDLOG_WARN("can not remove tmp file: {}, msg: {}",
                  client_input_path.c_str(), ec.message());
    }
    std::filesystem::remove(server_tmp_cache_path, ec);
    if (ec.value() != 0) {
      SPDLOG_WARN("can not remove tmp file: {}, msg: {}",
                  server_tmp_cache_path.c_str(), ec.message());
    }
  });

  WriteCsvFile(server_input_path.string(), items_a);
  WriteCsvFile(client_input_path.string(), items_b);

  spu::psi::EcdhOprfPsiOptions server_options;
  spu::psi::EcdhOprfPsiOptions client_options;

  server_options.link0 = ctxs[0];
  server_options.link1 = ctxs[0]->Spawn();
  server_options.curve_type = params.curve_type;

  client_options.link0 = ctxs[1];
  client_options.link1 = ctxs[1]->Spawn();
  client_options.curve_type = params.curve_type;

  // todo psi not support now
  // server_options.link0->SetThrottleWindowSize(server_options.window_size);
  // client_options.link0->SetThrottleWindowSize(server_options.window_size);

  std::shared_ptr<EcdhOprfPsiServer> dh_oprf_psi_server_offline =
      std::make_shared<EcdhOprfPsiServer>(server_options);
  std::shared_ptr<EcdhOprfPsiClient> dh_oprf_psi_client_offline =
      std::make_shared<EcdhOprfPsiClient>(client_options);

  //
  // save server side private key for online use
  //
  std::array<uint8_t, kEccKeySize> server_private_key =
      dh_oprf_psi_server_offline->GetPrivateKey();

  // server batch provider
  std::vector<std::string> cloumn_ids = {"id"};
  std::shared_ptr<CachedCsvBatchProvider> batch_provider_server =
      std::make_shared<CachedCsvBatchProvider>(server_input_path.string(),
                                               cloumn_ids, 100000, true);

  // server output
  auto memory_store_server = std::make_shared<MemoryCipherStore>();
  // client output
  auto memory_store_client = std::make_shared<MemoryCipherStore>();

  //
  // offline phase:  FullEvaluate server's data and store
  //
  std::shared_ptr<IUbPsiCache> ub_cache = std::make_shared<UbPsiCache>(
      server_tmp_cache_path.string(),
      dh_oprf_psi_server_offline->GetCompareLength(), cloumn_ids);

  dh_oprf_psi_server_offline->FullEvaluate(batch_provider_server, ub_cache);

  SPDLOG_INFO("Finished FullEvaluate");
  //
  // offline phase:  server send FullEvaluated data to client
  //
  std::future<size_t> f_sever_send_fullevaluate = std::async([&] {
    std::shared_ptr<IBatchProvider> batch_provider =
        std::make_shared<UbPsiCacheProvider>(
            server_tmp_cache_path.string(),
            dh_oprf_psi_server_offline->GetCompareLength());

    return dh_oprf_psi_server_offline->SendFinalEvaluatedItems(batch_provider);
  });

  std::future<void> f_client_recv_full_evaluate = std::async([&] {
    dh_oprf_psi_client_offline->RecvFinalEvaluatedItems(memory_store_client);
  });

  f_sever_send_fullevaluate.get();
  f_client_recv_full_evaluate.get();

  std::vector<std::string> intersection_std =
      test::GetIntersection(items_a, items_b);

  SPDLOG_INFO("a:{} b:{} intersection_std:{}", items_a.size(), items_b.size(),
              intersection_std.size());

  // online phase
  // online server, load private key saved by offline phase
  std::shared_ptr<EcdhOprfPsiServer> dh_oprf_psi_server_online =
      std::make_shared<EcdhOprfPsiServer>(server_options, server_private_key);

  if (params.shuffle_online) {
    std::vector<uint8_t> client_private_key =
        yacl::crypto::RandBytes(kEccKeySize);

    std::shared_ptr<EcdhOprfPsiClient> dh_oprf_psi_client_online =
        std::make_shared<EcdhOprfPsiClient>(client_options, client_private_key);

    std::future<void> f_sever_recv_blind_shuffle = std::async(
        [&] { dh_oprf_psi_server_online->RecvBlindAndShuffleSendEvaluate(); });

    std::future<void> f_client_send_blind = std::async([&] {
      std::shared_ptr<IBatchProvider> batch_provider_client =
          std::make_shared<CsvBatchProvider>(client_input_path.string(),
                                             cloumn_ids);

      dh_oprf_psi_client_online->SendBlindedItems(batch_provider_client);
      dh_oprf_psi_client_online->RecvEvaluatedItems(memory_store_client);
    });

    f_sever_recv_blind_shuffle.get();
    f_client_send_blind.get();

    std::vector<std::string> &client_peer_evaluate_items =
        memory_store_client->peer_results();

    std::vector<std::string> &client_self_evaluate_items =
        memory_store_client->self_results();

    std::sort(client_peer_evaluate_items.begin(),
              client_peer_evaluate_items.end());

    for (size_t index = 0; index < client_self_evaluate_items.size(); ++index) {
      SPDLOG_INFO("{}: {} {}", index,
                  absl::BytesToHexString(client_self_evaluate_items[index]),
                  client_self_evaluate_items[index].length());
    }

    // intersection
    std::vector<std::string> intersection;
    for (size_t index = 0; index < client_self_evaluate_items.size(); ++index) {
      if (std::binary_search(client_peer_evaluate_items.begin(),
                             client_peer_evaluate_items.end(),
                             client_self_evaluate_items[index])) {
        intersection.push_back(client_self_evaluate_items[index]);
      }
    }

    SPDLOG_INFO("intersection: {}", intersection.size());

    std::future<void> f_send_intersection_mask = std::async([&] {
      std::shared_ptr<IBatchProvider> batch_provider =
          std::make_shared<MemoryBatchProvider>(intersection);
      dh_oprf_psi_client_online->SendIntersectionMaskedItems(batch_provider);
    });

    std::future<std::pair<std::vector<uint64_t>, size_t>>
        f_recv_intersection_mask = std::async([&] {
          std::shared_ptr<IShuffleBatchProvider> batch_provider =
              std::make_shared<UbPsiCacheProvider>(
                  server_tmp_cache_path.string(),
                  dh_oprf_psi_server_offline->GetCompareLength());
          return dh_oprf_psi_server_online->RecvIntersectionMaskedItems(
              batch_provider, 30000);
        });

    f_send_intersection_mask.get();
    std::vector<uint64_t> indices;
    size_t server_items_size;
    std::tie(indices, server_items_size) = f_recv_intersection_mask.get();
    SPDLOG_INFO("indices: {} server_items_size:{}", indices.size(),
                server_items_size);
  } else {
    std::shared_ptr<EcdhOprfPsiClient> dh_oprf_psi_client_online =
        std::make_shared<EcdhOprfPsiClient>(client_options);

    std::future<void> f_sever_recv_blind = std::async(
        [&] { dh_oprf_psi_server_online->RecvBlindAndSendEvaluate(); });

    std::future<void> f_client_send_blind = std::async([&] {
      std::shared_ptr<IBatchProvider> batch_provider_client =
          std::make_shared<CsvBatchProvider>(client_input_path.string(),
                                             cloumn_ids);

      dh_oprf_psi_client_online->SendBlindedItems(batch_provider_client);
    });

    std::future<void> f_client_recv_evaluate = std::async([&] {
      dh_oprf_psi_client_online->RecvEvaluatedItems(memory_store_client);
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
        SPU_ENFORCE(index < items_b.size());
        intersection.push_back(items_b[index]);
      }
    }

    EXPECT_EQ(intersection_std, intersection);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, BasicEcdhOprfTest,
    testing::Values(
        // CURVE_FOURQ
        TestParams{1},      //
        TestParams{10},     //
        TestParams{50},     //
        TestParams{4095},   // less than one batch
        TestParams{4096},   // exactly one batch
        TestParams{10000},  // more than one batch
        // CURVE_SM2
        TestParams{1000, CurveType::CURVE_SM2},  // more than one batch
        // Curve256k1
        TestParams{1000, CurveType::CURVE_SECP256K1}  // more than one batch
        )                                             //
);

}  // namespace spu::psi
