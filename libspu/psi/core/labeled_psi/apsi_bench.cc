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

#include <algorithm>
#include <chrono>
#include <future>
#include <random>
#include <string>

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/core/labeled_psi/sender.h"
#include "libspu/psi/core/labeled_psi/sender_kvdb.h"

namespace spu::psi {

namespace {

constexpr char kLinkAddrAB[] = "127.0.0.1:9532,127.0.0.1:9533";
constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;
constexpr uint32_t kLinkWindowSize = 16;

using duration_millis = std::chrono::duration<double, std::milli>;

constexpr size_t kPsiStartPos = 100;

std::vector<std::string> GenerateData(size_t seed, size_t item_count) {
  yacl::crypto::Prg<uint128_t> prg(seed);

  std::vector<std::string> items;

  for (size_t i = 0; i < item_count; ++i) {
    std::string item(16, '\0');
    prg.Fill(absl::MakeSpan(item.data(), item.length()));
    items.emplace_back(absl::BytesToHexString(item));
  }

  return items;
}

std::vector<std::string> GenerateSenderData(
    size_t seed, size_t item_count,
    const absl::Span<std::string>& receiver_items,
    std::vector<size_t>* intersection_idx) {
  std::vector<std::string> sender_items;

  yacl::crypto::Prg<uint128_t> prg(seed);

  for (size_t i = 0; i < item_count; ++i) {
    std::string item(16, '\0');
    prg.Fill(absl::MakeSpan(item.data(), item.size()));
    sender_items.emplace_back(absl::BytesToHexString(item));
  }

  for (size_t i = 0; i < receiver_items.size(); i += 3) {
    if ((kPsiStartPos + i * 5) >= sender_items.size()) {
      break;
    }
    sender_items[kPsiStartPos + i * 5] = receiver_items[i];
    (*intersection_idx).emplace_back(i);
  }

  return sender_items;
}

std::shared_ptr<yacl::link::Context> CreateContext(
    int self_rank, yacl::link::ContextDesc& lctx_desc) {
  std::shared_ptr<yacl::link::Context> link_ctx;

  yacl::link::FactoryBrpc factory;
  link_ctx = factory.CreateContext(lctx_desc, self_rank);
  link_ctx->ConnectToMesh();

  return link_ctx;
}

std::vector<std::shared_ptr<yacl::link::Context>> CreateLinks(
    std::string host_str) {
  std::vector<std::string> hosts = absl::StrSplit(host_str, ',');
  yacl::link::ContextDesc lctx_desc;
  for (size_t rank = 0; rank < hosts.size(); rank++) {
    const std::string id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, hosts[rank]});
  }

  auto proc = [&](int self_randk) -> std::shared_ptr<yacl::link::Context> {
    return CreateContext(self_randk, lctx_desc);
  };

  size_t world_size = hosts.size();
  std::vector<std::future<std::shared_ptr<yacl::link::Context>>> f_links(
      world_size);
  for (size_t i = 0; i < world_size; i++) {
    f_links[i] = std::async(proc, i);
  }

  std::vector<std::shared_ptr<yacl::link::Context>> links(world_size);
  for (size_t i = 0; i < world_size; i++) {
    links[i] = f_links[i].get();
  }

  return links;
}

}  // namespace

static void BM_LabeledPsi(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t nr = state.range(0);
    size_t ns = state.range(1);

    auto ctxs = CreateLinks(kLinkAddrAB);

    ctxs[0]->SetThrottleWindowSize(kLinkWindowSize);
    ctxs[1]->SetThrottleWindowSize(kLinkWindowSize);

    ctxs[0]->SetRecvTimeout(kLinkRecvTimeout);
    ctxs[1]->SetRecvTimeout(kLinkRecvTimeout);

    state.ResumeTiming();

    apsi::PSIParams psi_params = spu::psi::GetPsiParams(nr, ns);

    // step 1: PsiParams Request and Response
    std::future<void> f_sender_params =
        std::async([&] { return LabelPsiSender::RunPsiParams(ns, ctxs[0]); });

    std::future<apsi::PSIParams> f_receiver_params = std::async(
        [&] { return LabelPsiReceiver::RequestPsiParams(nr, ctxs[1]); });

    f_sender_params.get();
    apsi::PSIParams psi_params2 = f_receiver_params.get();

    EXPECT_EQ(psi_params.table_params().table_size,
              psi_params2.table_params().table_size);

    size_t item_count = ns;
    size_t nonce_byte_count = 16;

    std::random_device rd;
    yacl::crypto::Prg<uint128_t> prg(rd());

    std::array<uint8_t, 32> oprf_key;
    prg.Fill(absl::MakeSpan(oprf_key));

    bool compressed = false;
    std::shared_ptr<spu::psi::ISenderDB> sender_db =
        std::make_shared<spu::psi::SenderKvDB>(psi_params, oprf_key, "::memory",
                                               0, nonce_byte_count, compressed);

    std::vector<std::string> receiver_items = GenerateData(rd(), nr);

    std::vector<size_t> intersection_idx;
    std::vector<std::string> intersection_label;

    // step 2: set database

    const auto setdb_start = std::chrono::system_clock::now();

    std::vector<std::string> sender_items = GenerateSenderData(
        rd(), item_count, absl::MakeSpan(receiver_items), &intersection_idx);

    std::shared_ptr<IBatchProvider> batch_provider =
        std::make_shared<MemoryBatchProvider>(sender_items);

    sender_db->SetData(batch_provider);

    const auto setdb_end = std::chrono::system_clock::now();
    const duration_millis setdb_duration = setdb_end - setdb_start;
    SPDLOG_INFO("*** step2 set db duration:{}", setdb_duration.count());

    EXPECT_EQ(ns, sender_db->GetItemCount());

    SPDLOG_INFO("after set db, bin_bundle_count:{}, packing_rate:{}",
                sender_db->GetBinBundleCount(), sender_db->GetPackingRate());

    std::unique_ptr<spu::psi::IEcdhOprfServer> oprf_server =
        spu::psi::CreateEcdhOprfServer(oprf_key, spu::psi::OprfType::Basic,
                                       spu::psi::CurveType::CURVE_FOURQ);

    LabelPsiSender sender(sender_db);

    LabelPsiReceiver receiver(psi_params, false);

    // step 3: oprf request and response

    const auto oprf_start = std::chrono::system_clock::now();

    std::future<void> f_sender_oprf = std::async(
        [&] { return sender.RunOPRF(std::move(oprf_server), ctxs[0]); });

    std::future<
        std::pair<std::vector<apsi::HashedItem>, std::vector<apsi::LabelKey>>>
        f_receiver_oprf = std::async(
            [&] { return receiver.RequestOPRF(receiver_items, ctxs[1]); });

    f_sender_oprf.get();
    std::pair<std::vector<apsi::HashedItem>, std::vector<apsi::LabelKey>>
        oprf_pair = f_receiver_oprf.get();

    const auto oprf_end = std::chrono::system_clock::now();
    const duration_millis oprf_duration = oprf_end - oprf_start;
    SPDLOG_INFO("*** step3 oprf duration:{}", oprf_duration.count());

    SPDLOG_INFO("hashed_item size:{} label keys size:{}",
                oprf_pair.first.size(), oprf_pair.second.size());

    // step 4: Query request and response

    const auto query_start = std::chrono::system_clock::now();

    std::future<void> f_sender_query =
        std::async([&] { return sender.RunQuery(ctxs[0]); });

    std::future<std::pair<std::vector<size_t>, std::vector<std::string>>>
        f_receiver_query = std::async([&] {
          return receiver.RequestQuery(oprf_pair.first, oprf_pair.second,
                                       ctxs[1]);
        });

    f_sender_query.get();
    std::pair<std::vector<size_t>, std::vector<std::string>> query_result =
        f_receiver_query.get();

    const auto query_end = std::chrono::system_clock::now();
    const duration_millis query_duration = query_end - query_start;
    SPDLOG_INFO("*** step4 query duration:{}", query_duration.count());

    EXPECT_EQ(query_result.first.size(), intersection_idx.size());
    SPDLOG_INFO("index vec size:{} intersection_idx size:{}",
                query_result.first.size(), intersection_idx.size());

    SPDLOG_INFO("intersection:{}", intersection_idx.size());
    auto stats0 = ctxs[0]->GetStats();
    auto stats1 = ctxs[1]->GetStats();
    SPDLOG_INFO("sender ctx0 sent_bytes:{} recv_bytes:{}", stats0->sent_bytes,
                stats0->recv_bytes);
    SPDLOG_INFO("receiver ctx1 sent_bytes:{} recv_bytes:{}", stats1->sent_bytes,
                stats1->recv_bytes);
  }
}

// [256k, 512k, 1m, 2m, 4m, 8m, 16m]
BENCHMARK(BM_LabeledPsi)
    ->Unit(benchmark::kMillisecond)
    ->Args({100, 256 << 10})
    ->Args({100, 512 << 10})
    ->Args({100, 1 << 20})
    ->Args({100, 2 << 20})
    ->Args({100, 4 << 20})
    ->Args({100, 8 << 20})
    ->Args({100, 16 << 20})
    ->Args({100, 1 << 25})
    ->Args({100, 1 << 26})
    ->Args({100, 1 << 27});

BENCHMARK_MAIN();

}  // namespace spu::psi
