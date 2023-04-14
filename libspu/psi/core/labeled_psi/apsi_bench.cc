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
#include <future>
#include <random>
#include <string>

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "benchmark/benchmark.h"
#include "spdlog/spdlog.h"
#include "spu/psi/core/labeled_psi/psi_params.h"
#include "spu/psi/core/labeled_psi/receiver.h"
#include "spu/psi/core/labeled_psi/sender.h"
#include "spu/psi/cryptor/ecdh_oprf/ecdh_oprf_selector.h"
#include "yacl/link/test_util.h"

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

constexpr char kLinkAddrAB[] = "127.0.0.1:9532,127.0.0.1:9533";
constexpr uint32_t kLinkRecvTimeout = 30 * 60 * 1000;
constexpr uint32_t kLinkWindowSize = 16;

}  // namespace

static void BM_LabeledPsi(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);

    auto ctxs = CreateLinks(kLinkAddrAB);

    ctxs[0]->SetThrottleWindowSize(kLinkWindowSize);
    ctxs[1]->SetThrottleWindowSize(kLinkWindowSize);

    ctxs[0]->SetRecvTimeout(kLinkRecvTimeout);
    ctxs[1]->SetRecvTimeout(kLinkRecvTimeout);

    state.ResumeTiming();

    apsi::PSIParams psi_params = spu::psi::GetPsiParams(params.nr, params.ns);

    // step 1: PsiParams Request and Response
    std::future<void> f_sender_params = std::async(
        [&] { return LabelPsiSender::RunPsiParams(params.ns, ctxs[0]); });

    std::future<apsi::PSIParams> f_receiver_params = std::async(
        [&] { return LabelPsiReceiver::RequestPsiParams(params.nr, ctxs[1]); });

    f_sender_params.get();
    apsi::PSIParams psi_params2 = f_receiver_params.get();

    EXPECT_EQ(psi_params.table_params().table_size,
              psi_params2.table_params().table_size);

    size_t item_count = params.ns;
    size_t label_byte_count = params.label_bytes;
    size_t nonce_byte_count = 16;

    std::random_device rd;
    yacl::crypto::Prg<uint128_t> prg(rd());

    std::array<uint8_t, 32> oprf_key;
    prg.Fill(absl::MakeSpan(oprf_key));

    bool compressed = false;
    std::shared_ptr<spu::psi::SenderDB> sender_db =
        std::make_shared<spu::psi::SenderDB>(psi_params, oprf_key,
                                             label_byte_count, nonce_byte_count,
                                             compressed);

    std::vector<std::string> receiver_items = GenerateData(rd(), params.nr);

    std::vector<size_t> intersection_idx;
    std::vector<std::string> intersection_label;

    // step 2: set database

    const auto setdb_start = std::chrono::system_clock::now();

    if (params.label_bytes == 0) {
      std::vector<apsi::Item> sender_items = GenerateSenderData(
          rd(), item_count, absl::MakeSpan(receiver_items), &intersection_idx);

      sender_db->SetData(sender_items);
    } else {
      std::vector<std::pair<apsi::Item, apsi::Label>> sender_items =
          GenerateSenderData(rd(), item_count, label_byte_count,
                             absl::MakeSpan(receiver_items), &intersection_idx,
                             &intersection_label);

      sender_db->SetData(sender_items);
    }

    const auto setdb_end = std::chrono::system_clock::now();
    const duration_millis setdb_duration = setdb_end - setdb_start;
    SPDLOG_INFO("*** step2 set db duration:{}", setdb_duration.count());

    EXPECT_EQ(params.ns, sender_db->GetItemCount());

    SPDLOG_INFO("after set db, bin_bundle_count:{}, packing_rate:{}",
                sender_db->GetBinBundleCount(), sender_db->GetPackingRate());

    std::unique_ptr<spu::IEcdhOprfServer> oprf_server =
        spu::CreateEcdhOprfServer(oprf_key, spu::OprfType::Basic,
                                  spu::CurveType::CurveFourQ);

    LabelPsiSender sender(sender_db);

    LabelPsiReceiver receiver(psi_params, params.label_bytes > 0);

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

    SPDLOG_INFO("index vec size:{} intersection_idx size:{}",
                query_result.first.size(), intersection_idx.size());

    SPDLOG_INFO("intersection:{}", intersection.size());
    auto stats0 = ctxs[0]->GetStats();
    auto stats1 = ctxs[1]->GetStats();
    SPDLOG_INFO("sender ctx0 sent_bytes:{} recv_bytes:{}", stats0->sent_bytes,
                stats0->recv_bytes);
    SPDLOG_INFO("receiver ctx1 sent_bytes:{} recv_bytes:{}", stats1->sent_bytes,
                stats1->recv_bytes);
  }
}

// [256k, 512k, 1m, 2m, 4m, 8m, 16m]
BENCHMARK(BM_PcgPsi)
    ->Unit(benchmark::kMillisecond)
    ->Arg(256 << 10)
    ->Arg(512 << 10)
    ->Arg(1 << 20)
    ->Arg(2 << 20)
    ->Arg(4 << 20)
    ->Arg(8 << 20)
    ->Arg(16 << 20);

BENCHMARK_MAIN();

}  // namespace spu::psi
