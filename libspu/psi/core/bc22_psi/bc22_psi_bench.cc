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
#include "yacl/link/test_util.h"

#include "libspu/psi/core/bc22_psi/bc22_psi.h"

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
    const std::string& host_str) {
  std::vector<std::string> hosts = absl::StrSplit(host_str, ',');
  yacl::link::ContextDesc lctx_desc;
  for (size_t rank = 0; rank < hosts.size(); rank++) {
    const std::string id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, hosts[rank]});
  }

  auto proc = [&](int self_rank) -> std::shared_ptr<yacl::link::Context> {
    return CreateContext(self_rank, lctx_desc);
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

static void BM_PcgPsi(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);
    auto alice_data = CreateRangeItems(1, n);
    auto bob_data = CreateRangeItems(2, n);

    auto ctxs = CreateLinks(kLinkAddrAB);

    ctxs[0]->SetThrottleWindowSize(kLinkWindowSize);
    ctxs[1]->SetThrottleWindowSize(kLinkWindowSize);

    ctxs[0]->SetRecvTimeout(kLinkRecvTimeout);
    ctxs[1]->SetRecvTimeout(kLinkRecvTimeout);

    state.ResumeTiming();

    Bc22PcgPsi pcg_psi_send(ctxs[0], PsiRoleType::Sender);
    Bc22PcgPsi pcg_psi_recv(ctxs[1], PsiRoleType::Receiver);

    std::future<void> send_thread =
        std::async([&] { pcg_psi_send.RunPsi(alice_data); });

    std::future<void> recv_thread =
        std::async([&] { return pcg_psi_recv.RunPsi(bob_data); });

    send_thread.get();
    recv_thread.get();

    std::vector<std::string> intersection = pcg_psi_recv.GetIntersection();

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

}  // namespace spu::psi
