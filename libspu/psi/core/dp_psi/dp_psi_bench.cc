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

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "benchmark/benchmark.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/core/dp_psi/dp_psi.h"

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
constexpr double kIntersectionRatio = 0.7;

std::map<size_t, DpPsiOptions> dp_psi_params_map = {
    {1 << 11, DpPsiOptions(0.9)},   {1 << 12, DpPsiOptions(0.9)},
    {1 << 13, DpPsiOptions(0.9)},   {1 << 14, DpPsiOptions(0.9)},
    {1 << 15, DpPsiOptions(0.9)},   {1 << 16, DpPsiOptions(0.9)},
    {1 << 17, DpPsiOptions(0.9)},   {1 << 18, DpPsiOptions(0.9)},
    {1 << 19, DpPsiOptions(0.9)},   {1 << 20, DpPsiOptions(0.995)},
    {1 << 21, DpPsiOptions(0.995)}, {1 << 22, DpPsiOptions(0.995)},
    {1 << 23, DpPsiOptions(0.995)}, {1 << 24, DpPsiOptions(0.995)},
    {1 << 25, DpPsiOptions(0.995)}, {1 << 26, DpPsiOptions(0.995)},
    {1 << 27, DpPsiOptions(0.995)}, {1 << 28, DpPsiOptions(0.995)},
    {1 << 29, DpPsiOptions(0.995)}, {1 << 30, DpPsiOptions(0.995)}};

}  // namespace

static void BM_DpPsi(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t items_size = state.range(0);

    std::vector<std::string> items_a = CreateRangeItems(0, items_size);
    std::vector<std::string> items_b =
        CreateRangeItems(items_size * (1 - kIntersectionRatio), items_size);

    auto ctxs = CreateLinks(kLinkAddrAB);

    ctxs[0]->SetThrottleWindowSize(kLinkWindowSize);
    ctxs[1]->SetThrottleWindowSize(kLinkWindowSize);

    ctxs[0]->SetRecvTimeout(kLinkRecvTimeout);
    ctxs[1]->SetRecvTimeout(kLinkRecvTimeout);

    std::vector<std::string> real_intersection =
        GetIntersection(items_a, items_b);

    const DpPsiOptions& options = dp_psi_params_map[items_size];

    state.ResumeTiming();

    size_t alice_rank = 0;
    size_t bob_rank = 1;

    size_t alice_sub_sample_size;
    size_t alice_up_sample_size;
    size_t bob_sub_sample_size;

    std::future<size_t> f_dp_psi_a = std::async([&] {
      return RunDpEcdhPsiAlice(options, ctxs[alice_rank], items_a,
                               &alice_sub_sample_size, &alice_up_sample_size);
    });

    std::future<std::vector<size_t>> f_dp_psi_b = std::async([&] {
      return RunDpEcdhPsiBob(options, ctxs[bob_rank], items_b,
                             &bob_sub_sample_size);
    });

    size_t alice_intersection_size = f_dp_psi_a.get();
    std::vector<size_t> dp_psi_result = f_dp_psi_b.get();

    SPDLOG_INFO(
        "alice_intersection_size:{} "
        "alice_sub_sample_size:{},alice_up_sample_size:{}",
        alice_intersection_size, alice_sub_sample_size, alice_up_sample_size);

    SPDLOG_INFO(
        "dp psi bob intersection size:{},bob_sub_sample_size:{} "
        "real_intersection size: {}",
        dp_psi_result.size(), bob_sub_sample_size, real_intersection.size());

    auto stats0 = ctxs[alice_rank]->GetStats();
    auto stats1 = ctxs[bob_rank]->GetStats();

    double total_comm_bytes = stats0->sent_bytes + stats0->recv_bytes;
    SPDLOG_INFO("bob: sent_bytes:{} recv_bytes:{}, total_comm_bytes:{}",
                stats1->sent_bytes, stats1->recv_bytes,
                total_comm_bytes / 1024 / 1024);
  }
}

// [256k, 512k, 1m, 2m, 4m, 8m, 16m]
BENCHMARK(BM_DpPsi)
    ->Unit(benchmark::kMillisecond)
    ->Arg(256 << 10)
    ->Arg(512 << 10)
    ->Arg(1 << 20)
    ->Arg(2 << 20)
    ->Arg(4 << 20)
    ->Arg(8 << 20)
    ->Arg(16 << 20)
    ->Arg(32 << 20)
    ->Arg(64 << 20)
    ->Arg(128 << 20);

}  // namespace spu::psi
