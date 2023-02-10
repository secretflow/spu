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
#include <cmath>
#include <future>
#include <random>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "benchmark/benchmark.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/core/dp_psi/dp_psi.h"
#include "libspu/psi/core/dp_psi/dp_psi_utils.h"

namespace spu::psi {

namespace {

// ErfInv code from
// https://github.com/abrari/block-cipher-testing/blob/master/stats.c
double ErfInv(double x) {
  // beware that the logarithm argument must be
  // computed as (1.0 - x) * (1.0 + x),
  // it must NOT be simplified as 1.0 - x * x as this
  // would induce rounding errors near the boundaries +/-1
  double w = -std::log((1.0 - x) * (1.0 + x));
  double p;

  if (w < 6.25) {
    w -= 3.125;
    p = -3.6444120640178196996e-21;
    p = -1.685059138182016589e-19 + p * w;
    p = 1.2858480715256400167e-18 + p * w;
    p = 1.115787767802518096e-17 + p * w;
    p = -1.333171662854620906e-16 + p * w;
    p = 2.0972767875968561637e-17 + p * w;
    p = 6.6376381343583238325e-15 + p * w;
    p = -4.0545662729752068639e-14 + p * w;
    p = -8.1519341976054721522e-14 + p * w;
    p = 2.6335093153082322977e-12 + p * w;
    p = -1.2975133253453532498e-11 + p * w;
    p = -5.4154120542946279317e-11 + p * w;
    p = 1.051212273321532285e-09 + p * w;
    p = -4.1126339803469836976e-09 + p * w;
    p = -2.9070369957882005086e-08 + p * w;
    p = 4.2347877827932403518e-07 + p * w;
    p = -1.3654692000834678645e-06 + p * w;
    p = -1.3882523362786468719e-05 + p * w;
    p = 0.0001867342080340571352 + p * w;
    p = -0.00074070253416626697512 + p * w;
    p = -0.0060336708714301490533 + p * w;
    p = 0.24015818242558961693 + p * w;
    p = 1.6536545626831027356 + p * w;
  } else if (w < 16.0) {
    w = std::sqrt(w) - 3.25;
    p = 2.2137376921775787049e-09;
    p = 9.0756561938885390979e-08 + p * w;
    p = -2.7517406297064545428e-07 + p * w;
    p = 1.8239629214389227755e-08 + p * w;
    p = 1.5027403968909827627e-06 + p * w;
    p = -4.013867526981545969e-06 + p * w;
    p = 2.9234449089955446044e-06 + p * w;
    p = 1.2475304481671778723e-05 + p * w;
    p = -4.7318229009055733981e-05 + p * w;
    p = 6.8284851459573175448e-05 + p * w;
    p = 2.4031110387097893999e-05 + p * w;
    p = -0.0003550375203628474796 + p * w;
    p = 0.00095328937973738049703 + p * w;
    p = -0.0016882755560235047313 + p * w;
    p = 0.0024914420961078508066 + p * w;
    p = -0.0037512085075692412107 + p * w;
    p = 0.005370914553590063617 + p * w;
    p = 1.0052589676941592334 + p * w;
    p = 3.0838856104922207635 + p * w;
  } else if (!std::isinf(w)) {
    w = std::sqrt(w) - 5.0;
    p = -2.7109920616438573243e-11;
    p = -2.5556418169965252055e-10 + p * w;
    p = 1.5076572693500548083e-09 + p * w;
    p = -3.7894654401267369937e-09 + p * w;
    p = 7.6157012080783393804e-09 + p * w;
    p = -1.4960026627149240478e-08 + p * w;
    p = 2.9147953450901080826e-08 + p * w;
    p = -6.7711997758452339498e-08 + p * w;
    p = 2.2900482228026654717e-07 + p * w;
    p = -9.9298272942317002539e-07 + p * w;
    p = 4.5260625972231537039e-06 + p * w;
    p = -1.9681778105531670567e-05 + p * w;
    p = 7.5995277030017761139e-05 + p * w;
    p = -0.00021503011930044477347 + p * w;
    p = -0.00013871931833623122026 + p * w;
    p = 1.0103004648645343977 + p * w;
    p = 4.8499064014085844221 + p * w;
  } else {
    // this branch does not appears in the original code, it
    // was added because the previous branch does not handle
    // x = +/-1 correctly. In this case, w is positive infinity
    // and as the first coefficient (-2.71e-11) is negative.
    // Once the first multiplication is done, p becomes negative
    // infinity and remains so throughout the polynomial evaluation.
    // So the branch above incorrectly returns negative infinity
    // instead of the correct positive infinity.
    p = INFINITY;
  }

  return p * x;
}

double TruncatedNormalSample(double mu, double sigma, double a, double b,
                             uint64_t seed) {
  double alpha = (a - mu) / sigma;
  double beta = (b - mu) / sigma;

  double sqrt2 = std::sqrt(2);
  double lower = std::erf(alpha / sqrt2);
  double upper = std::erf(beta / sqrt2);

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(lower, upper);
  double u = distribution(generator);
  double out = sqrt2 * ErfInv(u);
  // SPDLOG_INFO("lower:{} upper:{} a:{} b:{} alpha:{} beta:{} sqrt2:{}", lower,
  //             upper, a, b, alpha, beta, sqrt2);
  // SPDLOG_INFO("u:{} out:{}", u, out);

  out = mu + sigma * out;
  // SPDLOG_INFO("u:{} out:{}", u, out);

  return out;
}

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

constexpr double kPayloadMu = 50;
constexpr double kPayloadSigma = 8;
constexpr double kPayloadMinValue = 0;
constexpr double kPayloadMaxValue = 100;

std::vector<uint32_t> CreateItemsPayload(size_t size) {
  std::vector<uint32_t> random_data(size);

  double mu = kPayloadMu;
  double sigma = kPayloadSigma;

  double a = kPayloadMinValue;
  double b = kPayloadMaxValue;

  std::random_device rd;

  std::mt19937 rand_mt(rd());

  std::generate(begin(random_data), end(random_data), [&]() {
    uint64_t seed = (static_cast<uint64_t>(rand_mt()) << 32) | rand_mt();
    double res = TruncatedNormalSample(mu, sigma, a, b, seed);
    return std::round(res);
  });

  return random_data;
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

void PayloadMeanWithDp(const DpPsiOptions& dp_psi_options,
                       const std::vector<size_t>& intersection_idx,
                       const std::vector<uint32_t>& payloads, double* mean,
                       double* dp_mean, double* sigma, size_t max_value = 100,
                       double epsilon = 3) {
  double sum = 0;

  for (const auto& idx : intersection_idx) {
    sum += payloads[idx];
  }

  *mean = sum / intersection_idx.size();

  std::pair<uint64_t, uint64_t> seed_pair =
      yacl::DecomposeUInt128(yacl::crypto::RandSeed());
  std::default_random_engine rng{
      static_cast<std::random_device::result_type>(seed_pair.first)};

  double GS = static_cast<double>(max_value) / intersection_idx.size();
  double delta = 1.0 / (10 * payloads.size());
  // p1 * p2 * delta_f
  // delta = delta * dp_psi_options.p1 * dp_psi_options.p2;

  // SPDLOG_INFO("item size:{},intersection size: {}", payloads.size(),
  //             intersection_idx.size());

  SPDLOG_INFO("GS: {}, delta:{}, epsilon:{}", GS, delta, epsilon);

  *sigma = CalibrateAnalyticGaussianMechanism(epsilon, delta, GS);

  double mu{0};
  std::normal_distribution<> norm{mu, *sigma};

  // mean + dp
  *dp_mean = *mean + norm(rng);
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
    double payload_epsilon = state.range(1);

    std::vector<std::string> items_a = CreateRangeItems(0, items_size);
    std::vector<std::string> items_b =
        CreateRangeItems(items_size * (1 - kIntersectionRatio), items_size);

    std::vector<uint32_t> payloads_b = CreateItemsPayload(items_size);

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
        "dp psi bob intersection size:{}, bob_sub_sample_size:{} "
        "real_intersection size: {}",
        dp_psi_result.size(), bob_sub_sample_size, real_intersection.size());

    double mean;
    double dp_mean;
    double sigma;
    PayloadMeanWithDp(options, dp_psi_result, payloads_b, &mean, &dp_mean,
                      &sigma, kPayloadMaxValue, payload_epsilon);

    SPDLOG_INFO("mean:{} dp_mean:{}, sigma:{}", mean, dp_mean, sigma);

    auto stats0 = ctxs[alice_rank]->GetStats();
    auto stats1 = ctxs[bob_rank]->GetStats();

    double total_comm_bytes = stats0->sent_bytes + stats0->recv_bytes;
    SPDLOG_INFO("bob: sent_bytes:{} recv_bytes:{} total_comm_bytes:{}",
                stats1->sent_bytes, stats1->recv_bytes,
                total_comm_bytes / 1024 / 1024);
  }
}

// [256k, 512k, 1m, 2m, 4m, 8m, 16m]
BENCHMARK(BM_DpPsi)
    ->Unit(benchmark::kMillisecond)
    ->Args({256 << 10, 3})
    ->Args({512 << 10, 3})
    ->Args({1 << 20, 3})
    ->Args({2 << 20, 3})
    ->Args({4 << 20, 3})
    ->Args({8 << 20, 3})
    ->Args({16 << 20, 3})
    ->Args({32 << 20, 3})
    ->Args({64 << 20, 3})
    ->Args({128 << 20, 3})
    ->Args({1 << 20, 1})
    ->Args({1 << 20, 2})
    ->Args({1 << 20, 3})
    ->Args({1 << 20, 4})
    ->Args({1 << 20, 5});

}  // namespace spu::psi
