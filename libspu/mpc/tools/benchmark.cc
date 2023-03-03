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

#include "libspu/mpc/tools/benchmark.h"

#include <iostream>

#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"

#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/semi2k/protocol.h"

namespace {

static constexpr uint32_t kUnSetMagic = 0x123456;
const std::string kTwoPartyHosts = "127.0.0.1:9540,127.0.0.1:9541";
const std::string kThreePartyHosts =
    "127.0.0.1:9540,127.0.0.1:9541,127.0.0.1:9542";

llvm::cl::opt<std::string> cli_gbenchmark(
    "benchmark_**",
    llvm::cl::desc("google benchmark options, eg: \n"
                   "    --benchmark_out=<filename>,\n"
                   "    --benchmark_out_format={json|console|csv},\n"
                   "    --benchmark_filter=<regex>,\n"
                   "    --benchmark_counters_tabular = true,\n"
                   "    --benchmark_time_unit={ns|us|ms|s}"),
    llvm::cl::init("gbenchmark"));

llvm::cl::opt<uint32_t> cli_rank("rank", llvm::cl::init(0),
                                 llvm::cl::desc("self rank, starts with 0"));
llvm::cl::opt<std::string> cli_parties(
    "parties",
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));
llvm::cl::opt<uint32_t> cli_party_num("party_num", llvm::cl::init(0),
                                      llvm::cl::desc("server numbers"));
llvm::cl::opt<std::string> cli_protocol(
    "protocol", llvm::cl::init("aby3"),
    llvm::cl::desc("benchmark protocol, supported protocols: semi2k / aby3, "
                   "default: aby3"));
llvm::cl::opt<uint32_t> cli_numel(
    "numel", llvm::cl::init(kUnSetMagic),
    llvm::cl::desc("number of benchmark elements, default: [10, 100, 1000]"));
llvm::cl::opt<uint32_t> cli_shiftbit(
    "shiftbit", llvm::cl::init(kUnSetMagic),
    llvm::cl::desc("benchmark shift bit, default: [2, 4, 8]"));
llvm::cl::opt<std::string> cli_mode(
    "mode", llvm::cl::init("standalone"),
    llvm::cl::desc(
        "benchmark mode : standalone / mparty, default: standalone"));
}  // namespace

namespace spu::mpc::bench {

void NumelAug(benchmark::internal::Benchmark* b) {
  b->ArgsProduct(
       {BenchConfig::bench_field_range, BenchConfig::bench_numel_range})
      ->Iterations(100);
}
void NumelShiftAug(benchmark::internal::Benchmark* b) {
  b->ArgsProduct({BenchConfig::bench_field_range,
                  BenchConfig::bench_numel_range,
                  BenchConfig::bench_shift_range})
      ->Iterations(100);
}
void MatrixSizeAug(benchmark::internal::Benchmark* b) {
  b->ArgsProduct({BenchConfig::bench_field_range,
                  BenchConfig::bench_matrix_m_range,
                  BenchConfig::bench_matrix_k_range})
      ->Iterations(100);
}

// register benchmarks with arguments
BENCHMARK(MPCBenchMark<BenchAddSS>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchMulSS>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchAndSS>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchXorSS>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchAddSP>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchMulSP>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchAndSP>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchXorSP>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchS2P>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchP2S>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchNotS>)->Apply(NumelAug);
BENCHMARK(MPCBenchMark<BenchNotP>)->Apply(NumelAug);

BENCHMARK(MPCBenchMark<BenchLShiftS>)->Apply(NumelShiftAug);
BENCHMARK(MPCBenchMark<BenchLShiftP>)->Apply(NumelShiftAug);
BENCHMARK(MPCBenchMark<BenchRShiftS>)->Apply(NumelShiftAug);
BENCHMARK(MPCBenchMark<BenchRShiftP>)->Apply(NumelShiftAug);
BENCHMARK(MPCBenchMark<BenchARShiftP>)->Apply(NumelShiftAug);
BENCHMARK(MPCBenchMark<BenchARShiftS>)->Apply(NumelShiftAug);
BENCHMARK(MPCBenchMark<BenchTruncS>)->Apply(NumelShiftAug);

BENCHMARK(MPCBenchMark<BenchMMulSP>)->Apply(MatrixSizeAug);
BENCHMARK(MPCBenchMark<BenchMMulSS>)->Apply(MatrixSizeAug);

}  // namespace spu::mpc::bench

void PrepareSemi2k(std::string& parties, uint32_t& party_num) {
  using BenchInteral = spu::mpc::bench::BenchConfig;
  if (parties.empty() && party_num == 0) {
    parties = kTwoPartyHosts;  // default ips for semi2k
    party_num = 2;
  }
  SPU_ENFORCE(party_num >= 2);
  BenchInteral::bench_factory =
      spu::mpc::makeSemi2kProtocol;  // semi2k protocol factory
}

void PrepareAby3(std::string& parties, uint32_t& party_num) {
  using BenchInteral = spu::mpc::bench::BenchConfig;
  if (parties.empty() && party_num == 0) {
    parties = kThreePartyHosts;  // default ips for aby3
    party_num = 3;
  }
  SPU_ENFORCE(party_num == 3);
  BenchInteral::bench_factory = spu::mpc::makeAby3Protocol;
}

void SetUpProtocol() {
  using BenchInteral = spu::mpc::bench::BenchConfig;
  auto protocol = cli_protocol.getValue();
  auto parties = cli_parties.getValue();
  auto party_num = cli_party_num.getValue();
  if (protocol == "semi2k") {
    PrepareSemi2k(parties, party_num);
  } else if (protocol == "aby3") {
    PrepareAby3(parties, party_num);
  } else {
    SPU_THROW("unknown protocol: {}, supported = semi2k/aby3", protocol);
  }
  benchmark::AddCustomContext("Benchmark Protocol", protocol);
  BenchInteral::bench_npc = party_num;
  benchmark::AddCustomContext("Benchmark Party Number",
                              std::to_string(party_num));
  BenchInteral::bench_parties = parties;
  ::benchmark::AddCustomContext("Benchmark Parties", parties);
}

void SetUpMode() {
  using BenchInteral = spu::mpc::bench::BenchConfig;

  auto mode = cli_mode.getValue();
  if (cli_mode.getValue() != "standalone") {
    auto rank = cli_rank.getValue();
    std::vector<std::string> host_ips =
        absl::StrSplit(BenchInteral::bench_parties, ',');
    SPU_ENFORCE(host_ips.size() == BenchInteral::bench_npc);

    yacl::link::ContextDesc lctx_desc;
    for (size_t i = 0; i < BenchInteral::bench_npc; i++) {
      const std::string id = fmt::format("party{}", i);
      lctx_desc.parties.push_back({id, host_ips[i]});
      benchmark::AddCustomContext(fmt::format("BenchmarkParty-{} IP", i),
                                  host_ips[i]);
    }
    // setup bench_lctx and link
    yacl::link::FactoryBrpc factory;
    BenchInteral::bench_lctx = factory.CreateContext(lctx_desc, rank);
    BenchInteral::bench_lctx->ConnectToMesh();
    benchmark::AddCustomContext("Benchmark Self Rank", std::to_string(rank));
  }
  BenchInteral::bench_mode = mode;
  benchmark::AddCustomContext("Benchmark Mode", BenchInteral::bench_mode);
}

void PrepareBenchmark() {
  using BenchInteral = spu::mpc::bench::BenchConfig;

  SetUpProtocol();
  SetUpMode();

  if (cli_numel.getValue() != kUnSetMagic) {
    BenchInteral::bench_numel_range.clear();
    BenchInteral::bench_numel_range.push_back(cli_numel.getValue());
    benchmark::AddCustomContext("Benchmark Data Size",
                                std::to_string(cli_numel.getValue()));
  }

  if (cli_shiftbit.getValue() != kUnSetMagic) {
    BenchInteral::bench_shift_range.clear();
    BenchInteral::bench_shift_range.push_back(cli_shiftbit.getValue());
    benchmark::AddCustomContext("Benchmark Shift Bits",
                                std::to_string(cli_shiftbit.getValue()));
  }
}

void ParseCommandLineOptions(int argc, char** argv) {
  std::vector<char*> llvm_opts;
  std::vector<char*> google_bm_opts;

  const char* google_bm_opt_prefix = "--benchmark";
  google_bm_opts.push_back(argv[0]);
  for (int i = 0; i != argc; ++i) {
    if (strncmp(google_bm_opt_prefix, argv[i], strlen(google_bm_opt_prefix)) ==
        0) {
      google_bm_opts.push_back(argv[i]);
    } else {
      llvm_opts.push_back(argv[i]);
    }
  }

  llvm::cl::ParseCommandLineOptions(llvm_opts.size(), llvm_opts.data());

  int google_bm_size = google_bm_opts.size();
  ::benchmark::Initialize(&google_bm_size, google_bm_opts.data());
  SPU_ENFORCE(!::benchmark::ReportUnrecognizedArguments(google_bm_size,
                                                        google_bm_opts.data()));
}

// the main function
int main(int argc, char** argv) {
  ParseCommandLineOptions(argc, argv);

  PrepareBenchmark();

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}
