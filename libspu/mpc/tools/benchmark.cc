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

#include <fstream>
#include <iostream>

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"

#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/semi2k/protocol.h"

namespace {

constexpr uint32_t kUnSetMagic = 0x123456;
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
llvm::cl::opt<std::string> cli_protocol(
    "protocol", llvm::cl::init("aby3"),
    llvm::cl::desc(
        "benchmark protocol, supported protocols: semi2k / aby3 / cheetah, "
        "default: aby3"));
llvm::cl::opt<uint32_t> cli_numel(
    "numel", llvm::cl::init(kUnSetMagic),
    llvm::cl::desc("number of benchmark elements, default: [2^10, 2^20]"));
llvm::cl::opt<uint32_t> cli_shiftbit(
    "shiftbit", llvm::cl::init(kUnSetMagic),
    llvm::cl::desc("benchmark shift bit, default: 2"));
llvm::cl::opt<uint32_t> cli_iteration(
    "iteration", llvm::cl::init(10),
    llvm::cl::desc("benchmark iteration, default: 10"));
llvm::cl::opt<std::string> cli_mode(
    "mode", llvm::cl::init("standalone"),
    llvm::cl::desc(
        "benchmark mode : standalone / mparty, default: standalone"));
}  // namespace

namespace spu::mpc::bench {

class BenchArgs {
 protected:
  std::string label_;
  benchmark::State& state;

 public:
  BenchArgs(const std::string& op_name, benchmark::State& st) : state(st) {
    label_ = "op_name:" + op_name;
  }
  virtual std::string StateInfo(benchmark::State& st) { return ""; }
  std::string Label() { return label_ + StateInfo(state); }
  virtual ~BenchArgs() = default;
};

class NumelArgs : public BenchArgs {
 public:
  using BenchArgs::BenchArgs;
  std::string StateInfo(benchmark::State& st) override {
    std::string ret;
    ret += "/field_type:" +
           std::to_string(8 * SizeOf(static_cast<FieldType>(st.range(0))));
    ret += "/buf_len:" + std::to_string(st.range(1));
    return ret;
  }
  static void AddArgs(benchmark::internal::Benchmark* b) {
    b->ArgsProduct(
         {BenchConfig::bench_field_range, BenchConfig::bench_numel_range})
        ->Iterations(cli_iteration.getValue())
        ->UseManualTime()
        ->MeasureProcessCPUTime();
  }
};

class NumelShiftArgs : public BenchArgs {
 public:
  using BenchArgs::BenchArgs;
  std::string StateInfo(benchmark::State& st) override {
    std::string ret;
    ret += "/field_type:" +
           std::to_string(8 * SizeOf(static_cast<FieldType>(st.range(0))));
    ret += "/buf_len:" + std::to_string(st.range(1));
    ret += "/shift_bit:" + std::to_string(st.range(2));
    return ret;
  }
  static void AddArgs(benchmark::internal::Benchmark* b) {
    b->ArgsProduct({BenchConfig::bench_field_range,
                    BenchConfig::bench_numel_range,
                    BenchConfig::bench_shift_range})
        ->Iterations(cli_iteration.getValue())
        ->UseManualTime()
        ->MeasureProcessCPUTime();
  }
};

class MatrixSizeArgs : public BenchArgs {
 public:
  using BenchArgs::BenchArgs;
  std::string StateInfo(benchmark::State& st) override {
    std::string ret;
    ret += "/field_type:" +
           std::to_string(8 * SizeOf(static_cast<FieldType>(st.range(0))));
    ret +=
        "/matrix_size:" + fmt::format("{{{0}, {0}}}*{{{0}, {0}}}", st.range(1));
    return ret;
  }
  static void AddArgs(benchmark::internal::Benchmark* b) {
    b->ArgsProduct(
         {BenchConfig::bench_field_range, BenchConfig::bench_matrix_range})
        ->Iterations(cli_iteration.getValue())
        ->UseManualTime()
        ->MeasureProcessCPUTime();
  }
};

// register benchmarks with arguments
#define DEFINE_BENCHMARK(OP, ARGS) \
  BENCHMARK(MPCBenchMark<OP, ARGS>)->Apply(ARGS::AddArgs)

DEFINE_BENCHMARK(BenchAddSS, NumelArgs);
DEFINE_BENCHMARK(BenchMulSS, NumelArgs);
DEFINE_BENCHMARK(BenchAndSS, NumelArgs);
DEFINE_BENCHMARK(BenchXorSS, NumelArgs);
DEFINE_BENCHMARK(BenchAddSP, NumelArgs);
DEFINE_BENCHMARK(BenchMulSP, NumelArgs);
DEFINE_BENCHMARK(BenchAndSP, NumelArgs);
DEFINE_BENCHMARK(BenchXorSP, NumelArgs);
DEFINE_BENCHMARK(BenchS2P, NumelArgs);
DEFINE_BENCHMARK(BenchP2S, NumelArgs);
DEFINE_BENCHMARK(BenchNotS, NumelArgs);
DEFINE_BENCHMARK(BenchNotP, NumelArgs);

DEFINE_BENCHMARK(BenchLShiftS, NumelShiftArgs);
DEFINE_BENCHMARK(BenchLShiftP, NumelShiftArgs);
DEFINE_BENCHMARK(BenchRShiftS, NumelShiftArgs);
DEFINE_BENCHMARK(BenchRShiftP, NumelShiftArgs);
DEFINE_BENCHMARK(BenchARShiftP, NumelShiftArgs);
DEFINE_BENCHMARK(BenchARShiftS, NumelShiftArgs);
DEFINE_BENCHMARK(BenchTruncS, NumelShiftArgs);

DEFINE_BENCHMARK(BenchMMulSP, MatrixSizeArgs);
DEFINE_BENCHMARK(BenchMMulSS, MatrixSizeArgs);

DEFINE_BENCHMARK(BenchRandA, NumelArgs);
DEFINE_BENCHMARK(BenchRandB, NumelArgs);
DEFINE_BENCHMARK(BenchP2A, NumelArgs);
DEFINE_BENCHMARK(BenchA2P, NumelArgs);
DEFINE_BENCHMARK(BenchMsbA2b, NumelArgs);
DEFINE_BENCHMARK(BenchNotA, NumelArgs);
DEFINE_BENCHMARK(BenchAddAP, NumelArgs);
DEFINE_BENCHMARK(BenchMulAP, NumelArgs);
DEFINE_BENCHMARK(BenchAddAA, NumelArgs);
DEFINE_BENCHMARK(BenchMulAA, NumelArgs);
DEFINE_BENCHMARK(BenchMulA1B, NumelArgs);
DEFINE_BENCHMARK(BenchLShiftA, NumelShiftArgs);
DEFINE_BENCHMARK(BenchTruncA, NumelShiftArgs);
DEFINE_BENCHMARK(BenchMMulAP, MatrixSizeArgs);
DEFINE_BENCHMARK(BenchMMulAA, MatrixSizeArgs);
DEFINE_BENCHMARK(BenchB2P, NumelArgs);
DEFINE_BENCHMARK(BenchP2B, NumelArgs);
DEFINE_BENCHMARK(BenchA2B, NumelArgs);
DEFINE_BENCHMARK(BenchB2A, NumelArgs);
DEFINE_BENCHMARK(BenchAddBB, NumelArgs);
DEFINE_BENCHMARK(BenchAndBP, NumelArgs);
DEFINE_BENCHMARK(BenchAndBB, NumelArgs);
DEFINE_BENCHMARK(BenchXorBP, NumelArgs);
DEFINE_BENCHMARK(BenchXorBB, NumelArgs);
DEFINE_BENCHMARK(BenchLShiftB, NumelShiftArgs);
DEFINE_BENCHMARK(BenchRShiftB, NumelShiftArgs);
DEFINE_BENCHMARK(BenchARShiftB, NumelShiftArgs);
DEFINE_BENCHMARK(BenchBitRevB, NumelArgs);
DEFINE_BENCHMARK(BenchBitIntlB, NumelArgs);
DEFINE_BENCHMARK(BenchBitDentlB, NumelArgs);

void PrepareSemi2k(std::string& parties, uint32_t& party_num) {
  using BenchInteral = spu::mpc::bench::BenchConfig;
  if (parties.empty() && party_num == 0) {
    parties = kTwoPartyHosts;  // default ips for semi2k
  }
  party_num = std::count(parties.begin(), parties.end(), ',') + 1;

  SPU_ENFORCE(party_num >= 2);
  BenchInteral::bench_factory =
      spu::mpc::makeSemi2kProtocol;  // semi2k protocol factory
}

void PrepareCheetah(std::string& parties, uint32_t& party_num) {
  using BenchInteral = spu::mpc::bench::BenchConfig;
  if (parties.empty() && party_num == 0) {
    parties = kTwoPartyHosts;
  }
  party_num = std::count(parties.begin(), parties.end(), ',') + 1;
  SPU_ENFORCE(party_num == 2);
  BenchInteral::bench_factory = spu::mpc::makeCheetahProtocol;
}

void PrepareAby3(std::string& parties, uint32_t& party_num) {
  using BenchInteral = spu::mpc::bench::BenchConfig;
  if (parties.empty() && party_num == 0) {
    parties = kThreePartyHosts;  // default ips for aby3
  }
  party_num = std::count(parties.begin(), parties.end(), ',') + 1;

  SPU_ENFORCE(party_num == 3);
  BenchInteral::bench_factory = spu::mpc::makeAby3Protocol;
}

void SetUpProtocol() {
  using BenchInteral = spu::mpc::bench::BenchConfig;
  auto protocol = cli_protocol.getValue();
  auto parties = cli_parties.getValue();
  uint32_t party_num = 0;
  if (protocol == "semi2k") {
    PrepareSemi2k(parties, party_num);
  } else if (protocol == "aby3") {
    PrepareAby3(parties, party_num);
  } else if (protocol == "cheetah") {
    PrepareCheetah(parties, party_num);
  } else {
    SPU_THROW("unknown protocol: {}, supported = semi2k/aby3/cheetah",
              protocol);
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
    lctx_desc.recv_timeout_ms = 120 * 1000;
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

  google_bm_opts.push_back(argv[0]);
  for (int i = 0; i != argc; ++i) {
    if (absl::StartsWith(argv[i], "--benchmark")) {
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

}  // namespace spu::mpc::bench

int main(int argc, char** argv) {
  spu::mpc::bench::ParseCommandLineOptions(argc, argv);

  spu::mpc::bench::PrepareBenchmark();

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}
