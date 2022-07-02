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

#include "mparty_bench.h"

#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "utils.h"
#include "yasl/link/transport/channel_brpc.h"

#include "spu/mpc/aby3/protocol.h"
#include "spu/mpc/semi2k/protocol.h"

namespace spu::mpc::bench {

void DefaultBMArguments(benchmark::internal::Benchmark* b) {
  b->Args({FieldType::FM32})->Args({FieldType::FM64})->Args({FieldType::FM128});

  // Uncomment the following if you want to manually config benchmark iterations
  // b->Args({FieldType::FM32})
  //     ->Args({FieldType::FM64})
  //     ->Args({FieldType::FM128})
  //     ->Iterations(100);
}

// register benchmarks with arguments
SPU_BM_PROTOCOL_REGISTER(DefaultBMArguments);

}  // namespace spu::mpc::bench

void PrepareSemi2k(std::string& parties, uint32_t& party_num) {
  using BenchInteral = spu::mpc::bench::ComputeBench;
  if (parties.empty() && party_num == 0) {
    parties = spu::mpc::bench::kTwoPartyHosts;  // default ips for semi2k
    party_num = 2;
  }
  YASL_ENFORCE(party_num >= 2);
  BenchInteral::bench_factory =
      spu::mpc::makeSemi2kProtocol;  // semi2k protocol factory
}

void PrepareAby3(std::string& parties, uint32_t& party_num) {
  using BenchInteral = spu::mpc::bench::ComputeBench;
  if (parties.empty() && party_num == 0) {
    parties = spu::mpc::bench::kThreePartyHosts;  // default ips for aby3
    party_num = 3;
  }
  YASL_ENFORCE(party_num == 3);
  BenchInteral::bench_factory = spu::mpc::makeAby3Protocol;
}

void PrepareBenchmark(uint32_t rank, std::string& parties, uint32_t party_num,
                      uint32_t numel, uint32_t shiftbit,
                      std::string& protocol) {
  using BenchInteral = spu::mpc::bench::ComputeBench;

  if (protocol == "semi2k") {
    PrepareSemi2k(parties, party_num);
  } else if (protocol == "aby3") {
    PrepareAby3(parties, party_num);
  } else {
    YASL_THROW("unknown protocol: {}, supported = semi2k/aby3", protocol);
  }

  std::vector<std::string> host_ips = absl::StrSplit(parties, ',');
  YASL_ENFORCE(host_ips.size() == party_num);

  yasl::link::ContextDesc lctx_desc;
  for (size_t i = 0; i < party_num; i++) {
    const std::string id = fmt::format("party{}", i);
    lctx_desc.parties.push_back({id, host_ips[i]});
    benchmark::AddCustomContext(fmt::format("Benchmark Partyr-{} IP", i),
                                host_ips[i]);
  }

  // setup bench_lctx and link
  yasl::link::FactoryBrpc factory;
  BenchInteral::bench_lctx = factory.CreateContext(lctx_desc, rank);
  BenchInteral::bench_lctx->ConnectToMesh();

  BenchInteral::bench_numel = numel;
  BenchInteral::bench_shiftbit = shiftbit;
  benchmark::AddCustomContext("Benchmark Data Size", std::to_string(numel));
  benchmark::AddCustomContext("Benchmark Shift Bits", std::to_string(shiftbit));
  benchmark::AddCustomContext("Benchmark Protocol", protocol);
}

// the main function
int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto bench_rank = cli_rank.getValue();
  auto bench_parties = cli_parties.getValue();
  auto bench_party_num = cli_party_num.getValue();
  auto bench_numel = cli_numel.getValue();
  auto bench_shiftbit = cli_shiftbit.getValue();
  auto bench_protocol = cli_protocol.getValue();

  PrepareBenchmark(bench_rank, bench_parties, bench_party_num, bench_numel,
                   bench_shiftbit, bench_protocol);

  // these entries are from BENCHMARK_MAIN
  // ::benchmark::Initialize(&argc, argv); // remove all benchmark flags
  // if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  // sync close
  spu::mpc::bench::ComputeBench::bench_lctx->WaitLinkTaskFinish();

  return 0;
}
