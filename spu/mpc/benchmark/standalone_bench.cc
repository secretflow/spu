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

#include "standalone_bench.h"

#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "utils.h"
#include "yasl/link/transport/channel_brpc.h"

#include "spu/mpc/aby3/protocol.h"
#include "spu/mpc/semi2k/protocol.h"

namespace spu::mpc::bench {

void DefaultBMArguments(benchmark::internal::Benchmark* b) {
  // b->Args({FieldType::FM32})->Args({FieldType::FM64})->Args({FieldType::FM128});

  // Uncomment the following if you want to manually config benchmark iterations
  b->Args({FieldType::FM32})
      ->Args({FieldType::FM64})
      ->Args({FieldType::FM128})
      ->Iterations(100);
}

// register benchmarks with arguments
SPU_BM_PROTOCOL_REGISTER(DefaultBMArguments);

}  // namespace spu::mpc::bench

void PrepareBenchmark(uint32_t party_num, uint32_t numel, uint32_t shiftbit,
                      std::string& protocol) {
  using BenchInteral = spu::mpc::bench::ComputeBench;

  if (protocol == "semi2k") {
    BenchInteral::bench_factory =
        spu::mpc::makeSemi2kProtocol;  // semi2k protocol factory
    if (party_num == 0) {
      party_num = 2;
    }
  } else if (protocol == "aby3") {
    BenchInteral::bench_factory =
        spu::mpc::makeAby3Protocol;  // semi2k protocol factory
    if (party_num == 0) {
      party_num = 3;
    }
  } else {
    YASL_THROW("unknown protocol: {}, supported = semi2k/aby", protocol);
  }

  BenchInteral::bench_npc = party_num;

  benchmark::AddCustomContext("Benchmark Party Number",
                              std::to_string(party_num));
  benchmark::AddCustomContext("Benchmark Data Size", std::to_string(numel));
  benchmark::AddCustomContext("Benchmark Shift Bits", std::to_string(shiftbit));
  benchmark::AddCustomContext("Benchmark Protocol", protocol);
}

// the main function
int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  auto bench_party_num = cli_party_num.getValue();
  auto bench_numel = cli_numel.getValue();
  auto bench_shiftbit = cli_shiftbit.getValue();
  auto bench_protocol = cli_protocol.getValue();

  PrepareBenchmark(bench_party_num, bench_numel, bench_shiftbit,
                   bench_protocol);

  // these entries are from BENCHMARK_MAIN
  // ::benchmark::Initialize(&argc, argv); // remove all benchmark flags
  // if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}
