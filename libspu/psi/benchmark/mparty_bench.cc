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

#include "libspu/psi/benchmark/mparty_bench.h"

#include <string>
#include <vector>

namespace spu::psi::bench {

void DefaultPsiArguments(benchmark::internal::Benchmark* b) {
  b->Args({1 << 18})
      ->Args({1 << 20})
      ->Args({1 << 22})
      ->Args({1 << 24})
      ->Args({1000000})
      ->Args({5000000})
      ->Args({10000000})
      ->Iterations(1)
      ->Unit(benchmark::kSecond);
}

// register benchmarks with arguments
BM_REGISTER_ALL_PSI(DefaultPsiArguments);
//
// Equivalent to the following:
//
// BM_REGISTER_ECDH_PSI(DefaultPsiArguments);
// BM_REGISTER_ECDH_OPRF_PSI(DefaultPsiArguments);
// BM_REGISTER_KKRT_PSI(DefaultPsiArguments);
// BM_REGISTER_BC22_PSI(DefaultPsiArguments);
// BM_REGISTER_MINI_PSI(DefaultPsiArguments);

}  // namespace spu::psi::bench

namespace {
void PreparePsiBench(const uint32_t rank, const std::string& parties) {
  std::vector<std::string> host_ips;
  if (parties.empty()) {
    // default ips for semi2k
    host_ips = absl::StrSplit(spu::psi::bench::kTwoPartyHosts, ',');
  } else {
    host_ips = absl::StrSplit(parties, ',');
  }
  SPU_ENFORCE(host_ips.size() == 2);

  yacl::link::ContextDesc lctx_desc;
  for (size_t i = 0; i < 2; i++) {
    const std::string id = fmt::format("party{}", i);
    lctx_desc.parties.push_back({id, host_ips[i]});
    benchmark::AddCustomContext(fmt::format("Benchmark Party-{} IP", i),
                                host_ips[i]);
  }

  // setup bench_lctx and link
  yacl::link::FactoryBrpc factory;
  spu::psi::bench::PsiBench::bench_lctx =
      factory.CreateContext(lctx_desc, rank);
  spu::psi::bench::PsiBench::bench_lctx->ConnectToMesh();
}
}  // namespace

// the main function
int main(int argc, char** argv) {
  if (!llvm::cl::ParseCommandLineOptions(argc, argv)) {
    llvm::cl::PrintHelpMessage();
    exit(EXIT_FAILURE);
  }
  try {
    auto rank = cli_rank.getValue();
    auto parties = cli_parties.getValue();
    PreparePsiBench(rank, parties);

    // these entries are from BENCHMARK_MAIN
    // ::benchmark::Initialize(&argc, argv); // remove all benchmark flags
    // if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    // sync close
    spu::psi::bench::PsiBench::bench_lctx->WaitLinkTaskFinish();
  } catch (std::exception& e) {
    exit(EXIT_FAILURE);
  }

  return 0;
}
