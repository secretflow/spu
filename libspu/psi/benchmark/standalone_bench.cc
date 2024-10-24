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

#include "libspu/psi/benchmark/standalone_bench.h"

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

// the main function
BENCHMARK_MAIN();
