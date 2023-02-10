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

#include <future>
#include <iostream>
#include <optional>

#include "benchmark/benchmark.h"
#include "yacl/link/test_util.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/ecdh_psi.h"
#include "libspu/psi/cryptor/cryptor_selector.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/cipher_store.h"

namespace {

std::vector<std::string> CreateRangeItems(size_t begin, size_t size) {
  std::vector<std::string> ret(size);
  for (size_t i = 0; i < size; i++) {
    ret[i] = std::to_string(begin + i);
  }
  return ret;
}

std::optional<spu::psi::CurveType> GetOverrideCurveType() {
  if (const auto* env = std::getenv("OVERRIDE_CURVE")) {
    if (std::strcmp(env, "25519") == 0) {
      return spu::psi::CurveType::CURVE_25519;
    }
    if (std::strcmp(env, "FOURQ") == 0) {
      return spu::psi::CurveType::CURVE_FOURQ;
    }
  }
  return {};
}

}  // namespace

static void BM_EcdhPsi(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);
    auto alice_items = CreateRangeItems(1, n);
    auto bob_items = CreateRangeItems(2, n);

    auto ctxs = yacl::link::test::SetupWorld(2);
    auto proc = [](const std::shared_ptr<yacl::link::Context>& ctx,
                   const std::vector<std::string>& items,
                   size_t target_rank) -> std::vector<std::string> {
      const auto curve = GetOverrideCurveType();
      return spu::psi::RunEcdhPsi(
          ctx, items, target_rank,
          curve.has_value() ? *curve : spu::psi::CurveType::CURVE_25519);
    };

    state.ResumeTiming();

    std::future<std::vector<std::string>> fa =
        std::async(proc, ctxs[0], alice_items, 0);
    std::future<std::vector<std::string>> fb =
        std::async(proc, ctxs[1], bob_items, 0);

    auto results_a = fa.get();
    auto results_b = fb.get();
  }
}

// [256k, 512k, 1m, 2m, 4m, 8m]
BENCHMARK(BM_EcdhPsi)
    ->Arg(256 << 10)
    ->Arg(512 << 10)
    ->Arg(1 << 20)
    ->Arg(2 << 20)
    ->Arg(4 << 20)
    ->Arg(8 << 20);
