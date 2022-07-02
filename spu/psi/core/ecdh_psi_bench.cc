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

#include <future>
#include <iostream>
#include <optional>

#include "benchmark/benchmark.h"
#include "yasl/base/exception.h"
#include "yasl/link/test_util.h"

#include "spu/psi/core/ecdh_psi.h"
#include "spu/psi/cryptor/cryptor_selector.h"
#include "spu/psi/provider/batch_provider_impl.h"
#include "spu/psi/store/cipher_store_impl.h"

namespace {

std::vector<std::string> CreateRangeItems(size_t begin, size_t size) {
  std::vector<std::string> ret(size);
  for (size_t i = 0; i < size; i++) {
    ret[i] = std::to_string(begin + i);
  }
  return ret;
}

std::optional<spu::CurveType> GetOverrideCurveType() {
  if (const auto* env = std::getenv("OVERRIDE_CURVE")) {
    if (std::strcmp(env, "25519") == 0) {
      return spu::CurveType::Curve25519;
    }
    if (std::strcmp(env, "FOURQ") == 0) {
      return spu::CurveType::CurveFourQ;
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

    auto ctxs = yasl::link::test::SetupWorld(2);
    auto proc = [](std::shared_ptr<yasl::link::Context> ctx,
                   const std::vector<std::string>& items,
                   size_t target_rank) -> std::vector<std::string> {
      spu::psi::PsiOptions options;
      auto memory_store = std::make_shared<spu::psi::MemoryCipherStore>();
      {
        const auto curve = GetOverrideCurveType();
        options.ecc_cryptor = spu::CreateEccCryptor(
            curve.has_value() ? *curve : spu::CurveType::Curve25519);
        options.batch_provider =
            std::make_shared<spu::psi::MemoryBatchProvider>(items);
        options.cipher_store = memory_store;
        options.link_ctx = ctx;
        options.target_rank = target_rank;
        options.ecc_cryptor = CreateEccCryptor(spu::CurveType::Curve25519);
      }

      spu::psi::RunEcdhPsi(options);

      std::vector<std::string> ret;
      std::vector<std::string> peer_results(memory_store->peer_results());
      std::sort(peer_results.begin(), peer_results.end());
      const auto& self_results = memory_store->self_results();
      for (uint32_t index = 0; index < self_results.size(); index++) {
        if (std::binary_search(peer_results.begin(), peer_results.end(),
                               self_results[index])) {
          YASL_ENFORCE(index < items.size());
          ret.push_back(items[index]);
        }
      }
      return ret;
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

BENCHMARK_MAIN();
