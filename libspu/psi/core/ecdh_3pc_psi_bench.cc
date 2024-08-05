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

#include "benchmark/benchmark.h"
#include "yacl/link/test_util.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/ecdh_3pc_psi.h"
#include "libspu/psi/utils/test_utils.h"

static void BM_Ecdh3PcPsi(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);
    auto alice_items = spu::psi::test::CreateRangeItems(1, n);
    auto bob_items = spu::psi::test::CreateRangeItems(2, n);
    auto carol_items = spu::psi::test::CreateRangeItems(3, n);

    auto contexts = yacl::link::test::SetupWorld(3);

    // simple runner
    auto psi_func =
        [&](const std::shared_ptr<spu::psi::ShuffleEcdh3PcPsi>& handler,
            const std::vector<std::string>& items,
            std::vector<std::string>* results) {
          std::vector<std::string> masked_master_items;
          std::vector<std::string> partner_psi_items;

          auto mask_master = std::async(
              [&] { return handler->MaskMaster(items, &masked_master_items); });
          auto partner_psi = std::async(
              [&] { return handler->PartnersPsi(items, &partner_psi_items); });

          mask_master.get();
          partner_psi.get();

          handler->FinalPsi(items, masked_master_items, partner_psi_items,
                            results);
        };

    state.ResumeTiming();

    std::vector<std::string> alice_res;
    std::vector<std::string> bob_res;
    std::vector<std::string> carol_res;
    auto alice_func = std::async([&]() {
      spu::psi::ShuffleEcdh3PcPsi::Options opts;
      opts.link_ctx = contexts[0];
      opts.master_rank = 0;
      auto op = std::make_shared<spu::psi::ShuffleEcdh3PcPsi>(opts);
      return psi_func(op, alice_items, &alice_res);
    });
    auto bob_func = std::async([&]() {
      spu::psi::ShuffleEcdh3PcPsi::Options opts;
      opts.link_ctx = contexts[1];
      opts.master_rank = 0;
      auto op = std::make_shared<spu::psi::ShuffleEcdh3PcPsi>(opts);
      return psi_func(op, bob_items, &bob_res);
    });
    auto carol_func = std::async([&]() {
      spu::psi::ShuffleEcdh3PcPsi::Options opts;
      opts.link_ctx = contexts[2];
      opts.master_rank = 0;
      auto op = std::make_shared<spu::psi::ShuffleEcdh3PcPsi>(opts);
      return psi_func(op, carol_items, &carol_res);
    });

    alice_func.get();
    bob_func.get();
    carol_func.get();
  }
}

// [256k, 512k, 1m, 2m, 4m, 8m]
BENCHMARK(BM_Ecdh3PcPsi)
    ->Unit(benchmark::kMillisecond)
    ->Arg(256 << 10)
    ->Arg(512 << 10)
    ->Arg(1 << 20)
    ->Arg(2 << 20);
