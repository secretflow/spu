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
#include "yasl/base/exception.h"
#include "yasl/crypto/hash_util.h"
#include "yasl/link/test_util.h"

#include "spu/psi/core/kkrt_psi.h"

namespace {
std::vector<uint128_t> CreateRangeItems(size_t begin, size_t size) {
  std::vector<uint128_t> ret(size);
  for (size_t i = 0; i < size; i++) {
    std::vector<uint8_t> hash = yasl::crypto::Blake3(std::to_string(begin + i));
    memcpy(&ret[i], hash.data(), sizeof(uint128_t));
  }
  return ret;
}

void KkrtPsiSend(const std::shared_ptr<yasl::link::Context>& link_ctx,
                 const std::vector<uint128_t>& items_hash) {
  yasl::BaseRecvOptions recv_opts;

  spu::psi::GetKkrtOtSenderOptions(link_ctx, 512, &recv_opts);

  return spu::psi::KkrtPsiSend(link_ctx, recv_opts, items_hash);
}

std::vector<std::size_t> KkrtPsiRecv(
    const std::shared_ptr<yasl::link::Context>& link_ctx,
    const std::vector<uint128_t>& items_hash) {
  yasl::BaseSendOptions send_opts;

  spu::psi::GetKkrtOtReceiverOptions(link_ctx, 512, &send_opts);

  return spu::psi::KkrtPsiRecv(link_ctx, send_opts, items_hash);
}

}  // namespace

static void BM_KkrtPsi(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t n = state.range(0);
    auto alice_items = CreateRangeItems(1, n);
    auto bob_items = CreateRangeItems(2, n);

    auto contexts = yasl::link::test::SetupWorld(2);

    state.ResumeTiming();

    std::future<void> kkrt_psi_sender =
        std::async([&] { return KkrtPsiSend(contexts[0], alice_items); });
    std::future<std::vector<std::size_t>> kkrt_psi_receiver =
        std::async([&] { return KkrtPsiRecv(contexts[1], bob_items); });

    kkrt_psi_sender.get();
    auto results_b = kkrt_psi_receiver.get();
  }
}

// [256k, 512k, 1m, 2m, 4m, 8m]
BENCHMARK(BM_KkrtPsi)
    ->Unit(benchmark::kMillisecond)
    ->Arg(256 << 10)
    ->Arg(512 << 10)
    ->Arg(1 << 20)
    ->Arg(2 << 20)
    ->Arg(4 << 20)
    ->Arg(8 << 20);

BENCHMARK_MAIN();
