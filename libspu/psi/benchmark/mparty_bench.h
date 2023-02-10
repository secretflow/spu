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

#pragma once

#include <algorithm>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "llvm/Support/CommandLine.h"
#include "yacl/base/int128.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/bc22_psi/bc22_psi.h"
#include "libspu/psi/core/ecdh_oprf_psi.h"
#include "libspu/psi/core/ecdh_psi.h"
#include "libspu/psi/core/kkrt_psi.h"
#include "libspu/psi/core/mini_psi.h"
#include "libspu/psi/cryptor/cryptor_selector.h"
#include "libspu/psi/utils/batch_provider.h"
#include "libspu/psi/utils/cipher_store.h"
#include "libspu/psi/utils/test_utils.h"

llvm::cl::opt<uint32_t> cli_rank("rank", llvm::cl::init(0),
                                 llvm::cl::desc("self rank, starts with 0"));
llvm::cl::opt<std::string> cli_parties(
    "parties",
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

namespace spu::psi::bench {

const char kTwoPartyHosts[] = "127.0.0.1:9540,127.0.0.1:9541";

class PsiBench : public benchmark::Fixture {
 public:
  static std::shared_ptr<yacl::link::Context> bench_lctx;
  PsiBench() {
    spdlog::set_level(spdlog::level::off);  // turn off spdlog
  }
};

std::shared_ptr<yacl::link::Context> PsiBench::bench_lctx = nullptr;

#define PSI_BM_DEFINE_ECDH_TYPE(CurveType)                                 \
  BENCHMARK_DEFINE_F(PsiBench, EcdhPsi_##CurveType)                        \
  (benchmark::State & state) {                                             \
    for (auto _ : state) {                                                 \
      state.PauseTiming();                                                 \
      size_t numel = state.range(0);                                       \
      auto items = psi::test::CreateRangeItems(bench_lctx->Rank(), numel); \
      const auto curve = psi::test::GetOverrideCurveType();                \
                                                                           \
      state.ResumeTiming();                                                \
                                                                           \
      psi::RunEcdhPsi(bench_lctx, items, 0,                                \
                      curve.has_value() ? *curve : (CurveType));           \
    }                                                                      \
  }

#define PSI_BM_DEFINE_ECDH()            \
  PSI_BM_DEFINE_ECDH_TYPE(CURVE_25519); \
  PSI_BM_DEFINE_ECDH_TYPE(CURVE_FOURQ); \
  PSI_BM_DEFINE_ECDH_TYPE(CURVE_SM2);   \
  PSI_BM_DEFINE_ECDH_TYPE(CURVE_SECP256K1);

PSI_BM_DEFINE_ECDH()

#define PSI_BM_DEFINE_ECDH_OPRF_FULL(CurveType)                              \
  BENCHMARK_DEFINE_F(PsiBench, EcdhPsiOprf_##CurveType)                      \
  (benchmark::State & state) {                                               \
    for (auto _ : state) {                                                   \
      state.PauseTiming();                                                   \
      size_t numel = state.range(0);                                         \
      auto items = psi::test::CreateRangeItems(bench_lctx->Rank(), numel);   \
                                                                             \
      /* We let bob obtains the final result */                              \
      state.ResumeTiming();                                                  \
      if (bench_lctx->Rank() == 0) {                                         \
        EcdhOprfPsiOptions options;                                          \
        options.curve_type = (CurveType);                                    \
        options.link0 = bench_lctx;                                          \
        options.link1 = bench_lctx->Spawn();                                 \
        auto memory_store = std::make_shared<MemoryCipherStore>();           \
        auto offline_proc = EcdhOprfPsiServer(options);                      \
        const auto sk = offline_proc.GetPrivateKey();                        \
        auto online_proc = EcdhOprfPsiServer(options, sk);                   \
                                                                             \
        /* offline: init */                                                  \
        auto item_provider = std::make_shared<MemoryBatchProvider>(items);   \
        offline_proc.FullEvaluate(item_provider, memory_store);              \
        auto& evaluate_items = memory_store->self_results();                 \
                                                                             \
        /* offline: shuffle */                                               \
        std::random_device rd;                                               \
        std::mt19937 rng(rd());                                              \
        std::shuffle(evaluate_items.begin(), evaluate_items.end(), rng);     \
                                                                             \
        /* offline: finalize */                                              \
        const auto cache_store =                                             \
            std::make_shared<MemoryBatchProvider>(evaluate_items);           \
        offline_proc.SendFinalEvaluatedItems(cache_store);                   \
                                                                             \
        /* online */                                                         \
        online_proc.RecvBlindAndSendEvaluate();                              \
      } else {                                                               \
        EcdhOprfPsiOptions options;                                          \
        options.curve_type = (CurveType);                                    \
        options.link0 = bench_lctx;                                          \
        options.link1 = bench_lctx->Spawn();                                 \
        auto memory_store = std::make_shared<MemoryCipherStore>();           \
        auto offline_proc = EcdhOprfPsiClient(options);                      \
        auto online_proc = EcdhOprfPsiClient(options);                       \
                                                                             \
        /* offline: recv and evaluate */                                     \
        offline_proc.RecvFinalEvaluatedItems(memory_store);                  \
                                                                             \
        /* online */                                                         \
        auto proc_send = std::async([&] {                                    \
          auto item_provider = std::make_shared<MemoryBatchProvider>(items); \
          online_proc.SendBlindedItems(item_provider);                       \
        });                                                                  \
                                                                             \
        auto proc_recv = std::async([&] {                                    \
          auto item_provider = std::make_shared<MemoryBatchProvider>(items); \
          online_proc.RecvEvaluatedItems(item_provider, memory_store);       \
        });                                                                  \
                                                                             \
        proc_send.get();                                                     \
        proc_recv.get();                                                     \
                                                                             \
        /* online: finalize */                                               \
        auto& peer_results = memory_store->peer_results();                   \
        auto& self_results = memory_store->self_results();                   \
        std::sort(peer_results.begin(), peer_results.end());                 \
                                                                             \
        std::vector<std::string> final_result;                               \
        for (size_t i = 0; i < self_results.size(); i++) {                   \
          if (std::binary_search(peer_results.begin(), peer_results.end(),   \
                                 self_results[i])) {                         \
            final_result.push_back(std::to_string(i + 1));                   \
          }                                                                  \
        }                                                                    \
      }                                                                      \
    }                                                                        \
  }

#define PSI_BM_DEFINE_ECDH_OPRF()            \
  PSI_BM_DEFINE_ECDH_OPRF_FULL(CURVE_25519); \
  PSI_BM_DEFINE_ECDH_OPRF_FULL(CURVE_FOURQ); \
  PSI_BM_DEFINE_ECDH_OPRF_FULL(CURVE_SM2);   \
  PSI_BM_DEFINE_ECDH_OPRF_FULL(CURVE_SECP256K1);

PSI_BM_DEFINE_ECDH_OPRF()

BENCHMARK_DEFINE_F(PsiBench, KkrtPsi)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t numel = state.range(0);
    auto items = psi::test::CreateItemHashes(bench_lctx->Rank(), numel);

    state.ResumeTiming();

    if (bench_lctx->Rank() == 0) { /* Sender */
      yacl::crypto::OtRecvStore recv_opts;
      psi::GetKkrtOtSenderOptions(bench_lctx, 512, &recv_opts);
      psi::KkrtPsiSend(bench_lctx, recv_opts, items);
    } else { /* Receiver */
      yacl::crypto::OtSendStore send_opts;
      psi::GetKkrtOtReceiverOptions(bench_lctx, 512, &send_opts);
      psi::KkrtPsiRecv(bench_lctx, send_opts, items);
    }
  }
}

BENCHMARK_DEFINE_F(PsiBench, Bc22Psi)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t numel = state.range(0);
    auto items = psi::test::CreateRangeItems(bench_lctx->Rank(), numel);

    state.ResumeTiming();

    if (bench_lctx->Rank() == 0) { /* Sender */
      Bc22PcgPsi party(bench_lctx, PsiRoleType::Sender);
      party.RunPsi(items);
    } else { /* Receiver */
      Bc22PcgPsi party(bench_lctx, PsiRoleType::Receiver);
      party.RunPsi(items);
      party.GetIntersection();
    }
  }
}

#define PSI_BM_DEFINE_MINI_TYPE(IsBatch)                                   \
  BENCHMARK_DEFINE_F(PsiBench, MiniPsi##_##IsBatch)                        \
  (benchmark::State & state) {                                             \
    for (auto _ : state) {                                                 \
      state.PauseTiming();                                                 \
      size_t numel = state.range(0);                                       \
      auto items = psi::test::CreateRangeItems(bench_lctx->Rank(), numel); \
                                                                           \
      state.ResumeTiming();                                                \
      if (bench_lctx->Rank() == 0) { /* Sender */                          \
        psi::MiniPsiSend(bench_lctx, items);                               \
      } else { /* Receiver */                                              \
        psi::MiniPsiRecv(bench_lctx, items);                               \
      }                                                                    \
    }                                                                      \
  }

#define PSI_BM_DEFINE_MINI()       \
  PSI_BM_DEFINE_MINI_TYPE(NoBatch) \
  PSI_BM_DEFINE_MINI_TYPE(Batch)
PSI_BM_DEFINE_MINI()

#define PSI_BM_REGISTER_CURVE_PSI_TYPE(PsiType, CurveType, Arguments) \
  BENCHMARK_REGISTER_F(PsiBench, PsiType##_##CurveType)->Apply(Arguments);

#define PSI_BM_REGISTER_CURVE_PSI(PsiType, Arguments)              \
  PSI_BM_REGISTER_CURVE_PSI_TYPE(PsiType, CURVE_25519, Arguments); \
  PSI_BM_REGISTER_CURVE_PSI_TYPE(PsiType, CURVE_FOURQ, Arguments); \
  PSI_BM_REGISTER_CURVE_PSI_TYPE(PsiType, CURVE_SM2, Arguments);   \
  PSI_BM_REGISTER_CURVE_PSI_TYPE(PsiType, CURVE_SECP256K1, Arguments);

#define BM_REGISTER_ECDH_PSI(Arguments) \
  PSI_BM_REGISTER_CURVE_PSI(EcdhPsi, Arguments)

#define BM_REGISTER_ECDH_OPRF_PSI(Arguments)                           \
  /* Currently, ECDH OPRF does not support Curve25518 donna */         \
  PSI_BM_REGISTER_CURVE_PSI_TYPE(EcdhPsiOprf, CURVE_FOURQ, Arguments); \
  PSI_BM_REGISTER_CURVE_PSI_TYPE(EcdhPsiOprf, CURVE_SM2, Arguments);   \
  PSI_BM_REGISTER_CURVE_PSI_TYPE(EcdhPsiOprf, CURVE_SECP256K1, Arguments);

#define BM_REGISTER_KKRT_PSI(Arguments) \
  BENCHMARK_REGISTER_F(PsiBench, KkrtPsi)->Apply(Arguments);

#define BM_REGISTER_BC22_PSI(Arguments) \
  BENCHMARK_REGISTER_F(PsiBench, Bc22Psi)->Apply(Arguments);

#define BM_REGISTER_MINI_PSI(Arguments)                              \
  BENCHMARK_REGISTER_F(PsiBench, MiniPsi_NoBatch)->Apply(Arguments); \
  BENCHMARK_REGISTER_F(PsiBench, MiniPsi_Batch)->Apply(Arguments);

#define BM_REGISTER_ALL_PSI(Arguments)  \
  BM_REGISTER_ECDH_PSI(Arguments);      \
  BM_REGISTER_ECDH_OPRF_PSI(Arguments); \
  BM_REGISTER_KKRT_PSI(Arguments);      \
  BM_REGISTER_BC22_PSI(Arguments);      \
  BM_REGISTER_MINI_PSI(Arguments);

}  // namespace spu::psi::bench
