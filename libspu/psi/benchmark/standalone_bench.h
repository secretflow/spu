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

#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "benchmark/benchmark.h"
#include "yacl/base/int128.h"
#include "yacl/link/test_util.h"

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

namespace spu::psi::bench {

namespace {

void WriteCsvFile(const std::string& file_name,
                  const std::vector<std::string>& items) {
  auto out = io::BuildOutputStream(io::FileIoOptions(file_name));
  out->Write("id\n");
  for (const auto& data : items) {
    out->Write(fmt::format("{}\n", data));
  }
  out->Close();
}

}  // namespace

class PsiBench : public benchmark::Fixture {
 public:
  PsiBench() {
    spdlog::set_level(spdlog::level::off);  // turn off spdlog
  }
};

#define PSI_BM_DEFINE_ECDH_TYPE(CurveType)                                \
  BENCHMARK_DEFINE_F(PsiBench, EcdhPsi_##CurveType)                       \
  (benchmark::State & state) {                                            \
    for (auto _ : state) {                                                \
      state.PauseTiming();                                                \
      size_t numel = state.range(0);                                      \
      auto a_items = psi::test::CreateRangeItems(1, numel);               \
      auto b_items = psi::test::CreateRangeItems(2, numel);               \
      auto ctxs = yacl::link::test::SetupWorld(2);                        \
      auto proc = [](const std::shared_ptr<yacl::link::Context>& ctx,     \
                     const std::vector<std::string>& items,               \
                     size_t target_rank) -> std::vector<std::string> {    \
        const auto curve = psi::test::GetOverrideCurveType();             \
        return psi::RunEcdhPsi(ctx, items, target_rank,                   \
                               curve.has_value() ? *curve : (CurveType)); \
      };                                                                  \
                                                                          \
      state.ResumeTiming();                                               \
                                                                          \
      auto fa = std::async(proc, ctxs[0], a_items, 0);                    \
      auto fb = std::async(proc, ctxs[1], b_items, 0);                    \
                                                                          \
      auto results_a = fa.get();                                          \
      auto results_b = fb.get();                                          \
    }                                                                     \
  }

#define PSI_BM_DEFINE_ECDH()            \
  PSI_BM_DEFINE_ECDH_TYPE(CURVE_25519); \
  PSI_BM_DEFINE_ECDH_TYPE(CURVE_FOURQ); \
  PSI_BM_DEFINE_ECDH_TYPE(CURVE_SM2);   \
  PSI_BM_DEFINE_ECDH_TYPE(CURVE_SECP256K1);

PSI_BM_DEFINE_ECDH()

#define ECDH_OPRF_SENDER_OFFLINE()                                             \
  {                                                                            \
    /* offline: init */                                                        \
    auto timestamp_str = std::to_string(absl::ToUnixNanos(absl::Now()));       \
    /* server input */                                                         \
    auto server_input_path =                                                   \
        std::filesystem::path(fmt::format("server-input-{}", timestamp_str));  \
                                                                               \
    /* server output */                                                        \
    auto server_tmp_cache_path =                                               \
        std::filesystem::path(fmt::format("tmp-cache-{}", timestamp_str));     \
    /* register remove of temp file. */                                        \
    ON_SCOPE_EXIT([&] {                                                        \
      std::error_code ec;                                                      \
      std::filesystem::remove(server_input_path, ec);                          \
      if (ec.value() != 0) {                                                   \
        SPDLOG_WARN("can not remove tmp file: {}, msg: {}",                    \
                    server_input_path.c_str(), ec.message());                  \
      }                                                                        \
      std::filesystem::remove(server_tmp_cache_path, ec);                      \
      if (ec.value() != 0) {                                                   \
        SPDLOG_WARN("can not remove tmp file: {}, msg: {}",                    \
                    server_tmp_cache_path.c_str(), ec.message());              \
      }                                                                        \
    });                                                                        \
                                                                               \
    WriteCsvFile(server_input_path.string(), items);                           \
    std::vector<std::string> cloumn_ids = {"id"};                              \
    std::shared_ptr<CachedCsvBatchProvider> item_provider =                    \
        std::make_shared<CachedCsvBatchProvider>(server_input_path.string(),   \
                                                 cloumn_ids, 100000, true);    \
                                                                               \
    std::shared_ptr<IUbPsiCache> ub_cache = std::make_shared<UbPsiCache>(      \
        server_tmp_cache_path.string(), offline_proc.GetCompareLength(),       \
        cloumn_ids);                                                           \
    offline_proc.FullEvaluate(item_provider, ub_cache);                        \
                                                                               \
    /* offline: finalize */                                                    \
    std::shared_ptr<IBatchProvider> batch_provider =                           \
        std::make_shared<UbPsiCacheProvider>(server_tmp_cache_path.string(),   \
                                             offline_proc.GetCompareLength()); \
    offline_proc.SendFinalEvaluatedItems(batch_provider);                      \
  }

#define ECDH_OPRF_SENDER_ONLINE()           \
  { /* online */                            \
    online_proc.RecvBlindAndSendEvaluate(); \
  }

#define ECDH_OPRF_RECEIVER_OFFLINE()                           \
  {                                                            \
    /* offline */                                              \
    auto memory_store = std::make_shared<MemoryCipherStore>(); \
    offline_proc.RecvFinalEvaluatedItems(memory_store);        \
  }

#define ECDH_OPRF_RECEIVER_ONLINE()                                        \
  { /* online */                                                           \
    auto memory_store = std::make_shared<MemoryCipherStore>();             \
    auto proc_send = std::async([&] {                                      \
      auto item_provider = std::make_shared<MemoryBatchProvider>(items);   \
      online_proc.SendBlindedItems(item_provider);                         \
    });                                                                    \
                                                                           \
    auto proc_recv =                                                       \
        std::async([&] { online_proc.RecvEvaluatedItems(memory_store); }); \
                                                                           \
    proc_send.get();                                                       \
    proc_recv.get();                                                       \
  }

#define PSI_BM_DEFINE_ECDH_OPRF_FULL(CurveType)                         \
  BENCHMARK_DEFINE_F(PsiBench, EcdhPsiOprf_##CurveType)                 \
  (benchmark::State & state) {                                          \
    for (auto _ : state) {                                              \
      state.PauseTiming();                                              \
      size_t numel = state.range(0);                                    \
      auto a_items = psi::test::CreateRangeItems(1, numel);             \
      auto b_items = psi::test::CreateRangeItems(2, numel);             \
      auto ctxs = yacl::link::test::SetupWorld(2);                      \
                                                                        \
      /* We let bob obtains the final result */                         \
      auto a_proc = [](const std::shared_ptr<yacl::link::Context>& ctx, \
                       const std::vector<std::string>& items) {         \
        EcdhOprfPsiOptions options;                                     \
        options.curve_type = (CurveType);                               \
        options.link0 = ctx;                                            \
        options.link1 = ctx->Spawn();                                   \
                                                                        \
        /* Offline Phase */                                             \
        auto offline_proc = EcdhOprfPsiServer(options);                 \
        ECDH_OPRF_SENDER_OFFLINE()                                      \
                                                                        \
        /* Online Phase */                                              \
        const auto sk = offline_proc.GetPrivateKey();                   \
        auto online_proc = EcdhOprfPsiServer(options, sk);              \
        ECDH_OPRF_SENDER_ONLINE()                                       \
      };                                                                \
                                                                        \
      auto b_proc = [](const std::shared_ptr<yacl::link::Context>& ctx, \
                       const std::vector<std::string>& items) {         \
        EcdhOprfPsiOptions options;                                     \
        options.curve_type = (CurveType);                               \
        options.link0 = ctx;                                            \
        options.link1 = ctx->Spawn();                                   \
                                                                        \
        /* Offline Phase */                                             \
        auto offline_proc = EcdhOprfPsiClient(options);                 \
        ECDH_OPRF_RECEIVER_OFFLINE()                                    \
                                                                        \
        /* Online Phase */                                              \
        auto online_proc = EcdhOprfPsiClient(options);                  \
        ECDH_OPRF_RECEIVER_ONLINE()                                     \
      };                                                                \
                                                                        \
      state.ResumeTiming();                                             \
                                                                        \
      auto fa = std::async(a_proc, ctxs[0], a_items);                   \
      auto fb = std::async(b_proc, ctxs[1], b_items);                   \
      fa.get();                                                         \
      fb.get();                                                         \
    }                                                                   \
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
    auto a_items = psi::test::CreateItemHashes(1, numel);
    auto b_items = psi::test::CreateItemHashes(2, numel);
    auto ctxs = yacl::link::test::SetupWorld(2);

    /* Sender */
    auto a_proc = [](const std::shared_ptr<yacl::link::Context>& ctx,
                     const std::vector<uint128_t>& items) {
      auto ot_recv = psi::GetKkrtOtSenderOptions(ctx, 512);
      psi::KkrtPsiSend(ctx, ot_recv, items);
    };

    /* Receiver */
    auto b_proc = [](const std::shared_ptr<yacl::link::Context>& ctx,
                     const std::vector<uint128_t>& items) {
      auto ot_send = psi::GetKkrtOtReceiverOptions(ctx, 512);
      return psi::KkrtPsiRecv(ctx, ot_send, items);
    };

    state.ResumeTiming();

    auto fa = std::async(a_proc, ctxs[0], a_items);
    auto fb = std::async(b_proc, ctxs[1], b_items);

    fa.get();
    auto results = fb.get();
  }
}

BENCHMARK_DEFINE_F(PsiBench, Bc22Psi)
(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    size_t numel = state.range(0);
    auto a_items = psi::test::CreateRangeItems(1, numel);
    auto b_items = psi::test::CreateRangeItems(2, numel);
    auto ctxs = yacl::link::test::SetupWorld(2);

    Bc22PcgPsi sender(ctxs[0], PsiRoleType::Sender);
    Bc22PcgPsi receiver(ctxs[1], PsiRoleType::Receiver);

    /* Sender */
    auto a_proc = [](const std::shared_ptr<yacl::link::Context>& ctx,
                     const std::vector<std::string>& items) {
      Bc22PcgPsi party(ctx, PsiRoleType::Sender);
      party.RunPsi(items);
    };

    /* Receiver */
    auto b_proc = [](const std::shared_ptr<yacl::link::Context>& ctx,
                     const std::vector<std::string>& items) {
      Bc22PcgPsi party(ctx, PsiRoleType::Receiver);
      party.RunPsi(items);
      return party.GetIntersection();
    };

    state.ResumeTiming();

    auto fa = std::async(a_proc, ctxs[0], a_items);
    auto fb = std::async(b_proc, ctxs[1], b_items);

    fa.get();
    auto results = fb.get();
  }
}

#define PSI_BM_DEFINE_MINI_TYPE(IsBatch)                                \
  BENCHMARK_DEFINE_F(PsiBench, MiniPsi##_##IsBatch)                     \
  (benchmark::State & state) {                                          \
    for (auto _ : state) {                                              \
      state.PauseTiming();                                              \
      size_t numel = state.range(0);                                    \
      auto a_items = psi::test::CreateRangeItems(1, numel);             \
      auto b_items = psi::test::CreateRangeItems(2, numel);             \
      auto ctxs = yacl::link::test::SetupWorld(2);                      \
                                                                        \
      /* Sender */                                                      \
      auto a_proc = [](const std::shared_ptr<yacl::link::Context>& ctx, \
                       const std::vector<std::string>& items) {         \
        psi::MiniPsiSend(ctx, items);                                   \
      };                                                                \
                                                                        \
      /* Receiver */                                                    \
      auto b_proc = [](const std::shared_ptr<yacl::link::Context>& ctx, \
                       const std::vector<std::string>& items) {         \
        psi::MiniPsiRecv(ctx, items);                                   \
      };                                                                \
                                                                        \
      state.ResumeTiming();                                             \
                                                                        \
      auto fa = std::async(a_proc, ctxs[0], a_items);                   \
      auto fb = std::async(b_proc, ctxs[1], b_items);                   \
                                                                        \
      fa.get();                                                         \
      fb.get();                                                         \
    }                                                                   \
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
