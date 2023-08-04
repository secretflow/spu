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

#include <chrono>
#include <filesystem>
#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/link/test_util.h"

#include "libspu/core/prelude.h"
#include "libspu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "libspu/psi/core/labeled_psi/psi_params.h"
#include "libspu/psi/core/labeled_psi/receiver.h"
#include "libspu/psi/core/labeled_psi/sender.h"
#include "libspu/psi/core/labeled_psi/sender_kvdb.h"
#include "libspu/psi/core/labeled_psi/sender_memdb.h"

namespace spu::psi {

namespace {

using DurationMillis = std::chrono::duration<double, std::milli>;

constexpr size_t kPsiStartPos = 100;
struct TestParams {
  size_t nr;
  size_t ns;
  bool use_kvdb = true;
};

std::vector<std::string> GenerateData(size_t seed, size_t item_count) {
  yacl::crypto::Prg<uint128_t> prg(seed);

  std::vector<std::string> items;

  for (size_t i = 0; i < item_count; ++i) {
    std::string item(16, '\0');
    prg.Fill(absl::MakeSpan(item.data(), item.length()));
    items.emplace_back(absl::BytesToHexString(item));
  }

  return items;
}

std::vector<std::string> GenerateSenderData(
    size_t seed, size_t item_count,
    const absl::Span<std::string> &receiver_items,
    std::vector<size_t> *intersection_idx) {
  std::vector<std::string> sender_items;

  yacl::crypto::Prg<uint128_t> prg(seed);

  for (size_t i = 0; i < item_count; ++i) {
    std::string item(16, '\0');
    prg.Fill(absl::MakeSpan(item.data(), item.size()));
    sender_items.emplace_back(absl::BytesToHexString(item));
  }

  for (size_t i = 0; i < receiver_items.size(); i += 3) {
    if ((kPsiStartPos + i * 5) >= sender_items.size()) {
      break;
    }
    sender_items[kPsiStartPos + i * 5] = receiver_items[i];
    (*intersection_idx).emplace_back(i);
  }

  return sender_items;
}

}  // namespace

class LabelPsiTest : public testing::TestWithParam<TestParams> {};

TEST_P(LabelPsiTest, Works) {
  auto params = GetParam();
  auto ctxs = yacl::link::test::SetupWorld(2);
  apsi::PSIParams psi_params = spu::psi::GetPsiParams(params.nr, params.ns);

  // step 1: PsiParams Request and Response
  std::future<void> f_sender_params = std::async(
      [&] { return LabelPsiSender::RunPsiParams(params.ns, ctxs[0]); });

  std::future<apsi::PSIParams> f_receiver_params = std::async(
      [&] { return LabelPsiReceiver::RequestPsiParams(params.nr, ctxs[1]); });

  f_sender_params.get();
  apsi::PSIParams psi_params2 = f_receiver_params.get();

  EXPECT_EQ(psi_params.table_params().table_size,
            psi_params2.table_params().table_size);

  size_t item_count = params.ns;
  size_t nonce_byte_count = 16;

  std::random_device rd;
  yacl::crypto::Prg<uint128_t> prg(rd());

  std::array<uint8_t, 32> oprf_key;
  prg.Fill(absl::MakeSpan(oprf_key));

  std::string kv_store_path = fmt::format("data_{}", params.ns);
  std::filesystem::create_directory(kv_store_path);
  // register remove of temp dir.
  ON_SCOPE_EXIT([&] {
    if (!kv_store_path.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(kv_store_path, ec);
      if (ec.value() != 0) {
        SPDLOG_WARN("can not remove tmp dir: {}, msg: {}", kv_store_path,
                    ec.message());
      }
    }
  });

  bool compressed = false;
  std::shared_ptr<spu::psi::ISenderDB> sender_db;
  if (params.use_kvdb) {
    sender_db = std::make_shared<spu::psi::SenderKvDB>(
        psi_params, oprf_key, kv_store_path, 0, nonce_byte_count, compressed);
  } else {
    sender_db = std::make_shared<spu::psi::SenderMemDB>(
        psi_params, oprf_key, 0, nonce_byte_count, compressed);
  }

  std::vector<std::string> receiver_items = GenerateData(rd(), params.nr);

  std::vector<size_t> intersection_idx;
  std::vector<std::string> intersection_label;

  // step 2: set database

  const auto setdb_start = std::chrono::system_clock::now();

  std::vector<std::string> sender_items = GenerateSenderData(
      rd(), item_count, absl::MakeSpan(receiver_items), &intersection_idx);

  std::shared_ptr<IBatchProvider> batch_provider =
      std::make_shared<MemoryBatchProvider>(sender_items);

  sender_db->SetData(batch_provider);

  const auto setdb_end = std::chrono::system_clock::now();
  const DurationMillis setdb_duration = setdb_end - setdb_start;
  SPDLOG_INFO("*** step2 set db duration:{}", setdb_duration.count());

  EXPECT_EQ(params.ns, sender_db->GetItemCount());

  SPDLOG_INFO("after set db, bin_bundle_count:{}, packing_rate:{}",
              sender_db->GetBinBundleCount(), sender_db->GetPackingRate());

  const apsi::PSIParams apsi_params = sender_db->GetParams();
  SPDLOG_INFO("params.bundle_idx_count={}", apsi_params.bundle_idx_count());
  for (size_t i = 0; i < apsi_params.bundle_idx_count(); ++i) {
    SPDLOG_INFO("i={},count={}", i, sender_db->GetBinBundleCount(i));
  }

  std::unique_ptr<IEcdhOprfServer> oprf_server =
      CreateEcdhOprfServer(oprf_key, OprfType::Basic, CurveType::CURVE_FOURQ);

  LabelPsiSender sender(sender_db);

  LabelPsiReceiver receiver(psi_params, false);

  // step 3: oprf request and response

  const auto oprf_start = std::chrono::system_clock::now();

  std::future<void> f_sender_oprf = std::async(
      [&] { return sender.RunOPRF(std::move(oprf_server), ctxs[0]); });

  std::future<
      std::pair<std::vector<apsi::HashedItem>, std::vector<apsi::LabelKey>>>
      f_receiver_oprf = std::async(
          [&] { return receiver.RequestOPRF(receiver_items, ctxs[1]); });

  f_sender_oprf.get();
  std::pair<std::vector<apsi::HashedItem>, std::vector<apsi::LabelKey>>
      oprf_pair = f_receiver_oprf.get();

  const auto oprf_end = std::chrono::system_clock::now();
  const DurationMillis oprf_duration = oprf_end - oprf_start;
  SPDLOG_INFO("*** step3 oprf duration:{}", oprf_duration.count());

  SPDLOG_INFO("hashed_item size:{} label keys size:{}", oprf_pair.first.size(),
              oprf_pair.second.size());

  // step 4: Query request and response

  const auto query_start = std::chrono::system_clock::now();

  std::future<void> f_sender_query =
      std::async([&] { return sender.RunQuery(ctxs[0]); });

  std::future<std::pair<std::vector<size_t>, std::vector<std::string>>>
      f_receiver_query = std::async([&] {
        return receiver.RequestQuery(oprf_pair.first, oprf_pair.second,
                                     ctxs[1]);
      });

  f_sender_query.get();
  std::pair<std::vector<size_t>, std::vector<std::string>> query_result =
      f_receiver_query.get();

  const auto query_end = std::chrono::system_clock::now();
  const DurationMillis query_duration = query_end - query_start;
  SPDLOG_INFO("*** step4 query duration:{}", query_duration.count());

  SPDLOG_INFO("index vec size:{} intersection_idx size:{}",
              query_result.first.size(), intersection_idx.size());

  EXPECT_EQ(intersection_idx, query_result.first);
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, LabelPsiTest,
                         testing::Values(  //
#if 0
                             TestParams{1, 10000},         // 1-10K
                             TestParams{1, 100000},        // 1-100K
                             TestParams{1, 256000},        // 1-256K
                             TestParams{1, 512000},        // 1-512K
                             TestParams{1, 1000000},       // 1-1M
                             TestParams{256, 100000},      // 256-100K
                             TestParams{512, 100000},      // 512-100K
                             TestParams{1024, 100000},     // 1024-100K
                             TestParams{2048, 100000},     // 2048-100K
                             TestParams{4096, 100000},     // 4096-100K
                             TestParams{10000, 100000},    // 10000-100K
#else
                             TestParams{1, 10000},            // 1-10K
                             TestParams{1, 10000, false},     // 1-10K memdb
                             TestParams{10, 10000},           // 1-10K
                             TestParams{100, 100000, false})  // 100-100K
#endif
);

}  // namespace spu::psi
