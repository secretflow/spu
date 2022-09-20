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
#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/crypto/pseudo_random_generator.h"
#include "yasl/link/test_util.h"

#include "spu/psi/core/ecdh_oprf/ecdh_oprf_selector.h"
#include "spu/psi/core/labeled_psi/psi_params.h"
#include "spu/psi/core/labeled_psi/receiver.h"
#include "spu/psi/core/labeled_psi/sender.h"

namespace spu::psi {

namespace {

using duration_millis = std::chrono::duration<double, std::milli>;

constexpr size_t kPsiStartPos = 100;
struct TestParams {
  size_t nr;
  size_t ns;
  size_t label_bytes;
};

std::vector<std::string> GenerateData(size_t seed, size_t item_count) {
  yasl::PseudoRandomGenerator<uint128_t> prg(seed);

  std::vector<std::string> items;

  for (size_t i = 0; i < item_count; ++i) {
    std::string item(16, '\0');
    prg.Fill(absl::MakeSpan(item.data(), item.length()));
    items.emplace_back(item);
  }

  return items;
}

std::vector<apsi::Item> GenerateSenderData(
    size_t seed, size_t item_count,
    const absl::Span<std::string> &receiver_items,
    std::vector<size_t> *intersection_idx) {
  std::vector<apsi::Item> sender_items;

  yasl::PseudoRandomGenerator<uint128_t> prg(seed);

  for (size_t i = 0; i < item_count; ++i) {
    apsi::Item::value_type value{};
    prg.Fill(absl::MakeSpan(value));
    sender_items.emplace_back(value);
  }

  for (size_t i = 0; i < receiver_items.size(); i += 3) {
    apsi::Item::value_type value{};
    std::memcpy(value.data(), receiver_items[i].data(),
                receiver_items[i].length());
    apsi::Item item(value);
    sender_items[kPsiStartPos + i * 5] = item;
    (*intersection_idx).emplace_back(i);
  }

  return sender_items;
}

std::vector<std::pair<apsi::Item, apsi::Label>> GenerateSenderData(
    size_t seed, size_t item_count, size_t label_byte_count,
    const absl::Span<std::string> &receiver_items,
    std::vector<size_t> *intersection_idx,
    std::vector<std::string> *intersection_label) {
  std::vector<std::pair<apsi::Item, apsi::Label>> sender_items;

  yasl::PseudoRandomGenerator<uint128_t> prg(seed);

  for (size_t i = 0; i < item_count; ++i) {
    apsi::Item item;
    apsi::Label label;
    label.resize(label_byte_count);
    prg.Fill(absl::MakeSpan(item.value()));
    prg.Fill(absl::MakeSpan(label));
    sender_items.emplace_back(item, label);
  }

  for (size_t i = 0; i < receiver_items.size(); i += 3) {
    apsi::Item item;
    std::memcpy(item.value().data(), receiver_items[i].data(),
                receiver_items[i].length());

    sender_items[kPsiStartPos + i * 5].first = item;
    (*intersection_idx).emplace_back(i);
    std::string label_string(sender_items[kPsiStartPos + i * 5].second.size(),
                             '\0');
    std::memcpy(&label_string[0],
                sender_items[kPsiStartPos + i * 5].second.data(),
                sender_items[kPsiStartPos + i * 5].second.size());
    (*intersection_label).emplace_back(label_string);
  }

  return sender_items;
}

}  // namespace

class LabelPsiTest : public testing::TestWithParam<TestParams> {};

TEST_P(LabelPsiTest, Works) {
  auto params = GetParam();
  auto ctxs = yasl::link::test::SetupWorld(2);
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
  size_t label_byte_count = params.label_bytes;
  size_t nonce_byte_count = 16;

  std::random_device rd;
  yasl::PseudoRandomGenerator<uint128_t> prg(rd());

  std::array<uint8_t, 32> oprf_key;
  prg.Fill(absl::MakeSpan(oprf_key));

  bool compressed = false;
  std::shared_ptr<spu::psi::SenderDB> sender_db =
      std::make_shared<spu::psi::SenderDB>(
          psi_params, oprf_key, label_byte_count, nonce_byte_count, compressed);

  std::vector<std::string> receiver_items = GenerateData(rd(), params.nr);

  std::vector<size_t> intersection_idx;
  std::vector<std::string> intersection_label;

  // step 2: set database

  const auto setdb_start = std::chrono::system_clock::now();

  if (params.label_bytes == 0) {
    std::vector<apsi::Item> sender_items = GenerateSenderData(
        rd(), item_count, absl::MakeSpan(receiver_items), &intersection_idx);

    sender_db->SetData(sender_items);
  } else {
    std::vector<std::pair<apsi::Item, apsi::Label>> sender_items =
        GenerateSenderData(rd(), item_count, label_byte_count,
                           absl::MakeSpan(receiver_items), &intersection_idx,
                           &intersection_label);

    sender_db->SetData(sender_items);
  }

  const auto setdb_end = std::chrono::system_clock::now();
  const duration_millis setdb_duration = setdb_end - setdb_start;
  SPDLOG_INFO("*** step2 set db duration:{}", setdb_duration.count());

  EXPECT_EQ(params.ns, sender_db->GetItemCount());

  SPDLOG_INFO("after set db, bin_bundle_count:{}, packing_rate:{}",
              sender_db->GetBinBundleCount(), sender_db->GetPackingRate());

  std::unique_ptr<IEcdhOprfServer> oprf_server =
      CreateEcdhOprfServer(oprf_key, OprfType::Basic, CurveType::CURVE_FOURQ);

  LabelPsiSender sender(sender_db);

  LabelPsiReceiver receiver(psi_params, params.label_bytes > 0);

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
  const duration_millis oprf_duration = oprf_end - oprf_start;
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
  const duration_millis query_duration = query_end - query_start;
  SPDLOG_INFO("*** step4 query duration:{}", query_duration.count());

  SPDLOG_INFO("index vec size:{} intersection_idx size:{}",
              query_result.first.size(), intersection_idx.size());

  EXPECT_EQ(intersection_idx, query_result.first);
  EXPECT_EQ(intersection_label, query_result.second);
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, LabelPsiTest,
                         testing::Values(  //
#if 0
                             TestParams{1, 10000, 0},         // 1-10K
                             TestParams{1, 100000, 0},        // 1-100K
                             TestParams{1, 256000, 0},        // 1-256K
                             TestParams{1, 512000, 0},        // 1-512K
                             TestParams{1, 1000000, 0},       // 1-1M
                             TestParams{256, 100000, 0},      // 256-100K
                             TestParams{512, 100000, 0},      // 512-100K
                             TestParams{1024, 100000, 0},     // 1024-100K
                             TestParams{2048, 100000, 0},     // 2048-100K
                             TestParams{4096, 100000, 0},     // 4096-100K
                             TestParams{10000, 100000, 0},    // 10000-100K
                             TestParams{1, 100000, 32},       // 1-100K-32
                             TestParams{1, 256000, 32},       // 1-256K-32
                             TestParams{1, 1000000, 32},      // 1-1M-32
                             TestParams{256, 100000, 32},     // 256-100K-32
                             TestParams{512, 100000, 32},     // 512-100K-32
                             TestParams{1024, 100000, 32},    // 1024-100K-32
                             TestParams{2048, 100000, 32},    // 2048-100K-32
                             TestParams{4096, 100000, 32},    // 4096-100K-32
                             TestParams{10000, 100000, 32},   // 10000-100K-32
                             TestParams{10000, 1000000, 32})  // 10000-1M-32
#else
                             TestParams{1, 10000, 0},       // 1-10K
                             TestParams{1, 100000, 0},      // 1-100K
                             TestParams{256, 100000, 0},    // 256-100K
                             TestParams{2048, 100000, 32},  // 2048-100K-32
                             TestParams{4096, 100000, 32})  // 4096-100K-32
#endif
);

}  // namespace spu::psi
