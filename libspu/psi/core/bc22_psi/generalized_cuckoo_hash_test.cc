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

#include "libspu/psi/core/bc22_psi/generalized_cuckoo_hash.h"

#include <random>
#include <string>

#include "absl/strings/escaping.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"

namespace spu::psi {

namespace {

std::vector<std::string> CreateRangeItems(size_t begin, size_t size) {
  std::vector<std::string> ret;
  for (size_t i = 0; i < size; i++) {
    ret.push_back(std::to_string(begin + i));
  }
  return ret;
}

}  // namespace

class GchTest : public testing::TestWithParam<size_t> {};

TEST(GchTest, BasicTest) {
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<std::string> items(1000);
  for (size_t idx = 1; idx < items.size(); ++idx) {
    items[idx] = std::to_string(rng());
  }

  size_t bin_size = 2;
  size_t hash_num = 2;
  CuckooIndex::Options cuckoo_options =
      GetCuckooHashOption(bin_size, hash_num, items.size());

  GeneralizedCuckooHashTable gch(cuckoo_options, bin_size, 0);
  SimpleHashTable simple_table(cuckoo_options);

  gch.Insert(absl::MakeSpan(items));
  simple_table.Insert(absl::MakeSpan(items));

  const std::vector<std::vector<CuckooIndex::Bin>> &bins = gch.bins();
  const std::vector<std::vector<CuckooIndex::Bin>> &simple_table_bins =
      simple_table.bins();

  SPDLOG_INFO("bin size: {}, simple_table_bins size: {}", bins.size(),
              simple_table_bins.size());

  size_t bin_items[4] = {
      0,
  };
  size_t c1 = 0;
  size_t c2 = 0;
  std::vector<size_t> hash_data_num(hash_num);
  memset(hash_data_num.data(), 0, hash_data_num.size() * sizeof(size_t));

  for (size_t i = 0; i < bins.size(); ++i) {
    if (simple_table_bins[i].size() < bins[i].size()) {
      SPDLOG_INFO("****{}****", i);
    }
    for (const auto &bin : bins[i]) {
      // size_t item_idx = bins[i][j].InputIdx();
      size_t hash_idx = bin.HashIdx();
      hash_data_num[hash_idx]++;
    }
    bin_items[bins[i].size()]++;
    c1 += bins[i].size();
    c2 += simple_table_bins[i].size();
  }
  SPDLOG_INFO("bin0:{}, bin1:{}, bin2:{}", bin_items[0], bin_items[1],
              bin_items[2]);
  SPDLOG_INFO("hash_data_num[0]:{} hash_data_num[1]:{}", hash_data_num[0],
              hash_data_num[1]);
  SPDLOG_INFO("c1:{} c2:{}", c1, c2);
  SPDLOG_INFO("FillRate:{}", gch.FillRate());

  //  conflict index
  const std::vector<size_t> conflict_idx = simple_table.GetConflictIdx();
  SPDLOG_INFO("conflict_idx: {}", conflict_idx.size());
}

TEST(GchTest, CuckooHashTest) {
  std::vector<std::string> alice_data = CreateRangeItems(20000000, 30000);

  size_t bin_size = 3;
  size_t hash_num = 2;
  CuckooIndex::Options cuckoo_options =
      GetCuckooHashOption(bin_size, hash_num, alice_data.size());

  GeneralizedCuckooHashTable gch(cuckoo_options, bin_size, 0);
  SimpleHashTable simple_table(cuckoo_options);

  gch.Insert(absl::MakeSpan(alice_data));
  simple_table.Insert(absl::MakeSpan(alice_data));

  const std::vector<std::vector<CuckooIndex::Bin>> &bins = gch.bins();

  const std::vector<uint64_t> &items_hash = gch.GetItemsHashLow64();

  SPDLOG_INFO("items size:{} bins:{} items:{}", alice_data.size(), bins.size(),
              items_hash.size());

  EXPECT_EQ(items_hash.size(), alice_data.size());

  //  conflict index
  const std::vector<size_t> conflict_idx = simple_table.GetConflictIdx();
  SPDLOG_INFO("conflict_idx: {}", conflict_idx.size());
}

}  // namespace spu::psi
