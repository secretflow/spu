// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/psi/utils/ub_psi_cache.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "yacl/crypto/utils/rand.h"

namespace spu::psi {

TEST(UbPsiCacheTest, Simple) {
  size_t data_len = 12;

  auto timestamp_str = std::to_string(absl::ToUnixNanos(absl::Now()));
  auto tmp_file_path =
      std::filesystem::path(fmt::format("tmp-cache-{}", timestamp_str));

  // register remove of temp file.
  ON_SCOPE_EXIT([&] {
    std::error_code ec;
    std::filesystem::remove(tmp_file_path, ec);
    if (ec.value() != 0) {
      SPDLOG_WARN("can not remove tmp file: {}, msg: {}", tmp_file_path.c_str(),
                  ec.message());
    }
  });

  std::vector<std::vector<uint8_t>> items;

  std::vector<std::string> selected_fields = {"id1", "id2"};
  UbPsiCache cache(tmp_file_path.string(), data_len, selected_fields);

  std::vector<uint8_t> rand_bytes = yacl::crypto::RandBytes(data_len);
  items.push_back(rand_bytes);

  cache.SaveData(rand_bytes, 0, 10);
  rand_bytes = yacl::crypto::RandBytes(data_len);
  items.push_back(rand_bytes);
  cache.SaveData(rand_bytes, 1, 11);
  cache.Flush();

  UbPsiCacheProvider provider(tmp_file_path.string(), data_len);

  const std::vector<std::string>& read_fields = provider.GetSelectedFields();
  EXPECT_EQ(read_fields.size(), selected_fields.size());

  std::vector<std::string> batch_data;
  std::vector<size_t> batch_indices;
  std::vector<size_t> shuffle_indices;
  std::tie(batch_data, batch_indices, shuffle_indices) =
      provider.ReadNextBatchWithIndex(items.size() + 1);

  EXPECT_EQ(batch_data.size(), items.size());

  for (size_t i = 0; i < batch_data.size(); i++) {
    EXPECT_EQ(batch_indices[i], i);
    EXPECT_EQ(std::memcmp(batch_data[i].data(), items[i].data(), data_len), 0);
  }
}

}  // namespace spu::psi
