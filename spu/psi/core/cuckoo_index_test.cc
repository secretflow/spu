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

#include "spu/psi/core/cuckoo_index.h"

#include <random>

#include "gtest/gtest.h"
#include "yasl/crypto/symmetric_crypto.h"

namespace spu::psi {

class CuckooIndexTest : public testing::TestWithParam<CuckooIndex::Options> {};

TEST_P(CuckooIndexTest, Works) {
  const auto& param = GetParam();
  {
    // Insert all in one call.
    CuckooIndex cuckoo_index(param);
    std::vector<uint128_t> inputs(param.num_input);
    yasl::FillAesRandom(std::random_device()(), 0, 0, absl::MakeSpan(inputs));

    ASSERT_NO_THROW(cuckoo_index.Insert(absl::MakeSpan(inputs)));
    ASSERT_NO_THROW(cuckoo_index.SanityCheck());

    EXPECT_EQ(cuckoo_index.bins().size(), param.NumBins());
    EXPECT_EQ(cuckoo_index.stash().size(), param.num_stash);
    EXPECT_EQ(cuckoo_index.hashes().size(), param.num_input);
  }
  {
    // Insert by batches.
    CuckooIndex cuckoo_index(param);
    std::vector<uint128_t> inputs(param.num_input);
    yasl::FillAesRandom(std::random_device()(), 0, 0, absl::MakeSpan(inputs));

    constexpr size_t kChunkSize = 1024;
    for (size_t i = 0; i < inputs.size(); i += kChunkSize) {
      size_t chunk_size = std::min(kChunkSize, inputs.size() - i);
      absl::Span<const uint128_t> chunk(inputs.data() + i, chunk_size);
      ASSERT_NO_THROW(cuckoo_index.Insert(chunk));
    }

    ASSERT_NO_THROW(cuckoo_index.SanityCheck());
    EXPECT_EQ(cuckoo_index.bins().size(), param.NumBins());
    EXPECT_EQ(cuckoo_index.stash().size(), param.num_stash);
    EXPECT_EQ(cuckoo_index.hashes().size(), param.num_input);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, CuckooIndexTest,
    testing::Values(
        CuckooIndex::Options{(1 << 16), 8, 3, 1.2},       // 3-way, 1.2, 8
        CuckooIndex::Options{(1 << 16) + 1, 4, 2, 2.4},   // 2-way, 2.4, 4
        CuckooIndex::Options{(1 << 16) - 1, 4, 2, 2.4},   // 2-way, 2.4, 4
        CuckooIndex::Options{(1 << 20) + 17, 4, 3, 1.2},  // 3-way, 1.2, 4
        CuckooIndex::Options{(1 << 0), 0, 3, 1.2}         // dummy
        ));

TEST(CuckooIndexTest, Bad_StashTooSmall) {
  CuckooIndex cuckoo_index(CuckooIndex::Options{1 << 16, 0, 3, 1.1});
  std::vector<uint128_t> inputs(1 << 16);
  yasl::FillAesRandom(std::random_device()(), 0, 0, absl::MakeSpan(inputs));

  ASSERT_THROW(cuckoo_index.Insert(absl::MakeSpan(inputs)), yasl::Exception);
}

TEST(CuckooIndexTest, Bad_SmallScaleFactor) {
  CuckooIndex cuckoo_index(CuckooIndex::Options{1 << 16, 8, 3, 1.01});
  std::vector<uint128_t> inputs(1 << 16);
  yasl::FillAesRandom(std::random_device()(), 0, 0, absl::MakeSpan(inputs));

  ASSERT_THROW(cuckoo_index.Insert(absl::MakeSpan(inputs)), yasl::Exception);
}

}  // namespace spu::psi
