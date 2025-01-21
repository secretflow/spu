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

#include "libspu/kernel/hlo/utils.h"

#include "gtest/gtest.h"

namespace spu {

TEST(UtilsTest, ForEachIndexScalar) {
  int64_t counter = 0;
  kernel::forEachIndex({}, {}, {}, {}, [&](absl::Span<const int64_t> idx) {
    EXPECT_TRUE(idx.empty());
    ++counter;
  });

  EXPECT_EQ(counter, 1);
}

TEST(UtilsTest, ForEachIndex1D) {
  int64_t counter = 0;
  std::vector<int64_t> expected_idx = {0, 1, 2};
  kernel::forEachIndex({3}, {0}, {3}, {1}, [&](absl::Span<const int64_t> idx) {
    EXPECT_EQ(idx.size(), 1);
    EXPECT_EQ(idx[0], expected_idx[counter]);
    ++counter;
  });

  EXPECT_EQ(counter, 3);
}

TEST(UtilsTest, ForEachIndex2D) {
  int64_t counter = 0;
  std::vector<std::vector<int64_t>> expected_idx = {
      {0, 0}, {0, 1}, {1, 0}, {1, 1}};

  kernel::forEachIndex({2, 2}, {0, 0}, {2, 2}, {1, 1},
                       [&](absl::Span<const int64_t> idx) {
                         EXPECT_EQ(idx.size(), 2);
                         EXPECT_EQ(idx[0], expected_idx[counter][0]);
                         EXPECT_EQ(idx[1], expected_idx[counter][1]);
                         ++counter;
                       });

  EXPECT_EQ(counter, 4);
}

}  // namespace spu
