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

#include "spu/mpc/util/bit_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu::mpc {

TEST(BitUtilsTest, Log2Floor) {
  ASSERT_EQ(log2Floor(1), 0);
  ASSERT_EQ(log2Floor(2), 1);
  ASSERT_EQ(log2Floor(3), 1);
  ASSERT_EQ(log2Floor(4), 2);
  ASSERT_EQ(log2Floor(5), 2);
  ASSERT_EQ(log2Floor(6), 2);
  ASSERT_EQ(log2Floor(7), 2);
  ASSERT_EQ(log2Floor(8), 3);
  ASSERT_EQ(log2Floor((1U << 31) - 1), 30);
  ASSERT_EQ(log2Floor(1U << 31), 31);
  ASSERT_EQ(log2Floor((1U << 31) + 1), 31);
}

TEST(BitUtilsTest, Log2Ceil) {
  ASSERT_EQ(log2Ceil(1), 0);
  ASSERT_EQ(log2Ceil(2), 1);
  ASSERT_EQ(log2Ceil(3), 2);
  ASSERT_EQ(log2Ceil(4), 2);
  ASSERT_EQ(log2Ceil(5), 3);
  ASSERT_EQ(log2Ceil(6), 3);
  ASSERT_EQ(log2Ceil(7), 3);
  ASSERT_EQ(log2Ceil(8), 3);
  ASSERT_EQ(log2Ceil(9), 4);
  ASSERT_EQ(log2Ceil((1U << 31) - 1), 31);
  ASSERT_EQ(log2Ceil(1U << 31), 31);
  ASSERT_EQ(log2Ceil((1U << 31) + 1), 32);
}

}  // namespace spu::mpc
