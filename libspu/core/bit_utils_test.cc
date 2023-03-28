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

#include "libspu/core/bit_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(BitUtilsTest, Log2Floor) {
  ASSERT_EQ(Log2Floor(1), 0);
  ASSERT_EQ(Log2Floor(2), 1);
  ASSERT_EQ(Log2Floor(3), 1);
  ASSERT_EQ(Log2Floor(4), 2);
  ASSERT_EQ(Log2Floor(5), 2);
  ASSERT_EQ(Log2Floor(6), 2);
  ASSERT_EQ(Log2Floor(7), 2);
  ASSERT_EQ(Log2Floor(8), 3);
  ASSERT_EQ(Log2Floor((1U << 31) - 1), 30);
  ASSERT_EQ(Log2Floor(1U << 31), 31);
  ASSERT_EQ(Log2Floor((1U << 31) + 1), 31);
}

TEST(BitUtilsTest, Log2Ceil) {
  ASSERT_EQ(Log2Ceil(1), 0);
  ASSERT_EQ(Log2Ceil(2), 1);
  ASSERT_EQ(Log2Ceil(3), 2);
  ASSERT_EQ(Log2Ceil(4), 2);
  ASSERT_EQ(Log2Ceil(5), 3);
  ASSERT_EQ(Log2Ceil(6), 3);
  ASSERT_EQ(Log2Ceil(7), 3);
  ASSERT_EQ(Log2Ceil(8), 3);
  ASSERT_EQ(Log2Ceil(9), 4);
  ASSERT_EQ(Log2Ceil((1U << 31) - 1), 31);
  ASSERT_EQ(Log2Ceil(1U << 31), 31);
  ASSERT_EQ(Log2Ceil((1U << 31) + 1), 32);
}

TEST(BitUtilsTest, BitWidth) {
  EXPECT_EQ(BitWidth(0u), 0);
  EXPECT_EQ(BitWidth(1u), 1);
  EXPECT_EQ(BitWidth(1u << 3), 3 + 1);
  EXPECT_EQ(BitWidth(1ull << 3), 3 + 1);
  EXPECT_EQ(BitWidth(1ull << 40), 40 + 1);
  EXPECT_EQ(BitWidth(yacl::MakeInt128(0, 1ull << 3)), 3 + 1);
  EXPECT_EQ(BitWidth(yacl::MakeInt128(1ull << 3, 0)), 3 + 1 + 64);
}

TEST(BitUtilsTest, BitDeintl32) {
  EXPECT_EQ(BitDeintl<uint32_t>(0x55555555, 0), 0x0000FFFF);
  EXPECT_EQ(BitDeintl<uint32_t>(0x33333333, 1), 0x0000FFFF);
  EXPECT_EQ(BitDeintl<uint32_t>(0x0F0F0F0F, 2), 0x0000FFFF);
  EXPECT_EQ(BitDeintl<uint32_t>(0x00FF00FF, 3), 0x0000FFFF);
  EXPECT_EQ(BitDeintl<uint32_t>(0x12345678, 4), 0x12345678);
  EXPECT_EQ(BitDeintl<uint32_t>(0x12345678, 5), 0x12345678);
}

TEST(BitUtilsTest, BitDeintl64) {
  EXPECT_EQ(BitDeintl<uint64_t>(0x5555555555555555, 0), 0x00000000FFFFFFFF);
  EXPECT_EQ(BitDeintl<uint64_t>(0x3333333333333333, 1), 0x00000000FFFFFFFF);
  EXPECT_EQ(BitDeintl<uint64_t>(0x0F0F0F0F0F0F0F0F, 2), 0x00000000FFFFFFFF);
  EXPECT_EQ(BitDeintl<uint64_t>(0x00FF00FF00FF00FF, 3), 0x00000000FFFFFFFF);
  EXPECT_EQ(BitDeintl<uint64_t>(0x0000FFFF0000FFFF, 4), 0x00000000FFFFFFFF);
  EXPECT_EQ(BitDeintl<uint64_t>(0x5555555555555555, 5), 0x5555555555555555);
  EXPECT_EQ(BitDeintl<uint64_t>(0x5555555555555555, 6), 0x5555555555555555);
}

TEST(BitUtilsTest, BitIntl32) {
  EXPECT_EQ(BitIntl<uint32_t>(0x0000FFFF, 0), 0x55555555);
  EXPECT_EQ(BitIntl<uint32_t>(0x0000FFFF, 1), 0x33333333);
  EXPECT_EQ(BitIntl<uint32_t>(0x0000FFFF, 2), 0x0F0F0F0F);
  EXPECT_EQ(BitIntl<uint32_t>(0x0000FFFF, 3), 0x00FF00FF);
  EXPECT_EQ(BitIntl<uint32_t>(0x0000FFFF, 4), 0x0000FFFF);
  EXPECT_EQ(BitIntl<uint32_t>(0x0000FFFF, 5), 0x0000FFFF);
  EXPECT_EQ(BitIntl<uint32_t>(0x0000FFFF, 6), 0x0000FFFF);
}

TEST(BitUtilsTest, BitIntl64) {
  EXPECT_EQ(BitIntl<uint64_t>(0x00000000FFFFFFFF, 0), 0x5555555555555555);
  EXPECT_EQ(BitIntl<uint64_t>(0x00000000FFFFFFFF, 1), 0x3333333333333333);
  EXPECT_EQ(BitIntl<uint64_t>(0x00000000FFFFFFFF, 2), 0x0F0F0F0F0F0F0F0F);
  EXPECT_EQ(BitIntl<uint64_t>(0x00000000FFFFFFFF, 3), 0x00FF00FF00FF00FF);
  EXPECT_EQ(BitIntl<uint64_t>(0x00000000FFFFFFFF, 4), 0x0000FFFF0000FFFF);
  EXPECT_EQ(BitIntl<uint64_t>(0x00000000FFFFFFFF, 5), 0x00000000FFFFFFFF);
  EXPECT_EQ(BitIntl<uint64_t>(0x00000000FFFFFFFF, 6), 0x00000000FFFFFFFF);
}

}  // namespace spu
