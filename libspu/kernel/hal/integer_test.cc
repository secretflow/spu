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

#include "libspu/kernel/hal/integer.h"

#include "gtest/gtest.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hal {

// TODO: UT is too cumbersome.
TEST(IntegralTest, Add) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  int ra = 3;
  int rb = 4;

  // WHAT
  Value a = constant(&ctx, ra, DT_I32);
  Value b = constant(&ctx, rb, DT_I32);
  ASSERT_TRUE(a.isInt()) << a.dtype();
  ASSERT_TRUE(b.isInt()) << b.dtype();

  //
  Value c = i_add(&ctx, a, b);
  ASSERT_TRUE(c.isInt());

  // THEN
  {
    const auto arr = dump_public(&ctx, c);
    EXPECT_EQ(arr.eltype(), I32);
    EXPECT_EQ(arr.shape().size(), 0);
    EXPECT_EQ(arr.at<int>({}), 7);
  }
}

TEST(IntegralTest, Sub) {
  // GIVEN
  SPUContext ctx = test::makeSPUContext();

  int ra = 3;
  int rb = 4;

  // WHAT
  Value a = constant(&ctx, ra, DT_I32);
  Value b = constant(&ctx, rb, DT_I32);
  ASSERT_TRUE(a.isInt()) << a.dtype();
  ASSERT_TRUE(b.isInt()) << b.dtype();

  //
  Value c = i_sub(&ctx, a, b);
  ASSERT_TRUE(c.isInt());

  // THEN
  {
    const auto arr = dump_public(&ctx, c);
    EXPECT_EQ(arr.eltype(), I32);
    EXPECT_EQ(arr.shape().size(), 0);
    EXPECT_EQ(arr.at<int>({}), -1);
  }
}

}  // namespace spu::kernel::hal
