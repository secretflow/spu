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

#include "spu/hal/integer.h"

#include "gtest/gtest.h"

#include "spu/hal/constants.h"
#include "spu/hal/test_util.h"

namespace spu::hal {

// TODO: UT is too cumbersome.
TEST(IntegralTest, Add) {
  // GIVEN
  HalContext ctx = test::makeRefHalContext();

  int ra = 3, rb = 4;

  // WHAT
  Value a = constant(&ctx, ra);
  Value b = constant(&ctx, rb);
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
  HalContext ctx = test::makeRefHalContext();

  int ra = 3, rb = 4;

  // WHAT
  Value a = constant(&ctx, ra);
  Value b = constant(&ctx, rb);
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

}  // namespace spu::hal
