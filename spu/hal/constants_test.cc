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

#include "spu/hal/constants.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "spu/hal/test_util.h"

namespace spu::hal {

TEST(ConstantsTest, Constant) {
  HalContext ctx = test::makeRefHalContext();

  // int scalar
  {
    Value x = constant(&ctx, 0);
    EXPECT_TRUE(x.shape().empty());
    EXPECT_TRUE(x.strides().empty());
    EXPECT_EQ(x.numel(), 1);
    EXPECT_TRUE(x.isPublic());
    EXPECT_TRUE(x.isInt());
  }

  // fxp scalar
  {
    Value x = constant(&ctx, 0.0f);
    EXPECT_TRUE(x.shape().empty());
    EXPECT_TRUE(x.strides().empty());
    EXPECT_EQ(x.numel(), 1);
    EXPECT_TRUE(x.isPublic());
    EXPECT_TRUE(x.isFxp());
  }

  // tensor
  {
    xt::xarray<float> raw{1.0f};
    Value x = constant(&ctx, raw);
    EXPECT_THAT(x.shape(), testing::ElementsAre(1));
    EXPECT_THAT(x.strides(), testing::ElementsAre(0));
    EXPECT_EQ(x.numel(), 1);
    EXPECT_TRUE(x.isPublic());
    EXPECT_TRUE(x.isFxp());
  }

  // tensor broadcast
  {
    xt::xarray<float> raw{{1.0f, 2.0f}};
    Value x = constant(&ctx, raw, {6, 2});
    EXPECT_THAT(x.shape(), testing::ElementsAre(6, 2));
    EXPECT_THAT(x.strides(), testing::ElementsAre(2, 1));
    EXPECT_EQ(x.numel(), 12);
    EXPECT_TRUE(x.isPublic());
    EXPECT_TRUE(x.isFxp());
  }
}

}  // namespace spu::hal
