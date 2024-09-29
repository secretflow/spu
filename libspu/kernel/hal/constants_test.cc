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

#include "libspu/kernel/hal/constants.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "libspu/kernel/test_util.h"

namespace spu::kernel::hal {

TEST(ConstantsTest, Scalar) {
  SPUContext ctx = test::makeSPUContext();

  MemRef i = constant(&ctx, 0);
  EXPECT_TRUE(i.shape().isScalar());
  EXPECT_TRUE(i.strides().empty());
  EXPECT_EQ(i.numel(), 1);
  EXPECT_TRUE(i.isPublic());

  MemRef f = constant(&ctx, 0.0F);
  EXPECT_TRUE(f.shape().isScalar());
  EXPECT_TRUE(f.strides().empty());
  EXPECT_EQ(f.numel(), 1);
  EXPECT_TRUE(f.isPublic());
}

TEST(ConstantsTest, Tensor) {
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> raw = {1.0F};
  MemRef x = constant(&ctx, raw);
  EXPECT_THAT(x.shape(), testing::ElementsAre(1));
  EXPECT_THAT(x.strides(), testing::ElementsAre(0));
  EXPECT_EQ(x.numel(), 1);
  EXPECT_TRUE(x.isPublic());
}

TEST(ConstantsTest, TensorBroadcast) {
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> raw = {
      {1.0, 2.0},
  };

  MemRef x = constant(&ctx, raw, {6, 2});
  EXPECT_THAT(x.shape(), testing::ElementsAre(6, 2));
  EXPECT_THAT(x.strides(), testing::ElementsAre(0, 1));
  EXPECT_EQ(x.numel(), 12);
  EXPECT_TRUE(x.isPublic());
}

}  // namespace spu::kernel::hal
