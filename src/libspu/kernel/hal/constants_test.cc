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

  Value i = constant(&ctx, 0, DT_I32);
  EXPECT_TRUE(i.shape().isScalar());
  EXPECT_TRUE(i.strides().empty());
  EXPECT_EQ(i.numel(), 1);
  EXPECT_TRUE(i.isPublic());
  EXPECT_TRUE(i.isInt());

  Value f = constant(&ctx, 0.0F, DT_F32);
  EXPECT_TRUE(f.shape().isScalar());
  EXPECT_TRUE(f.strides().empty());
  EXPECT_EQ(f.numel(), 1);
  EXPECT_TRUE(f.isPublic());
  EXPECT_TRUE(f.isFxp());
}

TEST(ConstantsTest, Tensor) {
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> raw = {1.0F};
  Value x = constant(&ctx, raw, DT_F32);
  EXPECT_THAT(x.shape(), testing::ElementsAre(1));
  EXPECT_THAT(x.strides(), testing::ElementsAre(0));
  EXPECT_EQ(x.numel(), 1);
  EXPECT_TRUE(x.isPublic());
  EXPECT_TRUE(x.isFxp());
}

TEST(ConstantsTest, TensorBroadcast) {
  SPUContext ctx = test::makeSPUContext();

  xt::xarray<float> raw = {
      {1.0, 2.0},
  };

  Value x = constant(&ctx, raw, DT_F32, {6, 2});
  EXPECT_THAT(x.shape(), testing::ElementsAre(6, 2));
  EXPECT_THAT(x.strides(), testing::ElementsAre(0, 1));
  EXPECT_EQ(x.numel(), 12);
  EXPECT_TRUE(x.isPublic());
  EXPECT_TRUE(x.isFxp());
}

TEST(ConstantsTest, Initializer) {
  SPUContext ctx = test::makeSPUContext();

  EXPECT_EQ(constant(&ctx, true, DT_I1).dtype(), DT_I1);
  EXPECT_EQ(constant(&ctx, static_cast<int8_t>(0), DT_I8).dtype(), DT_I8);
  EXPECT_EQ(constant(&ctx, static_cast<uint8_t>(0), DT_U8).dtype(), DT_U8);
  EXPECT_EQ(constant(&ctx, static_cast<int16_t>(0), DT_I16).dtype(), DT_I16);
  EXPECT_EQ(constant(&ctx, static_cast<uint16_t>(0), DT_U16).dtype(), DT_U16);
  EXPECT_EQ(constant(&ctx, static_cast<int32_t>(0), DT_I32).dtype(), DT_I32);
  EXPECT_EQ(constant(&ctx, static_cast<uint32_t>(0), DT_U32).dtype(), DT_U32);
  EXPECT_EQ(constant(&ctx, static_cast<int64_t>(0), DT_I64).dtype(), DT_I64);
  EXPECT_EQ(constant(&ctx, static_cast<uint64_t>(0), DT_U64).dtype(), DT_U64);

  EXPECT_EQ(constant(&ctx, 0.0F, DT_F32).dtype(), DT_F32);
  EXPECT_EQ(constant(&ctx, 0.0, DT_F64).dtype(), DT_F64);
}

}  // namespace spu::kernel::hal
