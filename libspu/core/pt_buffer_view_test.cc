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

#include "libspu/core/pt_buffer_view.h"

#include <array>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(PtBufferView, Scalar) {
  PtBufferView bv_i32(0);
  EXPECT_EQ(bv_i32.pt_type, PT_I32);
  EXPECT_TRUE(bv_i32.shape.empty());
  EXPECT_TRUE(bv_i32.strides.empty());

  PtBufferView bv_u32(0U);
  EXPECT_EQ(bv_u32.pt_type, PT_U32);
  EXPECT_TRUE(bv_u32.shape.empty());
  EXPECT_TRUE(bv_u32.strides.empty());

  PtBufferView bv_f32(.0F);
  EXPECT_EQ(bv_f32.pt_type, PT_F32);
  EXPECT_TRUE(bv_f32.shape.empty());
  EXPECT_TRUE(bv_f32.strides.empty());
}

TEST(PtBufferView, Vector) {
  std::vector<int32_t> raw_i32(10, 0);
  PtBufferView bv_i32(raw_i32);
  EXPECT_EQ(bv_i32.pt_type, PT_I32);
  EXPECT_THAT(bv_i32.shape, testing::ElementsAre(10));
  EXPECT_THAT(bv_i32.strides, testing::ElementsAre(1));

  std::array<float, 3> raw_f32 = {1.0, 2.0, 3.0};
  PtBufferView bv_f32(raw_f32);
  EXPECT_EQ(bv_f32.pt_type, PT_F32);
  EXPECT_THAT(bv_f32.shape, testing::ElementsAre(3));
  EXPECT_THAT(bv_f32.strides, testing::ElementsAre(1));
  EXPECT_FLOAT_EQ((*bv_f32.get<float>({0})), 1.0);
  EXPECT_FLOAT_EQ((*bv_f32.get<float>({1})), 2.0);
  EXPECT_FLOAT_EQ((*bv_f32.get<float>({2})), 3.0);
}

TEST(PtBufferView, ConvertToNdArray) {
  std::array<float, 3> raw_f32 = {1.0, 2.0, 3.0};

  auto arr = convertToNdArray(raw_f32);

  EXPECT_THAT(arr.eltype(), F32);
  EXPECT_THAT(arr.shape(), testing::ElementsAre(3));
  EXPECT_THAT(arr.strides(), testing::ElementsAre(1));
  EXPECT_NE(arr.data(), raw_f32.data());
  EXPECT_FLOAT_EQ((arr.at<float>({0})), 1.0);
  EXPECT_FLOAT_EQ((arr.at<float>({1})), 2.0);
  EXPECT_FLOAT_EQ((arr.at<float>({2})), 3.0);
}

}  // namespace spu
