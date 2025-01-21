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

#include "libspu/core/shape.h"

#include <array>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

TEST(ShapeTest, FlattenIndex1D) {
  // 1D
  Shape shape = {10};

  Index idx = {0};
  EXPECT_EQ(flattenIndex(idx, shape), 0);
  EXPECT_THAT(unflattenIndex(0, shape), testing::ElementsAre(0));

  idx = {5};
  EXPECT_EQ(flattenIndex(idx, shape), 5);
  EXPECT_THAT(unflattenIndex(5, shape), testing::ElementsAre(5));

  idx = {9};
  EXPECT_EQ(flattenIndex(idx, shape), 9);
  EXPECT_THAT(unflattenIndex(9, shape), testing::ElementsAre(9));
}

TEST(ShapeUtilTest, FlattenIndex2D) {
  // 2D
  Shape shape = {3, 3};

  Index idx = {0, 0};
  EXPECT_EQ(flattenIndex(idx, shape), 0);
  EXPECT_THAT(unflattenIndex(0, shape), testing::ElementsAre(0, 0));

  idx = {1, 1};
  EXPECT_EQ(flattenIndex(idx, shape), 4);
  EXPECT_THAT(unflattenIndex(4, shape), testing::ElementsAre(1, 1));

  idx = {2, 2};
  EXPECT_EQ(flattenIndex(idx, shape), 8);
  EXPECT_THAT(unflattenIndex(8, shape), testing::ElementsAre(2, 2));
}

TEST(ShapeTest, Empty) {
  {
    Shape s({0});
    EXPECT_TRUE(s.isTensor());
    EXPECT_FALSE(s.isScalar());
    EXPECT_EQ(s.numel(), 0);
  }

  {
    Shape s({0, 1});
    EXPECT_TRUE(s.isTensor());
    EXPECT_FALSE(s.isScalar());
  }

  {
    Shape s({1, 0});
    EXPECT_TRUE(s.isTensor());
    EXPECT_FALSE(s.isScalar());
  }

  {
    Shape s({});
    EXPECT_TRUE(s.isScalar());
    EXPECT_FALSE(s.isTensor());
    EXPECT_EQ(s.numel(), 1);
  }

  {
    Shape s({1});
    EXPECT_FALSE(s.isScalar());
    EXPECT_TRUE(s.isTensor());
    EXPECT_EQ(s.numel(), 1);
  }

  {
    Shape s({1, 1});
    EXPECT_FALSE(s.isScalar());
    EXPECT_EQ(s.numel(), 1);
  }
}

}  // namespace spu
