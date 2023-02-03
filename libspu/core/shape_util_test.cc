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

#include "libspu/core/shape_util.h"

#include <array>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spu {

class DeduceDotShapeTest
    : public ::testing::TestWithParam<std::tuple<
          std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>> {};

TEST_P(DeduceDotShapeTest, Works) {
  std::vector<int64_t> lhs_shape = std::get<0>(GetParam());
  std::vector<int64_t> rhs_shape = std::get<1>(GetParam());
  std::vector<int64_t> expected = std::get<2>(GetParam());

  auto ret = deduceDotShape(lhs_shape, rhs_shape);

  EXPECT_EQ(ret.size(), expected.size());
  EXPECT_EQ(ret, expected);
}

TEST(ShapeUtilTest, FlattenIndex1D) {
  // 1D
  std::array<int64_t, 1> shape = {10};

  std::array<int64_t, 1> idx = {0};
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
  std::array<int64_t, 2> shape = {3, 3};

  std::array<int64_t, 2> idx = {0, 0};
  EXPECT_EQ(flattenIndex(idx, shape), 0);
  EXPECT_THAT(unflattenIndex(0, shape), testing::ElementsAre(0, 0));

  idx = {1, 1};
  EXPECT_EQ(flattenIndex(idx, shape), 4);
  EXPECT_THAT(unflattenIndex(4, shape), testing::ElementsAre(1, 1));

  idx = {2, 2};
  EXPECT_EQ(flattenIndex(idx, shape), 8);
  EXPECT_THAT(unflattenIndex(8, shape), testing::ElementsAre(2, 2));
}

TEST(ShapeUilTest, Empty) {
  EXPECT_TRUE(isEmpty({0}));
  EXPECT_TRUE(isEmpty({0, 1}));
  EXPECT_TRUE(isEmpty({1, 0}));

  EXPECT_FALSE(isEmpty({}));
  EXPECT_FALSE(isEmpty({1}));
  EXPECT_FALSE(isEmpty({1, 1}));
}

INSTANTIATE_TEST_SUITE_P(
    DeduceDotShapeTestInstances, DeduceDotShapeTest,
    testing::Values(
        // hs: scalar, rhs: scalar
        std::make_tuple(std::vector<int64_t>(), std::vector<int64_t>(),
                        std::vector<int64_t>()),
        // lhs: tensor, rhs: scalar
        std::make_tuple(std::vector<int64_t>({2, 3, 4}), std::vector<int64_t>(),
                        std::vector<int64_t>({2, 3, 4})),
        // lhs: scalar, rhs: tensor
        std::make_tuple(std::vector<int64_t>(), std::vector<int64_t>({2, 3, 4}),
                        std::vector<int64_t>({2, 3, 4})),
        // lhs: vector, rhs: vector
        std::make_tuple(std::vector<int64_t>({3}), std::vector<int64_t>({3}),
                        std::vector<int64_t>({1})),
        // lhs: vector, rhs: 2d tensor
        std::make_tuple(std::vector<int64_t>({3}), std::vector<int64_t>({3, 2}),
                        std::vector<int64_t>({2})),
        // lhs: 2d tensor, rhs: vector
        std::make_tuple(std::vector<int64_t>({3, 2}), std::vector<int64_t>({2}),
                        std::vector<int64_t>({3})),
        // lhs: 2d tensor, rhs: 2d tensor
        std::make_tuple(std::vector<int64_t>({2, 3}),
                        std::vector<int64_t>({3, 4}),
                        std::vector<int64_t>({2, 4})),
        // lhs: vector, rhs: Nd(N > 2) tensor
        std::make_tuple(std::vector<int64_t>({3}),
                        std::vector<int64_t>({2, 3, 4}),
                        std::vector<int64_t>({2, 4})),
        // lhs: Nd(N > 2) tensor, rhs: vector
        std::make_tuple(std::vector<int64_t>({2, 3, 4}),
                        std::vector<int64_t>({4}),
                        std::vector<int64_t>({2, 3})),
        // lhs: Nd(N > 2) tensor, rhs: Md(M >= 2) tensor
        std::make_tuple(std::vector<int64_t>({2, 3, 3}),
                        std::vector<int64_t>({3, 4}),
                        std::vector<int64_t>({2, 3, 4})),
        std::make_tuple(std::vector<int64_t>({2, 3, 4}),
                        std::vector<int64_t>({3, 4, 5}),
                        std::vector<int64_t>({2, 3, 3, 5})),
        std::make_tuple(std::vector<int64_t>({2, 3, 4, 5}),
                        std::vector<int64_t>({3, 4, 5, 6}),
                        std::vector<int64_t>({2, 3, 4, 3, 4, 6}))));

}  // namespace spu
