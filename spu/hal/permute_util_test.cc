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

#include "spu/hal/permute_util.h"

#include <cstddef>
#include <cstdint>

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "spu/hal/test_util.h"

namespace spu::hal {
namespace {

TEST(PermuteTest, 1d) {
  // GIVEN
  const xt::xtensor<float, 1> x = {1, 2, 3, 4};

  // WHAT
  auto permute_wrapper_1 = [](HalContext* ctx, const Value& input) {
    return permute(ctx, input, 0, xt::xarray<size_t>({3, 2, 1, 0}));
  };

  auto permute_wrapper_2 = [](HalContext* ctx, const Value& input) {
    return permute(ctx, input, 0, xt::xarray<size_t>{0, 1, 2, 3});
  };
  auto permute_wrapper_3 = [](HalContext* ctx, const Value& input) {
    return permute(ctx, input, 0, xt::xarray<size_t>{2, 1, 2, 1});
  };

  // THEN
  {
    auto z = test::evalUnaryOp<float>(VIS_SECRET, permute_wrapper_1,
                                      xt::xarray<float>(x));
    const xt::xtensor<float, 1> expected_z = {4, 3, 2, 1};
    EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
        << expected_z << std::endl
        << z << std::endl;
  }
  {
    auto z = test::evalUnaryOp<float>(VIS_PUBLIC, permute_wrapper_2,
                                      xt::xarray<float>(x));
    const xt::xtensor<float, 1> expected_z = {1, 2, 3, 4};
    EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
        << expected_z << std::endl
        << z << std::endl;
  }
  {
    auto z = test::evalUnaryOp<float>(VIS_SECRET, permute_wrapper_3,
                                      xt::xarray<float>(x));
    const xt::xtensor<float, 1> expected_z = {3, 2, 3, 2};
    EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
        << expected_z << std::endl
        << z << std::endl;
  }
}

TEST(PermuteTest, 2d) {
  // GIVEN
  const xt::xtensor<float, 2> x = {{1, 2, 3, 4, 5},   //
                                   {6, 7, 8, 9, 10},  //
                                   {11, 12, 13, 14, 15}};

  // WHAT
  auto permute_wrapper_1 = [](HalContext* ctx, const Value& input) {
    return permute(ctx, input, 1,
                   xt::xtensor<size_t, 2>({{3, 2, 0, 1, 4},  //
                                           {0, 1, 2, 3, 4},  //
                                           {4, 3, 2, 1, 0}}));
  };
  auto permute_wrapper_2 = [](HalContext* ctx, const Value& input) {
    return permute(ctx, input, 0,
                   xt::xtensor<size_t, 2>({{0, 0, 0, 1, 1},  //
                                           {1, 1, 2, 2, 1},  //
                                           {2, 2, 1, 0, 2}}));
  };

  // THEN
  {
    auto z = test::evalUnaryOp<float>(VIS_SECRET, permute_wrapper_1,
                                      xt::xarray<float>(x));
    const xt::xtensor<float, 2> expected_z = {{4, 3, 1, 2, 5},   //
                                              {6, 7, 8, 9, 10},  //
                                              {15, 14, 13, 12, 11}};

    EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
        << expected_z << std::endl
        << z << std::endl;
  }
  {
    auto z = test::evalUnaryOp<float>(VIS_PUBLIC, permute_wrapper_2,
                                      xt::xarray<float>(x));
    const xt::xtensor<float, 2> expected_z = {{1, 2, 3, 9, 10},    //
                                              {6, 7, 13, 14, 10},  //
                                              {11, 12, 8, 4, 15}};

    EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
        << expected_z << std::endl
        << z << std::endl;
  }
}

TEST(PermuteTest, 3d) {
  // GIVEN
  const xt::xtensor<float, 3> x = {{{1, 2, 3, 4, 5},   //
                                    {6, 7, 8, 9, 10},  //
                                    {11, 12, 13, 14, 15}},
                                   {{10, 20, 30, 40, 50},   //
                                    {60, 70, 80, 90, 100},  //
                                    {110, 120, 130, 140, 150}}};

  // WHAT
  auto permute_wrapper_1 = [](HalContext* ctx, const Value& input) {
    return permute(ctx, input, 2,
                   xt::xtensor<size_t, 3>({{{3, 2, 0, 1, 4},  //
                                            {0, 1, 2, 3, 4},  //
                                            {4, 3, 2, 1, 0}},
                                           {{3, 2, 0, 1, 4},  //
                                            {0, 1, 2, 3, 4},  //
                                            {4, 3, 2, 1, 0}}}));
  };
  auto permute_wrapper_2 = [](HalContext* ctx, const Value& input) {
    return permute(ctx, input, 1,
                   xt::xtensor<size_t, 3>({{{0, 0, 0, 1, 1},  //
                                            {1, 1, 2, 2, 1},  //
                                            {2, 2, 1, 0, 2}},
                                           {{0, 0, 0, 1, 1},  //
                                            {1, 1, 2, 2, 1},  //
                                            {2, 2, 1, 0, 2}}}));
  };
  auto permute_wrapper_3 = [](HalContext* ctx, const Value& input) {
    return permute(ctx, input, 0,
                   xt::xtensor<size_t, 3>({{{0, 0, 0, 0, 0},  //
                                            {1, 1, 1, 1, 1},  //
                                            {0, 0, 0, 0, 0}},
                                           {{1, 1, 1, 1, 1},  //
                                            {0, 0, 0, 0, 0},  //
                                            {1, 1, 1, 1, 1}}}));
  };

  // THEN
  {
    auto z = test::evalUnaryOp<float>(VIS_SECRET, permute_wrapper_1,
                                      xt::xarray<float>(x));
    const xt::xtensor<float, 3> expected_z = {{{4, 3, 1, 2, 5},   //
                                               {6, 7, 8, 9, 10},  //
                                               {15, 14, 13, 12, 11}},
                                              {{40, 30, 10, 20, 50},   //
                                               {60, 70, 80, 90, 100},  //
                                               {150, 140, 130, 120, 110}}};

    EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
        << expected_z << std::endl
        << z << std::endl;
  }
  {
    auto z = test::evalUnaryOp<float>(VIS_PUBLIC, permute_wrapper_2,
                                      xt::xarray<float>(x));
    const xt::xtensor<float, 3> expected_z = {{{1, 2, 3, 9, 10},    //
                                               {6, 7, 13, 14, 10},  //
                                               {11, 12, 8, 4, 15}},
                                              {{10, 20, 30, 90, 100},    //
                                               {60, 70, 130, 140, 100},  //
                                               {110, 120, 80, 40, 150}}};

    EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
        << expected_z << std::endl
        << z << std::endl;
  }
  {
    auto z = test::evalUnaryOp<float>(VIS_PUBLIC, permute_wrapper_3,
                                      xt::xarray<float>(x));
    const xt::xtensor<float, 3> expected_z = {{{1, 2, 3, 4, 5},        //
                                               {60, 70, 80, 90, 100},  //
                                               {11, 12, 13, 14, 15}},
                                              {{10, 20, 30, 40, 50},  //
                                               {6, 7, 8, 9, 10},      //
                                               {110, 120, 130, 140, 150}}};
    EXPECT_TRUE(xt::allclose(expected_z, z, 0.01, 0.001))
        << expected_z << std::endl
        << z << std::endl;
  }
}

}  // namespace
}  // namespace spu::hal
