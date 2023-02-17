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

#include "libspu/kernel/hal/pad.h"

#include "gtest/gtest.h"
#include "xtensor/xbroadcast.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xshape.hpp"

#include "libspu/kernel/hal/test_util.h"

namespace spu::kernel::hal {

using secret_v = std::integral_constant<Visibility, VIS_SECRET>;
using public_v = std::integral_constant<Visibility, VIS_PUBLIC>;

using ShapeOpsUnaryTestTypes = ::testing::Types<
    // s
    std::tuple<float, secret_v, float>,      // (sfxp)
    std::tuple<int32_t, secret_v, int64_t>,  // (sint)
    // p
    std::tuple<float, public_v, float>,     // (pfxp)
    std::tuple<int32_t, public_v, int64_t>  // (pint)
    >;

template <typename S>
class ShapeOpsUnaryTest : public ::testing::Test {};
TYPED_TEST_SUITE(ShapeOpsUnaryTest, ShapeOpsUnaryTestTypes);

TEST(ShapeOpsUnaryTest, Pad) {
  // GIVEN
  xt::xarray<int32_t> x = {{{
      {1, 2},  // row 0
      {3, 4},  // row 1
      {5, 6},  // row 2
  }}};

  using P_VT = public_v::type;
  auto pad_wrapper = [](HalContext* ctx, const Value& in) {
    return pad(ctx, in, make_value(ctx, P_VT(), 35), {1, 0, 0, 0}, {0, 2, 0, 0},
               {2, 1, 0, 0});
  };

  // WHAT
  auto z = test::evalUnaryOp<int32_t>(P_VT(), pad_wrapper, x);

  // THEN
  auto expected = xt::xarray<int32_t>({{{{35, 35}, {35, 35}, {35, 35}},
                                        {{35, 35}, {35, 35}, {35, 35}},
                                        {{35, 35}, {35, 35}, {35, 35}}},
                                       {{{1, 2}, {3, 4}, {5, 6}},
                                        {{35, 35}, {35, 35}, {35, 35}},
                                        {{35, 35}, {35, 35}, {35, 35}}}});
  EXPECT_TRUE(xt::allclose(z, expected, 0.01, 0.001)) << z << std::endl
                                                      << expected;
}

TEST(ShapeOpsUnaryTest, InteriorPadding) {
  // GIVEN
  xt::xarray<int32_t> x = {{1, 2, 3, 4, 5},
                           {6, 7, 8, 9, 10},
                           {11, 12, 13, 14, 15},
                           {16, 17, 18, 19, 20}};

  using P_VT = public_v::type;
  auto pad_wrapper = [](HalContext* ctx, const Value& in) {
    return pad(ctx, in, make_value(ctx, P_VT(), 0), {0, 0}, {0, 0}, {1, 1});
  };

  // WHAT
  auto z = test::evalUnaryOp<int32_t>(P_VT(), pad_wrapper, x);

  // THEN
  xt::xarray<int32_t> expected = {{1, 0, 2, 0, 3, 0, 4, 0, 5},   //
                                  {0, 0, 0, 0, 0, 0, 0, 0, 0},   //
                                  {6, 0, 7, 0, 8, 0, 9, 0, 10},  //
                                  {0, 0, 0, 0, 0, 0, 0, 0, 0},   //
                                  {11, 0, 12, 0, 13, 0, 14, 0, 15},
                                  {0, 0, 0, 0, 0, 0, 0, 0, 0},  //
                                  {16, 0, 17, 0, 18, 0, 19, 0, 20}};
  EXPECT_EQ(z, expected) << z << std::endl << expected;
}

TEST(ShapeOpsUnaryTest, NegativeEdgePad) {
  // GIVEN
  xt::xarray<int32_t> x = {{1, 2, 3, 4, 5},
                           {6, 7, 8, 9, 10},
                           {11, 12, 13, 14, 15},
                           {16, 17, 18, 19, 20}};

  using P_VT = public_v::type;
  auto pad_wrapper = [](HalContext* ctx, const Value& in) {
    return pad(ctx, in, make_value(ctx, P_VT(), 0), {-1, -1}, {-1, -1}, {0, 0});
  };

  // WHAT
  auto z = test::evalUnaryOp<int32_t>(P_VT(), pad_wrapper, x);

  // THEN
  xt::xarray<int32_t> expected = {{7, 8, 9},  //
                                  {12, 13, 14}};
  EXPECT_EQ(z, expected) << z << std::endl << expected;
}

TEST(ShapeOpsUnaryTest, NegativeEdgePadWithInteriorPad) {
  // GIVEN
  xt::xarray<int32_t> x = {{1, 2, 3, 4, 5},
                           {6, 7, 8, 9, 10},
                           {11, 12, 13, 14, 15},
                           {16, 17, 18, 19, 20}};

  using P_VT = public_v::type;
  auto pad_wrapper = [](HalContext* ctx, const Value& in) {
    return pad(ctx, in, make_value(ctx, P_VT(), 0), {-1, -1}, {-1, -1}, {1, 1});
  };

  // WHAT
  auto z = test::evalUnaryOp<int32_t>(P_VT(), pad_wrapper, x);

  // THEN
  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 0, 0, 0},     //
                                  {0, 7, 0, 8, 0, 9, 0},     //
                                  {0, 0, 0, 0, 0, 0, 0},     //
                                  {0, 12, 0, 13, 0, 14, 0},  //
                                  {0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(z, expected) << z << std::endl << expected;
}

TEST(ShapeOpsUnaryTest, HighNegativeEdgePadWithInteriorPad) {
  // GIVEN
  xt::xarray<int32_t> x = {{1, 2, 3, 4, 5},
                           {6, 7, 8, 9, 10},
                           {11, 12, 13, 14, 15},
                           {16, 17, 18, 19, 20}};

  using P_VT = public_v::type;
  auto pad_wrapper = [](HalContext* ctx, const Value& in) {
    return pad(ctx, in, make_value(ctx, P_VT(), 0), {-3, -3}, {-1, -1}, {1, 1});
  };

  // WHAT
  auto z = test::evalUnaryOp<int32_t>(P_VT(), pad_wrapper, x);

  // THEN
  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 0},  //
                                  {0, 13, 0, 14, 0},
                                  {0, 0, 0, 0, 0}};
  EXPECT_EQ(z, expected) << z << std::endl << expected;
}

}  // namespace spu::kernel::hal
