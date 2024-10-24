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

#include "libspu/kernel/hal/shape_ops.h"

#include "gtest/gtest.h"
#include "xtensor/xbroadcast.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xshape.hpp"

#include "libspu/kernel/test_util.h"

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

TYPED_TEST(ShapeOpsUnaryTest, Transpose) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({2, 3, 4});

  // WHAT
  auto transpose_wrapper = [](SPUContext* ctx, const Value& x) {
    return transpose(ctx, x);
  };
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), transpose_wrapper, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::transpose(x), z, 0.01, 0.001)) << x << std::endl
                                                              << z;
}

TYPED_TEST(ShapeOpsUnaryTest, TransposeWithPermutation) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({2, 2, 4});

  auto transpose_wrapper = [](SPUContext* ctx, const Value& x) {
    return transpose(ctx, x, {1, 2, 0});
  };

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), transpose_wrapper, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::transpose(x, {1, 2, 0}), z, 0.01, 0.001))
      << x << std::endl
      << z;
}

TYPED_TEST(ShapeOpsUnaryTest, BroadcastTo) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 1, 6});

  auto broadcast_to_wrapper = [](SPUContext* ctx, const Value& in) {
    return broadcast_to(ctx, in, {5, 4, 6});
  };

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), broadcast_to_wrapper, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::broadcast(x, std::vector<int>{5, 4, 6}), z, 0.01,
                           0.001));
}

TYPED_TEST(ShapeOpsUnaryTest, BroadcastScalar) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({});

  auto broadcast_to_wrapper = [](SPUContext* ctx, const Value& in) {
    return broadcast_to(ctx, in, {1, 1});
  };

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), broadcast_to_wrapper, x);

  // THEN
  EXPECT_EQ(z.shape(), std::vector<size_t>(2, 1));
}

TYPED_TEST(ShapeOpsUnaryTest, BroadcastInDims) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = {1, 2, 3, 4};

  auto broadcast_to_wrapper = [](SPUContext* ctx, const Value& in) {
    return broadcast_to(ctx, in, {4, 2}, {0});
  };

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), broadcast_to_wrapper, x);

  // THEN
  EXPECT_TRUE(xt::allclose(
      xt::broadcast(xt::reshape_view(x, {4, 1}), std::vector<int>{4, 2}), z,
      0.01, 0.001));
}

TEST(SliceTest, Slice) {
  // GIVEN
  xt::xarray<int32_t> x = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}};
  using P_VT = public_v::type;

  auto slice_wrapper = [](SPUContext* ctx, const Value& in) {
    return slice(ctx, in, {2, 1}, {4, 3}, {});
  };
  auto z = test::evalUnaryOp<int64_t>(P_VT(), slice_wrapper, x);

  EXPECT_EQ(std::vector<int64_t>(z.shape().begin(), z.shape().end()),
            std::vector<int64_t>({2, 2}));

  EXPECT_EQ(xt::view(x, xt::range(2, 4), xt::range(1, 3)), z);
}

TEST(SliceTest, UpdateSlice) {
  // GIVEN
  xt::xarray<int32_t> x = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
  xt::xarray<int32_t> y = {{0, 1}, {3, 4}};
  using P_VT = public_v::type;

  auto update_slice_wrapper = [](SPUContext* ctx, const Value& in,
                                 const Value& update) {
    return update_slice(ctx, in, update, {1, 1});
  };
  auto z =
      test::evalBinaryOp<int64_t>(P_VT(), P_VT(), update_slice_wrapper, x, y);

  EXPECT_EQ(xt::view(z, xt::range(1, 3), xt::range(1, 3)), y);
}

TEST(SliceTest, SliceStride) {
  // GIVEN
  xt::xarray<int32_t> x = {0, 1, 2, 3};
  using P_VT = public_v::type;

  auto slice_wrapper = [](SPUContext* ctx, const Value& in) {
    return slice(ctx, in, {0}, {4}, {3});
  };
  auto z = test::evalUnaryOp<int64_t>(P_VT(), slice_wrapper, x);

  EXPECT_EQ(std::vector<int64_t>(z.shape().begin(), z.shape().end()),
            std::vector<int64_t>({2}));
  xt::xarray<int64_t> expected = {0, 3};

  EXPECT_EQ(z, expected);
}

TEST(ReshapeTest, Reshape) {
  // GIVEN
  xt::xarray<int32_t> x = {1, 2, 3, 4};
  using P_VT = public_v::type;

  auto reshape_wrapper = [](SPUContext* ctx, const Value& in) {
    return reshape(ctx, in, {2, 2});
  };
  auto z = test::evalUnaryOp<int64_t>(P_VT(), reshape_wrapper, x);

  EXPECT_EQ(std::vector<int64_t>(z.shape().begin(), z.shape().end()),
            std::vector<int64_t>({2, 2}));
}

TEST(ShapeOpsUnaryTest, Reverse) {
  // GIVEN
  xt::xarray<int32_t> x = {
      {{1, 2}, {3, 4}},
      {{5, 6}, {7, 8}},
  };
  using P_VT = public_v::type;

  auto reverse_wrapper = [](SPUContext* ctx, const Value& in) {
    return reverse(ctx, in, {0, 1, 2});
  };

  // WHAT
  auto z = test::evalUnaryOp<int32_t>(P_VT(), reverse_wrapper, x);
  xt::xarray<int32_t> expected = x;
  for (const auto& dim : {0, 1, 2}) {
    expected = xt::flip(expected, dim);
  }

  // THEN
  EXPECT_TRUE(xt::allclose(z, expected, 0.01, 0.001)) << z << std::endl
                                                      << expected;
}

TEST(ShapeOpsUnaryTest, BroadcastAfterReshape) {
  // GIVEN
  xt::xarray<int32_t> x = {{1, 2}, {3, 4}};

  using P_VT = public_v::type;
  auto pad_wrapper = [](SPUContext* ctx, const Value& in) {
    auto x1 = reshape(ctx, in, {1, 2, 2});
    return broadcast_to(ctx, x1, {2, 2, 2});
  };

  // WHAT
  auto z = test::evalUnaryOp<int32_t>(P_VT(), pad_wrapper, x);

  // THEN
  xt::xarray<int32_t> expected = {{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}};
  EXPECT_EQ(z, expected) << z << std::endl << expected;
}

TEST(ShapeOpsUnaryTest, Pad) {
  // GIVEN
  xt::xarray<int32_t> x = {{{
      {1, 2},  // row 0
      {3, 4},  // row 1
      {5, 6},  // row 2
  }}};

  using P_VT = public_v::type;
  auto pad_wrapper = [](SPUContext* ctx, const Value& in) {
    return pad(ctx, in, test::makeValue(ctx, 35, P_VT()), {1, 0, 0, 0},
               {0, 2, 0, 0}, {2, 1, 0, 0});
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
  auto pad_wrapper = [](SPUContext* ctx, const Value& in) {
    return pad(ctx, in, test::makeValue(ctx, 0, P_VT()), {0, 0}, {0, 0},
               {1, 1});
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
  auto pad_wrapper = [](SPUContext* ctx, const Value& in) {
    return pad(ctx, in, test::makeValue(ctx, 0, P_VT()), {-1, -1}, {-1, -1},
               {0, 0});
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
  auto pad_wrapper = [](SPUContext* ctx, const Value& in) {
    return pad(ctx, in, test::makeValue(ctx, 0, P_VT()), {-1, -1}, {-1, -1},
               {1, 1});
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
  auto pad_wrapper = [](SPUContext* ctx, const Value& in) {
    return pad(ctx, in, test::makeValue(ctx, 0, P_VT()), {-3, -3}, {-1, -1},
               {1, 1});
  };

  // WHAT
  auto z = test::evalUnaryOp<int32_t>(P_VT(), pad_wrapper, x);

  // THEN
  xt::xarray<int32_t> expected = {{0, 0, 0, 0, 0},  //
                                  {0, 13, 0, 14, 0},
                                  {0, 0, 0, 0, 0}};
  EXPECT_EQ(z, expected) << z << std::endl << expected;
}

using SecretV = std::integral_constant<Visibility, VIS_SECRET>;
using PublicV = std::integral_constant<Visibility, VIS_PUBLIC>;

using ConcatTestTypes = ::testing::Types<   //
    std::tuple<float, SecretV, SecretV>,    // concat(s, s)
    std::tuple<float, PublicV, PublicV>,    // concat(p, p)
    std::tuple<float, SecretV, PublicV>,    // concat(s, p)
    std::tuple<float, PublicV, SecretV>,    // concat(p, s)
                                            //
    std::tuple<int16_t, SecretV, SecretV>,  // concat(s, s)
    std::tuple<int16_t, PublicV, PublicV>,  // concat(p, p)
    std::tuple<int16_t, SecretV, PublicV>,  // concat(s, p)
    std::tuple<int16_t, PublicV, SecretV>   // concat(p, s)
    >;

template <typename S>
class ConcatTest : public ::testing::Test {};

TYPED_TEST_SUITE(ConcatTest, ConcatTestTypes);

TYPED_TEST(ConcatTest, Concatenate) {
  using DT = typename std::tuple_element<0, TypeParam>::type;
  using LhsVt = typename std::tuple_element<1, TypeParam>::type;
  using RhsVt = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<DT> x = test::xt_random<DT>({3, 3});
  xt::xarray<DT> y = test::xt_random<DT>({3, 3});

  auto concat_wrapper = [](SPUContext* ctx, const Value& lhs,
                           const Value& rhs) {
    return concatenate(ctx, {lhs, rhs}, 0);
  };
  // WHAT
  auto z = test::evalBinaryOp<DT>(LhsVt(), RhsVt(), concat_wrapper, x, y);

  // THEN
  EXPECT_TRUE(
      xt::allclose(xt::concatenate(xt::xtuple(x, y), 0), z, 0.01, 0.001))
      << x << std::endl
      << y << std::endl
      << z;
}

TYPED_TEST(ConcatTest, VConcatenate) {
  using DT = typename std::tuple_element<0, TypeParam>::type;
  using LhsVt = typename std::tuple_element<1, TypeParam>::type;
  using RhsVt = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<DT> x = test::xt_random<DT>({3, 3});
  xt::xarray<DT> y = test::xt_random<DT>({3, 3});

  auto concat_wrapper = [](SPUContext* ctx, const Value& lhs,
                           const Value& rhs) {
    return concatenate(ctx, {lhs, rhs}, 1);
  };
  // WHAT
  auto z = test::evalBinaryOp<DT>(LhsVt(), RhsVt(), concat_wrapper, x, y);

  // THEN
  EXPECT_TRUE(
      xt::allclose(xt::concatenate(xt::xtuple(x, y), 1), z, 0.01, 0.001))
      << x << std::endl
      << y << std::endl
      << z;
}

}  // namespace spu::kernel::hal
