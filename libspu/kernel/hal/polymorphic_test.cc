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

#include "libspu/kernel/hal/polymorphic.h"

#include <algorithm>
#include <type_traits>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xvectorize.hpp"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/linalg.h"

namespace spu::kernel::hal {

using secret_v = std::integral_constant<Visibility, VIS_SECRET>;
using public_v = std::integral_constant<Visibility, VIS_PUBLIC>;

template <typename S>
class MathTest : public ::testing::Test {};

using MathTestTypes = ::testing::Types<
    // ss
    std::tuple<float, secret_v, float, secret_v,
               std::common_type_t<float, float>>,  // (sfxp, sfxp)
    std::tuple<float, secret_v, int32_t, secret_v,
               std::common_type_t<float, int32_t>>,  // (sfxp, sint)
    std::tuple<int32_t, secret_v, float, secret_v,
               std::common_type_t<int32_t, float>>,  // (sint, sfxp)
    std::tuple<int32_t, secret_v, int32_t, secret_v,
               std::common_type_t<int32_t, int32_t>>,  // (sint, sint)
    std::tuple<int16_t, secret_v, int32_t, secret_v,
               std::common_type_t<int16_t, int32_t>>,  // (sint, sint)

    // sp
    std::tuple<float, secret_v, float, public_v,
               std::common_type_t<float, float>>,  // (sfxp, pfxp)
    std::tuple<float, secret_v, int32_t, public_v,
               std::common_type_t<float, int32_t>>,  // (sfxp, pint)
    std::tuple<int32_t, secret_v, float, public_v,
               std::common_type_t<int32_t, float>>,  // (sint, pfxp)
    std::tuple<int32_t, secret_v, int32_t, public_v,
               std::common_type_t<int32_t, int32_t>>,  // (sint, pint)
    std::tuple<int16_t, secret_v, int32_t, public_v,
               std::common_type_t<int16_t, int32_t>>,  // (sint, pint)

    // ps
    std::tuple<float, public_v, float, secret_v,
               std::common_type_t<float, float>>,  // (pfxp, sfxp)
    std::tuple<float, public_v, int32_t, secret_v,
               std::common_type_t<float, int32_t>>,  // (pfxp, sint)
    std::tuple<int32_t, public_v, float, secret_v,
               std::common_type_t<int32_t, float>>,  // (pint, sfxp)
    std::tuple<int32_t, public_v, int32_t, secret_v,
               std::common_type_t<int32_t, int32_t>>,  // (pint, sint)
    std::tuple<int16_t, public_v, int32_t, secret_v,
               std::common_type_t<int16_t, int32_t>>,  // (pint, pint)

    // pp
    std::tuple<float, public_v, float, public_v,
               std::common_type_t<float, float>>,  // (pfxp, pfxp)
    std::tuple<float, public_v, int32_t, public_v,
               std::common_type_t<float, int32_t>>,  // (pfxp, pint)
    std::tuple<int32_t, public_v, float, public_v,
               std::common_type_t<int32_t, float>>,  // (pint, pfxp)
    std::tuple<int32_t, public_v, int32_t, public_v,
               std::common_type_t<int32_t, int32_t>>,  // (pint, pint)
    std::tuple<int16_t, public_v, int32_t, public_v,
               std::common_type_t<int16_t, int32_t>>  // (pint, pint)
    >;

TYPED_TEST_SUITE(MathTest, MathTestTypes);

TYPED_TEST(MathTest, Add) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), add, x, y);

  // THEN
  EXPECT_TRUE(xt::allclose(x + y, z, 0.01, 0.001)) << (x + y) << std::endl
                                                   << z << std::endl;
}

TYPED_TEST(MathTest, Sub) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), sub, x, y);

  // THEN
  EXPECT_TRUE(xt::allclose(x - y, z, 0.01, 0.001)) << (x - y) << std::endl
                                                   << z << std::endl;
}

TYPED_TEST(MathTest, Mul) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), mul, x, y);

  // THEN
  EXPECT_TRUE(xt::allclose(x * y, z, 0.01, 0.001)) << (x * y) << std::endl
                                                   << z << std::endl;
}

TYPED_TEST(MathTest, Matmul) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // M x M
  // GIVEN
  {
    const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
    const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({6, 7});

    // WHAT
    auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), matmul, x, y);

    // THEN
    auto expected = xt::ones_like(z);
    if constexpr (std::is_same_v<LHS_DT, RHS_DT>) {
      mpc::linalg::matmul(5, 7, 6, x.data(), 6, 1, y.data(), 7, 1,
                          expected.data(), 7, 1);
    } else {
      using PT = std::common_type_t<LHS_DT, RHS_DT>;
      const xt::xarray<PT> xp = xt::cast<PT>(x);
      const xt::xarray<PT> yp = xt::cast<PT>(y);
      mpc::linalg::matmul(5, 7, 6, xp.data(), 6, 1, yp.data(), 7, 1,
                          expected.data(), 7, 1);
    }

    EXPECT_TRUE(xt::allclose(expected, z, 0.01, 0.001)) << expected << std::endl
                                                        << z << std::endl;
  }

  // M x V
  // GIVEN
  {
    const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
    const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({6});

    // WHAT
    auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), matmul, x, y);

    // THEN
    auto expected = xt::ones_like(z);
    if constexpr (std::is_same_v<LHS_DT, RHS_DT>) {
      mpc::linalg::matmul(5, 1, 6, x.data(), 6, 1, y.data(), 1, 1,
                          expected.data(), 1, 1);
    } else {
      using PT = std::common_type_t<LHS_DT, RHS_DT>;
      const xt::xarray<PT> xp = xt::cast<PT>(x);
      const xt::xarray<PT> yp = xt::cast<PT>(y);
      mpc::linalg::matmul(5, 1, 6, xp.data(), 6, 1, yp.data(), 1, 1,
                          expected.data(), 1, 1);
    }

    EXPECT_TRUE(xt::allclose(expected, z, 0.01, 0.001)) << expected << std::endl
                                                        << z << std::endl;
  }

  // V x M GIVEN
  // {
  //   const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5});
  //   const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  //   // WHAT
  //   auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), matmul, x, y);

  //   // THEN
  //   EXPECT_TRUE(xt::allclose(xt::linalg::dot(x, y), z, 0.01, 0.001))
  //       << xt::linalg::dot(x, y) << std::endl
  //       << z << std::endl;
  // }

  // // V x V
  // // GIVEN
  // {
  //   const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5});
  //   const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5});

  //   // WHAT
  //   auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), matmul, x, y);

  //   // THEN
  //   EXPECT_TRUE(xt::allclose(xt::linalg::dot(x, y), z, 0.01, 0.001))
  //       << xt::linalg::dot(x, y) << std::endl
  //       << z << std::endl;
  // }
}

using LogicOpTestTypes = ::testing::Types<
    // ss
    std::tuple<float, secret_v, float, secret_v, int64_t>,      // (sfxp, sfxp)
    std::tuple<float, secret_v, int32_t, secret_v, int64_t>,    // (sfxp, sint)
    std::tuple<int32_t, secret_v, float, secret_v, int64_t>,    // (sint, sfxp)
    std::tuple<int32_t, secret_v, int32_t, secret_v, int64_t>,  // (sint, sint)
    // sp
    std::tuple<float, secret_v, float, public_v, int64_t>,      // (sfxp, sfxp)
    std::tuple<float, secret_v, int32_t, public_v, int64_t>,    // (sfxp, sint)
    std::tuple<int32_t, secret_v, float, public_v, int64_t>,    // (sint, sfxp)
    std::tuple<int32_t, secret_v, int32_t, public_v, int64_t>,  // (sint, sint)
    // ps
    std::tuple<float, public_v, float, secret_v, int64_t>,      // (sfxp, sfxp)
    std::tuple<float, public_v, int32_t, secret_v, int64_t>,    // (sfxp, sint)
    std::tuple<int32_t, public_v, float, secret_v, int64_t>,    // (sint, sfxp)
    std::tuple<int32_t, public_v, int32_t, secret_v, int64_t>,  // (sint, sint)
    // pp
    std::tuple<float, public_v, float, public_v, int64_t>,     // (sfxp, sfxp)
    std::tuple<float, public_v, int32_t, public_v, int64_t>,   // (sfxp, sint)
    std::tuple<int32_t, public_v, float, public_v, int64_t>,   // (sint, sfxp)
    std::tuple<int32_t, public_v, int32_t, public_v, int64_t>  // (sint, sint)
    >;

template <typename S>
class LogicOpTest : public ::testing::Test {};

TYPED_TEST_SUITE(LogicOpTest, LogicOpTestTypes);

TYPED_TEST(LogicOpTest, Equal) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6}, 0, 2);
  xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6}, 0, 2);

  xt::row(x, 0) = xt::row(y, 0);

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), equal, x, y);

  // THEN
  EXPECT_EQ(xt::equal(x, y), z);
}

TYPED_TEST(LogicOpTest, NotEqual) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  xt::row(x, 0) = xt::row(y, 0);

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), not_equal, x, y);

  // THEN
  EXPECT_EQ(xt::not_equal(x, y), z);
}

TYPED_TEST(LogicOpTest, Less) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), less, x, y);

  // THEN
  EXPECT_EQ(xt::less(x, y), z);
}

TYPED_TEST(LogicOpTest, LessEqual) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});
  xt::row(x, 0) = xt::row(y, 0);

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), less_equal, x, y);

  // THEN
  EXPECT_EQ(xt::less_equal(x, y), z);
}

TYPED_TEST(LogicOpTest, Greater) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), greater, x, y);

  // THEN
  EXPECT_EQ(xt::greater(x, y), z);
}

TYPED_TEST(LogicOpTest, GreaterEqual) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});
  xt::row(x, 0) = xt::row(y, 0);

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), greater_equal, x, y);

  // THEN
  EXPECT_EQ(xt::greater_equal(x, y), z);
}

TYPED_TEST(MathTest, Max) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  if constexpr (!std::is_same_v<LHS_DT, RHS_DT> ||
                !std::is_same_v<LHS_VT, RHS_VT>) {
    return;
  }

  // GIVEN
  const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), max, x, y);

  // THEN
  auto expected = xt::maximum(x, y);
  EXPECT_TRUE(xt::allclose(expected, z, 0.01, 0.001)) << expected << std::endl
                                                      << z << std::endl;
}

TYPED_TEST(MathTest, Pow) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  if constexpr (!std::is_same_v<LHS_DT, RHS_DT> ||
                !std::is_same_v<LHS_VT, RHS_VT> || std::is_integral_v<RHS_DT>) {
    return;
  }

  // GIVEN
  const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6}, 0, 100);
  const xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6}, 0, 2);

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), power, x, y);

  // THEN
  auto expected = xt::pow(x, y);
  EXPECT_TRUE(xt::allclose(expected, z, 0.3, 0.03)) << x << std::endl
                                                    << y << std::endl
                                                    << expected << std::endl
                                                    << z << std::endl;
}

using MathUnaryTestTypes = ::testing::Types<
    // s
    std::tuple<float, secret_v, float>,      // (sfxp)
    std::tuple<int32_t, secret_v, int64_t>,  // (sint)
    // p
    std::tuple<float, public_v, float>,     // (pfxp)
    std::tuple<int32_t, public_v, int64_t>  // (pint)
    >;

template <typename S>
class MathUnaryTest : public ::testing::Test {};
TYPED_TEST_SUITE(MathUnaryTest, MathUnaryTestTypes);

using FpOnlyMathUnaryTestTypes = ::testing::Types<
    // s
    std::tuple<float, secret_v, float>,  // (sfxp)
    // p
    std::tuple<float, public_v, float>  // (pfxp)
    >;

template <typename S>
class FpOnlyMathUnaryTest : public ::testing::Test {};
TYPED_TEST_SUITE(FpOnlyMathUnaryTest, FpOnlyMathUnaryTestTypes);

TYPED_TEST(MathUnaryTest, Negate) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6});

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), negate, x);

  // THEN
  EXPECT_TRUE(xt::allclose(-x, z, 0.01, 0.001));
}

TYPED_TEST(MathUnaryTest, Abs) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = typename std::tuple_element<2, TypeParam>::type;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6});

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), abs, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::abs(x), z, 0.01, 0.001));
}

TYPED_TEST(FpOnlyMathUnaryTest, Exp) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = float;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6}, 0, 2);

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), exp, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::exp(x), z, 0.01, 0.001))
      << xt::exp(x) << std::endl
      << z;
}

TYPED_TEST(MathUnaryTest, Floor) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = float;

  if (std::is_integral_v<IN_DT>) {
    return;
  }

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6});

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), floor, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::floor(x), z, 0.01, 0.001))
      << xt::floor(x) << std::endl
      << z;
}

TYPED_TEST(MathUnaryTest, Reciprocal) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = float;

  // GIVEN
  if (std::is_integral_v<IN_DT>) {
    return;
  }
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6}, 0.1, 100);

  auto log_x = 1 / x;

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), reciprocal, x);

  // THEN
  EXPECT_TRUE(xt::allclose(log_x, z, 0.1, 0.05)) << log_x << std::endl << z;
}

TYPED_TEST(FpOnlyMathUnaryTest, Log) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = float;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6}, 0.5, 10);
  xt::xarray<float> log_x = xt::log(xt::cast<float>(x));

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), log, x);

  // THEN
  EXPECT_TRUE(xt::allclose(log_x, z, 0.1, 0.001)) << log_x << std::endl << z;
}

// TODO: can not pass MM1 & SEG3 test.
TYPED_TEST(MathUnaryTest, Logistic) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = float;

  if (std::is_integral_v<IN_DT>) {
    return;
  }

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6}, -5, 10);
  xt::xarray<float> x_logisitic = 1.0 / (1.0 + xt::exp(-xt::cast<float>(x)));

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), logistic, x);

  // THEN
  EXPECT_TRUE(xt::allclose(x_logisitic, z, 0.01, 0.001))
      << x << std::endl
      << x_logisitic << std::endl
      << z;
}

TYPED_TEST(FpOnlyMathUnaryTest, Tanh) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = float;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6}, 0, 4);

  // WHAT
  auto z = test::evalUnaryOp<RES_DT>(IN_VT(), tanh, x);

  // THEN
  EXPECT_TRUE(xt::allclose(xt::tanh(x), z, 0.01, 0.001))
      << xt::tanh(x) << std::endl
      << z;
}

TYPED_TEST(FpOnlyMathUnaryTest, RSqrt) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = float;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6}, 0, 20000);
  xt::xarray<float> expected_y = 1.0f / xt::sqrt(x);

  // WHAT
  auto y = test::evalUnaryOp<RES_DT>(IN_VT(), rsqrt, x);

  // THEN
  EXPECT_TRUE(xt::allclose(expected_y, y, 0.01, 0.001))
      << expected_y << std::endl
      << y;
}

TYPED_TEST(FpOnlyMathUnaryTest, Sqrt) {
  using IN_DT = typename std::tuple_element<0, TypeParam>::type;
  using IN_VT = typename std::tuple_element<1, TypeParam>::type;
  using RES_DT = float;

  // GIVEN
  xt::xarray<IN_DT> x = test::xt_random<IN_DT>({5, 6}, 0, 20000);
  xt::xarray<float> expected_y = xt::sqrt(x);

  // WHAT
  auto y = test::evalUnaryOp<RES_DT>(IN_VT(), sqrt, x);

  // THEN
  EXPECT_TRUE(xt::allclose(expected_y, y, 0.01, 0.001))
      << expected_y << std::endl
      << y;
}

TEST(MathTest, Select) {
  // GIVEN
  xt::xarray<int32_t> p =
      test::xt_random<int32_t>({5, 6}, 0, 2);  // pred is a [0, 2) boolean array
  xt::xarray<int32_t> x = test::xt_random<int32_t>({5, 6});
  xt::xarray<int32_t> y = test::xt_random<int32_t>({5, 6});
  using P_VT = public_v::type;

  auto z =
      test::evalTernaryOp<int64_t>(P_VT(), P_VT(), P_VT(), select, p, x, y);

  EXPECT_EQ(z.shape(), x.shape());
  EXPECT_EQ(z.shape(), y.shape());
  EXPECT_EQ(z.shape(), p.shape());

  for (size_t idx = 0; idx < p.size(); ++idx) {
    EXPECT_EQ(z[idx], p[idx] != 0 ? x[idx] : y[idx]);
  }
}

class LogisticTest
    : public ::testing::TestWithParam<RuntimeConfig::SigmoidMode> {};

TEST_P(LogisticTest, Logistic) {
  // GIVEN
  RuntimeConfig config;
  config.set_protocol(ProtocolKind::REF2K);
  config.set_field(FieldType::FM64);
  config.set_sigmoid_mode(GetParam());
  SPUContext ctx = test::makeSPUContext(config, nullptr);

  xt::xarray<float> x{{1.0, 2.0}, {0.5, 1.8}};

  // public logistic
  {
    Value a = constant(&ctx, x, DT_F32);
    Value c = logistic(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, c);
    EXPECT_TRUE(xt::allclose(1.0 / (1.0 + xt::exp(-x)), y, 0.1, 0.5))
        << 1.0 / (1.0 + xt::exp(-x)) << std::endl
        << y;
  }

  // secret logistic
  {
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value c = logistic(&ctx, a);
    EXPECT_EQ(c.dtype(), DT_F32);

    auto y = dump_public_as<float>(&ctx, reveal(&ctx, c));
    // low precision
    EXPECT_TRUE(xt::allclose(1.0 / (1.0 + xt::exp(-x)), y, 0.1, 0.5))
        << 1.0 / (1.0 + xt::exp(-x)) << std::endl
        << y;
  }
}

TEST(MathTest, Clamp) {
  // GIVEN
  xt::xarray<int32_t> minv = test::xt_random<int32_t>({5, 6});
  xt::xarray<int32_t> x = test::xt_random<int32_t>({5, 6});
  xt::xarray<int32_t> maxv = test::xt_random<int32_t>({5, 6});
  using P_VT = public_v::type;

  auto z = test::evalTernaryOp<int64_t>(P_VT(), P_VT(), P_VT(), clamp, minv, x,
                                        maxv);

  EXPECT_EQ(xt::minimum(xt::maximum(minv, x), maxv), z) << minv << x << maxv;
}

TEST(MathTest, LessWithLargeNumber) {
  // GIVEN
  xt::xarray<float> x = {std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
  xt::xarray<float> y = {0, 0, 0, 0};
  using P_VT = public_v::type;

  auto z = test::evalBinaryOp<float>(P_VT(), P_VT(), less, x, y);

  EXPECT_EQ(xt::less(x, y), z) << z;
}

TEST(MathTest, GreaterWithLargeNumber) {
  // GIVEN
  xt::xarray<float> x = {std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
  xt::xarray<float> y = {0, 0, 0, 0};
  using P_VT = public_v::type;

  auto z = test::evalBinaryOp<float>(P_VT(), P_VT(), greater, x, y);

  EXPECT_EQ(xt::greater(x, y), z) << z;
}

INSTANTIATE_TEST_SUITE_P(
    LogisticTestInstance, LogisticTest,
    testing::Values(RuntimeConfig::SIGMOID_MM1, RuntimeConfig::SIGMOID_SEG3,
                    RuntimeConfig::SIGMOID_REAL),
    [](const testing::TestParamInfo<LogisticTest::ParamType>& p) {
      return fmt::format("{}", p.param);
    });

TYPED_TEST(MathTest, Div) {
  using LHS_DT = typename std::tuple_element<0, TypeParam>::type;
  using LHS_VT = typename std::tuple_element<1, TypeParam>::type;
  using RHS_DT = typename std::tuple_element<2, TypeParam>::type;
  using RHS_VT = typename std::tuple_element<3, TypeParam>::type;
  using RES_DT = typename std::tuple_element<4, TypeParam>::type;

  // GIVEN
  const xt::xarray<LHS_DT> x = test::xt_random<LHS_DT>({5, 6});
  xt::xarray<RHS_DT> y = test::xt_random<RHS_DT>({5, 6});

  if constexpr (std::is_same_v<RHS_DT, int32_t>) {
    // If y is int, exclude 0
    std::for_each(y.begin(), y.end(), [](RHS_DT& v) {
      if (v == 0) {
        v = 1;
      }
    });
  }

  // WHAT
  auto z = test::evalBinaryOp<RES_DT>(LHS_VT(), RHS_VT(), div, x, y);

  // THEN
  EXPECT_TRUE(xt::allclose(x / y, z, 0.01, 0.001)) << (x / y) << std::endl
                                                   << z << std::endl;
}

// TEST(PopcountTest, Works) {
//   {
//     // GIVEN
//     xt::xarray<int32_t> x = xt::xarray<int32_t>{1, 100, 1000, -1, -100,
//     -1000};

//    // WHAT
//    auto z = test::evalUnaryOp<int>(secret_v(), popcount, x);
//    auto expected = xt::xarray<int>{1, 3, 6, 64, 60, 56};

//    // THEN
//    EXPECT_TRUE(xt::allclose(expected, z, 0.01, 0.001)) << expected <<
//    std::endl
//                                                        << z;
//  }

//  {
//    // GIVEN
//    xt::xarray<float> x = xt::xarray<float>{1, 100, 1000, -1, -100, -1000};

//    // WHAT
//    auto z = test::evalUnaryOp<int>(secret_v(), popcount, x);
//    auto expected = xt::xarray<int>{1, 3, 6, 46, 42, 38};

//    // THEN
//    EXPECT_TRUE(xt::allclose(expected, z, 0.01, 0.001)) << expected <<
//    std::endl
//                                                        << z;
//  }
//}

}  // namespace spu::kernel::hal
