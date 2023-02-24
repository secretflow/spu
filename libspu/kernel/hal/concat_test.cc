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

#include "libspu/kernel/hal/concat.h"

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"

#include "libspu/kernel/hal/test_util.h"

namespace spu::kernel::hal {

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

  auto concat_wrapper = [](HalContext* ctx, const Value& lhs,
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

  auto concat_wrapper = [](HalContext* ctx, const Value& lhs,
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
