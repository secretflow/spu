// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/sort.h"

#include <algorithm>
#include <random>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hlo {

TEST(SortTest, Simple) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.5, 0.24}, {5, 50, 2}};
  xt::xarray<float> sorted_x = {{0.05, 0.24, 0.5}, {2, 5, 50}};

  Value x_v = test::makeValue(&ctx, x, VIS_SECRET);

  std::vector<spu::Value> rets = Sort(
      &ctx, {x_v}, 1, false,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 1);

  auto sorted_x_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

  EXPECT_TRUE(xt::allclose(sorted_x, sorted_x_hat, 0.01, 0.001))
      << sorted_x << std::endl
      << sorted_x_hat << std::endl;
}

TEST(SortTest, SimpleWithNoPadding) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> x = {{0.05, 0.5, 0.24, 0.5}, {2, 5, 50, 2}};
  xt::xarray<float> sorted_x = {{0.05, 0.24, 0.5, 0.5}, {2, 2, 5, 50}};

  Value x_v = test::makeValue(&ctx, x, VIS_SECRET);

  std::vector<spu::Value> rets = Sort(
      &ctx, {x_v}, 1, false,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 1);

  auto sorted_x_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

  EXPECT_TRUE(xt::allclose(sorted_x, sorted_x_hat, 0.01, 0.001))
      << sorted_x << std::endl
      << sorted_x_hat << std::endl;
}

TEST(SortTest, MultiInputs) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> x1 = {{0.5, 0.05, 0.5, 0.24, 0.5, 0.5, 0.5}};
  xt::xarray<float> x2 = {{5, 1, 2, 1, 2, 3, 4}};
  xt::xarray<float> sorted_x1 = {{0.05, 0.24, 0.5, 0.5, 0.5, 0.5, 0.5}};
  xt::xarray<float> sorted_x2 = {{1, 1, 2, 2, 3, 4, 5}};

  Value x1_v = test::makeValue(&ctx, x1, VIS_SECRET);
  Value x2_v = test::makeValue(&ctx, x2, VIS_SECRET);

  std::vector<spu::Value> rets = Sort(
      &ctx, {x1_v, x2_v}, 1, false,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 2);

  auto sorted_x1_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

  EXPECT_TRUE(xt::allclose(sorted_x1, sorted_x1_hat, 0.01, 0.001))
      << sorted_x1 << std::endl
      << sorted_x1_hat << std::endl;

  // NOTE: Secret sort is unstable, so rets[1] need to be sort before check.
  auto sorted_x2_hat =
      xt::sort(hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1])));

  EXPECT_TRUE(xt::allclose(sorted_x2, sorted_x2_hat, 0.01, 0.001))
      << sorted_x2 << std::endl
      << sorted_x2_hat << std::endl;
}

TEST(SortTest, MultiOperands) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> k1 = {6, 6, 3, 4, 4, 5, 4};
  xt::xarray<float> k2 = {0.5, 0.1, 3.1, 6.5, 4.1, 6.7, 2.5};

  xt::xarray<float> sorted_k1 = {3, 4, 4, 4, 5, 6, 6};
  xt::xarray<float> sorted_k2 = {3.1, 2.5, 4.1, 6.5, 6.7, 0.1, 0.5};

  Value k1_v = test::makeValue(&ctx, k1, VIS_SECRET);
  Value k2_v = test::makeValue(&ctx, k2, VIS_SECRET);

  std::vector<spu::Value> rets = Sort(
      &ctx, {k1_v, k2_v}, 0, false,
      [&](absl::Span<const spu::Value> inputs) {
        auto pred_0 = hal::equal(&ctx, inputs[0], inputs[1]);
        auto pred_1 = hal::less(&ctx, inputs[0], inputs[1]);
        auto pred_2 = hal::less(&ctx, inputs[2], inputs[3]);

        return hal::select(&ctx, pred_0, pred_2, pred_1);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 2);

  auto sorted_k1_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));
  auto sorted_k2_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));

  EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
      << sorted_k1 << std::endl
      << sorted_k1_hat << std::endl;

  EXPECT_TRUE(xt::allclose(sorted_k2, sorted_k2_hat, 0.01, 0.001))
      << sorted_k2 << std::endl
      << sorted_k2_hat << std::endl;
}

TEST(SortTest, EmptyOperands) {
  SPUContext ctx = test::makeSPUContext();
  auto empty_x = Seal(&ctx, Constant(&ctx, 1, {0}));

  std::vector<spu::Value> rets = Sort(
      &ctx, {empty_x}, 0, false,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 1);
  EXPECT_EQ(rets[0].numel(), 0);
  EXPECT_EQ(rets[0].shape().size(), 1);
  EXPECT_EQ(rets[0].shape()[0], 0);
}

TEST(SortTest, LargeNumel) {
  SPUContext ctx = test::makeSPUContext();

  std::vector<std::size_t> numels = {63, 64, 65, 149, 170, 255, 256, 257, 500};

  for (auto numel : numels) {
    std::vector<int64_t> asc_arr(numel);
    std::vector<std::size_t> shape = {numel};
    std::iota(asc_arr.begin(), asc_arr.end(), 0);

    auto shuffled_arr = asc_arr;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(shuffled_arr.begin(), shuffled_arr.end(), rng);

    auto des_arr = asc_arr;
    std::reverse(des_arr.begin(), des_arr.end());

    auto x = xt::adapt(shuffled_arr, shape);
    auto asc_x = xt::adapt(asc_arr, shape);
    auto des_x = xt::adapt(des_arr, shape);

    Value x_v = test::makeValue(&ctx, x, VIS_SECRET);

    // ascending sort
    std::vector<spu::Value> rets = Sort(
        &ctx, {x_v}, 0, false,
        [&](absl::Span<const spu::Value> inputs) {
          return hal::less(&ctx, inputs[0], inputs[1]);
        },
        Visibility::VIS_SECRET);

    EXPECT_EQ(rets.size(), 1);

    auto asc_x_hat =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

    EXPECT_TRUE(xt::allclose(asc_x, asc_x_hat, 0.01, 0.001))
        << asc_x << std::endl
        << asc_x_hat << std::endl;

    // descending sort
    rets = Sort(
        &ctx, {x_v}, 0, false,
        [&](absl::Span<const spu::Value> inputs) {
          return hal::greater(&ctx, inputs[0], inputs[1]);
        },
        Visibility::VIS_SECRET);

    EXPECT_EQ(rets.size(), 1);

    auto des_x_hat =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

    EXPECT_TRUE(xt::allclose(des_x, des_x_hat, 0.01, 0.001))
        << des_x << std::endl
        << des_x_hat << std::endl;
  }
}

TEST(SimpleSortTest, MultiOperands) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> k1 = {7, 6, 5, 5, 4, 4, 4, 1, 3, 3};
  xt::xarray<float> k2 = {1, 2, 3, 6, 7, 6, 5, 2, 1, 2};

  xt::xarray<float> sorted_k1 = {1, 3, 3, 4, 4, 4, 5, 5, 6, 7};
  xt::xarray<float> sorted_k2 = {2, 1, 2, 5, 6, 7, 3, 6, 2, 1};

  Value k1_v = test::makeValue(&ctx, k1, VIS_SECRET);
  Value k2_v = test::makeValue(&ctx, k2, VIS_SECRET);

  std::vector<spu::Value> rets =
      SimpleSort(&ctx, {k1_v, k2_v}, 0, hal::SortDirection::Ascending, 2);

  EXPECT_EQ(rets.size(), 2);

  auto sorted_k1_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));
  auto sorted_k2_hat =
      hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));

  EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
      << sorted_k1 << std::endl
      << sorted_k1_hat << std::endl;

  EXPECT_TRUE(xt::allclose(sorted_k2, sorted_k2_hat, 0.01, 0.001))
      << sorted_k2 << std::endl
      << sorted_k2_hat << std::endl;
}

}  // namespace spu::kernel::hlo
