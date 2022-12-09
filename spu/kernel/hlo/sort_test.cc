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

#include "spu/kernel/hlo/sort.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "spu/kernel/hal/constants.h"
#include "spu/kernel/hal/polymorphic.h"
#include "spu/kernel/hal/test_util.h"

namespace spu::kernel::hlo {

TEST(SortTest, Simple) {
  HalContext ctx = hal::test::makeRefHalContext();
  xt::xarray<float> x{{0.05, 0.5, 0.24},
                      {
                          5,
                          50,
                          2,
                      }};
  xt::xarray<float> sorted_x{{0.05, 0.24, 0.5}, {2, 5, 50}};

  Value x_v = hal::const_secret(&ctx, x);

  std::vector<spu::Value> rets = Sort(
      &ctx, {x_v}, 1, true,
      [&](absl::Span<const spu::Value> inputs) {
        return hal::less(&ctx, inputs[0], inputs[1]);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 1);

  auto sorted_x_hat =
      hal::test::dump_public_as<float>(&ctx, hal::_s2p(&ctx, rets[0]).asFxp());

  EXPECT_TRUE(xt::allclose(sorted_x, sorted_x_hat, 0.01, 0.001))
      << sorted_x << std::endl
      << sorted_x_hat << std::endl;
}

TEST(SortTest, MultiOperands) {
  HalContext ctx = hal::test::makeRefHalContext();
  xt::xarray<float> k1{6, 6, 3, 4, 4, 5, 4};
  xt::xarray<float> k2{0.5, 0.1, 3.1, 6.5, 4.1, 6.7, 2.5};

  xt::xarray<float> sorted_k1{3, 4, 4, 4, 5, 6, 6};
  xt::xarray<float> sorted_k2{3.1, 2.5, 4.1, 6.5, 6.7, 0.1, 0.5};

  Value k1_v = hal::const_secret(&ctx, k1);
  Value k2_v = hal::const_secret(&ctx, k2);

  std::vector<spu::Value> rets = Sort(
      &ctx, {k1_v, k2_v}, 0, true,
      [&](absl::Span<const spu::Value> inputs) {
        auto pred_0 = hal::equal(&ctx, inputs[0], inputs[1]);
        auto pred_1 = hal::less(&ctx, inputs[0], inputs[1]);
        auto pred_2 = hal::less(&ctx, inputs[2], inputs[3]);

        return hal::select(&ctx, pred_0, pred_2, pred_1);
      },
      Visibility::VIS_SECRET);

  EXPECT_EQ(rets.size(), 2);

  auto sorted_k1_hat =
      hal::test::dump_public_as<float>(&ctx, hal::_s2p(&ctx, rets[0]).asFxp());
  auto sorted_k2_hat =
      hal::test::dump_public_as<float>(&ctx, hal::_s2p(&ctx, rets[1]).asFxp());

  EXPECT_TRUE(xt::allclose(sorted_k1, sorted_k1_hat, 0.01, 0.001))
      << sorted_k1 << std::endl
      << sorted_k1_hat << std::endl;

  EXPECT_TRUE(xt::allclose(sorted_k2, sorted_k2_hat, 0.01, 0.001))
      << sorted_k2 << std::endl
      << sorted_k2_hat << std::endl;
}

}  // namespace spu::kernel::hlo
