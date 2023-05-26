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

#include "libspu/kernel/hlo/shuffle.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"
#include "xtensor/xsort.hpp"

#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hlo {

TEST(SortTest, Array) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> x = xt::random::rand<float>({10});
  std::vector<Value> x_v = {test::makeValue(&ctx, x, VIS_SECRET)};
  spu::Value ret1 = Shuffle(&ctx, x_v, 0)[0];
  auto ret1_hat = hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret1));

  spu::Value ret2 = Shuffle(&ctx, x_v, 0)[0];
  auto ret2_hat = hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret2));

  EXPECT_TRUE(xt::allclose(xt::sort(x), xt::sort(ret1_hat), 0.01, 0.001))
      << xt::sort(x) << std::endl
      << xt::sort(ret1_hat) << std::endl;

  EXPECT_TRUE(xt::allclose(xt::sort(x), xt::sort(ret2_hat), 0.01, 0.001))
      << xt::sort(x) << std::endl
      << xt::sort(ret2_hat) << std::endl;
  EXPECT_FALSE(xt::allclose(ret1_hat, ret2_hat, 0.01, 0.001));
}

TEST(SortTest, 2D) {
  SPUContext ctx = test::makeSPUContext();
  xt::xarray<float> x = xt::random::rand<float>({10, 15});
  std::vector<Value> x_v = {test::makeValue(&ctx, x, VIS_SECRET)};

  spu::Value ret1 = Shuffle(&ctx, x_v, 1)[0];
  auto ret1_hat = hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret1));

  spu::Value ret2 = Shuffle(&ctx, x_v, 1)[0];
  auto ret2_hat = hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret2));

  EXPECT_TRUE(xt::allclose(xt::sort(x, 1), xt::sort(ret1_hat, 1), 0.01, 0.001))
      << xt::sort(x, 1) << std::endl
      << xt::sort(ret1_hat, 1) << std::endl;

  EXPECT_TRUE(xt::allclose(xt::sort(x, 1), xt::sort(ret2_hat, 1), 0.01, 0.001))
      << xt::sort(x, 1) << std::endl
      << xt::sort(ret2_hat, 1) << std::endl;
  EXPECT_FALSE(xt::allclose(ret1_hat, ret2_hat, 0.01, 0.001));
}

}  // namespace spu::kernel::hlo
