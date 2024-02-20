// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/kernel/hal/utils.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hal {
namespace {

TEST(UtilsTest, associative_scan) {
  SPUContext ctx = test::makeSPUContext();

  {
    const xt::xarray<int32_t> x = {1, 1, 1, 1, 1};
    const xt::xarray<int32_t> prefix_sum = {1, 2, 3, 4, 5};
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value b = associative_scan(hal::add, &ctx, a);
    auto ret = dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, b));
    EXPECT_TRUE(prefix_sum == ret) << x << std::endl
                                   << prefix_sum << std::endl
                                   << ret;
  }

  {
    const xt::xarray<int32_t> x = {1, 2, 3, 4, 5};
    const xt::xarray<int32_t> prefix_prod = {1, 2, 6, 24, 120};
    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value b = associative_scan(hal::mul, &ctx, a);
    auto ret = dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, b));
    EXPECT_TRUE(prefix_prod == ret) << x << std::endl
                                    << prefix_prod << std::endl
                                    << ret;
  }

  {
    const xt::xarray<bool> x = {
        true, true, true, false, true, false,
    };
    const xt::xarray<bool> prefix_and = {
        true, true, true, false, false, false,
    };

    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value b = associative_scan(hal::bitwise_and, &ctx, a);
    auto ret = dump_public_as<bool>(&ctx, hal::reveal(&ctx, b));
    EXPECT_TRUE(prefix_and == ret) << x << std::endl
                                   << prefix_and << std::endl
                                   << ret;
  }

  {
    const xt::xarray<bool> x = {
        true, true, true, false, true, false,
    };
    const xt::xarray<bool> prefix_or = {
        true, true, true, true, true, true,
    };

    Value a = test::makeValue(&ctx, x, VIS_SECRET);
    Value b = associative_scan(hal::bitwise_or, &ctx, a);
    auto ret = dump_public_as<bool>(&ctx, hal::reveal(&ctx, b));
    EXPECT_TRUE(prefix_or == ret) << x << std::endl
                                  << prefix_or << std::endl
                                  << ret;
  }
}

}  // namespace
}  // namespace spu::kernel::hal
