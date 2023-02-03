
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

#include "libspu/kernel/hlo/dynamic_slice.h"

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/test_util.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hlo {

TEST(ConstTest, UpdateSliceScalar) {
  HalContext hctx = hal::test::makeRefHalContext();
  auto input = Constant(&hctx, static_cast<int64_t>(1), {5});
  auto update = Constant(&hctx, static_cast<int64_t>(2), {1});

  UpdateSliceInPlace(&hctx, input, update, {1});

  EXPECT_EQ(input.numel(), 5);

  auto p_ret = hal::test::dump_public_as<int64_t>(&hctx, input);
  xt::xarray<int64_t> expected{1, 2, 1, 1, 1};
  EXPECT_EQ(p_ret, expected);
}

TEST(ConstTest, UpdateSliceMultiValues) {
  HalContext hctx = hal::test::makeRefHalContext();
  auto input = Constant(&hctx, static_cast<int64_t>(1), {5});
  auto update = Constant(&hctx, static_cast<int64_t>(2), {2});

  UpdateSliceInPlace(&hctx, input, update, {3});

  EXPECT_EQ(input.numel(), 5);

  auto p_ret = hal::test::dump_public_as<int64_t>(&hctx, input);
  xt::xarray<int64_t> expected{1, 1, 1, 2, 2};
  EXPECT_EQ(p_ret, expected);
}

}  // namespace spu::kernel::hlo
