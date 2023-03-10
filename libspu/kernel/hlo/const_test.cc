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

#include "libspu/kernel/hlo/const.h"

#include "gtest/gtest.h"
#include "spdlog/fmt/bin_to_hex.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/test_util.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hlo {

TEST(ConstTest, Empty) {
  HalContext hctx = hal::test::makeRefHalContext();

  auto empty_c = Constant(&hctx, true, {0});

  EXPECT_EQ(empty_c.numel(), 0);
  EXPECT_EQ(empty_c.shape().size(), 1);
  EXPECT_EQ(empty_c.shape()[0], 0);
}

TEST(ConstTest, Epsilon) {
  HalContext hctx = hal::test::makeRefHalContext();

  auto eps = Epsilon(&hctx);

  auto v = hal::dump_public_as<float>(&hctx, eps);

  EXPECT_FLOAT_EQ(v[0], 1 / (std::pow(2, getDefaultFxpBits(hctx.rt_config()))));
}

}  // namespace spu::kernel::hlo
