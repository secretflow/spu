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

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/value.h"
#include "libspu/kernel/test_util.h"

namespace spu::kernel::hlo {

TEST(ConstTest, Empty) {
  SPUContext sctx = test::makeSPUContext();

  auto empty_c = Constant(&sctx, true, {0});

  EXPECT_EQ(empty_c.numel(), 0);
  EXPECT_EQ(empty_c.shape().size(), 1);
  EXPECT_EQ(empty_c.shape()[0], 0);
}

TEST(ConstTest, Epsilon) {
  SPUContext sctx = test::makeSPUContext();

  auto eps = Epsilon(&sctx, DT_F32);

  auto v = hal::dump_public_as<float>(&sctx, eps);

  EXPECT_FLOAT_EQ(v[0], 1 / (std::pow(2, sctx.config().fxp_fraction_bits())));
}

}  // namespace spu::kernel::hlo
