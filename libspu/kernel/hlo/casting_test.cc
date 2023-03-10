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

#include "libspu/kernel/hlo/casting.h"

#include "gtest/gtest.h"

#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/test_util.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hlo {

TEST(ConstTest, Empty) {
  HalContext hctx = hal::test::makeRefHalContext();

  auto empty_c = Constant(&hctx, true, {0});

  // Seal
  auto s_empty = Seal(&hctx, empty_c);

  // Reveal
  auto p_empty = Reveal(&hctx, s_empty);

  EXPECT_EQ(p_empty.numel(), 0);
  EXPECT_EQ(p_empty.shape().size(), 1);
  EXPECT_EQ(p_empty.shape()[0], 0);
}

}  // namespace spu::kernel::hlo
