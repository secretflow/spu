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

#include "spu/kernel/hlo/geometrical.h"

#include "gtest/gtest.h"

#include "spu/core/ndarray_ref.h"
#include "spu/kernel/context.h"
#include "spu/kernel/hal/test_util.h"
#include "spu/kernel/hlo/casting.h"
#include "spu/kernel/hlo/const.h"
#include "spu/kernel/value.h"

namespace spu::kernel::hlo {

TEST(ConstTest, EmptySlice) {
  HalContext hctx = hal::test::makeRefHalContext();
  auto in = Iota<int64_t>(&hctx, 10, VIS_PUBLIC);

  auto ret = Slice(&hctx, in, {1}, {1}, {1});

  EXPECT_EQ(ret.numel(), 0);
  EXPECT_EQ(ret.shape().size(), 1);
  EXPECT_EQ(ret.shape()[0], 0);

  ret = Slice(&hctx, in, {2}, {1}, {1});
  EXPECT_EQ(ret.numel(), 0);
  EXPECT_EQ(ret.shape().size(), 1);
  EXPECT_EQ(ret.shape()[0], 0);
}

}  // namespace spu::kernel::hlo
