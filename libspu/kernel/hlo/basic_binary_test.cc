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

#include "libspu/kernel/hlo/basic_binary.h"

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/test_util.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hlo {

#define BINARY_EMPTY_TEST(NAME)                       \
  TEST(ConstTest, Empty_##NAME) {                     \
    HalContext hctx = hal::test::makeRefHalContext(); \
    auto empty_c0 = Constant(&hctx, 1, {0});          \
    auto empty_c1 = Constant(&hctx, 1, {0});          \
    auto s_empty = NAME(&hctx, empty_c0, empty_c1);   \
    EXPECT_EQ(s_empty.numel(), 0);                    \
    EXPECT_EQ(s_empty.shape().size(), 1);             \
    EXPECT_EQ(s_empty.shape()[0], 0);                 \
  }

BINARY_EMPTY_TEST(Add)
BINARY_EMPTY_TEST(Equal);
BINARY_EMPTY_TEST(NotEqual)
BINARY_EMPTY_TEST(LessEqual)
BINARY_EMPTY_TEST(GreaterEqual)
BINARY_EMPTY_TEST(Sub)
BINARY_EMPTY_TEST(Less)
BINARY_EMPTY_TEST(Greater)
BINARY_EMPTY_TEST(Mul)
BINARY_EMPTY_TEST(Power)
BINARY_EMPTY_TEST(Max)
BINARY_EMPTY_TEST(Min)
BINARY_EMPTY_TEST(And)
BINARY_EMPTY_TEST(Or)
BINARY_EMPTY_TEST(Xor)
BINARY_EMPTY_TEST(Div)
BINARY_EMPTY_TEST(Remainder)

TEST(ConstTest, Empty_Dot) {
  HalContext hctx = hal::test::makeRefHalContext();
  auto empty_c0 = Constant(&hctx, 1, {1, 0});
  auto empty_c1 = Constant(&hctx, 1, {0, 1});

  // M = 1, N = 1, K = 0, result should be 1x1
  auto ret = Dot(&hctx, empty_c0, empty_c1);

  EXPECT_EQ(ret.numel(), 1);
  EXPECT_EQ(ret.shape().size(), 2);
  EXPECT_EQ(ret.shape()[0], 1);
  EXPECT_EQ(ret.shape()[1], 1);

  auto pt_ret = hal::test::dump_public_as<int64_t>(&hctx, Reveal(&hctx, ret));
  EXPECT_EQ(pt_ret[0], 0);

  // M = 0, N = 0, K = 1, result should be 0x0
  ret = Dot(&hctx, empty_c1, empty_c0);

  EXPECT_EQ(ret.numel(), 0);
  EXPECT_EQ(ret.shape().size(), 2);
  EXPECT_EQ(ret.shape()[0], 0);
  EXPECT_EQ(ret.shape()[1], 0);
}

TEST(ConstTest, MixedIntMul) {
  HalContext hctx = hal::test::makeRefHalContext();
  auto c0 = Constant(&hctx, static_cast<int32_t>(1), {1, 1});
  auto c1 = Constant(&hctx, static_cast<int16_t>(1), {1, 1});

  // M = 1, N = 1, K = 0, result should be 1x1
  auto ret = Mul(&hctx, c0, c1);

  EXPECT_EQ(ret.dtype(), DataType::DT_I32);
}

}  // namespace spu::kernel::hlo
