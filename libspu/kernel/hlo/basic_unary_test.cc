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

#include "libspu/kernel/hlo/basic_unary.h"

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/test_util.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hlo {

#define UNARY_EMPTY_TEST(NAME)                        \
  TEST(ConstTest, Empty_##NAME) {                     \
    HalContext hctx = hal::test::makeRefHalContext(); \
    auto empty_c = Constant(&hctx, 1.0F, {0});        \
    auto s_empty = NAME(&hctx, empty_c);              \
    EXPECT_EQ(s_empty.numel(), 0);                    \
    EXPECT_EQ(s_empty.shape().size(), 1);             \
    EXPECT_EQ(s_empty.shape()[0], 0);                 \
  }

UNARY_EMPTY_TEST(Reciprocal)
UNARY_EMPTY_TEST(Neg)
UNARY_EMPTY_TEST(Exp)
UNARY_EMPTY_TEST(Expm1)
UNARY_EMPTY_TEST(Log)
UNARY_EMPTY_TEST(Log1p)
UNARY_EMPTY_TEST(Floor)
UNARY_EMPTY_TEST(Ceil)
UNARY_EMPTY_TEST(Abs)
UNARY_EMPTY_TEST(Logistic)
UNARY_EMPTY_TEST(Tanh)
UNARY_EMPTY_TEST(Not)
UNARY_EMPTY_TEST(Rsqrt)
UNARY_EMPTY_TEST(Sqrt)
UNARY_EMPTY_TEST(Sign)
UNARY_EMPTY_TEST(Round_AFZ)

}  // namespace spu::kernel::hlo
