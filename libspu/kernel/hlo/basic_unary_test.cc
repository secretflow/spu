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

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

class UnaryTest
    : public ::testing::TestWithParam<std::tuple<FieldType, ProtocolKind>> {};

#define UNARY_EMPTY_TEST(NAME)                                       \
  TEST_P(UnaryTest, Empty_##NAME) {                                  \
    FieldType field = std::get<0>(GetParam());                       \
    ProtocolKind prot = std::get<1>(GetParam());                     \
    mpc::utils::simulate(                                            \
        3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {   \
          SPUContext sctx = test::makeSPUContext(prot, field, lctx); \
          auto empty_c = Constant(&sctx, 1.0F, {0});                 \
          auto s_empty = NAME(&sctx, empty_c);                       \
          EXPECT_EQ(s_empty.numel(), 0);                             \
          EXPECT_EQ(s_empty.shape().size(), 1);                      \
          EXPECT_EQ(s_empty.shape()[0], 0);                          \
        });                                                          \
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

INSTANTIATE_TEST_SUITE_P(
    UnaryTestInstances, UnaryTest,
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<UnaryTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

}  // namespace spu::kernel::hlo
