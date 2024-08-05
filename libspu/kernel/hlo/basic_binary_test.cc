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

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

class BinaryTest
    : public ::testing::TestWithParam<std::tuple<FieldType, ProtocolKind>> {};

#define BINARY_EMPTY_TEST(NAME)                                      \
  TEST_P(BinaryTest, Empty_##NAME) {                                 \
    FieldType field = std::get<0>(GetParam());                       \
    ProtocolKind prot = std::get<1>(GetParam());                     \
    mpc::utils::simulate(                                            \
        3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {   \
          SPUContext sctx = test::makeSPUContext(prot, field, lctx); \
          auto empty_c0 = Seal(&sctx, Constant(&sctx, 1, {0}));      \
          auto empty_c1 = Seal(&sctx, Constant(&sctx, 1, {0}));      \
          auto s_empty = NAME(&sctx, empty_c0, empty_c1);            \
          EXPECT_EQ(s_empty.numel(), 0);                             \
          EXPECT_EQ(s_empty.shape().size(), 1);                      \
          EXPECT_EQ(s_empty.shape()[0], 0);                          \
        });                                                          \
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
BINARY_EMPTY_TEST(Max)
BINARY_EMPTY_TEST(Min)
BINARY_EMPTY_TEST(And)
BINARY_EMPTY_TEST(Or)
BINARY_EMPTY_TEST(Xor)
BINARY_EMPTY_TEST(Div)
BINARY_EMPTY_TEST(Remainder)

TEST_P(BinaryTest, Empty_Power) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());
  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto empty_c0 = Seal(&sctx, Constant(&sctx, 1.0, {0}));
        auto empty_c1 = Seal(&sctx, Constant(&sctx, 1.0, {0}));
        auto s_empty = Power(&sctx, empty_c0, empty_c1);
        EXPECT_EQ(s_empty.numel(), 0);
        EXPECT_EQ(s_empty.shape().size(), 1);
        EXPECT_EQ(s_empty.shape()[0], 0);
      });
}

TEST_P(BinaryTest, Empty_Dot) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto empty_c0 = Seal(&sctx, Constant(&sctx, 1, {1, 0}));
        auto empty_c1 = Seal(&sctx, Constant(&sctx, 1, {0, 1}));

        // M = 1, N = 1, K = 0, result should be 1x1
        auto ret = Dot(&sctx, empty_c0, empty_c1);

        EXPECT_EQ(ret.numel(), 1);
        EXPECT_EQ(ret.shape().size(), 2);
        EXPECT_EQ(ret.shape()[0], 1);
        EXPECT_EQ(ret.shape()[1], 1);

        // M = 0, N = 0, K = 1, result should be 0x0
        ret = Dot(&sctx, empty_c1, empty_c0);

        EXPECT_EQ(ret.numel(), 0);
        EXPECT_EQ(ret.shape().size(), 2);
        EXPECT_EQ(ret.shape()[0], 0);
        EXPECT_EQ(ret.shape()[1], 0);
      });
}

TEST_P(BinaryTest, MixedIntMul) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto c0 = Constant(&sctx, static_cast<int32_t>(1), {1, 1});
        auto c1 = Constant(&sctx, static_cast<int16_t>(1), {1, 1});

        // M = 1, N = 1, K = 0, result should be 1x1
        auto ret = Mul(&sctx, c0, c1);

        EXPECT_EQ(ret.dtype(), DataType::DT_I32);
      });
}

INSTANTIATE_TEST_SUITE_P(
    BinaryTestInstances, BinaryTest,
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<BinaryTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

}  // namespace spu::kernel::hlo
