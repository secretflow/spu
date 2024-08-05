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

#include "libspu/kernel/hlo/basic_ternary.h"

#include "gtest/gtest.h"

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

class TernaryTest
    : public ::testing::TestWithParam<std::tuple<FieldType, ProtocolKind>> {};

TEST_P(TernaryTest, SelectEmpty) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto empty_p = Seal(&sctx, Constant(&sctx, true, {0}));
        auto empty_true = Seal(&sctx, Constant(&sctx, 1, {0}));
        auto empty_false = Seal(&sctx, Constant(&sctx, 2, {0}));

        auto ret = Select(&sctx, empty_p, empty_true, empty_false);

        EXPECT_EQ(ret.numel(), 0);
        EXPECT_EQ(ret.shape().size(), 1);
        EXPECT_EQ(ret.shape()[0], 0);
      });
}

TEST_P(TernaryTest, ClampEmpty) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto empty_in = Seal(&sctx, Constant(&sctx, 0, {0}));
        auto empty_min = Seal(&sctx, Constant(&sctx, 1, {0}));
        auto empty_max = Seal(&sctx, Constant(&sctx, 2, {0}));

        auto ret = Clamp(&sctx, empty_in, empty_min, empty_max);

        EXPECT_EQ(ret.numel(), 0);
        EXPECT_EQ(ret.shape().size(), 1);
        EXPECT_EQ(ret.shape()[0], 0);
      });
}

INSTANTIATE_TEST_SUITE_P(
    TernaryTestInstances, TernaryTest,
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<TernaryTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

}  // namespace spu::kernel::hlo
