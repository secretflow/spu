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

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

class CastingTest
    : public ::testing::TestWithParam<std::tuple<FieldType, ProtocolKind>> {};

TEST_P(CastingTest, Empty) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto empty_c = Constant(&sctx, true, {0});

        // Seal
        auto s_empty = Seal(&sctx, empty_c);

        // Reveal
        auto p_empty = Reveal(&sctx, s_empty);

        // RevealTo
        auto v_empty = RevealTo(&sctx, s_empty, 0);

        EXPECT_EQ(p_empty.numel(), 0);
        EXPECT_EQ(p_empty.shape().size(), 1);
        EXPECT_EQ(p_empty.shape()[0], 0);

        EXPECT_EQ(v_empty.numel(), 0);
        EXPECT_EQ(v_empty.shape().size(), 1);
        EXPECT_EQ(v_empty.shape()[0], 0);
      });
}

INSTANTIATE_TEST_SUITE_P(
    CastingTestInstances, CastingTest,
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<CastingTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

}  // namespace spu::kernel::hlo
