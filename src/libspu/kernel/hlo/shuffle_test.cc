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

#include "libspu/kernel/hlo/shuffle.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"
#include "xtensor/xsort.hpp"

#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

class ShuffleTest : public ::testing::TestWithParam<
                        std::tuple<size_t, FieldType, ProtocolKind>> {};

TEST_P(ShuffleTest, SingleArray) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  xt::xarray<float> x = xt::random::rand<float>({10});

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);
        std::vector<Value> x_v = {test::makeValue(&ctx, x, VIS_SECRET)};

        auto ret = Shuffle(&ctx, x_v, 0)[0];
        auto ret_hat = hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret));

        EXPECT_TRUE(xt::allclose(xt::sort(x), xt::sort(ret_hat), 0.01, 0.001))
            << xt::sort(x) << std::endl
            << xt::sort(ret_hat) << std::endl;
      });
}

TEST_P(ShuffleTest, MultiOperands) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  xt::xarray<float> x = xt::random::rand<float>({10});
  xt::xarray<float> y = xt::random::rand<float>({10});

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        Value x_v = test::makeValue(&ctx, x, VIS_SECRET);
        Value y_v = test::makeValue(&ctx, y, VIS_SECRET);

        auto ret = Shuffle(&ctx, {x_v, y_v}, 0);
        auto ret_hat0 =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret[0]));
        auto ret_hat1 =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret[1]));

        EXPECT_TRUE(xt::allclose(xt::sort(x), xt::sort(ret_hat0), 0.01, 0.001))
            << xt::sort(x) << std::endl
            << xt::sort(ret_hat0) << std::endl;

        EXPECT_TRUE(xt::allclose(xt::sort(y), xt::sort(ret_hat1), 0.01, 0.001))
            << xt::sort(y) << std::endl
            << xt::sort(ret_hat1) << std::endl;
      });
}

TEST_P(ShuffleTest, MultiOperands2D) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  xt::xarray<float> x = xt::random::rand<float>({10, 15});
  xt::xarray<float> y = xt::random::rand<float>({10, 15});

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);
        Value x_v = test::makeValue(&ctx, x, VIS_SECRET);
        Value y_v = test::makeValue(&ctx, y, VIS_SECRET);

        auto ret = Shuffle(&ctx, {x_v, y_v}, 1);
        auto ret_hat0 =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret[0]));
        auto ret_hat1 =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret[1]));

        EXPECT_TRUE(
            xt::allclose(xt::sort(x, 1), xt::sort(ret_hat0, 1), 0.01, 0.001))
            << xt::sort(x, 1) << std::endl
            << xt::sort(ret_hat0, 1) << std::endl;

        EXPECT_TRUE(
            xt::allclose(xt::sort(y, 1), xt::sort(ret_hat1, 1), 0.01, 0.001))
            << xt::sort(y, 1) << std::endl
            << xt::sort(ret_hat1, 1) << std::endl;
      });
}

TEST_P(ShuffleTest, SpecialCases) {
  size_t npc = std::get<0>(GetParam());
  FieldType field = std::get<1>(GetParam());
  ProtocolKind prot = std::get<2>(GetParam());

  xt::xarray<float> x = xt::random::rand<float>({1});

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        RuntimeConfig cfg;
        cfg.protocol = prot;
        cfg.field = field;
        SPUContext ctx = test::makeSPUContext(cfg, lctx);

        // special case: single element
        std::vector<Value> x_v = {test::makeValue(&ctx, x, VIS_SECRET)};
        // special case: empty input
        std::vector<Value> y_v = {Seal(&ctx, Constant(&ctx, 1, {0}))};

        auto ret_x = Shuffle(&ctx, x_v, 0);
        auto ret_y = Shuffle(&ctx, y_v, 0);

        auto ret_x_hat =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, ret_x[0]));
        EXPECT_TRUE(xt::allclose(x, ret_x_hat, 0.01, 0.001))
            << x << std::endl
            << ret_x_hat << std::endl;

        EXPECT_EQ(ret_y[0].numel(), 0);
        EXPECT_EQ(ret_y[0].shape().size(), 1);
        EXPECT_EQ(ret_y[0].shape()[0], 0);
      });
}

INSTANTIATE_TEST_SUITE_P(
    Shuffle2PCTestInstances, ShuffleTest,
    testing::Combine(testing::Values(2),
                     testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::SEMI2K,
                                     ProtocolKind::REF2K)),
    [](const testing::TestParamInfo<ShuffleTest::ParamType> &p) {
      return fmt::format("{}pcx{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Shuffle3PCTestInstances, ShuffleTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::SEMI2K, ProtocolKind::ABY3,
                                     ProtocolKind::REF2K)),
    [](const testing::TestParamInfo<ShuffleTest::ParamType> &p) {
      return fmt::format("{}pcx{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param));
    });

}  // namespace spu::kernel::hlo
