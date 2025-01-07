// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/soprf.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

class SoPrfTest
    : public ::testing::TestWithParam<std::tuple<FieldType, ProtocolKind>> {};

INSTANTIATE_TEST_SUITE_P(
    SoPrfTestInstances, SoPrfTest,
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::SEMI2K)),
    [](const testing::TestParamInfo<SoPrfTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(SoPrfTest, EmptyWork) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);

        auto empty_x = Seal(&sctx, Constant(&sctx, 1, {0}));
        auto empty_ret = SoPrf(&sctx, empty_x);

        EXPECT_EQ(empty_ret.numel(), 0);
        EXPECT_EQ(empty_ret.shape().size(), 1);
        EXPECT_EQ(empty_ret.shape()[0], 0);
      });
}

TEST_P(SoPrfTest, Work) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);

        const Shape shape = {20, 17};
        xt::xarray<uint64_t> x = xt::random::randint<uint64_t>(shape, 0);
        xt::xarray<uint64_t> y = xt::random::randint<uint64_t>(shape, 0);

        auto x_share = Seal(&sctx, Constant(&sctx, x, shape));
        auto y_share = Seal(&sctx, Constant(&sctx, y, shape));

        auto ret_x = SoPrf(&sctx, x_share);
        auto ret_y = SoPrf(&sctx, y_share);
        EXPECT_EQ(ret_x.shape(), shape);
        EXPECT_EQ(ret_x.shape(), ret_y.shape());

        auto ret_x_pub = Reveal(&sctx, ret_x);
        auto ret_y_pub = Reveal(&sctx, ret_y);

        EXPECT_FALSE(mpc::ring_all_equal(ret_x_pub.data(), ret_y_pub.data()));
      });
}

class MultiKeySoPrfTest : public ::testing::TestWithParam<
                              std::tuple<FieldType, ProtocolKind, size_t>> {};

INSTANTIATE_TEST_SUITE_P(
    MultiKeySoPrfTestInstances, MultiKeySoPrfTest,
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::SEMI2K),
                     testing::Values(1, 2, 4)),  // num of keys
    [](const testing::TestParamInfo<MultiKeySoPrfTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

TEST_P(MultiKeySoPrfTest, Work) {
  FieldType field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());
  size_t num_keys = std::get<2>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);

        const Shape shape = {20, 17};
        std::vector<spu::Value> inputs;
        inputs.reserve(num_keys);
        for (size_t i = 0; i < num_keys; ++i) {
          xt::xarray<uint64_t> tmp = xt::random::randint<uint64_t>(shape, 0);
          auto v = Seal(&sctx, Constant(&sctx, tmp, shape));
          inputs.push_back(v);
        }

        auto ret = SoPrf(&sctx, absl::MakeSpan(inputs));

        EXPECT_EQ(ret.shape(), shape);
      });
}

}  // namespace spu::kernel::hlo
