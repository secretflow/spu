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

#include "libspu/kernel/hlo/geometrical.h"

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/test_utils.h"
#include "libspu/kernel/value.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

class GeoTest
    : public ::testing::TestWithParam<std::tuple<FieldType, ProtocolKind>> {};

TEST_P(GeoTest, EmptySlice) {
  auto cfg =
      test::makeRefConfig(std::get<0>(GetParam()), std::get<1>(GetParam()));

  mpc::utils::simulate(3,
                       [&](const std::shared_ptr<yacl::link::Context> &lctx) {
                         HalContext hctx(cfg, lctx);
                         auto in = Iota(&hctx, DT_I64, 10);

                         auto ret = Slice(&hctx, in, {1}, {1}, {1});

                         EXPECT_EQ(ret.numel(), 0);
                         EXPECT_EQ(ret.shape().size(), 1);
                         EXPECT_EQ(ret.shape()[0], 0);

                         ret = Slice(&hctx, in, {2}, {1}, {1});
                         EXPECT_EQ(ret.numel(), 0);
                         EXPECT_EQ(ret.shape().size(), 1);
                         EXPECT_EQ(ret.shape()[0], 0);
                       });
}

TEST_P(GeoTest, MixedStoragePad) {
  auto cfg =
      test::makeRefConfig(std::get<0>(GetParam()), std::get<1>(GetParam()));

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HalContext hctx(cfg, lctx);
        auto in = Iota(&hctx, DT_I64, 10);
        in = Seal(&hctx, in);
        auto pv = Constant(&hctx, static_cast<int64_t>(0), {1});
        pv = Xor(&hctx, pv, pv);  // Force a bshr

        auto ret = Pad(&hctx, in, pv, {1}, {0}, {0});

        EXPECT_EQ(ret.numel(), 11);
        EXPECT_EQ(ret.vtype(), VIS_SECRET);
        EXPECT_EQ(ret.dtype(), DT_I64);
      });
}

TEST_P(GeoTest, Pad) {
  auto cfg =
      test::makeRefConfig(std::get<0>(GetParam()), std::get<1>(GetParam()));

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HalContext hctx(cfg, lctx);
        auto in = Seal(
            &hctx, Cast(&hctx, Constant(&hctx, std::vector<uint8_t>(2, 0), {2}),
                        VIS_PUBLIC, DT_I1));
        auto pv = Seal(
            &hctx, Cast(&hctx, Constant(&hctx, static_cast<uint8_t>(0), {1}),
                        VIS_PUBLIC, DT_I1));

        auto ret = Pad(&hctx, in, pv, {0}, {1}, {0});

        EXPECT_EQ(ret.numel(), 3);
        EXPECT_EQ(ret.vtype(), VIS_SECRET);
        EXPECT_EQ(ret.dtype(), DT_I1);
      });
}

INSTANTIATE_TEST_SUITE_P(
    GeoTestInstances, GeoTest,
    testing::Combine(testing::Values(FieldType::FM64, FieldType::FM128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<GeoTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

}  // namespace spu::kernel::hlo
