// Copyright 2021 Ant Group Co., Ltd.
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

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "spu/device/io.h"
#include "spu/mpc/util/simulate.h"

namespace spu::device {

class IoClientTest
    : public ::testing::TestWithParam<
          std::tuple<size_t, ProtocolKind, FieldType, Visibility>> {};

TEST_P(IoClientTest, Float) {
  const size_t kWorldSize = std::get<0>(GetParam());
  const Visibility kVisibility = std::get<3>(GetParam());

  RuntimeConfig hconf;
  hconf.set_protocol(std::get<1>(GetParam()));
  hconf.set_field(std::get<2>(GetParam()));
  IoClient io(kWorldSize, hconf);

  xt::xarray<float> in_data({{1, -2, 3, 0}});

  auto shares = io.makeShares(in_data, kVisibility);
  EXPECT_EQ(shares.size(), kWorldSize);

  auto out = io.combineShares(shares);
  EXPECT_EQ(out.eltype().as<PtTy>()->pt_type(), PT_F32);

  auto out_data = xt_adapt<float>(out);
  EXPECT_EQ(in_data, out_data);
}

TEST_P(IoClientTest, Int) {
  const size_t kWorldSize = std::get<0>(GetParam());
  const Visibility kVisibility = std::get<3>(GetParam());

  RuntimeConfig hconf;
  hconf.set_protocol(std::get<1>(GetParam()));
  hconf.set_field(std::get<2>(GetParam()));
  IoClient io(kWorldSize, hconf);

  xt::xarray<int> in_data({{1, -2, 3, 0}});

  auto shares = io.makeShares(in_data, kVisibility);
  EXPECT_EQ(shares.size(), kWorldSize);

  auto out = io.combineShares(shares);
  EXPECT_EQ(out.eltype().as<PtTy>()->pt_type(), PT_I32);

  auto out_data = xt_adapt<int32_t>(out);
  EXPECT_EQ(in_data, out_data);
}

INSTANTIATE_TEST_SUITE_P(
    IoClientTestInstance, IoClientTest,
    testing::Combine(
        testing::Values(4, 3, 2),
        testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(Visibility::VIS_PUBLIC, Visibility::VIS_SECRET)),
    [](const testing::TestParamInfo<IoClientTest::ParamType> &p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<3>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    IoClientTestABY3Instance, IoClientTest,
    testing::Combine(
        testing::Values(3), testing::Values(ProtocolKind::ABY3),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(Visibility::VIS_PUBLIC, Visibility::VIS_SECRET)),
    [](const testing::TestParamInfo<IoClientTest::ParamType> &p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<3>(p.param));
    });

class ColocatedIoTest
    : public ::testing::TestWithParam<
          std::tuple<size_t, ProtocolKind, FieldType, Visibility>> {};

TEST_P(ColocatedIoTest, Works) {
  const size_t kWorldSize = std::get<0>(GetParam());
  const Visibility kVisibility = std::get<3>(GetParam());

  RuntimeConfig hconf;
  hconf.set_protocol(std::get<1>(GetParam()));
  hconf.set_field(std::get<2>(GetParam()));

  mpc::util::simulate(kWorldSize, [&](auto lctx) {
    HalContext hctx(hconf, lctx);
    ColocatedIo cio(&hctx);

    // WHEN
    if (lctx->Rank() == 0) {
      cio.hostSetVar("x", xt::xarray<int>{{1, -2, 3, 0}}, kVisibility);
    } else if (lctx->Rank() == 1) {
      cio.hostSetVar("y", xt::xarray<float>{{1, -2, 3, 0}}, kVisibility);
    }
    cio.sync();

    // THEN
    EXPECT_TRUE(cio.deviceHasVar("x"));
    auto x = cio.deviceGetVar("x");
    EXPECT_EQ(x.isPublic(), (kVisibility == VIS_PUBLIC)) << x;
    EXPECT_TRUE(x.isInt());
    EXPECT_TRUE(cio.deviceHasVar("y"));
    auto y = cio.deviceGetVar("y");
    EXPECT_EQ(x.isPublic(), (kVisibility == VIS_PUBLIC)) << y;
    EXPECT_TRUE(y.isFxp());
    EXPECT_FALSE(cio.deviceHasVar("z"));
  });
}

INSTANTIATE_TEST_SUITE_P(
    ColocatedIoTestInstance, ColocatedIoTest,
    testing::Combine(
        testing::Values(4, 3, 2),
        testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(Visibility::VIS_PUBLIC, Visibility::VIS_SECRET)),
    [](const testing::TestParamInfo<ColocatedIoTest::ParamType> &p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<3>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    ColocatedIoTestABY3Instance, ColocatedIoTest,
    testing::Combine(
        testing::Values(3), testing::Values(ProtocolKind::ABY3),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(Visibility::VIS_PUBLIC, Visibility::VIS_SECRET)),
    [](const testing::TestParamInfo<ColocatedIoTest::ParamType> &p) {
      return fmt::format("{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         std::get<3>(p.param));
    });

} // namespace spu::device
