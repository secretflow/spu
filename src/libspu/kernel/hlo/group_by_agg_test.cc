// Copyright 2025 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/group_by_agg.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

namespace {
SPUContext makeSPUContextWithProfile(
    ProtocolKind prot_kind, FieldType field,
    const std::shared_ptr<yacl::link::Context> &lctx) {
  RuntimeConfig cfg;
  cfg.protocol = prot_kind;
  cfg.field = field;
  cfg.enable_action_trace = false;

  if (lctx->Rank() == 0) {
    cfg.enable_hal_profile = true;
    cfg.enable_pphlo_profile = true;
  }
  return test::makeSPUContext(cfg, lctx);
}

enum class VisType {
  VisPriv0 = 0,  // private, own by party 0
  VisPriv1 = 1,  // private, own by party 1
  VisPub = 2,
  VisSec = 3,
};

const std::vector<VisType> kVisTypes = {VisType::VisPub, VisType::VisSec,
                                        VisType::VisPriv0, VisType::VisPriv1};

inline std::string get_vis_str(VisType type) {
  switch (type) {
    case VisType::VisPub:
      return "VisPub";
    case VisType::VisSec:
      return "VisSec";
    case VisType::VisPriv0:
      return "VisPriv0";
    case VisType::VisPriv1:
      return "VisPriv1";
    default:
      return "Unknown";
  }
}

Value makeValue(SPUContext *ctx, PtBufferView init, VisType vis_type,
                DataType dtype = DT_INVALID, const Shape &shape = {}) {
  if (vis_type == VisType::VisPub) {
    return test::makeValue(ctx, init, VIS_PUBLIC, dtype, shape);
  } else if (vis_type == VisType::VisSec) {
    return test::makeValue(ctx, init, VIS_SECRET, dtype, shape);
  } else if (vis_type == VisType::VisPriv0) {
    return test::makeValue(ctx, init, VIS_PRIVATE, dtype, shape, 0);
  } else if (vis_type == VisType::VisPriv1) {
    return test::makeValue(ctx, init, VIS_PRIVATE, dtype, shape, 1);
  }
  SPU_THROW("Unknown vis type");
}

}  // namespace

class SingleKeyGroupBySumTest
    : public ::testing::TestWithParam<
          std::tuple<size_t, FieldType, ProtocolKind, VisType, VisType>> {};

INSTANTIATE_TEST_SUITE_P(
    PrivateGroupBySum3PCTestInstances, SingleKeyGroupBySumTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K, ProtocolKind::ABY3),
                     testing::ValuesIn(kVisTypes),
                     testing::ValuesIn(kVisTypes)),
    [](const testing::TestParamInfo<SingleKeyGroupBySumTest::ParamType> &p) {
      return fmt::format("{}x{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         get_vis_str(std::get<3>(p.param)),
                         get_vis_str(std::get<4>(p.param)));
    });

TEST_P(SingleKeyGroupBySumTest, Basic) {
  const size_t npc = std::get<0>(GetParam());
  const FieldType field = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());
  const VisType key_vis = std::get<3>(GetParam());
  const VisType payload_vis = std::get<4>(GetParam());

  const AggFunc agg_func = AggFunc::Sum;

  // Skip test if key is secret now.
  if (key_vis == VisType::VisSec) {
    return;
  }

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>
                                    &lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);

    // GIVEN
    xt::xarray<int32_t> keys = {1, 2, 3, 1, 2, 1};
    xt::xarray<int32_t> payloads = {10, 20, 30, 100, 200, 1000};
    // 1,2,3
    int64_t valid_key_count = 3;

    xt::xarray<int32_t> expected_keys = {1, 2, 3};
    xt::xarray<int32_t> expected_payloads = {1110, 220, 30};

    auto keys_v = makeValue(&ctx, keys, key_vis);
    auto payloads_v = makeValue(&ctx, payloads, payload_vis);

    // WHEN
    // setupTrace(&ctx, ctx.config());
    std::vector<spu::Value> rets =
        GroupByAgg(&ctx, {keys_v}, {payloads_v}, agg_func, /*valid_bits*/ {});
    // test::printProfileData(&ctx);

    // THEN
    EXPECT_EQ(rets.size(), 2);

    Value groupby_key = rets[0];
    Value groupby_payload = rets[1];
    if (!groupby_key.isPublic()) {
      groupby_key = hal::reveal(&ctx, groupby_key);
    }
    if (!groupby_payload.isPublic()) {
      groupby_payload = hal::reveal(&ctx, groupby_payload);
    }

    auto keys_hat = hal::dump_public_as<int32_t>(&ctx, groupby_key);
    auto sum_payloads_hat = hal::dump_public_as<int32_t>(&ctx, groupby_payload);

    // only the first valid_key_count are valid
    auto keys_valid = xt::view(keys_hat, xt::range(0, valid_key_count));
    auto payloads_valid =
        xt::view(sum_payloads_hat, xt::range(0, valid_key_count));

    EXPECT_TRUE(xt::allclose(expected_keys, keys_valid, 0.001, 0.001))
        << expected_keys << std::endl
        << keys_valid << std::endl;
    EXPECT_TRUE(xt::allclose(expected_payloads, payloads_valid, 0.001, 0.001))
        << expected_payloads << std::endl
        << payloads_valid << std::endl;
  });
}

TEST_P(SingleKeyGroupBySumTest, Empty) {
  const size_t npc = std::get<0>(GetParam());
  const FieldType field = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());
  const VisType key_vis = std::get<3>(GetParam());
  const VisType payload_vis = std::get<4>(GetParam());

  const AggFunc agg_func = AggFunc::Sum;

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);

        auto keys_v = makeValue(&ctx, 1, key_vis, DT_I32, {0});
        auto payloads_v = makeValue(&ctx, 1, payload_vis, DT_I32, {0});

        // WHEN
        std::vector<spu::Value> rets = GroupByAgg(&ctx, {keys_v}, {payloads_v},
                                                  agg_func, /*valid_bits*/ {});

        // THEN
        EXPECT_EQ(rets.size(), 2);
        EXPECT_EQ(rets[0].numel(), 0);
        EXPECT_EQ(rets[1].numel(), 0);
        EXPECT_EQ(rets[0].shape().size(), 1);
        EXPECT_EQ(rets[1].shape().size(), 1);
        EXPECT_EQ(rets[0].shape()[0], 0);
        EXPECT_EQ(rets[1].shape()[0], 0);
      });
}

}  // namespace spu::kernel::hlo
