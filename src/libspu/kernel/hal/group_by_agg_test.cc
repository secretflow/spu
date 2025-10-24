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

#include "libspu/kernel/hal/group_by_agg.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {

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

template <typename K, typename P>
std::pair<std::vector<K>, std::vector<P>> ComputePlaintextGroupBySum(
    const xt::xarray<K> &keys, const xt::xarray<P> &payloads) {
  SPU_ENFORCE(keys.size() == payloads.size(), "size mismatch");
  // Compute plaintext groupby sum for verification
  std::map<K, P> plaintext_groupby_sum;
  for (size_t i = 0; i < keys.size(); ++i) {
    plaintext_groupby_sum[keys[i]] += payloads[i];
  }

  // Extract expected results (sorted by key)
  std::vector<K> expected_keys;
  std::vector<P> expected_sums;
  for (const auto &[key, sum] : plaintext_groupby_sum) {
    expected_keys.push_back(key);
    expected_sums.push_back(sum);
  }

  return {expected_keys, expected_sums};
}
}  // namespace

class SingleKeyPrivateGroupBySumTest
    : public ::testing::TestWithParam<
          std::tuple<size_t, FieldType, ProtocolKind, VisType>> {};

INSTANTIATE_TEST_SUITE_P(
    PrivateGroupBySum3PCTestInstances, SingleKeyPrivateGroupBySumTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K, ProtocolKind::ABY3),
                     testing::ValuesIn(kVisTypes)),
    [](const testing::TestParamInfo<SingleKeyPrivateGroupBySumTest::ParamType>
           &p) {
      return fmt::format("{}x{}x{}xpayload_{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         get_vis_str(std::get<3>(p.param)));
    });

TEST_P(SingleKeyPrivateGroupBySumTest, Basic) {
  const size_t npc = std::get<0>(GetParam());
  const FieldType field = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());
  const VisType payload_vis = std::get<3>(GetParam());
  const VisType key_vis = VisType::VisPriv0;

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
        private_groupby_sum_1d(&ctx, {keys_v}, {payloads_v});
    // test::printProfileData(&ctx);

    // THEN
    EXPECT_EQ(rets.size(), 2);
    EXPECT_TRUE(rets[0].isPrivate());

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

TEST_P(SingleKeyPrivateGroupBySumTest, RandomTest) {
  const size_t npc = std::get<0>(GetParam());
  const FieldType field = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());
  const VisType payload_vis = std::get<3>(GetParam());
  const VisType key_vis = VisType::VisPriv0;

  // GIVEN
  //   uint64_t n = static_cast<uint64_t>(1) << 20;
  uint64_t n = static_cast<uint64_t>(1) << 4;
  xt::xarray<int32_t> keys = xt::random::randint<int32_t>({n}, 1, 100);
  xt::xarray<float> payloads = xt::random::rand<float>({n}, 0, 10.0);

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);

        // Extract expected results (sorted by key)
        std::vector<int32_t> expected_keys;
        std::vector<float> expected_sums;
        std::tie(expected_keys, expected_sums) =
            ComputePlaintextGroupBySum<int32_t, float>(keys, payloads);

        auto keys_v = makeValue(&ctx, keys, key_vis);
        auto payloads_v = makeValue(&ctx, payloads, payload_vis);

        // WHEN
        // setupTrace(&ctx, ctx.config());
        // test::CommunicationStats comm_stats;
        // comm_stats.reset(ctx.lctx());
        std::vector<spu::Value> rets =
            private_groupby_sum_1d(&ctx, {keys_v}, {payloads_v});
        // comm_stats.diff(ctx.lctx());
        // test::printProfileData(&ctx);
        // comm_stats.print_link_comm_stats(ctx.lctx());

        // THEN - Verify correctness
        EXPECT_EQ(rets.size(), 2);
        EXPECT_TRUE(rets[0].isPrivate());

        Value groupby_key = rets[0];
        Value groupby_payload = rets[1];
        if (!groupby_key.isPublic()) {
          groupby_key = hal::reveal(&ctx, groupby_key);
        }
        if (!groupby_payload.isPublic()) {
          groupby_payload = hal::reveal(&ctx, groupby_payload);
        }

        auto keys_hat = hal::dump_public_as<int32_t>(&ctx, groupby_key);
        auto sum_payloads_hat =
            hal::dump_public_as<float>(&ctx, groupby_payload);

        // Extract valid results (number of unique keys)
        auto valid_key_count = static_cast<int64_t>(expected_keys.size());
        auto keys_valid = xt::view(keys_hat, xt::range(0, valid_key_count));
        auto payloads_valid =
            xt::view(sum_payloads_hat, xt::range(0, valid_key_count));

        // Verify keys match
        EXPECT_TRUE(
            xt::allclose(xt::adapt(expected_keys), keys_valid, 0.001, 0.001))
            << "Keys mismatch!" << std::endl
            << "Expected: " << xt::adapt(expected_keys) << std::endl
            << "Got: " << keys_valid << std::endl;

        // Verify sums match (with tolerance for floating point)
        EXPECT_TRUE(
            xt::allclose(xt::adapt(expected_sums), payloads_valid, 0.01, 0.01))
            << "Sums mismatch!" << std::endl
            << "Expected: " << xt::adapt(expected_sums) << std::endl
            << "Got: " << payloads_valid << std::endl;
      });
}

class TwoKeyPrivateGroupBySumTest
    : public ::testing::TestWithParam<
          std::tuple<size_t, FieldType, ProtocolKind, VisType, VisType>> {};

INSTANTIATE_TEST_SUITE_P(
    PrivateGroupBySum3PCTestInstances, TwoKeyPrivateGroupBySumTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K, ProtocolKind::ABY3),
                     testing::Values(VisType::VisPub,
                                     VisType::VisPriv0),  // key
                     testing::ValuesIn(kVisTypes)),       // payload
    [](const testing::TestParamInfo<TwoKeyPrivateGroupBySumTest::ParamType>
           &p) {
      return fmt::format("{}x{}x{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param),
                         get_vis_str(std::get<3>(p.param)),
                         get_vis_str(std::get<4>(p.param)));
    });

TEST_P(TwoKeyPrivateGroupBySumTest, MultipleKeysAndPayloads) {
  const size_t npc = std::get<0>(GetParam());
  const FieldType field = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());

  const VisType key_vis = std::get<3>(GetParam());
  const VisType payload_vis = std::get<4>(GetParam());

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>
                                    &lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);

    // GIVEN
    xt::xarray<int32_t> key0 = {1, 2, 3, 1, 2, 1};
    xt::xarray<int32_t> key1 = {4, 5, 6, 4, 5, 3};

    xt::xarray<int32_t> payload0 = {10, 20, 30, 100, 200, 1000};
    xt::xarray<int32_t> payload1 = {40, 50, 60, 40, 50, 30};

    // (1,3), (1,4), (2,5), (3,6)
    int64_t valid_key_count = 4;

    xt::xarray<int32_t> expected_key0 = {1, 1, 2, 3};
    xt::xarray<int32_t> expected_key1 = {3, 4, 5, 6};
    xt::xarray<int32_t> expected_payload0 = {1000, 110, 220, 30};
    xt::xarray<int32_t> expected_payload1 = {30, 80, 100, 60};

    auto keys0_v = makeValue(&ctx, key0, key_vis);
    auto keys1_v = makeValue(&ctx, key1, key_vis);
    auto payload0_v = makeValue(&ctx, payload0, payload_vis);
    auto payload1_v = makeValue(&ctx, payload1, payload_vis);

    // WHEN
    // setupTrace(&ctx, ctx.config());
    std::vector<spu::Value> rets = private_groupby_sum_1d(
        &ctx, {keys0_v, keys1_v}, {payload0_v, payload1_v});
    // test::printProfileData(&ctx);

    // THEN
    EXPECT_EQ(rets.size(), 4);
    Value groupby_key0 = rets[0];
    Value groupby_key1 = rets[1];
    Value groupby_payload0 = rets[2];
    Value groupby_payload1 = rets[3];
    if (!groupby_key0.isPublic()) {
      groupby_key0 = hal::reveal(&ctx, groupby_key0);
    }
    if (!groupby_key1.isPublic()) {
      groupby_key1 = hal::reveal(&ctx, groupby_key1);
    }
    if (!groupby_payload0.isPublic()) {
      groupby_payload0 = hal::reveal(&ctx, groupby_payload0);
    }
    if (!groupby_payload1.isPublic()) {
      groupby_payload1 = hal::reveal(&ctx, groupby_payload1);
    }

    auto keys0_hat = hal::dump_public_as<int32_t>(&ctx, groupby_key0);
    auto keys1_hat = hal::dump_public_as<int32_t>(&ctx, groupby_key1);
    auto sum_payloads0_hat =
        hal::dump_public_as<int32_t>(&ctx, groupby_payload0);
    auto sum_payloads1_hat =
        hal::dump_public_as<int32_t>(&ctx, groupby_payload1);

    // only the first valid_key_count are valid
    auto keys0_valid = xt::view(keys0_hat, xt::range(0, valid_key_count));
    auto keys1_valid = xt::view(keys1_hat, xt::range(0, valid_key_count));
    auto payloads0_valid =
        xt::view(sum_payloads0_hat, xt::range(0, valid_key_count));
    auto payloads1_valid =
        xt::view(sum_payloads1_hat, xt::range(0, valid_key_count));

    EXPECT_TRUE(xt::allclose(expected_key0, keys0_valid, 0.001, 0.001))
        << expected_key0 << std::endl
        << keys0_valid << std::endl;
    EXPECT_TRUE(xt::allclose(expected_key1, keys1_valid, 0.001, 0.001))
        << expected_key1 << std::endl
        << keys1_valid << std::endl;
    EXPECT_TRUE(xt::allclose(expected_payload0, payloads0_valid, 0.001, 0.001))
        << expected_payload0 << std::endl
        << payloads0_valid << std::endl;
    EXPECT_TRUE(xt::allclose(expected_payload1, payloads1_valid, 0.001, 0.001))
        << expected_payload1 << std::endl
        << payloads1_valid << std::endl;
  });
}

}  // namespace spu::kernel::hal
