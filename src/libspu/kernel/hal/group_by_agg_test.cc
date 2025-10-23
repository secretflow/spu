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

Value makePrivateValue(SPUContext *ctx, PtBufferView init, int64_t owner = -1,
                       Visibility vtype = VIS_PRIVATE,
                       DataType dtype = DT_INVALID, const Shape &shape = {}) {
  return test::makeValue(ctx, init, vtype, dtype, shape, owner);
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

class PrivateGroupBySumTest : public ::testing::TestWithParam<
                                  std::tuple<size_t, FieldType, ProtocolKind>> {
};

INSTANTIATE_TEST_SUITE_P(
    PrivateGroupBySum2PCTestInstances, PrivateGroupBySumTest,
    testing::Combine(testing::Values(2),
                     testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K)),
    [](const testing::TestParamInfo<PrivateGroupBySumTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    PrivateGroupBySum3PCTestInstances, PrivateGroupBySumTest,
    testing::Combine(testing::Values(3),
                     testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(ProtocolKind::SEMI2K, ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<PrivateGroupBySumTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

TEST_P(PrivateGroupBySumTest, Basic) {
  const size_t npc = std::get<0>(GetParam());
  const FieldType field = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);

        // GIVEN
        xt::xarray<int32_t> keys = {1, 2, 3, 1, 2, 1};
        xt::xarray<int32_t> payloads = {10, 20, 30, 100, 200, 1000};
        // 1,2,3
        int64_t valid_key_count = 3;

        xt::xarray<int32_t> expected_keys = {1, 2, 3};
        xt::xarray<int32_t> expected_payloads = {1110, 220, 30};

        auto keys_v = makePrivateValue(&ctx, keys, 0);
        auto payloads_v = makePrivateValue(&ctx, payloads, 1);

        // WHEN
        // setupTrace(&ctx, ctx.config());
        std::vector<spu::Value> rets =
            private_groupby_sum_1d(&ctx, {keys_v}, {payloads_v});
        // test::printProfileData(&ctx);

        // THEN
        EXPECT_EQ(rets.size(), 2);
        EXPECT_TRUE(rets[0].isPrivate());
        EXPECT_TRUE(rets[1].isSecret());

        auto keys_hat =
            hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[0]));
        auto sum_payloads_hat =
            hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[1]));

        // only the first valid_key_count are valid
        auto keys_valid = xt::view(keys_hat, xt::range(0, valid_key_count));
        auto payloads_valid =
            xt::view(sum_payloads_hat, xt::range(0, valid_key_count));

        EXPECT_TRUE(xt::allclose(expected_keys, keys_valid, 0.001, 0.001))
            << expected_keys << std::endl
            << keys_valid << std::endl;
        EXPECT_TRUE(
            xt::allclose(expected_payloads, payloads_valid, 0.001, 0.001))
            << expected_payloads << std::endl
            << payloads_valid << std::endl;
      });
}

TEST_P(PrivateGroupBySumTest, MultipleKeysAndPayloads) {
  const size_t npc = std::get<0>(GetParam());
  const FieldType field = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());

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

    auto keys0_v = makePrivateValue(&ctx, key0, 0);
    auto keys1_v = makePrivateValue(&ctx, key1, 0);
    auto payload0_v = makePrivateValue(&ctx, payload0, 1);
    auto payload1_v = makePrivateValue(&ctx, payload1, 1);

    // WHEN
    // setupTrace(&ctx, ctx.config());
    std::vector<spu::Value> rets = private_groupby_sum_1d(
        &ctx, {keys0_v, keys1_v}, {payload0_v, payload1_v});
    // test::printProfileData(&ctx);

    // THEN
    EXPECT_EQ(rets.size(), 4);
    EXPECT_TRUE(rets[0].isPrivate());
    EXPECT_TRUE(rets[1].isPrivate());
    EXPECT_TRUE(rets[2].isSecret());
    EXPECT_TRUE(rets[3].isSecret());

    auto keys0_hat =
        hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[0]));
    auto keys1_hat =
        hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[1]));
    auto sum_payloads0_hat =
        hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[2]));
    auto sum_payloads1_hat =
        hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[3]));

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

TEST_P(PrivateGroupBySumTest, RandomTest) {
  const size_t npc = std::get<0>(GetParam());
  const FieldType field = std::get<1>(GetParam());
  const ProtocolKind protocol = std::get<2>(GetParam());

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

        auto keys_v = makePrivateValue(&ctx, keys, 0);
        auto payloads_v = makePrivateValue(&ctx, payloads, 1);

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
        EXPECT_TRUE(rets[1].isSecret());

        auto keys_hat =
            hal::dump_public_as<int32_t>(&ctx, hal::reveal(&ctx, rets[0]));
        auto sum_payloads_hat =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[1]));

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

}  // namespace spu::kernel::hal
