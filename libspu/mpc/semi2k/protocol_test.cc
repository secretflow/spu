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

#include "libspu/mpc/semi2k/protocol.h"

#include <mutex>

#include "gtest/gtest.h"
#include "yacl/crypto/key_utils.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/ab_api_test.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/api_test.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/ttp_server/beaver_server.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(size_t field) {
  RuntimeConfig conf;
  ProtocolConfig proto_conf;
  Semi2kConfig semi2k_conf;
  semi2k_conf.set_beaver_type(BeaverType::TrustedFirstParty);
  proto_conf.set_kind(ProtocolKind::SEMI2K);
  proto_conf.set_field(field);
  *proto_conf.mutable_semi2k_config() = std::move(semi2k_conf);
  *conf.mutable_protocol() = std::move(proto_conf);
  return conf;
}

std::once_flag init_server;
std::unique_ptr<brpc::Server> server;
std::string server_host;
std::pair<yacl::Buffer, yacl::Buffer> key_pair;

void InitBeaverServer() {
  std::call_once(init_server, []() {
    key_pair = yacl::crypto::GenSm2KeyPairToPemBuf();
    semi2k::beaver::ttp_server::ServerOptions options;
    options.asym_crypto_schema = "sm2";
    options.server_private_key = key_pair.second;
    options.port = 0;
    server = semi2k::beaver::ttp_server::RunServer(options);
    server_host = fmt::format("127.0.0.1:{}", server->listen_address().port);
  });
}

std::unique_ptr<SPUContext> makeTTPSemi2kProtocol(
    const RuntimeConfig& rt, const std::shared_ptr<yacl::link::Context>& lctx) {
  InitBeaverServer();
  RuntimeConfig ttp_rt = rt;

  auto* semi2k_cfg = ttp_rt.mutable_protocol()->mutable_semi2k_config();
  semi2k_cfg->set_beaver_type(BeaverType::TrustedThirdParty);
  auto* ttp = semi2k_cfg->mutable_ttp_beaver_config();
  ttp->set_adjust_rank(lctx->WorldSize() - 1);
  ttp->set_server_host(server_host);
  ttp->set_asym_crypto_schema("SM2");
  ttp->set_server_public_key(key_pair.first.data(), key_pair.first.size());

  return makeSemi2kProtocol(ttp_rt, lctx);
}

SemanticType GetSemanticType(int64_t field) {
  switch (field) {
    case 32:
      return SE_I32;
    case 64:
      return SE_I64;
    case 128:
      return SE_I128;
  }
  return SE_INVALID;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ApiTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),  //
                     testing::Values(makeConfig(32),          //
                                     makeConfig(64),          //
                                     makeConfig(128)),        //
                     testing::Values(2, 3, 5)),               //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ArithmeticTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),  //
                     testing::Values(makeConfig(32),          //
                                     makeConfig(64),          //
                                     makeConfig(128)),        //
                     testing::Values(2, 3, 5)),               //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
      ;
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, BooleanTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),  //
                     testing::Values(makeConfig(32),          //
                                     makeConfig(64),          //
                                     makeConfig(128)),        //
                     testing::Values(2, 3, 5)),               //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
      ;
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ConversionTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),  //
                     testing::Values(makeConfig(32),          //
                                     makeConfig(64),          //
                                     makeConfig(128)),        //
                     testing::Values(2, 3, 5)),               //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
      ;
    });

class BeaverCacheTest : public ::testing::TestWithParam<OpTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Semi2k, BeaverCacheTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),  //
                     testing::Values(makeConfig(32),          //
                                     makeConfig(64),          //
                                     makeConfig(128)),        //
                     testing::Values(2, 3, 5)),               //
    [](const testing::TestParamInfo<BeaverCacheTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).protocol().field(),
                         std::get<2>(p.param));
      ;
    });

TEST_P(BeaverCacheTest, MatMulAA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  // please keep shape.numel() * SizeOf(conf.field()) > 32kb
  // otherwise, transpose test will failed.
  const Shape shape_A = {383, 29};
  const Shape shape_B = {29, 383};

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    auto* beaver_cache = obj->getState<mpc::Semi2kState>()->beaver_cache();

    auto p_a1 =
        rand_p(obj.get(), GetSemanticType(conf.protocol().field()), shape_A);
    auto p_a2 =
        rand_p(obj.get(), GetSemanticType(conf.protocol().field()), shape_A);
    auto p_a2_t = transpose(obj.get(), p_a2, {});
    SPU_ENFORCE(p_a2_t.numel() * p_a2_t.elsize() > 32 * 1024);
    auto p_b1 =
        rand_p(obj.get(), GetSemanticType(conf.protocol().field()), shape_B);
    auto p_b1_t = transpose(obj.get(), p_b1, {});
    auto p_b2 =
        rand_p(obj.get(), GetSemanticType(conf.protocol().field()), shape_B);

    auto a_a1 = p2a(obj.get(), p_a1);
    auto a_a2 = p2a(obj.get(), p_a2);
    auto a_a2_t = transpose(obj.get(), a_a2, {});
    auto a_b1 = p2a(obj.get(), p_b1);
    auto a_b1_t = transpose(obj.get(), a_b1, {});
    auto a_b2 = p2a(obj.get(), p_b2);

    beaver_cache->EnableCache(a_a1);
    beaver_cache->EnableCache(a_b1);

    auto verify = [&](const MemRef& r, const MemRef& p0, const MemRef& p1) {
      auto r_aa = a2p(obj.get(), r);
      auto r_pp = mmul_pp(obj.get(), p0, p1);
      EXPECT_EQ((r_aa).shape(), (r_pp).shape());
      EXPECT_TRUE(ring_all_equal(r_aa, r_pp));
    };

    auto test = [&](absl::Span<const MemRef> a, absl::Span<const MemRef> p) {
      auto prev = obj->prot()->getState<Communicator>()->getStats();
      auto r_xw = mmul_aa(obj.get(), a[0], a[1]);
      auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
      verify(r_xw, p[0], p[1]);

      return cost;
    };

    {
      // a1 * b1
      auto ab_no_cache = test({a_a1, a_b1}, {p_a1, p_b1});
      // a2 * b1, rhs hit cache
      auto ab_rhs_cache = test({a_a2, a_b1}, {p_a2, p_b1});
      // b1.T * a2.T, lhs hit cache with T
      auto ab_lhs_cache_t = test({a_b1_t, a_a2_t}, {p_b1_t, p_a2_t});
      // a1 * b2, lhs hit cache
      auto ab_lhs_cache = test({a_a1, a_b2}, {p_a1, p_b2});
      // a1 * b1, both hit cache
      auto ab_full_cache = test({a_a1, a_b1}, {p_a1, p_b1});

      // if hit cache, comm will drop.
      EXPECT_EQ(ab_no_cache.comm, ab_rhs_cache.comm * 2);
      EXPECT_EQ(ab_no_cache.comm, ab_lhs_cache_t.comm * 2);
      EXPECT_EQ(ab_no_cache.comm, ab_lhs_cache.comm * 2);
      EXPECT_EQ(0, ab_full_cache.comm);

      // drop a's cache
      beaver_cache->DisableCache(a_a1);
      // a1 * b1, rhs hit cache
      auto ab_rhs_cache_2 = test({a_a1, a_b1}, {p_a1, p_b1});
      EXPECT_EQ(ab_no_cache.comm, ab_rhs_cache_2.comm * 2);
      // drop b's cache
      beaver_cache->DisableCache(a_b1);
      // a1 * b1, no cache
      auto ab_no_cache_2 = test({a_a1, a_b1}, {p_a1, p_b1});
      EXPECT_EQ(ab_no_cache.comm, ab_no_cache_2.comm);
    }
  });
}

TEST_P(BeaverCacheTest, MulAA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  const Shape shape = {109, 107};

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* beaver_cache = obj->getState<mpc::Semi2kState>()->beaver_cache();

    // please keep slice_shape.numel() * SizeOf(conf.field()) > 32kb
    // otherwise, sliced test will failed.
    Index start{2, 2};
    Shape sizes{103, 103};
    Strides stride{1, 1};

    auto p_a1 =
        rand_p(obj.get(), GetSemanticType(conf.protocol().field()), shape);
    auto p_a1_slice = extract_slice(obj.get(), p_a1, start, sizes, stride);
    SPU_ENFORCE(p_a1_slice.numel() * p_a1_slice.elsize() > 32 * 1024);
    SPU_ENFORCE(!p_a1_slice.isCompact());
    auto p_a2 =
        rand_p(obj.get(), GetSemanticType(conf.protocol().field()), shape);
    auto p_b1 =
        rand_p(obj.get(), GetSemanticType(conf.protocol().field()), shape);
    auto p_b1_slice = extract_slice(obj.get(), p_b1, start, sizes, stride);
    auto p_b2 =
        rand_p(obj.get(), GetSemanticType(conf.protocol().field()), shape);
    auto p_b2_slice = extract_slice(obj.get(), p_b2, start, sizes, stride);

    auto a_a1 = p2a(obj.get(), p_a1);
    auto a_a1_slice = extract_slice(obj.get(), a_a1, start, sizes, stride);
    auto a_a2 = p2a(obj.get(), p_a2);
    auto a_b1 = p2a(obj.get(), p_b1);
    auto a_b1_slice = extract_slice(obj.get(), a_b1, start, sizes, stride);
    auto a_b2 = p2a(obj.get(), p_b2);
    auto a_b2_slice = extract_slice(obj.get(), a_b2, start, sizes, stride);

    beaver_cache->EnableCache(a_a1);
    beaver_cache->EnableCache(a_b1);

    auto verify = [&](const MemRef& r, const MemRef& p0, const MemRef& p1) {
      auto r_aa = a2p(obj.get(), r);
      auto r_pp = mul_pp(obj.get(), p0, p1);
      EXPECT_EQ((r_aa).shape(), (r_pp).shape());
      EXPECT_TRUE(ring_all_equal((r_aa), (r_pp)));
    };

    auto test = [&](absl::Span<const MemRef> a, absl::Span<const MemRef> p) {
      auto prev = obj->prot()->getState<Communicator>()->getStats();
      auto r_xw = mul_aa(obj.get(), a[0], a[1]);
      auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
      verify(r_xw, p[0], p[1]);

      return cost;
    };

    {
      // a1 * b1
      auto ab_no_cache = test({a_a1, a_b1}, {p_a1, p_b1});
      // a2 * b1, rhs hit cache
      auto ab_rhs_cache = test({a_a2, a_b1}, {p_a2, p_b1});
      // a1 * b2, lhs hit cache
      auto ab_lhs_cache = test({a_a1, a_b2}, {p_a1, p_b2});
      // a1 * b1, both hit cache
      auto ab_full_cache = test({a_a1, a_b1}, {p_a1, p_b1});
      // a1 * a1, both hit cache
      auto aa_full_cache = test({a_a1, a_a1}, {p_a1, p_a1});

      // if hit cache, comm will drop.
      EXPECT_EQ(ab_no_cache.comm, ab_rhs_cache.comm * 2);
      EXPECT_EQ(ab_no_cache.comm, ab_lhs_cache.comm * 2);
      EXPECT_EQ(0, ab_full_cache.comm);
      EXPECT_EQ(0, aa_full_cache.comm);

      // sliced, array is not compacted
      // a1_slice * b1_slice
      auto ab_sliced_no_cache =
          test({a_a1_slice, a_b1_slice}, {p_a1_slice, p_b1_slice});
      // a1_slice * b2_slice, lhs hit cache
      auto ab_sliced_lhs_cache =
          test({a_a1_slice, a_b2_slice}, {p_a1_slice, p_b2_slice});

      EXPECT_EQ(ab_sliced_no_cache.comm, ab_sliced_lhs_cache.comm * 2);

      // drop a's cache
      beaver_cache->DisableCache(a_a1);
      // a1 * b1, rhs hit cache
      auto ab_rhs_cache_2 = test({a_a1, a_b1}, {p_a1, p_b1});
      EXPECT_EQ(ab_no_cache.comm, ab_rhs_cache_2.comm * 2);
      // drop b's cache
      beaver_cache->DisableCache(a_b1);
      // a1 * b1, no cache
      auto ab_no_cache_2 = test({a_a1, a_b1}, {p_a1, p_b1});
      EXPECT_EQ(ab_no_cache.comm, ab_no_cache_2.comm);

      beaver_cache->EnableCache(a_a1);
      // a1 * a1, no cache
      auto aa_no_cache = test({a_a1, a_a1}, {p_a1, p_a1});
      // FIXME: how to save half the communication
      EXPECT_EQ(aa_no_cache.comm, ab_rhs_cache.comm * 2);
      beaver_cache->DisableCache(a_a1);
    }
  });
}

}  // namespace spu::mpc::test
