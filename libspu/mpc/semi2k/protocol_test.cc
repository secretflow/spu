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
#include "libspu/mpc/semi2k/exp.h"
#include "libspu/mpc/semi2k/prime_utils.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.set_protocol(ProtocolKind::SEMI2K);
  conf.set_field(field);
  if (field == FieldType::FM64) {
    conf.set_fxp_fraction_bits(17);
  } else if (field == FieldType::FM128) {
    conf.set_fxp_fraction_bits(40);
  }
  conf.set_experimental_enable_exp_prime(true);
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

  ttp_rt.set_beaver_type(RuntimeConfig_BeaverType_TrustedThirdParty);
  auto* ttp = ttp_rt.mutable_ttp_beaver_config();
  ttp->set_adjust_rank(lctx->WorldSize() - 1);
  ttp->set_server_host(server_host);
  ttp->set_asym_crypto_schema("SM2");
  ttp->set_server_public_key(key_pair.first.data(), key_pair.first.size());

  return makeSemi2kProtocol(ttp_rt, lctx);
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ApiTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ArithmeticTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
      ;
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, BooleanTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
      ;
    });

INSTANTIATE_TEST_SUITE_P(
    Semi2k, ConversionTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
      ;
    });

class BeaverCacheTest : public ::testing::TestWithParam<OpTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Semi2k, BeaverCacheTest,
    testing::Combine(testing::Values(CreateObjectFn(makeSemi2kProtocol, "tfp"),
                                     CreateObjectFn(makeTTPSemi2kProtocol,
                                                    "ttp")),         //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2, 3, 5)),                      //
    [](const testing::TestParamInfo<BeaverCacheTest::ParamType>& p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param).name(),
                         std::get<1>(p.param).field(), std::get<2>(p.param));
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

    auto p_a1 = rand_p(obj.get(), shape_A);
    auto p_a2 = rand_p(obj.get(), shape_A);
    auto p_a2_t = transpose(obj.get(), p_a2, {});
    SPU_ENFORCE(p_a2_t.numel() * p_a2_t.elsize() > 32 * 1024);
    auto p_b1 = rand_p(obj.get(), shape_B);
    auto p_b1_t = transpose(obj.get(), p_b1, {});
    auto p_b2 = rand_p(obj.get(), shape_B);

    auto a_a1 = p2a(obj.get(), p_a1);
    auto a_a2 = p2a(obj.get(), p_a2);
    auto a_a2_t = transpose(obj.get(), a_a2, {});
    auto a_b1 = p2a(obj.get(), p_b1);
    auto a_b1_t = transpose(obj.get(), a_b1, {});
    auto a_b2 = p2a(obj.get(), p_b2);

    beaver_cache->EnableCache(a_a1.data());
    beaver_cache->EnableCache(a_b1.data());

    auto verify = [&](const Value& r, const Value& p0, const Value& p1) {
      auto r_aa = a2p(obj.get(), r);
      auto r_pp = mmul_pp(obj.get(), p0, p1);
      EXPECT_EQ((r_aa).shape(), (r_pp).shape());
      EXPECT_TRUE(ring_all_equal((r_aa).data(), (r_pp).data()));
    };

    auto test = [&](absl::Span<const Value> a, absl::Span<const Value> p) {
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
      beaver_cache->DisableCache(a_a1.data());
      // a1 * b1, rhs hit cache
      auto ab_rhs_cache_2 = test({a_a1, a_b1}, {p_a1, p_b1});
      EXPECT_EQ(ab_no_cache.comm, ab_rhs_cache_2.comm * 2);
      // drop b's cache
      beaver_cache->DisableCache(a_b1.data());
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
    Index end{105, 105};
    Strides stride{1, 1};

    auto p_a1 = rand_p(obj.get(), shape);
    auto p_a1_slice = extract_slice(obj.get(), p_a1, start, end, stride);
    SPU_ENFORCE(p_a1_slice.numel() * p_a1_slice.elsize() > 32 * 1024);
    SPU_ENFORCE(!p_a1_slice.data().isCompact());
    auto p_a2 = rand_p(obj.get(), shape);
    auto p_b1 = rand_p(obj.get(), shape);
    auto p_b1_slice = extract_slice(obj.get(), p_b1, start, end, stride);
    auto p_b2 = rand_p(obj.get(), shape);
    auto p_b2_slice = extract_slice(obj.get(), p_b2, start, end, stride);

    auto a_a1 = p2a(obj.get(), p_a1);
    auto a_a1_slice = extract_slice(obj.get(), a_a1, start, end, stride);
    auto a_a2 = p2a(obj.get(), p_a2);
    auto a_b1 = p2a(obj.get(), p_b1);
    auto a_b1_slice = extract_slice(obj.get(), a_b1, start, end, stride);
    auto a_b2 = p2a(obj.get(), p_b2);
    auto a_b2_slice = extract_slice(obj.get(), a_b2, start, end, stride);

    beaver_cache->EnableCache(a_a1.data());
    beaver_cache->EnableCache(a_b1.data());

    auto verify = [&](const Value& r, const Value& p0, const Value& p1) {
      auto r_aa = a2p(obj.get(), r);
      auto r_pp = mul_pp(obj.get(), p0, p1);
      EXPECT_EQ((r_aa).shape(), (r_pp).shape());
      EXPECT_TRUE(ring_all_equal((r_aa).data(), (r_pp).data()));
    };

    auto test = [&](absl::Span<const Value> a, absl::Span<const Value> p) {
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
      beaver_cache->DisableCache(a_a1.data());
      // a1 * b1, rhs hit cache
      auto ab_rhs_cache_2 = test({a_a1, a_b1}, {p_a1, p_b1});
      EXPECT_EQ(ab_no_cache.comm, ab_rhs_cache_2.comm * 2);
      // drop b's cache
      beaver_cache->DisableCache(a_b1.data());
      // a1 * b1, no cache
      auto ab_no_cache_2 = test({a_a1, a_b1}, {p_a1, p_b1});
      EXPECT_EQ(ab_no_cache.comm, ab_no_cache_2.comm);

      beaver_cache->EnableCache(a_a1.data());
      // a1 * a1, no cache
      auto aa_no_cache = test({a_a1, a_a1}, {p_a1, p_a1});
      // FIXME: how to save half the communication
      EXPECT_EQ(aa_no_cache.comm, ab_rhs_cache.comm * 2);
      beaver_cache->DisableCache(a_a1.data());
    }
  });
}

TEST_P(BeaverCacheTest, SquareA) {
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
    Index end{105, 105};
    Strides stride{1, 1};

    auto p_a1 = rand_p(obj.get(), shape);
    auto p_a1_slice = extract_slice(obj.get(), p_a1, start, end, stride);
    SPU_ENFORCE(p_a1_slice.numel() * p_a1_slice.elsize() > 32 * 1024);
    SPU_ENFORCE(!p_a1_slice.data().isCompact());

    auto a_a1 = p2a(obj.get(), p_a1);
    auto a_a1_slice = extract_slice(obj.get(), a_a1, start, end, stride);

    beaver_cache->EnableCache(a_a1.data());

    auto verify = [&](const Value& r, const Value& p0) {
      auto r_aa = a2p(obj.get(), r);
      auto r_pp = square_p(obj.get(), p0);
      EXPECT_EQ((r_aa).shape(), (r_pp).shape());
      EXPECT_TRUE(ring_all_equal((r_aa).data(), (r_pp).data()));
    };

    auto test = [&](absl::Span<const Value> a, absl::Span<const Value> p) {
      auto prev = obj->prot()->getState<Communicator>()->getStats();
      auto r_xw = square_a(obj.get(), a[0]);
      auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
      verify(r_xw, p[0]);

      return cost;
    };

    {
      // square(a1)
      auto no_cache = test({a_a1}, {p_a1});
      // square(a1), hit cache
      auto hit_cache = test({a_a1}, {p_a1});

      // if hit cache, comm will drop.
      EXPECT_NE(0, no_cache.comm);
      EXPECT_EQ(0, hit_cache.comm);

      // sliced, array is not compacted
      // square(a_a1_slice)
      auto sliced_no_cache = test({a_a1_slice}, {p_a1_slice});
      // square(a_a1_slice), hit cache
      auto sliced_hit_cache = test({a_a1_slice}, {p_a1_slice});

      EXPECT_NE(0, sliced_no_cache.comm);
      EXPECT_EQ(0, sliced_hit_cache.comm);

      // drop a's cache
      beaver_cache->DisableCache(a_a1.data());
      // square(a1), no cache
      no_cache = test({a_a1}, {p_a1});
      EXPECT_NE(0, no_cache.comm);
    }
  });
}

TEST_P(BeaverCacheTest, priv_mul_test) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());
  // only supports 2 party (not counting beaver)
  if (npc != 2) {
    return;
  }
  NdArrayRef ring2k_shr[2];

  int64_t numel = 1;
  FieldType field = conf.field();

  std::vector<double> real_vec(numel);
  for (int64_t i = 0; i < numel; ++i) {
    real_vec[i] = 2;
  }

  auto rnd_msg = gfmp_zeros(field, {numel});

  DISPATCH_ALL_FIELDS(field, [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> xmsg(rnd_msg);
    pforeach(0, numel, [&](int64_t i) { xmsg[i] = std::round(real_vec[i]); });
  });

  ring2k_shr[0] = rnd_msg;
  ring2k_shr[1] = rnd_msg;

  NdArrayRef input, outp_pub;
  NdArrayRef outp[2];

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    KernelEvalContext kcontext(obj.get());

    int rank = lctx->Rank();

    outp[rank] = spu::mpc::semi2k::MulPrivModMP(&kcontext, ring2k_shr[rank]);
  });
  auto got = gfmp_add_mod(outp[0], outp[1]);
  DISPATCH_ALL_FIELDS(field, [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> got_view(got);

    double max_err = 0.0;
    double min_err = 99.0;
    for (int64_t i = 0; i < numel; ++i) {
      double expected = real_vec[i] * real_vec[i];
      double got = static_cast<double>(got_view[i]);
      max_err = std::max(max_err, std::abs(expected - got));
      min_err = std::min(min_err, std::abs(expected - got));
    }
    ASSERT_LE(min_err, 1e-3);
    ASSERT_LE(max_err, 1e-3);
  });
}

TEST_P(BeaverCacheTest, exp_mod_test) {
  const RuntimeConfig& conf = std::get<1>(GetParam());
  FieldType field = conf.field();

  DISPATCH_ALL_FIELDS(field, [&]() {
    // exponents < 32
    ring2k_t exponents[5] = {10, 21, 27};

    for (ring2k_t exponent : exponents) {
      ring2k_t y = exp_mod<ring2k_t>(2, exponent);
      ring2k_t prime = ScalarTypeToPrime<ring2k_t>::prime;
      ring2k_t prime_minus_one = (prime - 1);
      ring2k_t shifted_bit = 1;
      shifted_bit <<= exponent;
      EXPECT_EQ(y, shifted_bit % prime_minus_one);
    }
  });
}

TEST_P(BeaverCacheTest, ExpA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());
  // exp only supports 2 party (not counting beaver)
  // only supports FM128 for now
  // note not using ctx->hasKernel("exp_a") because we are testing kernel
  // registration as well.
  if (npc != 2 || conf.field() != FieldType::FM128) {
    return;
  }
  auto fxp = conf.fxp_fraction_bits();

  NdArrayRef ring2k_shr[2];

  int64_t numel = 100;
  FieldType field = conf.field();

  // how to define and achieve high pricision for e^20
  std::uniform_real_distribution<double> dist(-18.0, 15.0);
  std::default_random_engine rd;
  std::vector<double> real_vec(numel);
  for (int64_t i = 0; i < numel; ++i) {
    // make the input a fixed point number, eliminate the fixed point encoding
    // error
    real_vec[i] =
        static_cast<double>(std::round((dist(rd) * (1L << fxp)))) / (1L << fxp);
  }

  auto rnd_msg = ring_zeros(field, {numel});

  DISPATCH_ALL_FIELDS(field, [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> xmsg(rnd_msg);
    pforeach(0, numel, [&](int64_t i) {
      xmsg[i] = std::round(real_vec[i] * (1L << fxp));
    });
  });

  ring2k_shr[0] = ring_rand(field, rnd_msg.shape())
                      .as(makeType<spu::mpc::semi2k::AShrTy>(field));
  ring2k_shr[1] = ring_sub(rnd_msg, ring2k_shr[0])
                      .as(makeType<spu::mpc::semi2k::AShrTy>(field));

  NdArrayRef outp_pub;
  NdArrayRef outp[2];

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    KernelEvalContext kcontext(obj.get());

    int rank = lctx->Rank();

    size_t bytes = lctx->GetStats()->sent_bytes;
    size_t action = lctx->GetStats()->sent_actions;

    spu::mpc::semi2k::ExpA exp;
    outp[rank] = exp.proc(&kcontext, ring2k_shr[rank]);

    bytes = lctx->GetStats()->sent_bytes - bytes;
    action = lctx->GetStats()->sent_actions - action;
    SPDLOG_INFO("ExpA ({}) for n = {}, sent {} MiB ({} B per), actions {}",
                field, numel, bytes * 1. / 1024. / 1024., bytes * 1. / numel,
                action);
  });
  assert(outp[0].eltype() == ring2k_shr[0].eltype());
  auto got = ring_add(outp[0], outp[1]);
  ring_print(got, "exp result");
  DISPATCH_ALL_FIELDS(field, [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> got_view(got);

    double max_err = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
      double expected = std::exp(real_vec[i]);
      expected = static_cast<double>(std::round((expected * (1L << fxp)))) /
                 (1L << fxp);
      double got = static_cast<double>(got_view[i]) / (1L << fxp);
      // cout left here for future improvement
      std::cout << "expected: " << fmt::format("{0:f}", expected)
                << ", got: " << fmt::format("{0:f}", got) << std::endl;
      std::cout << "expected: "
                << fmt::format("{0:b}",
                               static_cast<ring2k_t>(expected * (1L << fxp)))
                << ", got: " << fmt::format("{0:b}", got_view[i]) << std::endl;
      max_err = std::max(max_err, std::abs(expected - got));
    }
    ASSERT_LE(max_err, 1e-0);
  });
}
}  // namespace spu::mpc::test
