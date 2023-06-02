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

#include "libspu/mpc/api_test.h"

#include "gtest/gtest.h"

#include "libspu/core/shape_util.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

Shape kShape = {30, 40};
const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

#define EXPECT_VALUE_EQ(X, Y)                         \
  {                                                   \
    EXPECT_EQ((X).shape(), (Y).shape());              \
    auto [x_data, x_shape, x_dtype] = UnwrapValue(X); \
    auto [y_data, y_shape, y_dtype] = UnwrapValue(Y); \
    EXPECT_TRUE(ring_all_equal(x_data, y_data));      \
  }

#define EXPECT_VALUE_ALMOST_EQ(X, Y, ERR)             \
  {                                                   \
    EXPECT_EQ((X).shape(), (Y).shape());              \
    auto [x_data, x_shape, x_dtype] = UnwrapValue(X); \
    auto [y_data, y_shape, y_dtype] = UnwrapValue(Y); \
    EXPECT_TRUE(ring_all_equal(x_data, y_data, ERR)); \
  }

}  // namespace

#define TEST_BINARY_OP_SS(OP)                                               \
  TEST_P(ApiTest, OP##_ss) {                                                \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc,                                                    \
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) { \
                      auto sctx = factory(conf, lctx);                      \
                                                                            \
                      /* GIVEN */                                           \
                      auto x_p = rand_p(sctx.get(), kShape);                \
                      auto y_p = rand_p(sctx.get(), kShape);                \
                      auto x_s = p2s(sctx.get(), x_p);                      \
                      auto y_s = p2s(sctx.get(), y_p);                      \
                                                                            \
                      /* WHEN */                                            \
                      auto tmp = OP##_ss(sctx.get(), x_s, y_s);             \
                      auto r_e = s2p(sctx.get(), tmp);                      \
                      auto r_p = OP##_pp(sctx.get(), x_p, y_p);             \
                                                                            \
                      /* THEN */                                            \
                      EXPECT_VALUE_EQ(r_e, r_p);                            \
                    });                                                     \
  }

#define TEST_BINARY_OP_SV(OP)                                               \
  TEST_P(ApiTest, OP##_sv) {                                                \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc,                                                    \
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) { \
                      auto sctx = factory(conf, lctx);                      \
                                                                            \
                      /* GIVEN */                                           \
                      auto x_p = rand_p(sctx.get(), kShape);                \
                      auto y_p = rand_p(sctx.get(), kShape);                \
                      for (size_t idx = 0; idx < npc; idx++) {              \
                        auto x_s = p2s(sctx.get(), x_p);                    \
                        auto y_v = p2v(sctx.get(), y_p, idx);               \
                                                                            \
                        /* WHEN */                                          \
                        auto tmp = OP##_sv(sctx.get(), x_s, y_v);           \
                        auto r_e = s2p(sctx.get(), tmp);                    \
                        auto r_p = OP##_pp(sctx.get(), x_p, y_p);           \
                                                                            \
                        /* THEN */                                          \
                        EXPECT_VALUE_EQ(r_e, r_p);                          \
                      }                                                     \
                    });                                                     \
  }

#define TEST_BINARY_OP_SP(OP)                                               \
  TEST_P(ApiTest, OP##_sp) {                                                \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc,                                                    \
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) { \
                      auto sctx = factory(conf, lctx);                      \
                                                                            \
                      /* GIVEN */                                           \
                      auto x_p = rand_p(sctx.get(), kShape);                \
                      auto y_p = rand_p(sctx.get(), kShape);                \
                      auto x_s = p2s(sctx.get(), x_p);                      \
                                                                            \
                      /* WHEN */                                            \
                      auto tmp = OP##_sp(sctx.get(), x_s, y_p);             \
                      auto r_e = s2p(sctx.get(), tmp);                      \
                      auto r_p = OP##_pp(sctx.get(), x_p, y_p);             \
                                                                            \
                      /* THEN */                                            \
                      EXPECT_VALUE_EQ(r_e, r_p);                            \
                    });                                                     \
  }

#define TEST_BINARY_OP_VP(OP)                                               \
  TEST_P(ApiTest, OP##_vp) {                                                \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc,                                                    \
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) { \
                      auto sctx = factory(conf, lctx);                      \
                                                                            \
                      /* GIVEN */                                           \
                      auto x_p = rand_p(sctx.get(), kShape);                \
                      auto y_p = rand_p(sctx.get(), kShape);                \
                                                                            \
                      /* baseline */                                        \
                      auto r_pp = OP##_pp(sctx.get(), x_p, y_p);            \
                                                                            \
                      for (size_t idx = 0; idx < npc; idx++) {              \
                        auto x_v = p2v(sctx.get(), x_p, idx);               \
                                                                            \
                        /* VP == PP */                                      \
                        auto r_vp = OP##_vp(sctx.get(), x_v, y_p);          \
                        r_vp = v2p(sctx.get(), r_vp);                       \
                        EXPECT_VALUE_EQ(r_vp, r_pp);                        \
                                                                            \
                        /* VV == PP */                                      \
                        auto y_v = p2v(sctx.get(), y_p, idx);               \
                        auto r_vv = OP##_vv(sctx.get(), x_v, y_v);          \
                        r_vv = v2p(sctx.get(), r_vv);                       \
                        EXPECT_VALUE_EQ(r_vv, r_pp);                        \
                                                                            \
                        /* VV (with differnt owner) == PP */                \
                        auto y_v2 = p2v(sctx.get(), y_p, (idx + 1) % npc);  \
                        auto r_vv2 = OP##_vv(sctx.get(), x_v, y_v2);        \
                        r_vv2 = s2p(sctx.get(), r_vv2);                     \
                        EXPECT_VALUE_EQ(r_vv2, r_pp);                       \
                      }                                                     \
                    });                                                     \
  }

#define TEST_BINARY_OP(OP) \
  TEST_BINARY_OP_SS(OP)    \
  TEST_BINARY_OP_SV(OP)    \
  TEST_BINARY_OP_SP(OP)    \
  TEST_BINARY_OP_VP(OP)

TEST_BINARY_OP(add)
TEST_BINARY_OP(mul)
TEST_BINARY_OP(and)
TEST_BINARY_OP(xor)

#define TEST_UNARY_OP_S(OP)                                                 \
  TEST_P(ApiTest, OP##_s) {                                                 \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc,                                                    \
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) { \
                      auto sctx = factory(conf, lctx);                      \
                                                                            \
                      /* GIVEN */                                           \
                      auto x_p = rand_p(sctx.get(), kShape);                \
                      auto x_s = p2s(sctx.get(), x_p);                      \
                                                                            \
                      /* WHEN */                                            \
                      auto r_s = s2p(sctx.get(), OP##_s(sctx.get(), x_s));  \
                      auto r_p = OP##_p(sctx.get(), x_p);                   \
                                                                            \
                      /* THEN */                                            \
                      EXPECT_VALUE_EQ(r_s, r_p);                            \
                    });                                                     \
  }

#define TEST_UNARY_OP_V(OP)                                                  \
  TEST_P(ApiTest, OP##_v) {                                                  \
    const auto factory = std::get<0>(GetParam());                            \
    const RuntimeConfig& conf = std::get<1>(GetParam());                     \
    const size_t npc = std::get<2>(GetParam());                              \
                                                                             \
    utils::simulate(npc,                                                     \
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) {  \
                      auto sctx = factory(conf, lctx);                       \
                      for (size_t rank = 0; rank < npc; rank++) {            \
                        /* GIVEN */                                          \
                        auto x_p = rand_p(sctx.get(), kShape);               \
                        auto x_v = p2v(sctx.get(), x_p, rank);               \
                                                                             \
                        /* WHEN */                                           \
                        auto r_v = v2p(sctx.get(), OP##_v(sctx.get(), x_v)); \
                        auto r_p = OP##_p(sctx.get(), x_p);                  \
                                                                             \
                        /* THEN */                                           \
                        EXPECT_VALUE_EQ(r_v, r_p);                           \
                      }                                                      \
                    });                                                      \
  }

#define TEST_UNARY_OP_P(OP)                                                 \
  TEST_P(ApiTest, OP##_p) {                                                 \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc,                                                    \
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) { \
                      auto sctx = factory(conf, lctx);                      \
                                                                            \
                      /* GIVEN */                                           \
                      auto p0 = rand_p(sctx.get(), kShape);                 \
                                                                            \
                      /* FIXME: verify self?*/                              \
                      auto r_p = OP##_p(sctx.get(), p0);                    \
                      auto r_pp = OP##_p(sctx.get(), p0);                   \
                                                                            \
                      /* THEN */                                            \
                      EXPECT_VALUE_EQ(r_p, r_pp);                           \
                    });                                                     \
  }

#define TEST_UNARY_OP(OP) \
  TEST_UNARY_OP_S(OP)     \
  TEST_UNARY_OP_V(OP)     \
  TEST_UNARY_OP_P(OP)

TEST_UNARY_OP(not )
TEST_UNARY_OP(msb)

#define TEST_UNARY_OP_WITH_BIT_S(OP)                                   \
  TEST_P(ApiTest, OP##S) {                                             \
    const auto factory = std::get<0>(GetParam());                      \
    const RuntimeConfig& conf = std::get<1>(GetParam());               \
    const size_t npc = std::get<2>(GetParam());                        \
                                                                       \
    utils::simulate(                                                   \
        npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {   \
          auto sctx = factory(conf, lctx);                             \
                                                                       \
          /* GIVEN */                                                  \
          auto x_p = rand_p(sctx.get(), kShape);                       \
          auto x_s = p2s(sctx.get(), x_p);                             \
                                                                       \
          for (auto bits : kShiftBits) {                               \
            if (bits >= SizeOf(conf.field()) * 8) {                    \
              continue;                                                \
            }                                                          \
            /* WHEN */                                                 \
            auto r_s = s2p(sctx.get(), OP##_s(sctx.get(), x_s, bits)); \
            auto r_p = OP##_p(sctx.get(), x_p, bits);                  \
                                                                       \
            /* THEN */                                                 \
            EXPECT_VALUE_EQ(r_s, r_p);                                 \
          }                                                            \
        });                                                            \
  }

#define TEST_UNARY_OP_WITH_BIT_V(OP)                                     \
  TEST_P(ApiTest, OP##V) {                                               \
    const auto factory = std::get<0>(GetParam());                        \
    const RuntimeConfig& conf = std::get<1>(GetParam());                 \
    const size_t npc = std::get<2>(GetParam());                          \
                                                                         \
    utils::simulate(                                                     \
        npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {     \
          auto sctx = factory(conf, lctx);                               \
                                                                         \
          for (size_t rank = 0; rank < npc; rank++) {                    \
            /* GIVEN */                                                  \
            auto x_p = rand_p(sctx.get(), kShape);                       \
            auto x_v = p2v(sctx.get(), x_p, rank);                       \
                                                                         \
            for (auto bits : kShiftBits) {                               \
              if (bits >= SizeOf(conf.field()) * 8) {                    \
                continue;                                                \
              }                                                          \
              /* WHEN */                                                 \
              auto r_v = v2p(sctx.get(), OP##_v(sctx.get(), x_v, bits)); \
              auto r_p = OP##_p(sctx.get(), x_p, bits);                  \
                                                                         \
              /* THEN */                                                 \
              EXPECT_VALUE_EQ(r_v, r_p);                                 \
            }                                                            \
          }                                                              \
        });                                                              \
  }

#define TEST_UNARY_OP_WITH_BIT_P(OP)                                        \
  TEST_P(ApiTest, OP##P) {                                                  \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc,                                                    \
                    [&](const std::shared_ptr<yacl::link::Context>& lctx) { \
                      auto sctx = factory(conf, lctx);                      \
                                                                            \
                      /* GIVEN */                                           \
                      auto p0 = rand_p(sctx.get(), kShape);                 \
                                                                            \
                      for (auto bits : kShiftBits) { /* WHEN */             \
                        if (bits >= SizeOf(conf.field()) * 8) {             \
                          continue;                                         \
                        }                                                   \
                        auto r_p = OP##_p(sctx.get(), p0, bits);            \
                        auto r_pp = OP##_p(sctx.get(), p0, bits);           \
                                                                            \
                        /* THEN */                                          \
                        EXPECT_VALUE_EQ(r_p, r_pp);                         \
                      }                                                     \
                    });                                                     \
  }

#define TEST_UNARY_OP_WITH_BIT(OP) \
  TEST_UNARY_OP_WITH_BIT_S(OP)     \
  TEST_UNARY_OP_WITH_BIT_V(OP)     \
  TEST_UNARY_OP_WITH_BIT_P(OP)

TEST_UNARY_OP_WITH_BIT(lshift)
TEST_UNARY_OP_WITH_BIT(rshift)
TEST_UNARY_OP_WITH_BIT(arshift)

TEST_P(ApiTest, TruncS) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    auto p0 = rand_p(sctx.get(), kShape);

    // TODO: here we assume has msb error, only use lowest 10 bits.
    p0 = arshift_p(sctx.get(), p0, SizeOf(conf.field()) * 8 - 10);

    const size_t bits = 2;
    auto r_s = s2p(sctx.get(), trunc_s(sctx.get(), p2s(sctx.get(), p0), bits));
    auto r_p = arshift_p(sctx.get(), p0, bits);

    /* THEN */
    EXPECT_VALUE_ALMOST_EQ(r_s, r_p, npc);
  });
}

TEST_P(ApiTest, MatMulSS) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  const int64_t M = 70;
  const int64_t K = 400;
  const int64_t N = 60;
  const int64_t N2 = 90;
  const Shape shape_A = {M, K};
  const Shape shape_B = {K, N};
  const Shape shape_B2 = {K, N2};

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(sctx.get(), shape_A);
    auto p1 = rand_p(sctx.get(), shape_B);
    auto p2 = rand_p(sctx.get(), shape_B2);

    auto s0 = p2s(sctx.get(), p0);
    auto s1 = p2s(sctx.get(), p1);
    auto s2 = p2s(sctx.get(), p2);

    /* WHEN */
    auto tmp = mmul_ss(sctx.get(), s0, s1, M, N, K);
    auto tmp2 = mmul_ss(sctx.get(), s0, s2, M, N2, K);

    auto r_ss = s2p(sctx.get(), tmp);
    auto r_pp = mmul_pp(sctx.get(), p0, p1, M, N, K);

    auto r_ss2 = s2p(sctx.get(), tmp2);
    auto r_pp2 = mmul_pp(sctx.get(), p0, p2, M, N2, K);

    /* THEN */
    EXPECT_VALUE_EQ(r_ss, r_pp);
    EXPECT_VALUE_EQ(r_ss2, r_pp2);
  });
}

TEST_P(ApiTest, MmulSP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const Shape shape_A = {M, K};
  const Shape shape_B = {K, N};

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(sctx.get(), shape_A);
    auto p1 = rand_p(sctx.get(), shape_B);

    /* WHEN */
    auto tmp = mmul_sp(sctx.get(), p2s(sctx.get(), p0), p1, M, N, K);
    auto r_ss = s2p(sctx.get(), tmp);
    auto r_pp = mmul_pp(sctx.get(), p0, p1, M, N, K);

    /* THEN */
    EXPECT_VALUE_EQ(r_ss, r_pp);
  });
}

TEST_P(ApiTest, P2S_S2P) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(sctx.get(), kShape);

    /* WHEN */
    auto s = p2s(sctx.get(), p0);
    auto p1 = s2p(sctx.get(), s);

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
  });
}

TEST_P(ApiTest, P2V_V2P) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    for (size_t idx = 0; idx < npc; idx++) {
      /* GIVEN */
      auto x_p = rand_p(sctx.get(), kShape);
      /* WHEN */
      auto x_v = p2v(sctx.get(), x_p, idx);
      auto r_p = v2p(sctx.get(), x_v);

      /* THEN */
      EXPECT_VALUE_EQ(x_p, r_p);
    }
  });
}

TEST_P(ApiTest, P2V_V2S_S2P) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    for (size_t idx = 0; idx < npc; idx++) {
      /* GIVEN */
      auto x_p = rand_p(sctx.get(), kShape);
      /* WHEN */
      auto x_v = p2v(sctx.get(), x_p, idx);
      auto x_s = v2s(sctx.get(), x_v);
      auto r_p = s2p(sctx.get(), x_s);

      /* THEN */
      EXPECT_VALUE_EQ(x_p, r_p);
    }
  });
}

TEST_P(ApiTest, P2S_S2V_V2P) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    for (size_t idx = 0; idx < npc; idx++) {
      /* GIVEN */
      auto x_p = rand_p(sctx.get(), kShape);
      /* WHEN */
      auto x_s = p2s(sctx.get(), x_p);
      auto x_v = s2v(sctx.get(), x_s, idx);
      auto r_p = v2p(sctx.get(), x_v);

      /* THEN */
      EXPECT_VALUE_EQ(x_p, r_p);
    }
  });
}

}  // namespace spu::mpc::test
