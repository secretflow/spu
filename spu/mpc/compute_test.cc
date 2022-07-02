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

#include "spu/mpc/compute_test.h"

#include "gtest/gtest.h"

#include "spu/core/shape_util.h"
#include "spu/mpc/interfaces.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/simulate.h"

namespace spu::mpc::test {
namespace {

constexpr int64_t kNumel = 7;
const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

}  // namespace

#define TEST_BINARY_OP_SS(OP)                                                \
  TEST_P(ComputeTest, OP##_ss) {                                             \
    const auto factory = std::get<0>(GetParam());                            \
    const size_t npc = std::get<1>(GetParam());                              \
    const FieldType field = std::get<2>(GetParam());                         \
                                                                             \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {     \
      auto obj = factory(lctx);                                              \
                                                                             \
      /* GIVEN */                                                            \
      auto p0 = rand_p(obj.get(), field, kNumel);                            \
      auto p1 = rand_p(obj.get(), field, kNumel);                            \
                                                                             \
      /* WHEN */                                                             \
      auto tmp = OP##_ss(obj.get(), p2s(obj.get(), p0), p2s(obj.get(), p1)); \
      auto re = s2p(obj.get(), tmp);                                         \
      auto rp = OP##_pp(obj.get(), p0, p1);                                  \
                                                                             \
      /* THEN */                                                             \
      EXPECT_TRUE(ring_all_equal(re, rp));                                   \
    });                                                                      \
  }

#define TEST_BINARY_OP_SP(OP)                                            \
  TEST_P(ComputeTest, OP##_sp) {                                         \
    const auto factory = std::get<0>(GetParam());                        \
    const size_t npc = std::get<1>(GetParam());                          \
    const FieldType field = std::get<2>(GetParam());                     \
                                                                         \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) { \
      auto obj = factory(lctx);                                          \
                                                                         \
      /* GIVEN */                                                        \
      auto p0 = rand_p(obj.get(), field, kNumel);                        \
      auto p1 = rand_p(obj.get(), field, kNumel);                        \
                                                                         \
      /* WHEN */                                                         \
      auto tmp = OP##_sp(obj.get(), p2s(obj.get(), p0), p1);             \
      auto re = s2p(obj.get(), tmp);                                     \
      auto rp = OP##_pp(obj.get(), p0, p1);                              \
                                                                         \
      /* THEN */                                                         \
      EXPECT_TRUE(ring_all_equal(re, rp));                               \
    });                                                                  \
  }

#define TEST_BINARY_OP(OP) \
  TEST_BINARY_OP_SS(OP)    \
  TEST_BINARY_OP_SP(OP)

TEST_BINARY_OP(add)
TEST_BINARY_OP(mul)
TEST_BINARY_OP(and)
TEST_BINARY_OP(xor)

#define TEST_UNARY_OP_S(OP)                                              \
  TEST_P(ComputeTest, OP##_s) {                                          \
    const auto factory = std::get<0>(GetParam());                        \
    const size_t npc = std::get<1>(GetParam());                          \
    const FieldType field = std::get<2>(GetParam());                     \
                                                                         \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) { \
      auto obj = factory(lctx);                                          \
                                                                         \
      /* GIVEN */                                                        \
      auto p0 = rand_p(obj.get(), field, kNumel);                        \
                                                                         \
      /* WHEN */                                                         \
      auto r_s = s2p(obj.get(), OP##_s(obj.get(), p2s(obj.get(), p0)));  \
      auto r_p = OP##_p(obj.get(), p0);                                  \
                                                                         \
      /* THEN */                                                         \
      EXPECT_TRUE(ring_all_equal(r_s, r_p));                             \
    });                                                                  \
  }

#define TEST_UNARY_OP_P(OP)                                              \
  TEST_P(ComputeTest, OP##_p) {                                          \
    const auto factory = std::get<0>(GetParam());                        \
    const size_t npc = std::get<1>(GetParam());                          \
    const FieldType field = std::get<2>(GetParam());                     \
                                                                         \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) { \
      auto obj = factory(lctx);                                          \
                                                                         \
      /* GIVEN */                                                        \
      auto p0 = rand_p(obj.get(), field, kNumel);                        \
                                                                         \
      /* WHEN */                                                         \
      auto r_p = OP##_p(obj.get(), p0);                                  \
      auto r_pp = OP##_p(obj.get(), p0);                                 \
                                                                         \
      /* THEN */                                                         \
      EXPECT_TRUE(ring_all_equal(r_p, r_pp));                            \
    });                                                                  \
  }

#define TEST_UNARY_OP(OP) \
  TEST_UNARY_OP_S(OP)     \
  TEST_UNARY_OP_P(OP)

TEST_UNARY_OP(not )
TEST_UNARY_OP(msb)

#define TEST_UNARY_OP_WITH_BIT_S(OP)                                     \
  TEST_P(ComputeTest, OP##S) {                                           \
    const auto factory = std::get<0>(GetParam());                        \
    const size_t npc = std::get<1>(GetParam());                          \
    const FieldType field = std::get<2>(GetParam());                     \
                                                                         \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) { \
      auto obj = factory(lctx);                                          \
                                                                         \
      /* GIVEN */                                                        \
      auto p0 = rand_p(obj.get(), field, kNumel);                        \
                                                                         \
      for (auto bits : kShiftBits) {                                     \
        /* WHEN */                                                       \
        auto r_s =                                                       \
            s2p(obj.get(), OP##_s(obj.get(), p2s(obj.get(), p0), bits)); \
        auto r_p = OP##_p(obj.get(), p0, bits);                          \
                                                                         \
        /* THEN */                                                       \
        EXPECT_TRUE(ring_all_equal(r_s, r_p));                           \
      }                                                                  \
    });                                                                  \
  }

#define TEST_UNARY_OP_WITH_BIT_P(OP)                                     \
  TEST_P(ComputeTest, OP##P) {                                           \
    const auto factory = std::get<0>(GetParam());                        \
    const size_t npc = std::get<1>(GetParam());                          \
    const FieldType field = std::get<2>(GetParam());                     \
                                                                         \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) { \
      auto obj = factory(lctx);                                          \
                                                                         \
      /* GIVEN */                                                        \
      auto p0 = rand_p(obj.get(), field, kNumel);                        \
                                                                         \
      for (auto bits : kShiftBits) { /* WHEN */                          \
        auto r_p = OP##_p(obj.get(), p0, bits);                          \
        auto r_pp = OP##_p(obj.get(), p0, bits);                         \
                                                                         \
        /* THEN */                                                       \
        EXPECT_TRUE(ring_all_equal(r_p, r_pp));                          \
      }                                                                  \
    });                                                                  \
  }

#define TEST_UNARY_OP_WITH_BIT(OP) \
  TEST_UNARY_OP_WITH_BIT_S(OP)     \
  TEST_UNARY_OP_WITH_BIT_P(OP)

TEST_UNARY_OP_WITH_BIT(lshift)
TEST_UNARY_OP_WITH_BIT(rshift)
TEST_UNARY_OP_WITH_BIT(arshift)

TEST_P(ComputeTest, TruncPrS) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  // trunc_pr only work for smalle range.
  auto p0 = ring_rand_range(field, kNumel, 0, 10000);
  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    const size_t bits = 2;
    auto r_s = s2p(obj.get(), truncpr_s(obj.get(), p2s(obj.get(), p0), bits));
    auto r_p = arshift_p(obj.get(), p0, bits);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_s, r_p, npc));
  });
}

TEST_P(ComputeTest, MatMulSS) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const std::vector<int64_t> shape_A{M, K};
  const std::vector<int64_t> shape_B{K, N};

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, calcNumel(shape_A));
    auto p1 = rand_p(obj.get(), field, calcNumel(shape_B));

    /* WHEN */
    auto tmp =
        mmul_ss(obj.get(), p2s(obj.get(), p0), p2s(obj.get(), p1), M, N, K);
    auto r_ss = s2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1, M, N, K);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_ss, r_pp));
  });
}

TEST_P(ComputeTest, mmulSP) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const std::vector<int64_t> shape_A{M, K};
  const std::vector<int64_t> shape_B{K, N};

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, calcNumel(shape_A));
    auto p1 = rand_p(obj.get(), field, calcNumel(shape_B));

    /* WHEN */
    auto tmp = mmul_sp(obj.get(), p2s(obj.get(), p0), p1, M, N, K);
    auto r_ss = s2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1, M, N, K);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_ss, r_pp));
  });
}

TEST_P(ComputeTest, P2S_S2P) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);

    /* WHEN */
    auto s = p2s(obj.get(), p0);
    auto p1 = s2p(obj.get(), s);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(p0, p1));
  });
}

}  // namespace spu::mpc::test
