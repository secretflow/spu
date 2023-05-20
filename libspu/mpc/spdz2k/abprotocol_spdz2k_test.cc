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

#include "libspu/mpc/spdz2k/abprotocol_spdz2k_test.h"

#include "libspu/core/shape_util.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

Shape kShape = {30, 40};
const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

#define EXPECT_VALUE_EQ(X, Y)                         \
  {                                                   \
    EXPECT_EQ((X).dtype(), (Y).dtype());              \
    EXPECT_EQ((X).shape(), (Y).shape());              \
    auto [x_data, x_shape, x_dtype] = UnwrapValue(X); \
    auto [y_data, y_shape, y_dtype] = UnwrapValue(Y); \
    EXPECT_TRUE(ring_all_equal(x_data, y_data));      \
  }

#define EXPECT_VALUE_ALMOST_EQ(X, Y, ERR)             \
  {                                                   \
    EXPECT_EQ((X).dtype(), (Y).dtype());              \
    EXPECT_EQ((X).shape(), (Y).shape());              \
    auto [x_data, x_shape, x_dtype] = UnwrapValue(X); \
    auto [y_data, y_shape, y_dtype] = UnwrapValue(Y); \
    EXPECT_TRUE(ring_all_equal(x_data, y_data, ERR)); \
  }

bool verifyCost(Kernel* kernel, std::string_view name, FieldType field,
                const Shape& shape, size_t npc,
                const Communicator::Stats& cost) {
  if (kernel->kind() == Kernel::Kind::Dynamic) {
    return true;
  }

  auto comm = kernel->comm();
  auto latency = kernel->latency();

  size_t numel = shape.numel();

  bool succeed = true;
  constexpr size_t kBitsPerBytes = 8;
  ce::Params params = {{"K", SizeOf(field) * 8}, {"N", npc}};
  if (comm->eval(params) * numel != cost.comm * kBitsPerBytes) {
    fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
               comm->eval(params) * numel, cost.comm * kBitsPerBytes);
    succeed = false;
  }
  if (latency->eval(params) != cost.latency) {
    fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
               latency->eval(params), cost.latency);
    succeed = false;
  }

  return succeed;
}

}  // namespace

TEST_P(ArithmeticTest, P2A) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(sctx.get(), kShape);

    /* WHEN */
    auto prev = sctx->prot()->getState<Communicator>()->getStats();
    auto a0 = p2a(sctx.get(), p0);
    auto cost = sctx->prot()->getState<Communicator>()->getStats() - prev;
    auto p1 = a2p(sctx.get(), a0);

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(sctx->prot()->getKernel("p2a"), "p2a", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(ArithmeticTest, A2P) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(sctx.get(), kShape);

    /* WHEN */
    auto a0 = p2a(sctx.get(), p0);
    auto prev = sctx->prot()->getState<Communicator>()->getStats();
    auto p1 = a2p(sctx.get(), a0);
    [[maybe_unused]] auto cost =
        sctx->prot()->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(sctx->prot()->getKernel("a2p"), "a2p", conf.field(),
                           kShape, npc, cost));
  });
}

#define TEST_ARITHMETIC_BINARY_OP_AA(OP)                                    \
  TEST_P(ArithmeticTest, OP##AA) {                                          \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {   \
      auto obj = factory(conf, lctx);                                       \
                                                                            \
      /* GIVEN */                                                           \
      auto p0 = rand_p(obj.get(), kShape);                                  \
      auto p1 = rand_p(obj.get(), kShape);                                  \
                                                                            \
      /* WHEN */                                                            \
      auto a0 = p2a(obj.get(), p0);                                         \
      auto a1 = p2a(obj.get(), p1);                                         \
      auto prev = obj->prot()->getState<Communicator>()->getStats();        \
      auto tmp = OP##_aa(obj.get(), a0, a1);                                \
      auto cost = obj->prot()->getState<Communicator>()->getStats() - prev; \
      auto re = a2p(obj.get(), tmp);                                        \
      auto rp = OP##_pp(obj.get(), p0, p1);                                 \
                                                                            \
      /* THEN */                                                            \
      EXPECT_VALUE_EQ(re, rp);                                              \
      EXPECT_TRUE(verifyCost(obj->prot()->getKernel(#OP "_aa"), #OP "_aa",  \
                             conf.field(), kShape, npc, cost));             \
    });                                                                     \
  }

#define TEST_ARITHMETIC_BINARY_OP_AP(OP)                                    \
  TEST_P(ArithmeticTest, OP##AP) {                                          \
    const auto factory = std::get<0>(GetParam());                           \
    const RuntimeConfig& conf = std::get<1>(GetParam());                    \
    const size_t npc = std::get<2>(GetParam());                             \
                                                                            \
    utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {   \
      auto obj = factory(conf, lctx);                                       \
                                                                            \
      /* GIVEN */                                                           \
      auto p0 = rand_p(obj.get(), kShape);                                  \
      auto p1 = rand_p(obj.get(), kShape);                                  \
                                                                            \
      /* WHEN */                                                            \
      auto a0 = p2a(obj.get(), p0);                                         \
      auto prev = obj->prot()->getState<Communicator>()->getStats();        \
      auto tmp = OP##_ap(obj.get(), a0, p1);                                \
      auto cost = obj->prot()->getState<Communicator>()->getStats() - prev; \
      auto re = a2p(obj.get(), tmp);                                        \
      auto rp = OP##_pp(obj.get(), p0, p1);                                 \
                                                                            \
      /* THEN */                                                            \
      EXPECT_VALUE_EQ(re, rp);                                              \
      EXPECT_TRUE(verifyCost(obj->prot()->getKernel(#OP "_ap"), #OP "_ap",  \
                             conf.field(), kShape, npc, cost));             \
    });                                                                     \
  }

#define TEST_ARITHMETIC_BINARY_OP(OP) \
  TEST_ARITHMETIC_BINARY_OP_AA(OP)    \
  TEST_ARITHMETIC_BINARY_OP_AP(OP)

TEST_ARITHMETIC_BINARY_OP(add)
TEST_ARITHMETIC_BINARY_OP(mul)

TEST_P(ArithmeticTest, NotA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto r_a = not_a(obj.get(), a0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_p = a2p(obj.get(), r_a);
    auto r_pp = a2p(obj.get(), not_a(obj.get(), a0));

    /* THEN */
    EXPECT_VALUE_EQ(r_p, r_pp);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("not_a"), "not_a",
                           conf.field(), kShape, npc, cost));
  });
}

TEST_P(ArithmeticTest, MatMulAP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const Shape shape_A = {M, K};
  const Shape shape_B = {K, N};
  const Shape shape_C = {M, N};

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), shape_A);
    auto p1 = rand_p(obj.get(), shape_B);
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = mmul_ap(obj.get(), a0, p1, M, N, K);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1, M, N, K);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mmul_ap"), "mmul_ap",
                           conf.field(), shape_C, npc, cost));
  });
}

TEST_P(ArithmeticTest, MatMulAA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const Shape shape_A = {M, K};
  const Shape shape_B = {K, N};
  const Shape shape_C = {M, N};

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), shape_A);
    auto p1 = rand_p(obj.get(), shape_B);
    auto a0 = p2a(obj.get(), p0);
    auto a1 = p2a(obj.get(), p1);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = mmul_aa(obj.get(), a0, a1, M, N, K);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1, M, N, K);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mmul_aa"), "mmul_aa",
                           conf.field(), shape_C, npc, cost));
  });
}

TEST_P(ArithmeticTest, LShiftA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto a0 = p2a(obj.get(), p0);

    for (auto bits : kShiftBits) {
      if (bits >= p0.elsize() * 8) {
        // Shift more than elsize is a UB
        continue;
      }
      /* WHEN */
      auto prev = obj->prot()->getState<Communicator>()->getStats();
      auto tmp = lshift_a(obj.get(), a0, bits);
      auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
      auto r_b = a2p(obj.get(), tmp);
      auto r_p = lshift_p(obj.get(), p0, bits);

      /* THEN */
      EXPECT_VALUE_EQ(r_b, r_p);
      EXPECT_TRUE(verifyCost(obj->prot()->getKernel("lshift_a"), "lshift_a",
                             conf.field(), kShape, npc, cost));
    }
  });
}

TEST_P(ArithmeticTest, TruncA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto p0 = rand_p(obj.get(), kShape);

    EXPECT_TRUE(static_cast<TruncAKernel*>(obj->prot()->getKernel("trunc_a"))
                    ->hasMsbError());
    // has msb error, only use lowest 10 bits.
    p0 = arshift_p(obj.get(), p0, SizeOf(conf.field()) * 8 - 10);

    /* GIVEN */
    const size_t bits = 2;
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto a1 = trunc_a(obj.get(), a0, bits);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_a = a2p(obj.get(), a1);
    auto r_p = arshift_p(obj.get(), p0, bits);

    /* THEN */
    EXPECT_VALUE_ALMOST_EQ(r_a, r_p, npc);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("trunc_a"), "trunc_a",
                           conf.field(), kShape, npc, cost));
  });
}

}  // namespace spu::mpc::test
