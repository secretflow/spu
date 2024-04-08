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

#include "libspu/mpc/ab_api_test.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

Shape kShape = {20, 30};
const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

#define EXPECT_VALUE_EQ(X, Y)                            \
  {                                                      \
    EXPECT_EQ((X).shape(), (Y).shape());                 \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data())); \
  }

#define EXPECT_VALUE_ALMOST_EQ(X, Y, ERR)                     \
  {                                                           \
    EXPECT_EQ((X).shape(), (Y).shape());                      \
    EXPECT_TRUE(ring_all_equal((X).data(), (Y).data(), ERR)); \
  }

bool verifyCost(Kernel* kernel, std::string_view name, const ce::Params& params,
                const Communicator::Stats& cost, size_t repeated = 1) {
  if (kernel->kind() == Kernel::Kind::Dynamic) {
    return true;
  }

  auto comm = kernel->comm();
  auto latency = kernel->latency();

  bool succeed = true;
  constexpr size_t kBitsPerBytes = 8;
  const auto expectedComm = comm->eval(params) * repeated;
  const auto realComm = cost.comm * kBitsPerBytes;

  float diff;
  if (expectedComm == 0) {
    diff = realComm;
  } else {
    diff = std::abs(static_cast<float>(realComm - expectedComm)) / expectedComm;
  }
  if (realComm < expectedComm || diff > kernel->getCommTolerance()) {
    fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
               expectedComm, realComm);
    succeed = false;
  }

  if (latency->eval(params) != cost.latency) {
    fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
               latency->eval(params), cost.latency);
    succeed = false;
  }

  return succeed;
}

bool verifyCost(Kernel* kernel, std::string_view name, FieldType field,
                const Shape& shape, size_t npc,
                const Communicator::Stats& cost) {
  ce::Params params = {{"K", SizeOf(field) * 8}, {"N", npc}};
  return verifyCost(kernel, name, params, cost, shape.numel() /*repeated*/);
}

}  // namespace

#define TEST_ARITHMETIC_BINARY_OP_AA(OP)                                       \
  TEST_P(ArithmeticTest, OP##AA) {                                             \
    const auto factory = std::get<0>(GetParam());                              \
    const RuntimeConfig& conf = std::get<1>(GetParam());                       \
    const size_t npc = std::get<2>(GetParam());                                \
                                                                               \
    utils::simulate(                                                           \
        npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {           \
          auto obj = factory(conf, lctx);                                      \
                                                                               \
          /* GIVEN */                                                          \
          auto p0 = rand_p(obj.get(), kShape);                                 \
          auto p1 = rand_p(obj.get(), kShape);                                 \
                                                                               \
          /* WHEN */                                                           \
          auto a0 = p2a(obj.get(), p0);                                        \
          auto a1 = p2a(obj.get(), p1);                                        \
          auto prev = obj->prot()->getState<Communicator>()->getStats();       \
          auto tmp = OP##_aa(obj.get(), a0, a1);                               \
          auto cost =                                                          \
              obj->prot()->getState<Communicator>()->getStats() - prev;        \
          auto re = a2p(obj.get(), tmp);                                       \
          auto rp = OP##_pp(obj.get(), p0, p1);                                \
                                                                               \
          /* THEN */                                                           \
          EXPECT_VALUE_EQ(re, rp);                                             \
          EXPECT_TRUE(verifyCost(obj->prot()->getKernel(#OP "_aa"), #OP "_aa", \
                                 conf.field(), kShape, npc, cost));            \
        });                                                                    \
  }

#define TEST_ARITHMETIC_BINARY_OP_AP(OP)                                       \
  TEST_P(ArithmeticTest, OP##AP) {                                             \
    const auto factory = std::get<0>(GetParam());                              \
    const RuntimeConfig& conf = std::get<1>(GetParam());                       \
    const size_t npc = std::get<2>(GetParam());                                \
                                                                               \
    utils::simulate(                                                           \
        npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {           \
          auto obj = factory(conf, lctx);                                      \
                                                                               \
          /* GIVEN */                                                          \
          auto p0 = rand_p(obj.get(), kShape);                                 \
          auto p1 = rand_p(obj.get(), kShape);                                 \
                                                                               \
          /* WHEN */                                                           \
          auto a0 = p2a(obj.get(), p0);                                        \
          auto prev = obj->prot()->getState<Communicator>()->getStats();       \
          auto tmp = OP##_ap(obj.get(), a0, p1);                               \
          auto cost =                                                          \
              obj->prot()->getState<Communicator>()->getStats() - prev;        \
          auto re = a2p(obj.get(), tmp);                                       \
          auto rp = OP##_pp(obj.get(), p0, p1);                                \
                                                                               \
          /* THEN */                                                           \
          EXPECT_VALUE_EQ(re, rp);                                             \
          EXPECT_TRUE(verifyCost(obj->prot()->getKernel(#OP "_ap"), #OP "_ap", \
                                 conf.field(), kShape, npc, cost));            \
        });                                                                    \
  }

#define TEST_ARITHMETIC_BINARY_OP(OP) \
  TEST_ARITHMETIC_BINARY_OP_AA(OP)    \
  TEST_ARITHMETIC_BINARY_OP_AP(OP)

TEST_ARITHMETIC_BINARY_OP(add)
TEST_ARITHMETIC_BINARY_OP(mul)

TEST_P(ArithmeticTest, SquareA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->prot()->hasKernel("square_a")) {
      return;
    }

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = square_a(obj.get(), a0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = square_p(obj.get(), p0);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("square_a"), "square_a",
                           conf.field(), kShape, npc, cost));
  });
}

TEST_P(ArithmeticTest, MulA1B) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->prot()->hasKernel("mul_a1b")) {
      return;
    }

    const size_t K = spu::SizeOf(conf.field()) * 8;

    /* GIVEN */
    auto p0 = rand_p(obj.get(), conf.protocol() == ProtocolKind::CHEETAH
                                    ? Shape({200, 26})
                                    : kShape);
    auto p1 = rand_p(obj.get(), conf.protocol() == ProtocolKind::CHEETAH
                                    ? Shape({200, 26})
                                    : kShape);
    p1 = rshift_p(obj.get(), p1, K - 1);
    auto a0 = p2a(obj.get(), p0);
    auto a1 = p2b(obj.get(), p1);
    // hint runtime this is a 1bit value.
    a1 = lshift_b(obj.get(), a1, K - 1);
    a1 = rshift_b(obj.get(), a1, K - 1);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = mul_a1b(obj.get(), a0, a1);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mul_a1b"), "mul_a1b",
                           conf.field(), kShape, npc, cost));
  });
}

TEST_P(ArithmeticTest, MulAV) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->prot()->hasKernel("mul_av")) {
      return;
    }

    const size_t K = spu::SizeOf(conf.field()) * 8;

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);
    p1 = rshift_p(obj.get(), p1, K - 1);
    auto a0 = p2a(obj.get(), p0);
    auto a1 = p2v(obj.get(), p1, 0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = mul_av(obj.get(), a0, a1).value();
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mul_av"), "mul_av",
                           conf.field(), kShape, npc, cost));
  });
}

TEST_P(ArithmeticTest, MulA1BV) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    // MulA1BV available for cheetah only for now.
    if (!obj->prot()->hasKernel("mul_a1bv")) {
      return;
    }

    const size_t K = spu::SizeOf(conf.field()) * 8;

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);
    p1 = rshift_p(obj.get(), p1, K - 1);
    auto a0 = p2a(obj.get(), p0);
    auto a1 = p2v(obj.get(), p1, 0);
    // hint runtime this is a 1bit value.
    a1 = lshift_v(obj.get(), a1, K - 1);
    a1 = rshift_v(obj.get(), a1, K - 1);
    // auto a1 = b2v(obj.get(), _a1, 0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto tmp = mul_a1bv(obj.get(), a0, a1).value();
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mul_a1bv"), "mul_a1bv",
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
    auto tmp = mmul_ap(obj.get(), a0, p1);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);

    auto r_pp = mmul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
    ce::Params params = {{"K", SizeOf(conf.field()) * 8},
                         {"N", npc},
                         {"m", M},
                         {"n", N},
                         {"k", K}};
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mmul_ap"), "mmul_ap", params,
                           cost, 1));
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
    auto tmp = mmul_aa(obj.get(), a0, a1);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
    ce::Params params = {{"K", SizeOf(conf.field()) * 8},
                         {"N", npc},
                         {"m", M},
                         {"n", N},
                         {"k", K}};
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mmul_aa"), "mmul_aa", params,
                           cost, 1));
  });
}

TEST_P(ArithmeticTest, MatMulAV) {
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
    if (not obj->hasKernel("mmul_av")) {
      return;
    }

    /* GIVEN */
    auto r = rand_p(obj.get(), shape_A);
    auto b0 = p2b(obj.get(), r);
    auto a0 = p2a(obj.get(), r);
    auto p1 = rand_p(obj.get(), shape_B);
    auto v1 = p2v(obj.get(), p1, 0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    // mmul_sv -> a * v
    auto tmp1 = mmul_sv(obj.get(), a0, v1);
    // mmul_sv -> b * v -> a * v
    auto tmp0 = mmul_sv(obj.get(), b0, v1);

    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r0_aa = a2p(obj.get(), tmp0);
    auto r1_aa = a2p(obj.get(), tmp1);
    auto r_pp = mmul_pp(obj.get(), b2p(obj.get(), b0), p1);

    /* THEN */
    EXPECT_VALUE_EQ(r0_aa, r_pp);
    EXPECT_VALUE_EQ(r1_aa, r_pp);
    ce::Params params = {{"K", SizeOf(conf.field()) * 8},
                         {"N", npc},
                         {"m", M},
                         {"n", N},
                         {"k", K}};
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("mmul_av"), "mmul_av", params,
                           cost, 1));
  });
}

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

  // ArrayRef p0_large =
  //     ring_rand_range(conf.field(), kShape, -(1 << 28), -(1 << 27));
  // ArrayRef p0_small = ring_rand_range(conf.field(), kShape, 1, 10000);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* kernel =
        static_cast<TruncAKernel*>(obj->prot()->getKernel("trunc_a"));

    auto p0 = rand_p(obj.get(), kShape);

    if (!kernel->hasMsbError()) {
      // trunc requires MSB to be zero.
      p0 = arshift_p(obj.get(), p0, 1);
    } else {
      // has msb error, only use lowest 10 bits.
      p0 = arshift_p(obj.get(), p0, SizeOf(conf.field()) * 8 - 10);
    }

    /* GIVEN */
    const size_t bits = 2;
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto a1 = trunc_a(obj.get(), a0, bits, SignType::Unknown);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    auto r_a = a2p(obj.get(), a1);
    auto r_p = arshift_p(obj.get(), p0, bits);

    /* THEN */
    EXPECT_VALUE_ALMOST_EQ(r_a, r_p, npc);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("trunc_a"), "trunc_a",
                           conf.field(), kShape, npc, cost));
  });
}

TEST_P(ArithmeticTest, P2A) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto a0 = p2a(obj.get(), p0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
    auto p1 = a2p(obj.get(), a0);

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("p2a"), "p2a", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(ArithmeticTest, A2P) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2a(obj.get(), p0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto p1 = a2p(obj.get(), a0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("a2p"), "a2p", conf.field(),
                           kShape, npc, cost));
  });
}

#define TEST_BOOLEAN_BINARY_OP_BB(OP)                                          \
  TEST_P(BooleanTest, OP##BB) {                                                \
    const auto factory = std::get<0>(GetParam());                              \
    const RuntimeConfig& conf = std::get<1>(GetParam());                       \
    const size_t npc = std::get<2>(GetParam());                                \
                                                                               \
    utils::simulate(                                                           \
        npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {           \
          auto obj = factory(conf, lctx);                                      \
                                                                               \
          /* GIVEN */                                                          \
          auto p0 = rand_p(obj.get(), kShape);                                 \
          auto p1 = rand_p(obj.get(), kShape);                                 \
                                                                               \
          /* WHEN */                                                           \
          auto b0 = p2b(obj.get(), p0);                                        \
          auto b1 = p2b(obj.get(), p1);                                        \
          auto prev = obj->prot()->getState<Communicator>()->getStats();       \
          auto tmp = OP##_bb(obj.get(), b0, b1);                               \
          auto cost =                                                          \
              obj->prot()->getState<Communicator>()->getStats() - prev;        \
          auto re = b2p(obj.get(), tmp);                                       \
          auto rp = OP##_pp(obj.get(), p0, p1);                                \
                                                                               \
          /* THEN */                                                           \
          EXPECT_VALUE_EQ(re, rp);                                             \
          EXPECT_TRUE(verifyCost(obj->prot()->getKernel(#OP "_bb"), #OP "_bb", \
                                 conf.field(), kShape, npc, cost));            \
        });                                                                    \
  }

#define TEST_BOOLEAN_BINARY_OP_BP(OP)                                          \
  TEST_P(BooleanTest, OP##BP) {                                                \
    const auto factory = std::get<0>(GetParam());                              \
    const RuntimeConfig& conf = std::get<1>(GetParam());                       \
    const size_t npc = std::get<2>(GetParam());                                \
                                                                               \
    utils::simulate(                                                           \
        npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {           \
          auto obj = factory(conf, lctx);                                      \
                                                                               \
          /* GIVEN */                                                          \
          auto p0 = rand_p(obj.get(), kShape);                                 \
          auto p1 = rand_p(obj.get(), kShape);                                 \
                                                                               \
          /* WHEN */                                                           \
          auto b0 = p2b(obj.get(), p0);                                        \
          auto prev = obj->prot()->getState<Communicator>()->getStats();       \
          auto tmp = OP##_bp(obj.get(), b0, p1);                               \
          auto cost =                                                          \
              obj->prot()->getState<Communicator>()->getStats() - prev;        \
          auto re = b2p(obj.get(), tmp);                                       \
          auto rp = OP##_pp(obj.get(), p0, p1);                                \
                                                                               \
          /* THEN */                                                           \
          EXPECT_VALUE_EQ(re, rp);                                             \
          EXPECT_TRUE(verifyCost(obj->prot()->getKernel(#OP "_bp"), #OP "_bp", \
                                 conf.field(), kShape, npc, cost));            \
        });                                                                    \
  }

#define TEST_BOOLEAN_BINARY_OP(OP) \
  TEST_BOOLEAN_BINARY_OP_BB(OP)    \
  TEST_BOOLEAN_BINARY_OP_BP(OP)

TEST_BOOLEAN_BINARY_OP(and)
TEST_BOOLEAN_BINARY_OP(xor)

#define TEST_UNARY_OP_WITH_BIT_B(OP)                                           \
  TEST_P(BooleanTest, OP##B) {                                                 \
    const auto factory = std::get<0>(GetParam());                              \
    const RuntimeConfig& conf = std::get<1>(GetParam());                       \
    const size_t npc = std::get<2>(GetParam());                                \
                                                                               \
    utils::simulate(                                                           \
        npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {           \
          auto obj = factory(conf, lctx);                                      \
                                                                               \
          /* GIVEN */                                                          \
          auto p0 = rand_p(obj.get(), kShape);                                 \
          auto b0 = p2b(obj.get(), p0);                                        \
                                                                               \
          for (auto bits : kShiftBits) {                                       \
            if (bits >= p0.elsize() * 8) {                                     \
              continue;                                                        \
            }                                                                  \
            /* WHEN */                                                         \
            auto prev = obj->prot()->getState<Communicator>()->getStats();     \
            auto tmp = OP##_b(obj.get(), b0, bits);                            \
            auto cost =                                                        \
                obj->prot()->getState<Communicator>()->getStats() - prev;      \
            auto r_b = b2p(obj.get(), tmp);                                    \
            auto r_p = OP##_p(obj.get(), p0, bits);                            \
                                                                               \
            /* THEN */                                                         \
            EXPECT_VALUE_EQ(r_b, r_p);                                         \
            EXPECT_TRUE(verifyCost(obj->prot()->getKernel(#OP "_b"), #OP "_b", \
                                   conf.field(), kShape, npc, cost));          \
          }                                                                    \
        });                                                                    \
  }

TEST_UNARY_OP_WITH_BIT_B(lshift)
TEST_UNARY_OP_WITH_BIT_B(rshift)
TEST_UNARY_OP_WITH_BIT_B(arshift)

TEST_P(BooleanTest, P2B) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto b0 = p2b(obj.get(), p0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
    auto p1 = b2p(obj.get(), b0);

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("p2b"), "p2b", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(BooleanTest, B2P) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto p1 = b2p(obj.get(), b0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("b2p"), "b2p", conf.field(),
                           kShape, npc, cost));
  });
}

TEST_P(BooleanTest, BitrevB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);

    for (size_t i = 0; i < SizeOf(conf.field()); i++) {
      for (size_t j = i; j < SizeOf(conf.field()); j++) {
        auto prev = obj->prot()->getState<Communicator>()->getStats();
        auto b1 = bitrev_b(obj.get(), b0, i, j);
        auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

        auto p1 = b2p(obj.get(), b1);
        auto pp1 = bitrev_p(obj.get(), p0, i, j);
        EXPECT_VALUE_EQ(p1, pp1);

        EXPECT_TRUE(verifyCost(obj->prot()->getKernel("bitrev_b"), "bitrev_b",
                               conf.field(), kShape, npc, cost));
      }
    }
  });
}

TEST_P(ConversionTest, A2B) {
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
    auto b1 = a2b(obj.get(), a0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("a2b"), "a2b", conf.field(),
                           kShape, npc, cost));
    EXPECT_VALUE_EQ(p0, b2p(obj.get(), b1));
  });
}

TEST_P(ConversionTest, B2A) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto b1 = a2b(obj.get(), a0);
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto a1 = b2a(obj.get(), b1);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("b2a"), "b2a", conf.field(),
                           kShape, npc, cost));
    EXPECT_VALUE_EQ(p0, a2p(obj.get(), a1));
  });
}

TEST_P(ConversionTest, MSB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->prot()->hasKernel("msb_a2b")) {
      return;
    }

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->prot()->getState<Communicator>()->getStats();
    auto b1 = msb_a2b(obj.get(), a0);
    auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_TRUE(verifyCost(obj->prot()->getKernel("msb_a2b"), "msb_a2b",
                           conf.field(), kShape, npc, cost));
    EXPECT_VALUE_EQ(rshift_p(obj.get(), p0, SizeOf(conf.field()) * 8 - 1),
                    b2p(obj.get(), b1));
  });
}

TEST_P(ConversionTest, EqualAA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->hasKernel("equal_aa")) {
      return;
    }
    /* GIVEN */
    // NOTE(lwj) for Cheetah, set a lager case to test the tield dispatch
    auto r0 = rand_p(obj.get(), conf.protocol() == ProtocolKind::CHEETAH
                                    ? Shape({10, 20, 30})
                                    : kShape);
    auto r1 = rand_p(obj.get(), conf.protocol() == ProtocolKind::CHEETAH
                                    ? Shape({10, 20, 30})
                                    : kShape);
    auto r2 = rand_p(obj.get(), conf.protocol() == ProtocolKind::CHEETAH
                                    ? Shape({10, 20, 30})
                                    : kShape);
    std::memcpy(r2.data().data(), r0.data().data(), 16);
    std::vector<Value> test_values = {r0, r1, r2};

    for (auto& test_value : test_values) {
      auto l_value = p2a(obj.get(), r0);
      auto r_value = p2a(obj.get(), test_value);
      auto prev = obj->prot()->getState<Communicator>()->getStats();
      auto tmp = dynDispatch(obj.get(), "equal_aa", l_value, r_value);
      auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
      auto out_value = b2p(obj.get(), tmp);
      auto t_value = equal_pp(obj.get(), r0, test_value);

      /* THEN */
      EXPECT_VALUE_EQ(out_value, t_value);
      EXPECT_TRUE(verifyCost(obj->prot()->getKernel("equal_aa"), "equal_aa",
                             conf.field(), kShape, npc, cost));
    }
  });
}

TEST_P(ConversionTest, EqualAP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](std::shared_ptr<yacl::link::Context> lctx) {
    auto obj = factory(conf, lctx);

    if (!obj->hasKernel("equal_ap")) {
      return;
    }
    /* GIVEN */
    auto r0 = rand_p(obj.get(), kShape);
    auto r1 = rand_p(obj.get(), kShape);
    auto r2 = rand_p(obj.get(), kShape);
    std::memcpy(r2.data().data(), r0.data().data(), 16);
    std::vector<Value> test_values = {r0, r1, r2};

    for (auto& test_value : test_values) {
      auto l_value = p2a(obj.get(), r0);
      auto r_value = test_value;
      auto prev = obj->prot()->getState<Communicator>()->getStats();
      auto tmp = dynDispatch(obj.get(), "equal_ap", l_value, r_value);
      auto cost = obj->prot()->getState<Communicator>()->getStats() - prev;
      auto out_value = b2p(obj.get(), tmp);
      auto t_value = equal_pp(obj.get(), r0, test_value);

      /* THEN */
      EXPECT_VALUE_EQ(out_value, t_value);
      EXPECT_TRUE(verifyCost(obj->prot()->getKernel("equal_ap"), "equal_ap",
                             conf.field(), kShape, npc, cost));
    }
  });
}

}  // namespace spu::mpc::test
