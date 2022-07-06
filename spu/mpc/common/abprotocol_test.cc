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

#include "spu/mpc/common/abprotocol_test.h"

#include "spu/core/shape_util.h"
#include "spu/mpc/interfaces.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"
#include "spu/mpc/util/simulate.h"

namespace spu::mpc::test {
namespace {

constexpr int64_t kNumel = 1000;
const std::vector<size_t> kShiftBits = {0, 1, 2, 31, 32, 33, 64, 1000};

bool verifyCost(Kernel* kernel, std::string_view name, FieldType field,
                size_t numel, size_t npc, const Communicator::Stats& cost) {
  if (kernel->kind() == Kernel::Kind::kDynamic) {
    return true;
  }

  auto comm = kernel->comm();
  auto latency = kernel->latency();

  bool succeed = true;
  constexpr size_t kBitsPerBytes = 8;
  if (comm->eval(field, npc) * numel != cost.comm * kBitsPerBytes) {
    fmt::print("Failed: {} comm mismatch, expected={}, got={}\n", name,
               comm->eval(field, npc) * numel, cost.comm * kBitsPerBytes);
    succeed = false;
  }
  if (latency->eval(field, npc) != cost.latency) {
    fmt::print("Failed: {} latency mismatch, expected={}, got={}\n", name,
               latency->eval(field, npc), cost.latency);
    succeed = false;
  }

  return succeed;
}

}  // namespace

#define TEST_ARITHMETIC_BINARY_OP_AA(OP)                                  \
  TEST_P(ArithmeticTest, OP##AA) {                                        \
    const auto factory = std::get<0>(GetParam());                         \
    const size_t npc = std::get<1>(GetParam());                           \
    const FieldType field = std::get<2>(GetParam());                      \
                                                                          \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {  \
      auto obj = factory(lctx);                                           \
                                                                          \
      /* GIVEN */                                                         \
      auto p0 = rand_p(obj.get(), field, kNumel);                         \
      auto p1 = rand_p(obj.get(), field, kNumel);                         \
                                                                          \
      /* WHEN */                                                          \
      auto a0 = p2a(obj.get(), p0);                                       \
      auto a1 = p2a(obj.get(), p1);                                       \
      auto prev = obj->getState<Communicator>()->getStats();              \
      auto tmp = OP##_aa(obj.get(), a0, a1);                              \
      auto cost = obj->getState<Communicator>()->getStats() - prev;       \
      auto re = a2p(obj.get(), tmp);                                      \
      auto rp = OP##_pp(obj.get(), p0, p1);                               \
                                                                          \
      /* THEN */                                                          \
      EXPECT_TRUE(ring_all_equal(re, rp));                                \
      EXPECT_TRUE(verifyCost(obj->getKernel(#OP "_aa"), #OP "_aa", field, \
                             kNumel, npc, cost));                         \
    });                                                                   \
  }

#define TEST_ARITHMETIC_BINARY_OP_AP(OP)                                  \
  TEST_P(ArithmeticTest, OP##AP) {                                        \
    const auto factory = std::get<0>(GetParam());                         \
    const size_t npc = std::get<1>(GetParam());                           \
    const FieldType field = std::get<2>(GetParam());                      \
                                                                          \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {  \
      auto obj = factory(lctx);                                           \
                                                                          \
      /* GIVEN */                                                         \
      auto p0 = rand_p(obj.get(), field, kNumel);                         \
      auto p1 = rand_p(obj.get(), field, kNumel);                         \
                                                                          \
      /* WHEN */                                                          \
      auto a0 = p2a(obj.get(), p0);                                       \
      auto prev = obj->getState<Communicator>()->getStats();              \
      auto tmp = OP##_ap(obj.get(), a0, p1);                              \
      auto cost = obj->getState<Communicator>()->getStats() - prev;       \
      auto re = a2p(obj.get(), tmp);                                      \
      auto rp = OP##_pp(obj.get(), p0, p1);                               \
                                                                          \
      /* THEN */                                                          \
      EXPECT_TRUE(ring_all_equal(re, rp));                                \
      EXPECT_TRUE(verifyCost(obj->getKernel(#OP "_ap"), #OP "_ap", field, \
                             kNumel, npc, cost));                         \
    });                                                                   \
  }

#define TEST_ARITHMETIC_BINARY_OP(OP) \
  TEST_ARITHMETIC_BINARY_OP_AA(OP)    \
  TEST_ARITHMETIC_BINARY_OP_AP(OP)

TEST_ARITHMETIC_BINARY_OP(add)
TEST_ARITHMETIC_BINARY_OP(mul)

TEST_P(ArithmeticTest, MatMulAP) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const std::vector<int64_t> shape_A{M, K};
  const std::vector<int64_t> shape_B{K, N};
  const std::vector<int64_t> shape_C{M, N};

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, calcNumel(shape_A));
    auto p1 = rand_p(obj.get(), field, calcNumel(shape_B));
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->getState<Communicator>()->getStats();
    auto tmp = mmul_ap(obj.get(), a0, p1, M, N, K);
    auto cost = obj->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1, M, N, K);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_aa, r_pp));
    EXPECT_TRUE(verifyCost(obj->getKernel("mmul_ap"), "mmul_ap", field,
                           calcNumel(shape_C), npc, cost));
  });
}

TEST_P(ArithmeticTest, MatMulAA) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 3;
  const std::vector<int64_t> shape_A{M, K};
  const std::vector<int64_t> shape_B{K, N};
  const std::vector<int64_t> shape_C{M, N};

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, calcNumel(shape_A));
    auto p1 = rand_p(obj.get(), field, calcNumel(shape_B));
    auto a0 = p2a(obj.get(), p0);
    auto a1 = p2a(obj.get(), p1);

    /* WHEN */
    auto prev = obj->getState<Communicator>()->getStats();
    auto tmp = mmul_aa(obj.get(), a0, a1, M, N, K);
    auto cost = obj->getState<Communicator>()->getStats() - prev;

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1, M, N, K);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_aa, r_pp));
    EXPECT_TRUE(verifyCost(obj->getKernel("mmul_aa"), "mmul_aa", field,
                           calcNumel(shape_C), npc, cost));
  });
}

TEST_P(ArithmeticTest, NotA) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->getState<Communicator>()->getStats();
    auto r_a = not_a(obj.get(), a0);
    auto cost = obj->getState<Communicator>()->getStats() - prev;

    auto r_p = a2p(obj.get(), r_a);
    auto r_pp = a2p(obj.get(), not_a(obj.get(), a0));

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_p, r_pp));
    EXPECT_TRUE(
        verifyCost(obj->getKernel("not_a"), "not_a", field, kNumel, npc, cost));
  });
}

TEST_P(ArithmeticTest, LShiftA) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);
    auto b0 = p2a(obj.get(), p0);

    for (auto bits : kShiftBits) {
      if (bits >= p0.elsize() * 8) {
        // Shift more than elsize is a UB
        continue;
      }
      /* WHEN */
      auto prev = obj->getState<Communicator>()->getStats();
      auto tmp = lshift_a(obj.get(), b0, bits);
      auto cost = obj->getState<Communicator>()->getStats() - prev;
      auto r_b = a2p(obj.get(), tmp);
      auto r_p = lshift_p(obj.get(), p0, bits);

      /* THEN */
      EXPECT_TRUE(ring_all_equal(r_b, r_p));
      EXPECT_TRUE(verifyCost(obj->getKernel("lshift_a"), "lshift_a", field,
                             kNumel, npc, cost));
    }
  });
}

TEST_P(ArithmeticTest, TruncPrA) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  ArrayRef p0_large = ring_rand_range(field, kNumel, -(1 << 28), -(1 << 27));
  ArrayRef p0_small = ring_rand_range(field, kNumel, 1, 10000);

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);
    ArrayRef p0;
    if (static_cast<TruncPrAKernel*>(obj->getKernel("truncpr_a"))
            ->isPrecise()) {
      p0 = p0_large;
    } else {
      p0 = p0_small;
    }

    /* GIVEN */
    const size_t bits = 2;
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->getState<Communicator>()->getStats();
    auto a1 = truncpr_a(obj.get(), a0, bits);
    auto cost = obj->getState<Communicator>()->getStats() - prev;

    auto r_a = a2p(obj.get(), a1);
    auto r_p = arshift_p(obj.get(), p0, bits);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(r_a, r_p, npc));
    EXPECT_TRUE(verifyCost(obj->getKernel("truncpr_a"), "truncpr_a", field,
                           kNumel, npc, cost));
  });
}

TEST_P(ArithmeticTest, P2A) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);

    /* WHEN */
    auto prev = obj->getState<Communicator>()->getStats();
    auto a0 = p2a(obj.get(), p0);
    auto cost = obj->getState<Communicator>()->getStats() - prev;
    auto p1 = a2p(obj.get(), a0);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(p0, p1));
    EXPECT_TRUE(
        verifyCost(obj->getKernel("p2a"), "p2a", field, kNumel, npc, cost));
  });
}

TEST_P(ArithmeticTest, A2P) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);

    /* WHEN */
    auto a0 = p2a(obj.get(), p0);
    auto prev = obj->getState<Communicator>()->getStats();
    auto p1 = a2p(obj.get(), a0);
    auto cost = obj->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_TRUE(ring_all_equal(p0, p1));
    EXPECT_TRUE(
        verifyCost(obj->getKernel("a2p"), "a2p", field, kNumel, npc, cost));
  });
}

#define TEST_BOOLEAN_BINARY_OP_BB(OP)                                     \
  TEST_P(BooleanTest, OP##BB) {                                           \
    const auto factory = std::get<0>(GetParam());                         \
    const size_t npc = std::get<1>(GetParam());                           \
    const FieldType field = std::get<2>(GetParam());                      \
                                                                          \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {  \
      auto obj = factory(lctx);                                           \
                                                                          \
      /* GIVEN */                                                         \
      auto p0 = rand_p(obj.get(), field, kNumel);                         \
      auto p1 = rand_p(obj.get(), field, kNumel);                         \
                                                                          \
      /* WHEN */                                                          \
      auto b0 = p2b(obj.get(), p0);                                       \
      auto b1 = p2b(obj.get(), p1);                                       \
      auto prev = obj->getState<Communicator>()->getStats();              \
      auto tmp = OP##_bb(obj.get(), b0, b1);                              \
      auto cost = obj->getState<Communicator>()->getStats() - prev;       \
      auto re = b2p(obj.get(), tmp);                                      \
      auto rp = OP##_pp(obj.get(), p0, p1);                               \
                                                                          \
      /* THEN */                                                          \
      EXPECT_TRUE(ring_all_equal(re, rp));                                \
      EXPECT_TRUE(verifyCost(obj->getKernel(#OP "_bb"), #OP "_bb", field, \
                             kNumel, npc, cost));                         \
    });                                                                   \
  }

#define TEST_BOOLEAN_BINARY_OP_BP(OP)                                     \
  TEST_P(BooleanTest, OP##BP) {                                           \
    const auto factory = std::get<0>(GetParam());                         \
    const size_t npc = std::get<1>(GetParam());                           \
    const FieldType field = std::get<2>(GetParam());                      \
                                                                          \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {  \
      auto obj = factory(lctx);                                           \
                                                                          \
      /* GIVEN */                                                         \
      auto p0 = rand_p(obj.get(), field, kNumel);                         \
      auto p1 = rand_p(obj.get(), field, kNumel);                         \
                                                                          \
      /* WHEN */                                                          \
      auto b0 = p2b(obj.get(), p0);                                       \
      auto prev = obj->getState<Communicator>()->getStats();              \
      auto tmp = OP##_bp(obj.get(), b0, p1);                              \
      auto cost = obj->getState<Communicator>()->getStats() - prev;       \
      auto re = b2p(obj.get(), tmp);                                      \
      auto rp = OP##_pp(obj.get(), p0, p1);                               \
                                                                          \
      /* THEN */                                                          \
      EXPECT_TRUE(ring_all_equal(re, rp));                                \
      EXPECT_TRUE(verifyCost(obj->getKernel(#OP "_bp"), #OP "_bp", field, \
                             kNumel, npc, cost));                         \
    });                                                                   \
  }

#define TEST_BOOLEAN_BINARY_OP(OP) \
  TEST_BOOLEAN_BINARY_OP_BB(OP)    \
  TEST_BOOLEAN_BINARY_OP_BP(OP)

TEST_BOOLEAN_BINARY_OP(and)
TEST_BOOLEAN_BINARY_OP(xor)

#define TEST_UNARY_OP_WITH_BIT_B(OP)                                      \
  TEST_P(BooleanTest, OP##B) {                                            \
    const auto factory = std::get<0>(GetParam());                         \
    const size_t npc = std::get<1>(GetParam());                           \
    const FieldType field = std::get<2>(GetParam());                      \
                                                                          \
    util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {  \
      auto obj = factory(lctx);                                           \
                                                                          \
      /* GIVEN */                                                         \
      auto p0 = rand_p(obj.get(), field, kNumel);                         \
      auto b0 = p2b(obj.get(), p0);                                       \
                                                                          \
      for (auto bits : kShiftBits) {                                      \
        if (bits >= p0.elsize() * 8) {                                    \
          continue;                                                       \
        }                                                                 \
        /* WHEN */                                                        \
        auto prev = obj->getState<Communicator>()->getStats();            \
        auto tmp = OP##_b(obj.get(), b0, bits);                           \
        auto cost = obj->getState<Communicator>()->getStats() - prev;     \
        auto r_b = b2p(obj.get(), tmp);                                   \
        auto r_p = OP##_p(obj.get(), p0, bits);                           \
                                                                          \
        /* THEN */                                                        \
        EXPECT_TRUE(ring_all_equal(r_b, r_p));                            \
        EXPECT_TRUE(verifyCost(obj->getKernel(#OP "_b"), #OP "_b", field, \
                               kNumel, npc, cost));                       \
      }                                                                   \
    });                                                                   \
  }

TEST_UNARY_OP_WITH_BIT_B(lshift)
TEST_UNARY_OP_WITH_BIT_B(rshift)
TEST_UNARY_OP_WITH_BIT_B(arshift)

TEST_P(BooleanTest, P2B) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);

    /* WHEN */
    auto prev = obj->getState<Communicator>()->getStats();
    auto b0 = p2b(obj.get(), p0);
    auto cost = obj->getState<Communicator>()->getStats() - prev;
    auto p1 = b2p(obj.get(), b0);

    /* THEN */
    EXPECT_TRUE(ring_all_equal(p0, p1));
    EXPECT_TRUE(
        verifyCost(obj->getKernel("p2b"), "p2b", field, kNumel, npc, cost));
  });
}

TEST_P(BooleanTest, B2P) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto prev = obj->getState<Communicator>()->getStats();
    auto p1 = b2p(obj.get(), b0);
    auto cost = obj->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_TRUE(ring_all_equal(p0, p1));
    EXPECT_TRUE(
        verifyCost(obj->getKernel("b2p"), "b2p", field, kNumel, npc, cost));
  });
}

TEST_P(BooleanTest, BitrevB) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);

    for (size_t i = 0; i < SizeOf(field); i++) {
      for (size_t j = i; j < SizeOf(field); j++) {
        auto prev = obj->getState<Communicator>()->getStats();
        auto b1 = bitrev_b(obj.get(), b0, i, j);
        auto cost = obj->getState<Communicator>()->getStats() - prev;

        auto p1 = b2p(obj.get(), b1);
        auto pp1 = bitrev_p(obj.get(), p0, i, j);
        EXPECT_TRUE(ring_all_equal(p1, pp1));

        EXPECT_TRUE(verifyCost(obj->getKernel("bitrev_b"), "bitrev_b", field,
                               kNumel, npc, cost));
      }
    }
  });
}

TEST_P(ConversionTest, A2B) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto prev = obj->getState<Communicator>()->getStats();
    auto b1 = a2b(obj.get(), a0);
    auto cost = obj->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_TRUE(
        verifyCost(obj->getKernel("a2b"), "a2b", field, kNumel, npc, cost));
    EXPECT_TRUE(ring_all_equal(p0, b2p(obj.get(), b1)));
  });
}

TEST_P(ConversionTest, B2A) {
  const auto factory = std::get<0>(GetParam());
  const size_t npc = std::get<1>(GetParam());
  const FieldType field = std::get<2>(GetParam());

  util::simulate(npc, [&](std::shared_ptr<yasl::link::Context> lctx) {
    auto obj = factory(lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), field, kNumel);
    auto a0 = p2a(obj.get(), p0);

    /* WHEN */
    auto b1 = a2b(obj.get(), a0);
    auto prev = obj->getState<Communicator>()->getStats();
    auto a1 = b2a(obj.get(), b1);
    auto cost = obj->getState<Communicator>()->getStats() - prev;

    /* THEN */
    EXPECT_TRUE(
        verifyCost(obj->getKernel("b2a"), "b2a", field, kNumel, npc, cost));
    EXPECT_TRUE(ring_all_equal(p0, a2p(obj.get(), a1)));
  });
}

}  // namespace spu::mpc::test
