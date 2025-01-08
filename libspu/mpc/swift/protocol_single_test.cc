#include "libspu/mpc/swift/protocol_single_test.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/swift/arithmetic.h"
#include "libspu/mpc/swift/type.h"
#include "libspu/mpc/swift/value.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

Shape kShape = {2, 3};
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

TEST_P(ArithmeticTest, A2P_P2A) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    // auto rank = obj->prot()->getState<Communicator>()->getRank();

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    // auto p0 = ring_rand(conf.field(), kShape);

    /* WHEN */
    auto a0 = p2a(obj.get(), p0);
    // ring_print(p0.data());
    // if (rank == 0) {
    //   fmt::print("P_0:  ");
    //   ring_print(spu::mpc::swift::getFirstShare(a0.data()));
    //   fmt::print("\n");
    // }

    // if (rank == 1) {
    //   fmt::print("P_1:  ");
    //   ring_print(spu::mpc::swift::getFirstShare(a0.data()));
    //   fmt::print("\n");
    // }

    // if (rank == 2) {
    //   fmt::print("P_2:  ");
    //   ring_print(spu::mpc::swift::getFirstShare(a0.data()));
    //   fmt::print("\n");
    // }

    auto p1 = a2p(obj.get(), a0);
    // if (rank == 0) {
    //   fmt::print("output P_0:  ");
    //   ring_print(p1.data());
    //   fmt::print("\n");
    // }

    // if (rank == 1) {
    //   fmt::print("output P_1:  ");
    //   ring_print(p1.data());
    //   fmt::print("\n");
    // }

    // if (rank == 2) {
    //   fmt::print("output P_2:  ");
    //   ring_print(p1.data());
    //   fmt::print("\n");
    // }

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
  });
}

// TEST_P(ArithmeticTest, ShairngTest) {
//   const auto factory = std::get<0>(GetParam());
//   const RuntimeConfig& conf = std::get<1>(GetParam());
//   const size_t npc = std::get<2>(GetParam());

//   utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx)
//   {
//     auto obj = factory(conf, lctx);
//     // auto rank = obj->prot()->getState<Communicator>()->getRank();

//     /* GIVEN */
//     auto p0 = rand_p(obj.get(), kShape);

//     /* WHEN */
//     auto a0 = negate_a(obj.get(), p0);

//     // auto p1 = a2p(obj.get(), a0);

//     /* THEN */
//     // EXPECT_VALUE_EQ(p0, p1);
//   });
// }

TEST_P(ArithmeticTest, AddAP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2a(obj.get(), p0);

    auto tmp = add_ap(obj.get(), a0, p1);
    auto re = a2p(obj.get(), tmp);
    auto rp = add_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
  });
}

TEST_P(ArithmeticTest, AddAA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2a(obj.get(), p0);
    auto a1 = p2a(obj.get(), p1);

    auto tmp = add_aa(obj.get(), a0, a1);
    auto re = a2p(obj.get(), tmp);
    auto rp = add_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
  });
}

TEST_P(ArithmeticTest, MulAP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2a(obj.get(), p0);

    auto tmp = mul_ap(obj.get(), a0, p1);
    auto re = a2p(obj.get(), tmp);
    auto rp = mul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
  });
}

TEST_P(ArithmeticTest, MulAA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2a(obj.get(), p0);
    auto a1 = p2a(obj.get(), p1);

    auto tmp = mul_aa(obj.get(), a0, a1);
    auto re = a2p(obj.get(), tmp);
    auto rp = mul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
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
    auto tmp = mmul_ap(obj.get(), a0, p1);

    auto r_aa = a2p(obj.get(), tmp);

    auto r_pp = mmul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
  });
}

TEST_P(ArithmeticTest, MatMulAA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  const int64_t M = 3;
  const int64_t K = 4;
  const int64_t N = 5;
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
    auto tmp = mmul_aa(obj.get(), a0, a1);

    auto r_aa = a2p(obj.get(), tmp);
    auto r_pp = mmul_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(r_aa, r_pp);
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
      auto tmp = lshift_a(obj.get(), a0, {static_cast<int64_t>(bits)});
      auto r_b = a2p(obj.get(), tmp);
      auto r_p = lshift_p(obj.get(), p0, {static_cast<int64_t>(bits)});

      /* THEN */
      EXPECT_VALUE_EQ(r_b, r_p);
    }
  });
}

TEST_P(ArithmeticTest, NegateA) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    // auto rank = obj->prot()->getState<Communicator>()->getRank();

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto a0 = p2a(obj.get(), p0);
    auto neg_p0 = negate_p(obj.get(), p0);

    /* WHEN */
    auto r_a = negate_a(obj.get(), a0);

    auto r_p = a2p(obj.get(), r_a);
    auto r_pp = a2p(obj.get(), negate_a(obj.get(), a0));

    /* THEN */
    EXPECT_VALUE_EQ(r_p, r_pp);
    EXPECT_VALUE_EQ(r_p, neg_p0);
  });
}

// TEST_P(ArithmeticTest, TruncA) {
//   const auto factory = std::get<0>(GetParam());
//   const RuntimeConfig& conf = std::get<1>(GetParam());
//   const size_t npc = std::get<2>(GetParam());

//   // ArrayRef p0_large =
//   //     ring_rand_range(conf.field(), kShape, -(1 << 28), -(1 << 27));
//   // ArrayRef p0_small = ring_rand_range(conf.field(), kShape, 1, 10000);

//   utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx)
//   {
//     auto obj = factory(conf, lctx);
//     auto* kernel =
//         static_cast<TruncAKernel*>(obj->prot()->getKernel("trunc_a"));

//     auto p0 = rand_p(obj.get(), kShape);
//     // auto p0 = rand_p(obj.get(), {4, 5});

//     if (!kernel->hasMsbError()) {
//       // trunc requires MSB to be zero.
//       p0 = arshift_p(obj.get(), p0, {1});
//     } else {
//       // has msb error, only use lowest 10 bits.
//       p0 = arshift_p(obj.get(), p0,
//                      {static_cast<int64_t>(SizeOf(conf.field()) * 8 - 10)});
//     }

//     /* GIVEN */
//     const size_t bits = 2;
//     auto a0 = p2a(obj.get(), p0);

//     /* WHEN */
//     auto a1 = trunc_a(obj.get(), a0, bits, SignType::Unknown);

//     auto r_a = a2p(obj.get(), a1);
//     auto r_p = arshift_p(obj.get(), p0, {static_cast<int64_t>(bits)});

//     /* THEN */
//     EXPECT_VALUE_ALMOST_EQ(r_a, r_p, npc);
//   });
// }

// BooleanTest
TEST_P(BooleanTest, P2B_B2P) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto p1 = b2p(obj.get(), b0);

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
  });
}

TEST_P(BooleanTest, XorBP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2b(obj.get(), p0);

    auto tmp = xor_bp(obj.get(), a0, p1);
    auto re = b2p(obj.get(), tmp);
    auto rp = xor_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
  });
}

TEST_P(BooleanTest, XorBB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2b(obj.get(), p0);
    auto a1 = p2b(obj.get(), p1);

    auto tmp = xor_bb(obj.get(), a0, a1);
    auto re = b2p(obj.get(), tmp);
    auto rp = xor_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
  });
}

TEST_P(BooleanTest, AndBP) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2b(obj.get(), p0);

    auto tmp = and_bp(obj.get(), a0, p1);
    auto re = b2p(obj.get(), tmp);
    auto rp = and_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
  });
}

TEST_P(BooleanTest, AndBB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto p1 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto a0 = p2b(obj.get(), p0);
    auto a1 = p2b(obj.get(), p1);

    auto tmp = and_bb(obj.get(), a0, a1);
    auto re = b2p(obj.get(), tmp);
    auto rp = and_pp(obj.get(), p0, p1);

    /* THEN */
    EXPECT_VALUE_EQ(re, rp);
  });
}

TEST_P(BooleanTest, LshiftB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto b0 = p2b(obj.get(), p0);

    for (auto bits : kShiftBits) {
      if (bits >= p0.elsize() * 8) {
        continue;
      }
      /* WHEN */
      auto tmp = lshift_b(obj.get(), b0, {static_cast<int64_t>(bits)});
      auto r_b = b2p(obj.get(), tmp);
      auto r_p = lshift_p(obj.get(), p0, {static_cast<int64_t>(bits)});

      /* THEN */
      EXPECT_VALUE_EQ(r_b, r_p);
    }
  });
}

TEST_P(BooleanTest, RshiftB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto b0 = p2b(obj.get(), p0);

    for (auto bits : kShiftBits) {
      if (bits >= p0.elsize() * 8) {
        continue;
      }
      /* WHEN */
      auto tmp = rshift_b(obj.get(), b0, {static_cast<int64_t>(bits)});
      auto r_b = b2p(obj.get(), tmp);
      auto r_p = rshift_p(obj.get(), p0, {static_cast<int64_t>(bits)});

      /* THEN */
      EXPECT_VALUE_EQ(r_b, r_p);
    }
  });
}

TEST_P(BooleanTest, ARshiftB) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    auto b0 = p2b(obj.get(), p0);

    for (auto bits : kShiftBits) {
      if (bits >= p0.elsize() * 8) {
        continue;
      }
      /* WHEN */
      auto tmp = arshift_b(obj.get(), b0, {static_cast<int64_t>(bits)});
      auto r_b = b2p(obj.get(), tmp);
      auto r_p = arshift_p(obj.get(), p0, {static_cast<int64_t>(bits)});

      /* THEN */
      EXPECT_VALUE_EQ(r_b, r_p);
    }
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
        auto b1 = bitrev_b(obj.get(), b0, i, j);

        auto p1 = b2p(obj.get(), b1);
        auto pp1 = bitrev_p(obj.get(), p0, i, j);
        EXPECT_VALUE_EQ(p1, pp1);
      }
    }
  });
}

// ConversionTest
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
    auto b1 = a2b(obj.get(), a0);

    /* THEN */
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

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto a0 = b2a(obj.get(), b0);
    auto p1 = a2p(obj.get(), a0);

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
  });
}

TEST_P(ConversionTest, B2A_A2B) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);

    /* WHEN */
    auto b0 = p2b(obj.get(), p0);
    auto a0 = b2a(obj.get(), b0);
    auto b1 = a2b(obj.get(), a0);
    auto p1 = b2p(obj.get(), b1);

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
  });
}

TEST_P(ConversionTest, A2B_B2A) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    // auto rank = obj->prot()->getState<Communicator>()->getRank();

    /* GIVEN */
    auto p0 = rand_p(obj.get(), kShape);
    // p0 = add_pp(obj.get(), p0, negate_p(obj.get(), p0));
    auto a0 = p2a(obj.get(), p0);
    // ring_print(spu::mpc::swift::getFirstShare(a0.data()), "First Share");
    // ring_print(spu::mpc::swift::getSecondShare(a0.data()), "Second Share");
    // ring_print(spu::mpc::swift::getThirdShare(a0.data()), "Third Share");

    /* WHEN */
    auto b1 = a2b(obj.get(), a0);
    // if (rank == 0) {
    //   ring_print(spu::mpc::swift::getFirstShare(b1.data()),
    //              "(B) P0: First Share");
    //   ring_print(spu::mpc::swift::getSecondShare(b1.data()),
    //              "(B) P0: Second Share");
    //   ring_print(spu::mpc::swift::getThirdShare(b1.data()),
    //              "(B) P0: Third Share");
    // }

    // if (rank == 1) {
    //   ring_print(spu::mpc::swift::getFirstShare(b1.data()),
    //              "(B) P1: First Share");
    //   ring_print(spu::mpc::swift::getSecondShare(b1.data()),
    //              "(B) P1: Second Share");
    //   ring_print(spu::mpc::swift::getThirdShare(b1.data()),
    //              "(B) P1: Third Share");
    // }

    // if (rank == 2) {
    //   ring_print(spu::mpc::swift::getFirstShare(b1.data()),
    //              "(B) P2: First Share");
    //   ring_print(spu::mpc::swift::getSecondShare(b1.data()),
    //              "(B) P2: Second Share");
    //   ring_print(spu::mpc::swift::getThirdShare(b1.data()),
    //              "(B) P2: Third Share");
    // }

    auto a1 = b2a(obj.get(), b1);

    // if (rank == 0) {
    //   ring_print(spu::mpc::swift::getFirstShare(a1.data()),
    //              "(A) P0: First Share");
    //   ring_print(spu::mpc::swift::getSecondShare(a1.data()),
    //              "(A) P0: Second Share");
    //   ring_print(spu::mpc::swift::getThirdShare(a1.data()),
    //              "(A) P0: Third Share");
    // }

    // if (rank == 1) {
    //   ring_print(spu::mpc::swift::getFirstShare(a1.data()),
    //              "(A) P1: First Share");
    //   ring_print(spu::mpc::swift::getSecondShare(a1.data()),
    //              "(A) P1: Second Share");
    //   ring_print(spu::mpc::swift::getThirdShare(a1.data()),
    //              "(A) P1: Third Share");
    // }

    // if (rank == 2) {
    //   ring_print(spu::mpc::swift::getFirstShare(a1.data()),
    //              "(A) P2: First Share");
    //   ring_print(spu::mpc::swift::getSecondShare(a1.data()),
    //              "(A) P2: Second Share");
    //   ring_print(spu::mpc::swift::getThirdShare(a1.data()),
    //              "(A) P2: Third Share");
    // }
    auto p1 = a2p(obj.get(), a1);

    /* THEN */
    EXPECT_VALUE_EQ(p0, p1);
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
    auto b1 = msb_a2b(obj.get(), a0);

    /* THEN */
    EXPECT_VALUE_EQ(
        rshift_p(obj.get(), p0,
                 {static_cast<int64_t>(SizeOf(conf.field()) * 8 - 1)}),
        b2p(obj.get(), b1));
  });
}

}  // namespace
}  // namespace spu::mpc::test