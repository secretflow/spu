// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {
template <typename T>
T makeBitsMask(size_t nbits) {
  size_t max = sizeof(T) * 8;
  if (nbits == 0) {
    nbits = max;
  }
  SPU_ENFORCE(nbits <= max);
  T mask = static_cast<T>(-1);
  if (nbits < max) {
    mask = (static_cast<T>(1) << nbits) - 1;
  }
  return mask;
}

class BasicOTProtTest
    : public ::testing::TestWithParam<std::tuple<FieldType, CheetahOtKind>> {
  void SetUp() override {
// FIXME: figure out what happened on Linux arm
#if defined(__linux__) && defined(__aarch64__)
    GTEST_SKIP() << "Skipping all tests on Linux arm";
#endif
  }
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, BasicOTProtTest,
    testing::Combine(
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(CheetahOtKind::EMP_Ferret, CheetahOtKind::YACL_Ferret,
                        CheetahOtKind::YACL_Softspoken)),
    [](const testing::TestParamInfo<BasicOTProtTest::ParamType>& p) {
      std::string ot_s;
      switch (std::get<1>(p.param)) {
        case CheetahOtKind::YACL_Softspoken:
          ot_s = "Yacl_ss";
          break;
        default:
        case CheetahOtKind::YACL_Ferret:
          ot_s = "Yacl_ferret";
          break;
        case CheetahOtKind::EMP_Ferret:
          ot_s = "Emp_ferret";
          break;
      }
      return fmt::format("{}Ot{}", std::get<0>(p.param), ot_s);
    });

TEST_P(BasicOTProtTest, SingleB2A) {
  size_t kWorldSize = 2;
  Shape shape = {10, 30};
  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());

  size_t nbits = 8 * SizeOf(field) - 1;
  size_t packed_nbits = 8 * SizeOf(field) - nbits;
  auto boolean_t = makeType<BShrTy>(field, packed_nbits);

  auto bshr0 = ring_rand(field, shape).as(boolean_t);
  auto bshr1 = ring_rand(field, shape).as(boolean_t);
  DISPATCH_ALL_FIELDS(field, [&]() {
    auto mask = static_cast<ring2k_t>(-1);
    if (nbits > 0) {
      mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
      NdArrayView<ring2k_t> xb0(bshr0);
      NdArrayView<ring2k_t> xb1(bshr1);
      pforeach(0, xb0.numel(), [&](int64_t i) {
        xb0[i] &= mask;
        xb1[i] &= mask;
      });
    }
  });

  NdArrayRef ashr0;
  NdArrayRef ashr1;
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn, ot_type);
    if (ctx->Rank() == 0) {
      ashr0 = ot_prot.B2A(bshr0);
    } else {
      ashr1 = ot_prot.B2A(bshr1);
    }
  });

  EXPECT_EQ(ashr0.shape(), ashr1.shape());
  EXPECT_EQ(shape, ashr0.shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> b0(bshr0);
    NdArrayView<ring2k_t> b1(bshr1);
    NdArrayView<ring2k_t> a0(ashr0);
    NdArrayView<ring2k_t> a1(ashr1);
    auto mask = static_cast<ring2k_t>(-1);
    if (nbits > 0) {
      mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
    }
    for (int64_t i = 0; i < shape.numel(); ++i) {
      ring2k_t e = b0[i] ^ b1[i];
      ring2k_t c = (a0[i] + a1[i]) & mask;
      EXPECT_EQ(e, c);
    }
  });
}

TEST_P(BasicOTProtTest, PackedB2A) {
  size_t kWorldSize = 2;
  Shape shape = {2};
  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());
  for (size_t nbits : {3, 8, 9}) {
    size_t packed_nbits = nbits;
    auto boolean_t = makeType<BShrTy>(field, packed_nbits);

    auto bshr0 = ring_rand(field, shape).as(boolean_t);
    auto bshr1 = ring_rand(field, shape).as(boolean_t);
    DISPATCH_ALL_FIELDS(field, [&]() {
      auto mask = static_cast<ring2k_t>(-1);
      if (nbits > 0) {
        mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
        NdArrayView<ring2k_t> xb0(bshr0);
        NdArrayView<ring2k_t> xb1(bshr1);
        pforeach(0, xb0.numel(), [&](int64_t i) {
          xb0[i] &= mask;
          xb1[i] &= mask;
        });
      }
    });

    NdArrayRef ashr0;
    NdArrayRef ashr1;
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn, ot_type);
      if (ctx->Rank() == 0) {
        ashr0 = ot_prot.B2A(bshr0);
      } else {
        ashr1 = ot_prot.B2A(bshr1);
      }
    });
    EXPECT_EQ(ashr0.shape(), ashr1.shape());
    EXPECT_EQ(ashr0.shape(), shape);

    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> b0(bshr0);
      NdArrayView<ring2k_t> b1(bshr1);
      NdArrayView<ring2k_t> a0(ashr0);
      NdArrayView<ring2k_t> a1(ashr1);
      auto mask = static_cast<ring2k_t>(-1);

      if (nbits > 0) {
        mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
      }

      for (int64_t i = 0; i < shape.numel(); ++i) {
        ring2k_t e = b0[i] ^ b1[i];
        ring2k_t c = (a0[i] + a1[i]) & mask;
        EXPECT_EQ(e, c);
      }
    });
  }
}

TEST_P(BasicOTProtTest, PackedB2AFull) {
  size_t kWorldSize = 2;
  Shape shape = {1L};

  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());
  for (size_t nbits : {0}) {
    size_t packed_nbits = 8 * SizeOf(field) - nbits;
    auto boolean_t = makeType<BShrTy>(field, packed_nbits);

    auto bshr0 = ring_rand(field, shape).as(boolean_t);
    auto bshr1 = ring_rand(field, shape).as(boolean_t);
    DISPATCH_ALL_FIELDS(field, [&]() {
      auto mask = static_cast<ring2k_t>(-1);
      if (nbits > 0) {
        mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
        auto xb0 = NdArrayView<ring2k_t>(bshr0);
        auto xb1 = NdArrayView<ring2k_t>(bshr1);
        pforeach(0, xb0.numel(), [&](int64_t i) {
          xb0[i] &= mask;
          xb1[i] &= mask;
        });
      }
    });

    NdArrayRef ashr0;
    NdArrayRef ashr1;
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn, ot_type);
      size_t sent = ctx->GetStats()->sent_bytes;
      if (ctx->Rank() == 0) {
        ashr0 = ot_prot.B2A(bshr0);
      } else {
        ashr1 = ot_prot.B2A(bshr1);
      }
      sent = ctx->GetStats()->sent_bytes - sent;
      printf("B2A sent %f byte per\n", sent * 1. / shape.numel());
    });

    EXPECT_EQ(ashr0.shape(), ashr1.shape());
    EXPECT_EQ(ashr0.shape(), shape);

    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> b0(bshr0);
      NdArrayView<ring2k_t> b1(bshr1);
      NdArrayView<ring2k_t> a0(ashr0);
      NdArrayView<ring2k_t> a1(ashr1);
      auto mask = static_cast<ring2k_t>(-1);
      if (nbits > 0) {
        mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
      }
      for (int64_t i = 0; i < shape.numel(); ++i) {
        ring2k_t e = b0[i] ^ b1[i];
        ring2k_t c = (a0[i] + a1[i]) & mask;
        EXPECT_EQ(e, c);
      }
    });
  }
}

TEST_P(BasicOTProtTest, AndTripleSparse) {
  size_t kWorldSize = 2;
  Shape shape = {51, 10};

  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());
  size_t max_bit = 8 * SizeOf(field);

  for (size_t sparse : {1UL, 7UL, max_bit - 1}) {
    const size_t target_nbits = max_bit - sparse;
    std::vector<NdArrayRef> triple[2];

    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn, ot_type);

      for (const auto& t : ot_prot.AndTriple(field, shape, target_nbits)) {
        triple[ctx->Rank()].emplace_back(t);
      }
    });

    DISPATCH_ALL_FIELDS(field, [&]() {
      ring2k_t max = static_cast<ring2k_t>(1) << target_nbits;
      NdArrayView<ring2k_t> a0(triple[0][0]);
      NdArrayView<ring2k_t> b0(triple[0][1]);
      NdArrayView<ring2k_t> c0(triple[0][2]);
      NdArrayView<ring2k_t> a1(triple[1][0]);
      NdArrayView<ring2k_t> b1(triple[1][1]);
      NdArrayView<ring2k_t> c1(triple[1][2]);

      for (int64_t i = 0; i < shape.numel(); ++i) {
        EXPECT_TRUE(a0[i] < max && a1[i] < max);
        EXPECT_TRUE(b0[i] < max && b1[i] < max);
        EXPECT_TRUE(c0[i] < max && c1[i] < max);

        ring2k_t e = (a0[i] ^ a1[i]) & (b0[i] ^ b1[i]);
        ring2k_t c = (c0[i] ^ c1[i]);
        EXPECT_EQ(e, c);
      }
    });
  }
}

TEST_P(BasicOTProtTest, BitwiseAnd) {
  size_t kWorldSize = 2;
  Shape shape = {55};
  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());
  int bw = SizeOf(field) * 8;
  auto boolean_t = makeType<BShrTy>(field, bw);

  NdArrayRef lhs[2];
  NdArrayRef rhs[2];
  NdArrayRef out[2];

  for (int i : {0, 1}) {
    lhs[i] = ring_rand(field, shape).as(boolean_t);
    rhs[i] = ring_rand(field, shape).as(boolean_t);
    DISPATCH_ALL_FIELDS(field, [&]() {
      ring2k_t mask = makeBitsMask<ring2k_t>(bw);
      NdArrayView<ring2k_t> L(lhs[i]);
      NdArrayView<ring2k_t> R(rhs[i]);

      pforeach(0, shape.numel(), [&](int64_t j) { L[j] &= mask; });
      pforeach(0, shape.numel(), [&](int64_t j) { R[j] &= mask; });
    });
  }

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn, ot_type);
    int r = ctx->Rank();
    out[r] = ot_prot.BitwiseAnd(lhs[r].clone(), rhs[r].clone());
  });

  auto expected = ring_and(ring_xor(lhs[0], lhs[1]), ring_xor(rhs[0], rhs[1]));
  auto got = ring_xor(out[0], out[1]);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> e(expected);
    NdArrayView<ring2k_t> g(got);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      ASSERT_EQ(e[i], g[i]);
    }
  });
}

TEST_P(BasicOTProtTest, AndTripleFull) {
  size_t kWorldSize = 2;
  Shape shape = {55, 11};

  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());
  std::vector<NdArrayRef> packed_triple[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn, ot_type);
    for (const auto& t : ot_prot.AndTriple(field, shape, SizeOf(field) * 8)) {
      packed_triple[ctx->Rank()].emplace_back(t);
    }
  });

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> a0(packed_triple[0][0]);
    NdArrayView<ring2k_t> b0(packed_triple[0][1]);
    NdArrayView<ring2k_t> c0(packed_triple[0][2]);
    NdArrayView<ring2k_t> a1(packed_triple[1][0]);
    NdArrayView<ring2k_t> b1(packed_triple[1][1]);
    NdArrayView<ring2k_t> c1(packed_triple[1][2]);

    size_t nn = a0.numel();
    EXPECT_TRUE(nn * 8 * SizeOf(field) >= (size_t)shape.numel());

    for (size_t i = 0; i < nn; ++i) {
      ring2k_t e = (a0[i] ^ a1[i]) & (b0[i] ^ b1[i]);
      ring2k_t c = (c0[i] ^ c1[i]);

      EXPECT_EQ(e, c);
    }
  });
}

TEST_P(BasicOTProtTest, Multiplexer) {
  size_t kWorldSize = 2;
  Shape shape = {3, 4, 1, 3};

  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());
  auto boolean_t = makeType<BShrTy>(field, 1);

  auto ashr0 = ring_rand(field, shape);
  auto ashr1 = ring_rand(field, shape);
  auto bshr0 = ring_rand(field, shape).as(boolean_t);
  auto bshr1 = ring_rand(field, shape).as(boolean_t);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto mask = static_cast<ring2k_t>(1);
    NdArrayView<ring2k_t> xb0(bshr0);
    NdArrayView<ring2k_t> xb1(bshr1);
    pforeach(0, xb0.numel(), [&](int64_t i) {
      xb0[i] &= mask;
      xb1[i] &= mask;
    });
  });

  NdArrayRef computed[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn, ot_type);
    if (ctx->Rank() == 0) {
      computed[0] = ot_prot.Multiplexer(ashr0, bshr0);
    } else {
      computed[1] = ot_prot.Multiplexer(ashr1, bshr1);
    }
  });

  EXPECT_EQ(computed[0].shape(), computed[1].shape());
  EXPECT_EQ(computed[0].shape(), shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> a0(ashr0);
    NdArrayView<ring2k_t> a1(ashr1);
    NdArrayView<ring2k_t> b0(bshr0);
    NdArrayView<ring2k_t> b1(bshr1);
    NdArrayView<ring2k_t> c0(computed[0]);
    NdArrayView<ring2k_t> c1(computed[1]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      ring2k_t msg = (a0[i] + a1[i]);
      ring2k_t sel = (b0[i] ^ b1[i]);
      ring2k_t exp = msg * sel;
      ring2k_t got = (c0[i] + c1[i]);
      EXPECT_EQ(exp, got);
    }
  });
}

TEST_P(BasicOTProtTest, CorrelatedAndTriple) {
  size_t kWorldSize = 2;
  Shape shape = {10 * 8};
  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());
  std::array<NdArrayRef, 5> corr_triple[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn, ot_type);
    corr_triple[ctx->Rank()] = ot_prot.CorrelatedAndTriple(field, shape);
  });

  EXPECT_EQ(corr_triple[0][0].shape(), corr_triple[1][0].shape());
  for (int i = 1; i < 5; ++i) {
    EXPECT_EQ(corr_triple[0][0].shape(), corr_triple[0][i].shape());
    EXPECT_EQ(corr_triple[1][0].shape(), corr_triple[1][i].shape());
  }

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto a0 = NdArrayView<ring2k_t>(corr_triple[0][0]);
    auto b0 = NdArrayView<ring2k_t>(corr_triple[0][1]);
    auto c0 = NdArrayView<ring2k_t>(corr_triple[0][2]);
    auto d0 = NdArrayView<ring2k_t>(corr_triple[0][3]);
    auto e0 = NdArrayView<ring2k_t>(corr_triple[0][4]);

    auto a1 = NdArrayView<ring2k_t>(corr_triple[1][0]);
    auto b1 = NdArrayView<ring2k_t>(corr_triple[1][1]);
    auto c1 = NdArrayView<ring2k_t>(corr_triple[1][2]);
    auto d1 = NdArrayView<ring2k_t>(corr_triple[1][3]);
    auto e1 = NdArrayView<ring2k_t>(corr_triple[1][4]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      EXPECT_TRUE(a0[i] < 2 && a1[i] < 2);
      EXPECT_TRUE(b0[i] < 2 && b1[i] < 2);
      EXPECT_TRUE(c0[i] < 2 && c1[i] < 2);
      EXPECT_TRUE(d0[i] < 2 && d1[i] < 2);
      EXPECT_TRUE(e0[i] < 2 && e1[i] < 2);

      ring2k_t e = (a0[i] ^ a1[i]) & (b0[i] ^ b1[i]);
      ring2k_t c = (c0[i] ^ c1[i]);
      EXPECT_EQ(e, c);

      e = (a0[i] ^ a1[i]) & (d0[i] ^ d1[i]);
      c = (e0[i] ^ e1[i]);
      EXPECT_EQ(e, c);
    }
  });
}

TEST_P(BasicOTProtTest, PrivateMulx) {
  size_t kWorldSize = 2;
  Shape shape = {3, 4, 1, 3};
  FieldType field = std::get<0>(GetParam());
  auto ot_type = std::get<1>(GetParam());

  auto boolean_t = makeType<BShrTy>(field, 1);

  auto ashr0 = ring_rand(field, shape);
  auto ashr1 = ring_rand(field, shape);
  auto choices = ring_rand(field, shape).as(boolean_t);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto mask = static_cast<ring2k_t>(1);
    NdArrayView<ring2k_t> xb(choices);
    pforeach(0, xb.numel(), [&](int64_t i) { xb[i] &= mask; });
  });

  NdArrayRef computed[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn, ot_type);
    if (ctx->Rank() == 0) {
      computed[0] = ot_prot.PrivateMulxSend(ashr0);
    } else {
      computed[1] = ot_prot.PrivateMulxRecv(ashr1, choices);
    }
  });

  EXPECT_EQ(computed[0].shape(), computed[1].shape());
  EXPECT_EQ(computed[0].shape(), shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> a0(ashr0);
    NdArrayView<ring2k_t> a1(ashr1);
    NdArrayView<ring2k_t> c(choices);
    NdArrayView<ring2k_t> c0(computed[0]);
    NdArrayView<ring2k_t> c1(computed[1]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      ring2k_t msg = (a0[i] + a1[i]);
      ring2k_t exp = msg * c[i];
      ring2k_t got = (c0[i] + c1[i]);
      EXPECT_EQ(exp, got);
    }
  });
}
}  // namespace spu::mpc::cheetah::test
