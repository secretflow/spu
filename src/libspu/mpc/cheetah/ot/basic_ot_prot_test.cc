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
#include "yacl/utils/elapsed_timer.h"

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

namespace {
const auto ot_kind = CheetahOtKind::YACL_Softspoken;

const std::vector<size_t> lut_bw_values = {
    // full ring
    8, 16, 32, 64, 128,
    // not full ring
    3, 5, 13, 24, 37, 48,  //
};

// all field can pass, reduce the running time.
// const std::vector<FieldType> all_fields = {FM8, FM16, FM32, FM64, FM128};
const std::vector<FieldType> all_fields = {FM8};

const std::vector<uint64_t> all_table_sizes = {4, 16, 64, 128, 256};

}  // namespace

class LUTProtTest : public ::testing::TestWithParam<
                        std::tuple<size_t, FieldType, FieldType, uint64_t>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, LUTProtTest,
    ::testing::Combine(::testing::ValuesIn(lut_bw_values),  // out_bw
                       ::testing::ValuesIn(all_fields),     // index
                       ::testing::ValuesIn(all_fields),
                       ::testing::ValuesIn(all_table_sizes)));  // table

TEST_P(LUTProtTest, LookUpTable) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};
  auto ot_type = ot_kind;

  const auto out_bw = std::get<0>(GetParam());
  const auto out_field = FixGetProperFiled(out_bw);

  const auto index_field = std::get<1>(GetParam());
  const auto table_field = std::get<2>(GetParam());

  const auto table_size = std::get<3>(GetParam());
  const auto N_bits = Log2Ceil(table_size);
  const auto N_mask = makeBitsMask<uint8_t>(N_bits);

  NdArrayRef index[2];
  index[0] = ring_rand(index_field, shape);
  index[1] = ring_rand(index_field, shape);

  auto table = ring_rand(table_field, {static_cast<int64_t>(table_size)});

  NdArrayRef out[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn, ot_type);
    out[rank] = ot_prot.LookUpTable(index[rank], table, out_bw);
  });

  // meta check
  EXPECT_EQ(out[0].shape(), out[1].shape());
  EXPECT_EQ(out[0].shape(), shape);
  EXPECT_EQ(out[0].eltype().as<Ring2k>()->field(),
            out[1].eltype().as<Ring2k>()->field());
  EXPECT_EQ(out[0].eltype().as<Ring2k>()->field(), FixGetProperFiled(out_bw));

  // debug
  // SPDLOG_INFO("n_bits: {}, n_mask {:0b}", N_bits, N_mask);
  // ring_print(index[0], "index[0]");
  // ring_print(index[1], "index[1]");
  // ring_print(table, "table");

  // value check
  DISPATCH_ALL_FIELDS(index_field, [&]() {
    using ind_el_t = ring2k_t;
    NdArrayView<ind_el_t> _ind1(index[0]);
    NdArrayView<ind_el_t> _ind2(index[1]);

    DISPATCH_ALL_FIELDS(table_field, [&]() {
      using tbl_el_t = ring2k_t;
      NdArrayView<tbl_el_t> _table(table);

      DISPATCH_ALL_FIELDS(out_field, [&]() {
        using out_el_t = ring2k_t;
        const auto msk = makeBitsMask<out_el_t>(out_bw);

        NdArrayView<out_el_t> _out1(out[0]);
        NdArrayView<out_el_t> _out2(out[1]);

        for (int64_t i = 0; i < shape.numel(); ++i) {
          auto idx = (_ind1[i] + _ind2[i]) & N_mask;
          auto exp = static_cast<out_el_t>(_table[idx]) & msk;
          out_el_t got = (_out1[i] + _out2[i]) & msk;
          EXPECT_EQ(exp, got);
        }
      });
    });
  });
}

class GeneralMuxProtTest : public ::testing::TestWithParam<
                               std::tuple<size_t, size_t, CheetahOtKind>> {
  void SetUp() override {
// FIXME: figure out what happened on Linux arm
#if defined(__linux__) && defined(__aarch64__)
    GTEST_SKIP() << "Skipping all tests on Linux arm";
#endif
  }
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, GeneralMuxProtTest,
    testing::Combine(testing::Values(7, 16, 23, 32, 64, 128),
                     testing::Values(7, 16, 23, 32, 64, 128),
                     testing::Values(CheetahOtKind::YACL_Ferret,
                                     CheetahOtKind::YACL_Softspoken)),
    [](const testing::TestParamInfo<GeneralMuxProtTest::ParamType>& p) {
      std::string ot_s;
      switch (std::get<2>(p.param)) {
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
      return fmt::format("{}_{}_Ot{}", std::get<0>(p.param),
                         std::get<1>(p.param), ot_s);
    });

TEST_P(GeneralMuxProtTest, MultiplexerNotUniformBw) {
  size_t kWorldSize = 2;
  // 4096
  Shape shape = {64, 64};

  auto src_bw = std::get<0>(GetParam());
  auto dest_bw = std::get<1>(GetParam());
  // only support large to small conversion
  if (dest_bw > src_bw) {
    return;
  }

  FieldType src_field = FixGetProperFiled(src_bw);
  FieldType dest_field = FixGetProperFiled(dest_bw);

  auto ot_type = std::get<2>(GetParam());
  auto boolean_t = makeType<BShrTy>(FM8, 1);

  auto ashr0 = ring_rand(src_field, shape);
  auto ashr1 = ring_rand(src_field, shape);

  // bshr type is not necessary, so we just fix to FM8
  auto bshr0 = ring_rand(FM8, shape).as(boolean_t);
  auto bshr1 = ring_rand(FM8, shape).as(boolean_t);

  DISPATCH_ALL_FIELDS(FM8, [&]() {
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

    auto meta = BasicOTProtocols::MultiplexMeta();
    meta.src_ring = src_field;
    meta.src_width = src_bw;
    meta.dst_ring = dest_field;
    meta.dst_width = dest_bw;

    size_t b0 = ctx->GetStats()->sent_bytes;
    size_t r0 = ctx->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    if (ctx->Rank() == 0) {
      computed[0] = ot_prot.Multiplexer(ashr0, bshr0, meta);
    } else {
      computed[1] = ot_prot.Multiplexer(ashr1, bshr1, meta);
    }

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = ctx->GetStats()->sent_bytes;
    size_t r1 = ctx->GetStats()->sent_actions;

    SPDLOG_INFO(
        "Rank {}, {} samples, [Mux(Ashr*Bshr), {} bits -> {} bits], sent {} "
        "bits per element. "
        "Actions total {}, elapsed total time: {} ms.",
        ctx->Rank(), shape.numel(), src_bw, dest_bw,
        (b1 - b0) * 8. / shape.numel(), (r1 - r0) * 1.0, pack_time);
  });

  EXPECT_EQ(computed[0].shape(), computed[1].shape());
  EXPECT_EQ(computed[0].shape(), shape);
  EXPECT_EQ(computed[0].eltype().as<RingTy>()->field(),
            computed[1].eltype().as<RingTy>()->field());
  EXPECT_EQ(computed[0].eltype().as<RingTy>()->field(), dest_field);
  EXPECT_EQ(computed[0].fxp_bits(), computed[1].fxp_bits());
  EXPECT_EQ(computed[0].fxp_bits(), dest_bw);

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using src_el_t = ring2k_t;
    const auto src_mask = makeBitsMask<src_el_t>(src_bw);

    NdArrayView<src_el_t> a0(ashr0);
    NdArrayView<src_el_t> a1(ashr1);

    NdArrayView<uint8_t> b0(bshr0);
    NdArrayView<uint8_t> b1(bshr1);

    DISPATCH_ALL_FIELDS(dest_field, [&]() {
      using dest_el_t = ring2k_t;
      const auto dest_mask = makeBitsMask<dest_el_t>(dest_bw);

      NdArrayView<dest_el_t> c0(computed[0]);
      NdArrayView<dest_el_t> c1(computed[1]);

      for (int64_t i = 0; i < shape.numel(); ++i) {
        src_el_t msg = (a0[i] + a1[i]) & src_mask;
        uint8_t sel = (b0[i] ^ b1[i]) & 1;
        auto exp = static_cast<dest_el_t>((msg * sel) & dest_mask);
        auto got = (c0[i] + c1[i]) & dest_mask;
        EXPECT_EQ(exp, got);
      }
    });
  });
}

namespace {

const std::vector<std::tuple<int64_t, int64_t>> bit_and_bw_values = {
    // full ring test
    std::make_tuple(8, 8), std::make_tuple(16, 16), std::make_tuple(32, 32),
    std::make_tuple(64, 64), std::make_tuple(128, 128),
    // not full ring test
    std::make_tuple(3, 8), std::make_tuple(8, 3),      //
    std::make_tuple(13, 15), std::make_tuple(15, 13),  //
    std::make_tuple(5, 5), std::make_tuple(13, 13),    //
};
}  // namespace

class GeneralAndProtTest
    : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 13;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(Cheetah, GeneralAndProtTest,
                         testing::ValuesIn(bit_and_bw_values));

TEST_P(GeneralAndProtTest, Work) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  int64_t m;
  int64_t n;
  std::tie(m, n) = GetParam();

  auto field = FixGetProperFiled(m);
  auto field_y = FixGetProperFiled(n);

  SPU_ENFORCE(field == field_y, "only support same field");

  const auto bw = std::min(m, n);

  auto ty_x = makeType<BShrTy>(field, m);
  auto ty_y = makeType<BShrTy>(field, n);
  auto ty_ret = makeType<BShrTy>(field, bw);

  NdArrayRef lhs[2];
  NdArrayRef rhs[2];
  NdArrayRef out[2];

  for (int i : {0, 1}) {
    lhs[i] = ring_rand(field, shape);
    ring_reduce_(lhs[i], m);
    lhs[i] = lhs[i].as(ty_x);

    rhs[i] = ring_rand(field, shape);
    ring_reduce_(rhs[i], n);
    rhs[i] = rhs[i].as(ty_y);
  }

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lcxt) {
    int rank = lcxt->Rank();
    auto conn = std::make_shared<Communicator>(lcxt);
    BasicOTProtocols ot_prot(conn, CheetahOtKind::YACL_Softspoken);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    out[rank] = ot_prot.BitwiseAnd(lhs[rank], rhs[rank]);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    SPDLOG_INFO(
        "Rank {}, [Bitwise And {}x{} bits], sent {} bits per element. Actions "
        "total {}, elapsed total time: {} ms.",
        rank, m, n, (b1 - b0) * 8. / numel, (r1 - r0) * 1.0, pack_time);

    // check
    SPU_ENFORCE(out[rank].eltype().as<BShrTy>()->nbits() ==
                static_cast<size_t>(bw));
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

}  // namespace spu::mpc::cheetah::test
