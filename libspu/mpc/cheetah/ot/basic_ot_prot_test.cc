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

#include "libspu/core/xt_helper.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class BasicOTProtTest : public ::testing::TestWithParam<FieldType> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, BasicOTProtTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<BasicOTProtTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(BasicOTProtTest, SingleB2A) {
  size_t kWorldSize = 2;
  size_t n = 7;
  FieldType field = GetParam();

  size_t nbits = 8 * SizeOf(field) - 1;
  size_t packed_nbits = 8 * SizeOf(field) - nbits;
  auto boolean_t = makeType<semi2k::BShrTy>(field, packed_nbits);

  ArrayRef bshr0 = ring_rand(field, n).as(boolean_t);
  ArrayRef bshr1 = ring_rand(field, n).as(boolean_t);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto mask = static_cast<ring2k_t>(-1);
    if (nbits > 0) {
      mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
      auto xb0 = xt_mutable_adapt<ring2k_t>(bshr0);
      auto xb1 = xt_mutable_adapt<ring2k_t>(bshr1);
      std::transform(xb0.data(), xb0.data() + xb0.size(), xb0.data(),
                     [&](auto x) { return x & mask; });
      std::transform(xb1.data(), xb1.data() + xb1.size(), xb1.data(),
                     [&](auto x) { return x & mask; });
    }
  });

  ArrayRef ashr0, ashr1;
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn);
    if (ctx->Rank() == 0) {
      ashr0 = ot_prot.B2A(bshr0);
    } else {
      ashr1 = ot_prot.B2A(bshr1);
    }
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto b0 = xt_adapt<ring2k_t>(bshr0);
    auto b1 = xt_adapt<ring2k_t>(bshr1);
    auto a0 = xt_adapt<ring2k_t>(ashr0);
    auto a1 = xt_adapt<ring2k_t>(ashr1);
    auto mask = static_cast<ring2k_t>(-1);
    if (nbits > 0) {
      mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
    }
    for (size_t i = 0; i < n; ++i) {
      ring2k_t e = b0[i] ^ b1[i];
      ring2k_t c = (a0[i] + a1[i]) & mask;
      EXPECT_EQ(e, c);
    }
  });
}

TEST_P(BasicOTProtTest, PackedB2A) {
  size_t kWorldSize = 2;
  size_t n = 7;
  FieldType field = GetParam();

  for (size_t nbits : {1, 2}) {
    size_t packed_nbits = 8 * SizeOf(field) - nbits;
    auto boolean_t = makeType<semi2k::BShrTy>(field, packed_nbits);

    ArrayRef bshr0 = ring_rand(field, n).as(boolean_t);
    ArrayRef bshr1 = ring_rand(field, n).as(boolean_t);
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto mask = static_cast<ring2k_t>(-1);
      if (nbits > 0) {
        mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
        auto xb0 = xt_mutable_adapt<ring2k_t>(bshr0);
        auto xb1 = xt_mutable_adapt<ring2k_t>(bshr1);
        std::transform(xb0.data(), xb0.data() + xb0.size(), xb0.data(),
                       [&](auto x) { return x & mask; });
        std::transform(xb1.data(), xb1.data() + xb1.size(), xb1.data(),
                       [&](auto x) { return x & mask; });
      }
    });

    ArrayRef ashr0, ashr1;
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn);
      if (ctx->Rank() == 0) {
        ashr0 = ot_prot.B2A(bshr0);
      } else {
        ashr1 = ot_prot.B2A(bshr1);
      }
    });

    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto b0 = xt_adapt<ring2k_t>(bshr0);
      auto b1 = xt_adapt<ring2k_t>(bshr1);
      auto a0 = xt_adapt<ring2k_t>(ashr0);
      auto a1 = xt_adapt<ring2k_t>(ashr1);
      auto mask = static_cast<ring2k_t>(-1);
      if (nbits > 0) {
        mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
      }
      for (size_t i = 0; i < n; ++i) {
        ring2k_t e = b0[i] ^ b1[i];
        ring2k_t c = (a0[i] + a1[i]) & mask;
        EXPECT_EQ(e, c);
      }
    });
  }
}

TEST_P(BasicOTProtTest, PackedB2AFull) {
  size_t kWorldSize = 2;
  size_t n = 7;
  FieldType field = GetParam();

  for (size_t nbits : {0}) {
    size_t packed_nbits = 8 * SizeOf(field) - nbits;
    auto boolean_t = makeType<semi2k::BShrTy>(field, packed_nbits);

    ArrayRef bshr0 = ring_rand(field, n).as(boolean_t);
    ArrayRef bshr1 = ring_rand(field, n).as(boolean_t);
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto mask = static_cast<ring2k_t>(-1);
      if (nbits > 0) {
        mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
        auto xb0 = xt_mutable_adapt<ring2k_t>(bshr0);
        auto xb1 = xt_mutable_adapt<ring2k_t>(bshr1);
        std::transform(xb0.data(), xb0.data() + xb0.size(), xb0.data(),
                       [&](auto x) { return x & mask; });
        std::transform(xb1.data(), xb1.data() + xb1.size(), xb1.data(),
                       [&](auto x) { return x & mask; });
      }
    });

    ArrayRef ashr0, ashr1;
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn);
      if (ctx->Rank() == 0) {
        ashr0 = ot_prot.B2A(bshr0);
      } else {
        ashr1 = ot_prot.B2A(bshr1);
      }
    });

    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto b0 = xt_adapt<ring2k_t>(bshr0);
      auto b1 = xt_adapt<ring2k_t>(bshr1);
      auto a0 = xt_adapt<ring2k_t>(ashr0);
      auto a1 = xt_adapt<ring2k_t>(ashr1);
      auto mask = static_cast<ring2k_t>(-1);
      if (nbits > 0) {
        mask = (static_cast<ring2k_t>(1) << packed_nbits) - 1;
      }
      for (size_t i = 0; i < n; ++i) {
        ring2k_t e = b0[i] ^ b1[i];
        ring2k_t c = (a0[i] + a1[i]) & mask;
        EXPECT_EQ(e, c);
      }
    });
  }
}

TEST_P(BasicOTProtTest, AndTripleSparse) {
  size_t kWorldSize = 2;
  size_t n = 55;
  FieldType field = GetParam();
  size_t max_bit = 8 * SizeOf(field);

  for (size_t sparse : {1UL, 7UL, max_bit - 1}) {
    const size_t target_nbits = max_bit - sparse;
    std::array<ArrayRef, 3> triple[2];

    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn);
      triple[ctx->Rank()] = ot_prot.AndTriple(field, n, target_nbits);
    });

    DISPATCH_ALL_FIELDS(field, "", [&]() {
      ring2k_t max = static_cast<ring2k_t>(1) << target_nbits;
      auto a0 = xt_adapt<ring2k_t>(triple[0][0]);
      auto b0 = xt_adapt<ring2k_t>(triple[0][1]);
      auto c0 = xt_adapt<ring2k_t>(triple[0][2]);
      auto a1 = xt_adapt<ring2k_t>(triple[1][0]);
      auto b1 = xt_adapt<ring2k_t>(triple[1][1]);
      auto c1 = xt_adapt<ring2k_t>(triple[1][2]);

      for (size_t i = 0; i < n; ++i) {
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

TEST_P(BasicOTProtTest, AndTripleFull) {
  size_t kWorldSize = 2;
  size_t n = 55;
  FieldType field = GetParam();

  std::array<ArrayRef, 3> packed_triple[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn);
    packed_triple[ctx->Rank()] = ot_prot.AndTriple(field, n, SizeOf(field) * 8);
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto a0 = xt_adapt<ring2k_t>(packed_triple[0][0]);
    auto b0 = xt_adapt<ring2k_t>(packed_triple[0][1]);
    auto c0 = xt_adapt<ring2k_t>(packed_triple[0][2]);
    auto a1 = xt_adapt<ring2k_t>(packed_triple[1][0]);
    auto b1 = xt_adapt<ring2k_t>(packed_triple[1][1]);
    auto c1 = xt_adapt<ring2k_t>(packed_triple[1][2]);

    size_t nn = a0.size();
    EXPECT_TRUE(nn * 8 * SizeOf(field) >= n);

    for (size_t i = 0; i < nn; ++i) {
      ring2k_t e = (a0[i] ^ a1[i]) & (b0[i] ^ b1[i]);
      ring2k_t c = (c0[i] ^ c1[i]);

      EXPECT_EQ(e, c);
    }
  });
}

TEST_P(BasicOTProtTest, Multiplexer) {
  size_t kWorldSize = 2;
  size_t n = 7;
  FieldType field = GetParam();

  auto boolean_t = makeType<semi2k::BShrTy>(field, 1);

  ArrayRef ashr0 = ring_rand(field, n);
  ArrayRef ashr1 = ring_rand(field, n);
  ArrayRef bshr0 = ring_rand(field, n).as(boolean_t);
  ArrayRef bshr1 = ring_rand(field, n).as(boolean_t);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto mask = static_cast<ring2k_t>(1);
    auto xb0 = xt_mutable_adapt<ring2k_t>(bshr0);
    auto xb1 = xt_mutable_adapt<ring2k_t>(bshr1);
    std::transform(xb0.data(), xb0.data() + xb0.size(), xb0.data(),
                   [&](auto x) { return x & mask; });
    std::transform(xb1.data(), xb1.data() + xb1.size(), xb1.data(),
                   [&](auto x) { return x & mask; });
  });

  ArrayRef computed[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn);
    if (ctx->Rank() == 0) {
      computed[0] = ot_prot.Multiplexer(ashr0, bshr0);
    } else {
      computed[1] = ot_prot.Multiplexer(ashr1, bshr1);
    }
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto a0 = xt_adapt<ring2k_t>(ashr0);
    auto a1 = xt_adapt<ring2k_t>(ashr1);
    auto b0 = xt_adapt<ring2k_t>(bshr0);
    auto b1 = xt_adapt<ring2k_t>(bshr1);
    auto c0 = xt_adapt<ring2k_t>(computed[0]);
    auto c1 = xt_adapt<ring2k_t>(computed[1]);

    for (size_t i = 0; i < n; ++i) {
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
  size_t n = 10;
  FieldType field = GetParam();

  std::array<ArrayRef, 5> corr_triple[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn);
    corr_triple[ctx->Rank()] = ot_prot.CorrelatedAndTriple(field, n);
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto a0 = ArrayView<ring2k_t>(corr_triple[0][0]);
    auto b0 = ArrayView<ring2k_t>(corr_triple[0][1]);
    auto c0 = ArrayView<ring2k_t>(corr_triple[0][2]);
    auto d0 = ArrayView<ring2k_t>(corr_triple[0][3]);
    auto e0 = ArrayView<ring2k_t>(corr_triple[0][4]);

    auto a1 = ArrayView<ring2k_t>(corr_triple[1][0]);
    auto b1 = ArrayView<ring2k_t>(corr_triple[1][1]);
    auto c1 = ArrayView<ring2k_t>(corr_triple[1][2]);
    auto d1 = ArrayView<ring2k_t>(corr_triple[1][3]);
    auto e1 = ArrayView<ring2k_t>(corr_triple[1][4]);

    for (size_t i = 0; i < n; ++i) {
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

}  // namespace spu::mpc::cheetah::test
