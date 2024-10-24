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
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class BasicOTProtTest : public ::testing::TestWithParam<FieldType> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, BasicOTProtTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<BasicOTProtTest::ParamType>& p) {
      return fmt::format("{}", p.param);
    });

TEST_P(BasicOTProtTest, SingleB2A) {
  size_t kWorldSize = 2;
  Shape shape = {10, 30};
  FieldType field = GetParam();

  size_t nbits = 8 * SizeOf(field) - 1;
  size_t packed_nbits = 8 * SizeOf(field) - nbits;
  auto boolean_t = makeType<BShrTy>(field, packed_nbits);

  auto bshr0 = ring_rand(field, shape).as(boolean_t);
  auto bshr1 = ring_rand(field, shape).as(boolean_t);
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

  NdArrayRef ashr0;
  NdArrayRef ashr1;
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn);
    if (ctx->Rank() == 0) {
      ashr0 = ot_prot.B2A(bshr0);
    } else {
      ashr1 = ot_prot.B2A(bshr1);
    }
  });

  EXPECT_EQ(ashr0.shape(), ashr1.shape());
  EXPECT_EQ(shape, ashr0.shape());

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto b0 = xt_adapt<ring2k_t>(bshr0);
    auto b1 = xt_adapt<ring2k_t>(bshr1);
    auto a0 = xt_adapt<ring2k_t>(ashr0);
    auto a1 = xt_adapt<ring2k_t>(ashr1);
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
  Shape shape = {11, 12, 13};
  FieldType field = GetParam();

  for (size_t nbits : {1, 2}) {
    size_t packed_nbits = 8 * SizeOf(field) - nbits;
    auto boolean_t = makeType<BShrTy>(field, packed_nbits);

    auto bshr0 = ring_rand(field, shape).as(boolean_t);
    auto bshr1 = ring_rand(field, shape).as(boolean_t);
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

    NdArrayRef ashr0;
    NdArrayRef ashr1;
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn);
      if (ctx->Rank() == 0) {
        ashr0 = ot_prot.B2A(bshr0);
      } else {
        ashr1 = ot_prot.B2A(bshr1);
      }
    });
    EXPECT_EQ(ashr0.shape(), ashr1.shape());
    EXPECT_EQ(ashr0.shape(), shape);

    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto b0 = xt_adapt<ring2k_t>(bshr0);
      auto b1 = xt_adapt<ring2k_t>(bshr1);
      auto a0 = xt_adapt<ring2k_t>(ashr0);
      auto a1 = xt_adapt<ring2k_t>(ashr1);
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
  Shape shape = {1, 2, 3, 4, 5};
  FieldType field = GetParam();

  for (size_t nbits : {0}) {
    size_t packed_nbits = 8 * SizeOf(field) - nbits;
    auto boolean_t = makeType<BShrTy>(field, packed_nbits);

    auto bshr0 = ring_rand(field, shape).as(boolean_t);
    auto bshr1 = ring_rand(field, shape).as(boolean_t);
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

    NdArrayRef ashr0;
    NdArrayRef ashr1;
    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn);
      if (ctx->Rank() == 0) {
        ashr0 = ot_prot.B2A(bshr0);
      } else {
        ashr1 = ot_prot.B2A(bshr1);
      }
    });

    EXPECT_EQ(ashr0.shape(), ashr1.shape());
    EXPECT_EQ(ashr0.shape(), shape);

    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto b0 = xt_adapt<ring2k_t>(bshr0);
      auto b1 = xt_adapt<ring2k_t>(bshr1);
      auto a0 = xt_adapt<ring2k_t>(ashr0);
      auto a1 = xt_adapt<ring2k_t>(ashr1);
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
  Shape shape = {55, 100};
  FieldType field = GetParam();
  size_t max_bit = 8 * SizeOf(field);

  for (size_t sparse : {1UL, 7UL, max_bit - 1}) {
    const size_t target_nbits = max_bit - sparse;
    std::vector<NdArrayRef> triple[2];

    utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
      auto conn = std::make_shared<Communicator>(ctx);
      BasicOTProtocols ot_prot(conn);

      for (const auto& t : ot_prot.AndTriple(field, shape, target_nbits)) {
        triple[ctx->Rank()].emplace_back(t);
      }
    });

    DISPATCH_ALL_FIELDS(field, "", [&]() {
      ring2k_t max = static_cast<ring2k_t>(1) << target_nbits;
      auto a0 = xt_adapt<ring2k_t>(triple[0][0]);
      auto b0 = xt_adapt<ring2k_t>(triple[0][1]);
      auto c0 = xt_adapt<ring2k_t>(triple[0][2]);
      auto a1 = xt_adapt<ring2k_t>(triple[1][0]);
      auto b1 = xt_adapt<ring2k_t>(triple[1][1]);
      auto c1 = xt_adapt<ring2k_t>(triple[1][2]);

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

TEST_P(BasicOTProtTest, AndTripleFull) {
  size_t kWorldSize = 2;
  Shape shape = {55, 11};
  FieldType field = GetParam();

  std::vector<NdArrayRef> packed_triple[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn);
    for (const auto& t : ot_prot.AndTriple(field, shape, SizeOf(field) * 8)) {
      packed_triple[ctx->Rank()].emplace_back(t);
    }
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto a0 = xt_adapt<ring2k_t>(packed_triple[0][0]);
    auto b0 = xt_adapt<ring2k_t>(packed_triple[0][1]);
    auto c0 = xt_adapt<ring2k_t>(packed_triple[0][2]);
    auto a1 = xt_adapt<ring2k_t>(packed_triple[1][0]);
    auto b1 = xt_adapt<ring2k_t>(packed_triple[1][1]);
    auto c1 = xt_adapt<ring2k_t>(packed_triple[1][2]);

    size_t nn = a0.size();
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
  FieldType field = GetParam();

  auto boolean_t = makeType<BShrTy>(field, 1);

  auto ashr0 = ring_rand(field, shape);
  auto ashr1 = ring_rand(field, shape);
  auto bshr0 = ring_rand(field, shape).as(boolean_t);
  auto bshr1 = ring_rand(field, shape).as(boolean_t);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto mask = static_cast<ring2k_t>(1);
    auto xb0 = xt_mutable_adapt<ring2k_t>(bshr0);
    auto xb1 = xt_mutable_adapt<ring2k_t>(bshr1);
    std::transform(xb0.data(), xb0.data() + xb0.size(), xb0.data(),
                   [&](auto x) { return x & mask; });
    std::transform(xb1.data(), xb1.data() + xb1.size(), xb1.data(),
                   [&](auto x) { return x & mask; });
  });

  NdArrayRef computed[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn);
    if (ctx->Rank() == 0) {
      computed[0] = ot_prot.Multiplexer(ashr0, bshr0);
    } else {
      computed[1] = ot_prot.Multiplexer(ashr1, bshr1);
    }
  });

  EXPECT_EQ(computed[0].shape(), computed[1].shape());
  EXPECT_EQ(computed[0].shape(), shape);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto a0 = xt_adapt<ring2k_t>(ashr0);
    auto a1 = xt_adapt<ring2k_t>(ashr1);
    auto b0 = xt_adapt<ring2k_t>(bshr0);
    auto b1 = xt_adapt<ring2k_t>(bshr1);
    auto c0 = xt_adapt<ring2k_t>(computed[0]);
    auto c1 = xt_adapt<ring2k_t>(computed[1]);

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
  FieldType field = GetParam();

  std::array<NdArrayRef, 5> corr_triple[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    BasicOTProtocols ot_prot(conn);
    corr_triple[ctx->Rank()] = ot_prot.CorrelatedAndTriple(field, shape);
  });

  EXPECT_EQ(corr_triple[0][0].shape(), corr_triple[1][0].shape());
  for (int i = 1; i < 5; ++i) {
    EXPECT_EQ(corr_triple[0][0].shape(), corr_triple[0][i].shape());
    EXPECT_EQ(corr_triple[1][0].shape(), corr_triple[1][i].shape());
  }

  DISPATCH_ALL_FIELDS(field, "", [&]() {
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

}  // namespace spu::mpc::cheetah::test
