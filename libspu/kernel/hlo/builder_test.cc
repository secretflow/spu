// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "builder.h"
#include "capi.h"
#include "gtest/gtest.h"

#include "libspu/core/context.h"
#include "libspu/core/memref.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

SemanticType GetSemanticType(int64_t field) {
  switch (field) {
    case 32:
      return SE_I32;
    case 64:
      return SE_I64;
    case 128:
      return SE_I128;
  }
  return SE_INVALID;
}

class BinaryTest
    : public ::testing::TestWithParam<std::tuple<size_t, ProtocolKind>> {};

#define BINARY_EMPTY_TEST(_op_)                                               \
  TEST_P(BinaryTest, Empty_##_op_) {                                          \
    auto field = std::get<0>(GetParam());                                     \
    auto prot = std::get<1>(GetParam());                                      \
    mpc::utils::simulate(                                                     \
        3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {            \
          HloBuilder builder;                                                 \
          auto empty_c0 = builder.Seal(builder.Constant(1, {}));              \
          auto empty_c1 = builder.Seal(builder.Constant(1, {}));              \
          auto output = builder._op_(empty_c0, empty_c1);                     \
          builder.compile({output});                                          \
          SPUContext sctx = test::makeSPUContext(prot, field, lctx);          \
          auto s_empty = builder.execute(&sctx);                              \
          EXPECT_EQ(s_empty.size(), 1);                                       \
          EXPECT_EQ(s_empty[0].numel(), 1);                                   \
          EXPECT_EQ(s_empty[0].shape().size(), 0);                            \
        });                                                                   \
  }                                                                           \
                                                                              \
  TEST_P(BinaryTest, Empty_CAPI_##_op_) {                                     \
    auto field = std::get<0>(GetParam());                                     \
    auto prot = std::get<1>(GetParam());                                      \
    mpc::utils::simulate(                                                     \
        3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {            \
          SPUContext sctx = test::makeSPUContext(prot, field, lctx);          \
                                                                              \
          auto builder = spuHloBuilderCreate();                               \
                                                                              \
          PtBufferView view = {1};                                            \
          Shape shape = {};                                                   \
                                                                              \
          auto empty_c0 = spuHloSeal(                                         \
              builder,                                                        \
              spuHloConstant(builder, SpuHloPtBufferView{&view},              \
                             SpuHloShape{shape.data(), shape.size()}));       \
          auto empty_c1 = spuHloSeal(                                         \
              builder,                                                        \
              spuHloConstant(builder, SpuHloPtBufferView{&view},              \
                             SpuHloShape{shape.data(), shape.size()}));       \
                                                                              \
          auto output = spuHlo##_op_(builder, empty_c0, empty_c1);            \
                                                                              \
          spuHloCompile(builder, MlirValueArray{&output, 1});                 \
          auto s_empty = spuHloExecute(builder, SpuHloRtContext{&sctx},       \
                                       SpuHloValueArray{NULL, 0});            \
                                                                              \
          EXPECT_EQ(s_empty.size, 1);                                         \
          EXPECT_EQ(                                                          \
              static_cast<const spu::MemRef *>(s_empty.data[0].ptr)->numel(), \
              1);                                                             \
          EXPECT_EQ(static_cast<const spu::MemRef *>(s_empty.data[0].ptr)     \
                        ->shape()                                             \
                        .size(),                                              \
                    0);                                                       \
                                                                              \
          spuHloBuilderDestroy(builder);                                      \
          spuHloValueDestroy(s_empty.data[0]);                                \
        });                                                                   \
  }

BINARY_EMPTY_TEST(Add)
BINARY_EMPTY_TEST(Equal)
BINARY_EMPTY_TEST(NotEqual)
BINARY_EMPTY_TEST(LessEqual)
BINARY_EMPTY_TEST(GreaterEqual)
BINARY_EMPTY_TEST(Sub)
BINARY_EMPTY_TEST(Less)
BINARY_EMPTY_TEST(Greater)
BINARY_EMPTY_TEST(Mul)
BINARY_EMPTY_TEST(Max)
BINARY_EMPTY_TEST(Min)
BINARY_EMPTY_TEST(And)
BINARY_EMPTY_TEST(Or)
BINARY_EMPTY_TEST(Xor)
BINARY_EMPTY_TEST(Div)
BINARY_EMPTY_TEST(Remainder)

INSTANTIATE_TEST_SUITE_P(
    BinaryTestInstances, BinaryTest,
    testing::Combine(testing::Values(64, 128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<BinaryTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

class UnaryTest
    : public ::testing::TestWithParam<std::tuple<size_t, ProtocolKind>> {};

#define UNARY_EMPTY_TEST(_op_, _init_)                               \
  TEST_P(UnaryTest, Empty_##_op_) {                                  \
    auto field = std::get<0>(GetParam());                            \
    auto prot = std::get<1>(GetParam());                             \
    mpc::utils::simulate(                                            \
        3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {   \
          HloBuilder builder;                                        \
          auto input = builder.Seal(builder.Constant(_init_, {}));   \
          auto output = builder._op_(input);                         \
          builder.compile({output});                                 \
          SPUContext sctx = test::makeSPUContext(prot, field, lctx); \
          auto s_empty = builder.execute(&sctx);                     \
          EXPECT_EQ(s_empty.size(), 1);                              \
          EXPECT_EQ(s_empty[0].numel(), 1);                          \
          EXPECT_EQ(s_empty[0].shape().size(), 0);                   \
        });                                                          \
  }

UNARY_EMPTY_TEST(Not, 1U)
UNARY_EMPTY_TEST(Sine, 1.0)
UNARY_EMPTY_TEST(Cosine, 1.0)

INSTANTIATE_TEST_SUITE_P(
    UnaryTestInstances, UnaryTest,
    testing::Combine(testing::Values(64, 128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<UnaryTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

class GeoTest
    : public ::testing::TestWithParam<std::tuple<size_t, ProtocolKind>> {};

TEST_P(GeoTest, Empty_Pad) {
  size_t field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HloBuilder builder;

        auto in = builder.Constant(std::vector<uint8_t>(2, 0), {2});
        auto pv = builder.Constant(static_cast<uint8_t>(0), {});
        auto out = builder.Pad(in, pv, {0}, {1}, {0});

        builder.compile({out});
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto ret = builder.execute(&sctx);

        EXPECT_EQ(ret.size(), 1);
        EXPECT_EQ(ret[0].numel(), 3);
        EXPECT_EQ(ret[0].vtype(), VIS_PUBLIC);
        EXPECT_EQ(ret[0].eltype().semantic_type(), SE_U8);
      });
}

TEST_P(GeoTest, Empty_Concatenate) {
  size_t field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HloBuilder builder;

        auto in0 = builder.Constant(std::vector<uint8_t>(2, 0), {2});
        auto in1 = builder.Constant(std::vector<uint8_t>(2, 1), {2});
        auto in2 = builder.Constant(std::vector<uint8_t>(2, 2), {2});
        auto out = builder.Concatenate({in0, in1, in2}, 0);

        builder.compile({out});
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto ret = builder.execute(&sctx);

        EXPECT_EQ(ret.size(), 1);
        EXPECT_EQ(ret[0].numel(), 6);
        EXPECT_EQ(ret[0].vtype(), VIS_PUBLIC);
        EXPECT_EQ(ret[0].eltype().semantic_type(), SE_U8);
      });
}

TEST_P(GeoTest, Empty_Slice) {
  size_t field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HloBuilder builder;

        auto in = builder.Constant(std::vector<uint8_t>{1, 2, 3, 4, 5, 6}, {6});
        auto out = builder.Slice(in, {0}, {5}, {2});

        builder.compile({out});
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto ret = builder.execute(&sctx);

        EXPECT_EQ(ret.size(), 1);
        EXPECT_EQ(ret[0].numel(), 3);
        EXPECT_EQ(ret[0].shape().size(), 1);
        EXPECT_EQ(ret[0].shape()[0], 3);
      });
}

INSTANTIATE_TEST_SUITE_P(
    GeoTestInstances, GeoTest,
    testing::Combine(testing::Values(64, 128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<GeoTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

class TernaryTest
    : public ::testing::TestWithParam<std::tuple<size_t, ProtocolKind>> {};

TEST_P(TernaryTest, Empty_Select) {
  size_t field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HloBuilder builder;

        auto empty_p = builder.Constant(true, {});
        auto empty_true = builder.Constant(1, {});
        auto empty_false = builder.Constant(2, {});
        auto out = builder.Select(empty_p, empty_true, empty_false);

        builder.compile({out});
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto ret = builder.execute(&sctx);

        EXPECT_EQ(ret.size(), 1);
        EXPECT_EQ(ret[0].numel(), 1);
        EXPECT_EQ(ret[0].shape().size(), 0);
      });
}

TEST_P(TernaryTest, Empty_SimpleSort) {
  size_t field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HloBuilder builder;

        std::vector<float> k1 = {7, 6, 5, 5, 4, 4, 4, 1, 3, 3};
        std::vector<float> k2 = {1, 2, 3, 6, 7, 6, 5, 2, 1, 2};

        auto val_k1 = builder.Constant(k1, {static_cast<int64_t>(k1.size())});
        auto val_k2 = builder.Constant(k2, {static_cast<int64_t>(k2.size())});
        auto outputs = builder.SimpleSort({val_k1, val_k2}, 0,
                                          HloBuilder::SortDirection::ASC, 2);

        builder.compile(outputs);
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto ret = builder.execute(&sctx);

        EXPECT_EQ(ret.size(), 2);
        EXPECT_EQ(ret[0].numel(), k1.size());
        EXPECT_EQ(ret[1].numel(), k2.size());
        EXPECT_EQ(ret[0].shape().size(), 1);
        EXPECT_EQ(ret[1].shape().size(), 1);
      });
}

TEST_P(TernaryTest, Empty_Reduce) {
  size_t field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HloBuilder builder;

        std::vector<int64_t> inputs = {0, 1, 2, 3, 4, 5};
        int64_t init = 0;

        auto val_input = builder.Constant(inputs, {6});
        auto val_init = builder.Constant(init, {});
        auto output = builder.Reduce({val_input}, {val_init}, {0},
                                     HloBuilder::REDUCE_SUM);

        builder.compile({output});
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto ret = builder.execute(&sctx);

        EXPECT_EQ(ret.size(), 1);
        EXPECT_EQ(ret[0].numel(), 1);
        EXPECT_EQ(ret[0].shape().size(), 0);
      });
}

TEST_P(TernaryTest, Empty_Shuffle) {
  size_t field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  // FIXME: remove it when all protocol backends support shuffle op/intrinsic
  if (prot != ProtocolKind::SEMI2K && prot != ProtocolKind::ABY3) {
    return;  // FIXME
  }

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        HloBuilder builder;

        std::vector<int64_t> inputs = {0, 1, 2, 3, 4, 5};
        auto val_input = builder.Constant(inputs, {6});
        auto output = builder.Shuffle({val_input}, 0);

        builder.compile({output});
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);
        auto ret = builder.execute(&sctx);

        EXPECT_EQ(ret.size(), 1);
        EXPECT_EQ(ret[0].numel(), 6);
        EXPECT_EQ(ret[0].shape().size(), 1);
      });
}

TEST_P(TernaryTest, Empty_FilterByMask) {
  size_t field = std::get<0>(GetParam());
  ProtocolKind prot = std::get<1>(GetParam());

  mpc::utils::simulate(
      3, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);

        MemRef a(makeType<RingTy>(GetSemanticType(field), field), {5});

        auto *data_ptr = a.data<int64_t>();
        std::iota(data_ptr, data_ptr + 5, 0);

        MemRefView<int64_t> _a(a);
        for (int64_t idx = 0; idx < 5; ++idx) {
          auto v = _a[idx];
          EXPECT_EQ(v, idx);
        }

        HloBuilder builder;

        std::vector<uint8_t> mask = {0U, 1U, 1U, 0U, 0U};
        auto output = builder.FilterByMask(
            builder.Constant(PtBufferView(a.data(), PT_I64, {5}, {}), {5}),
            absl::Span<const uint8_t>(mask.data(), mask.size()));

        builder.compile({output});
        auto ret = builder.execute(&sctx);

        EXPECT_EQ(ret.size(), 1);
        EXPECT_EQ(ret[0].shape()[0], 2);
        EXPECT_EQ(ret[0].at<int64_t>(0), 1);
        EXPECT_EQ(ret[0].at<int64_t>(1), 2);
      });
}

INSTANTIATE_TEST_SUITE_P(
    TernaryTestInstances, TernaryTest,
    testing::Combine(testing::Values(64, 128),
                     testing::Values(ProtocolKind::REF2K, ProtocolKind::SEMI2K,
                                     ProtocolKind::ABY3)),
    [](const testing::TestParamInfo<TernaryTest::ParamType> &p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

}  // namespace spu::kernel::hlo
