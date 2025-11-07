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

#include "libspu/mpc/cheetah/nonlinear/truncate_and_reduce_prot.h"

#include <type_traits>

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class TRProtocolTest
    : public ::testing::TestWithParam<std::tuple<FieldType, bool, bool>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 12;
  // static constexpr int64_t kBenchSize = 1LL << 2;

  static constexpr int64_t kDefaultTRBits = 4;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, TRProtocolTest,
    testing::Combine(
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
        testing::Values(true, false),
        testing::Values(true, false)),  // whether dest bw is dynamic
    [](const testing::TestParamInfo<TRProtocolTest::ParamType> &p) {
      return fmt::format("{}x{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param) ? "Exact" : "Approx",
                         std::get<2>(p.param) ? "Dynamic" : "Fixed");
    }

);

namespace {

template <typename T>
typename std::make_unsigned<T>::type makeMask(size_t bw) {
  using U = typename std::make_unsigned<T>::type;
  if (bw == sizeof(U) * 8) {
    return static_cast<U>(-1);
  }
  return (static_cast<U>(1) << bw) - 1;
}

void MaskItInplace(NdArrayRef a, size_t width) {
  auto field = a.eltype().as<Ring2k>()->field();
  if (width == SizeOf(field) * 8) {
    return;
  }
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _a(a);
    ring2k_t msk = (static_cast<ring2k_t>(1) << width) - 1;
    pforeach(0, _a.numel(), [&](int64_t i) { _a[i] &= msk; });
  });
}

// 2**tr_bits, 2**tr_bits+1, ... , 0, -1, -2, ...
[[maybe_unused]] NdArrayRef ConstructUnderlyingArray(FieldType field,
                                                     const Shape &shape,
                                                     int64_t tr_bits) {
  NdArrayRef x = ring_iota(field, shape, static_cast<int64_t>(1) << tr_bits);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> x_(x);

    const auto start = shape.numel() / 2;
    const auto end = shape.numel();

    pforeach(start, end, [&](int64_t idx) { x_[idx] = start - idx; });
  });

  return x;
}

template <typename T>
bool safe_check(const T exp, const T got, int64_t bw) {
  if (exp == 0) {
    if (got == 0 || got == makeMask<T>(bw)) {
      return true;
    }
  } else {
    if (exp - got <= 1) {
      return true;
    }
  }

  return false;
}

template <typename T, typename U>
bool safe_check(const T exp, const U got, int64_t bw) {
  return safe_check<T>(exp, static_cast<T>(got), bw);
}

}  // namespace

TEST_P(TRProtocolTest, NotSameField) {
  size_t kWorldSize = 2;

  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  bool exact = std::get<1>(GetParam());
  bool dynamic = std::get<2>(GetParam());

  FieldType dst_field;
  // no fewer ring than fm32 now
  if (src_field == FM32 || !dynamic) {
    return;
  }
  if (src_field == FM64) {
    dst_field = FM32;
  } else {
    dst_field = FM64;
  }

  const auto src_width = SizeOf(src_field) * 8;
  size_t kTR_bits;
  if (dynamic) {
    kTR_bits = src_width / 2;
  } else {
    kTR_bits = kDefaultTRBits;
  }

  const auto dest_width = src_width - kTR_bits;

  // NdArrayRef expected = ConstructUnderlyingArray(src_field, shape, kTR_bits);
  NdArrayRef expected = ring_rand(src_field, shape);

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_sub(expected, inp[0]);

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    RingTruncateAndReduceProtocol prot(base);
    RingTruncateAndReduceProtocol::Meta meta;

    meta.exact = exact;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = src_width;
    meta.dst_width = dest_width;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    [[maybe_unused]] size_t r0 = ctx->GetStats()->sent_actions;

    // without wrap
    yacl::ElapsedTimer pack_timer;
    oup[rank] = prot.Compute(inp[rank], NdArrayRef(), meta);
    double pack_time = pack_timer.CountMs() * 1.0;

    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    [[maybe_unused]] size_t r1 = ctx->GetStats()->sent_actions;

    SPDLOG_INFO(
        "[TR]: rank {}, test size: {} ,tr {} bits to {} bits, sent {} bits per "
        "element. Actions total {}, elapsed total time: {} ms.",
        rank, kBenchSize, meta.src_width, meta.dst_width,
        (b1 - b0) * 8. / shape.numel(), (r1 - r0) * 1.0, pack_time);
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  const auto field_0 = oup[0].eltype().as<Ring2k>()->field();
  const auto field_1 = oup[1].eltype().as<Ring2k>()->field();
  EXPECT_EQ(field_0, dst_field);
  EXPECT_EQ(field_1, dst_field);
  EXPECT_EQ(oup[0].fxp_bits(), dest_width);
  EXPECT_EQ(oup[1].fxp_bits(), dest_width);

  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(expected, src_width);
  MaskItInplace(got, dest_width);

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> exp_(expected);

    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dt = ring2k_t;
      NdArrayView<dt> got_(got);

      auto mask = makeMask<dt>(dest_width);

      for (int64_t i = 0; i < shape.numel(); i++) {
        auto exp_v = static_cast<dt>((exp_[i] >> kTR_bits) & mask);
        auto got_v = got_[i] & mask;

        if (exact) {
          EXPECT_EQ(exp_v, got_v);
        } else {
          // 1-bit approx, exp - got = 0 or 1
          EXPECT_TRUE(safe_check(exp_v, got_v, dest_width));
        }
      }
    });
  });
}

TEST_P(TRProtocolTest, SameField) {
  size_t kWorldSize = 2;

  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  bool exact = std::get<1>(GetParam());
  bool dynamic = std::get<2>(GetParam());

  FieldType dst_field = src_field;

  const auto src_width = SizeOf(src_field) * 8;
  size_t kTR_bits;
  if (dynamic) {
    kTR_bits = src_width / 2;
  } else {
    kTR_bits = kDefaultTRBits;
  }

  const auto dest_width = src_width - kTR_bits;

  // NdArrayRef expected = ConstructUnderlyingArray(src_field, shape, kTR_bits);
  NdArrayRef expected = ring_rand(src_field, shape);

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_sub(expected, inp[0]);

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    RingTruncateAndReduceProtocol prot(base);
    RingTruncateAndReduceProtocol::Meta meta;

    meta.exact = exact;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = src_width;
    meta.dst_width = dest_width;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    [[maybe_unused]] size_t r0 = ctx->GetStats()->sent_actions;

    // without wrap
    yacl::ElapsedTimer pack_timer;
    oup[rank] = prot.Compute(inp[rank], NdArrayRef(), meta);
    double pack_time = pack_timer.CountMs() * 1.0;

    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    [[maybe_unused]] size_t r1 = ctx->GetStats()->sent_actions;

    SPDLOG_INFO(
        "[TR]: rank {}, test size: {} ,tr {} bits to {} bits, sent {} bits per "
        "element. Actions total {}, elapsed total time: {} ms.",
        rank, kBenchSize, meta.src_width, meta.dst_width,
        (b1 - b0) * 8. / shape.numel(), (r1 - r0) * 1.0, pack_time);
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  const auto field_0 = oup[0].eltype().as<Ring2k>()->field();
  const auto field_1 = oup[1].eltype().as<Ring2k>()->field();
  EXPECT_EQ(field_0, dst_field);
  EXPECT_EQ(field_1, dst_field);
  EXPECT_EQ(oup[0].fxp_bits(), dest_width);
  EXPECT_EQ(oup[1].fxp_bits(), dest_width);

  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(expected, src_width);
  MaskItInplace(got, dest_width);

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> exp_(expected);

    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dt = ring2k_t;
      NdArrayView<dt> got_(got);

      auto mask = makeMask<dt>(dest_width);

      for (int64_t i = 0; i < shape.numel(); i++) {
        auto exp_v = static_cast<dt>((exp_[i] >> kTR_bits) & mask);
        auto got_v = got_[i] & mask;

        if (exact) {
          EXPECT_EQ(exp_v, got_v);
        } else {
          // 1-bit approx, exp - got = 0 or 1
          EXPECT_TRUE(safe_check(exp_v, got_v, dest_width));
        }
      }
    });
  });
}

TEST_P(TRProtocolTest, NotSameFieldWithWrapS) {
  size_t kWorldSize = 2;

  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  bool exact = std::get<1>(GetParam());
  bool dynamic = std::get<2>(GetParam());

  FieldType dst_field;
  // no fewer ring than fm32 now
  // if wrap supplied, then must be exact
  if (src_field == FM32 || !dynamic || !exact) {
    return;
  }
  if (src_field == FM64) {
    dst_field = FM32;
  } else {
    dst_field = FM64;
  }

  const auto src_width = SizeOf(src_field) * 8;
  size_t kTR_bits;
  if (dynamic) {
    kTR_bits = src_width / 2;
  } else {
    kTR_bits = kDefaultTRBits;
  }

  const auto dest_width = src_width - kTR_bits;

  // NdArrayRef expected = ConstructUnderlyingArray(src_field, shape, kTR_bits);
  NdArrayRef expected = ring_rand(src_field, shape);

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_sub(expected, inp[0]);

  NdArrayRef wrap_exp = ring_zeros(src_field, shape);
  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> inp0_(inp[0]);
    NdArrayView<st> inp1_(inp[1]);
    NdArrayView<st> wrap_(wrap_exp);
    const auto mask = makeMask<st>(kTR_bits);

    pforeach(0, shape.numel(), [&](int64_t idx) {
      wrap_[idx] =
          static_cast<st>(((inp0_[idx] & mask) + (inp1_[idx] & mask)) > mask);
    });
  });

  NdArrayRef wrap[2];
  wrap[0] = ring_randbit(src_field, shape);
  wrap[1] = ring_xor(wrap_exp, wrap[0]);
  wrap[0] = wrap[0].as(makeType<BShrTy>(src_field, 1));
  wrap[1] = wrap[1].as(makeType<BShrTy>(src_field, 1));

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    RingTruncateAndReduceProtocol prot(base);
    RingTruncateAndReduceProtocol::Meta meta;

    meta.exact = exact;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = src_width;
    meta.dst_width = dest_width;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    [[maybe_unused]] size_t r0 = ctx->GetStats()->sent_actions;

    // with wrap
    yacl::ElapsedTimer pack_timer;
    oup[rank] = prot.Compute(inp[rank], wrap[rank], meta);
    double pack_time = pack_timer.CountMs() * 1.0;

    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    [[maybe_unused]] size_t r1 = ctx->GetStats()->sent_actions;

    SPDLOG_INFO(
        "[TR with msb]: rank {}, test size: {} ,tr {} bits to {} bits, sent {} "
        "bits per element. Actions total {}, elapsed total time: {} ms.",
        rank, kBenchSize, meta.src_width, meta.dst_width,
        (b1 - b0) * 8. / shape.numel(), (r1 - r0) * 1.0, pack_time);
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  const auto field_0 = oup[0].eltype().as<Ring2k>()->field();
  const auto field_1 = oup[1].eltype().as<Ring2k>()->field();
  EXPECT_EQ(field_0, dst_field);
  EXPECT_EQ(field_1, dst_field);
  EXPECT_EQ(oup[0].fxp_bits(), dest_width);
  EXPECT_EQ(oup[1].fxp_bits(), dest_width);

  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(expected, src_width);
  MaskItInplace(got, dest_width);

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> exp_(expected);

    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dt = ring2k_t;
      NdArrayView<dt> got_(got);

      auto mask = makeMask<dt>(dest_width);

      for (int64_t i = 0; i < shape.numel(); i++) {
        auto exp_v = static_cast<dt>((exp_[i] >> kTR_bits) & mask);
        auto got_v = got_[i] & mask;

        if (exact) {
          EXPECT_EQ(exp_v, got_v);
        } else {
          // 1-bit approx, exp - got = 0 or 1
          EXPECT_TRUE(safe_check(exp_v, got_v, dest_width));
        }
      }
    });
  });
}

TEST_P(TRProtocolTest, SameFieldWithWrapS) {
  size_t kWorldSize = 2;

  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  bool exact = std::get<1>(GetParam());
  bool dynamic = std::get<2>(GetParam());

  FieldType dst_field = src_field;

  // if wrap supplied, then must be exact
  if (!exact) {
    return;
  }

  const auto src_width = SizeOf(src_field) * 8;
  size_t kTR_bits;
  if (dynamic) {
    kTR_bits = src_width / 2;
  } else {
    kTR_bits = kDefaultTRBits;
  }

  const auto dest_width = src_width - kTR_bits;

  // NdArrayRef expected = ConstructUnderlyingArray(src_field, shape, kTR_bits);
  NdArrayRef expected = ring_rand(src_field, shape);

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_sub(expected, inp[0]);

  NdArrayRef wrap_exp = ring_zeros(src_field, shape);
  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> inp0_(inp[0]);
    NdArrayView<st> inp1_(inp[1]);
    NdArrayView<st> wrap_(wrap_exp);
    const auto mask = makeMask<st>(kTR_bits);

    pforeach(0, shape.numel(), [&](int64_t idx) {
      wrap_[idx] =
          static_cast<st>(((inp0_[idx] & mask) + (inp1_[idx] & mask)) > mask);
    });
  });

  NdArrayRef wrap[2];
  wrap[0] = ring_randbit(src_field, shape);
  wrap[1] = ring_xor(wrap_exp, wrap[0]);
  wrap[0] = wrap[0].as(makeType<BShrTy>(src_field, 1));
  wrap[1] = wrap[1].as(makeType<BShrTy>(src_field, 1));

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    RingTruncateAndReduceProtocol prot(base);
    RingTruncateAndReduceProtocol::Meta meta;

    meta.exact = exact;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = src_width;
    meta.dst_width = dest_width;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    [[maybe_unused]] size_t r0 = ctx->GetStats()->sent_actions;

    // with wrap
    yacl::ElapsedTimer pack_timer;
    oup[rank] = prot.Compute(inp[rank], wrap[rank], meta);
    double pack_time = pack_timer.CountMs() * 1.0;

    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    [[maybe_unused]] size_t r1 = ctx->GetStats()->sent_actions;

    SPDLOG_INFO(
        "[TR with msb]: rank {}, test size: {} ,tr {} bits to {} bits, sent {} "
        "bits per element. Actions total {}, elapsed total time: {} ms.",
        rank, kBenchSize, meta.src_width, meta.dst_width,
        (b1 - b0) * 8. / shape.numel(), (r1 - r0) * 1.0, pack_time);
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  const auto field_0 = oup[0].eltype().as<Ring2k>()->field();
  const auto field_1 = oup[1].eltype().as<Ring2k>()->field();
  EXPECT_EQ(field_0, dst_field);
  EXPECT_EQ(field_1, dst_field);
  EXPECT_EQ(oup[0].fxp_bits(), dest_width);
  EXPECT_EQ(oup[1].fxp_bits(), dest_width);

  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(expected, src_width);
  MaskItInplace(got, dest_width);

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> exp_(expected);

    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dt = ring2k_t;
      NdArrayView<dt> got_(got);

      auto mask = makeMask<dt>(dest_width);

      for (int64_t i = 0; i < shape.numel(); i++) {
        auto exp_v = static_cast<dt>((exp_[i] >> kTR_bits) & mask);
        auto got_v = got_[i] & mask;

        if (exact) {
          EXPECT_EQ(exp_v, got_v);
        } else {
          // 1-bit approx, exp - got = 0 or 1
          EXPECT_TRUE(safe_check(exp_v, got_v, dest_width));
        }
      }
    });
  });
}

namespace {
// yacl_ss runs faster in small dataset
// yacl_ferret & emp_ferret runs faster in large dataset (and smaller comm.)
// emp_ferret seems to be the fastest
const auto ot_kind = CheetahOtKind::YACL_Softspoken;
// const auto ot_kind = CheetahOtKind::YACL_Ferret;
// const auto ot_kind = CheetahOtKind::EMP_Ferret;

const std::vector<std::tuple<int64_t, int64_t>> bw_values = {
    std::make_tuple(16, 8),    // full bw tr
    std::make_tuple(32, 16),   //
    std::make_tuple(64, 32),   //
    std::make_tuple(128, 64),  //
                               // not full bw tr
    std::make_tuple(8, 5),     //
    std::make_tuple(14, 4),    //
    std::make_tuple(23, 16),   //
    std::make_tuple(48, 24),   //
};

}  // namespace

class TRGeneralTest : public ::testing::TestWithParam<
                          std::tuple<std::tuple<int64_t, int64_t>, bool>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, TRGeneralTest,
    testing::Combine(testing::ValuesIn(bw_values),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<TRGeneralTest::ParamType> &p) {
      return fmt::format("{}to{}x{}", std::get<0>(std::get<0>(p.param)),
                         std::get<1>(std::get<0>(p.param)),
                         std::get<1>(p.param));
    });

TEST_P(TRGeneralTest, Work) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  int64_t m;
  int64_t n;
  std::tie(m, n) = std::get<0>(GetParam());
  const auto src_field = FixGetProperFiled(m);
  const auto dst_field = FixGetProperFiled(n);

  bool exact = std::get<1>(GetParam());

  NdArrayRef x = ring_rand(src_field, shape);
  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_sub(x, inp[0]);

  ring_reduce_(x, m);
  ring_reduce_(inp[0], m);
  ring_reduce_(inp[1], m);

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn, ot_kind);
    RingTruncateAndReduceProtocol prot(base);
    RingTruncateAndReduceProtocol::Meta meta;

    meta.exact = exact;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = m;
    meta.dst_width = n;

    size_t b0 = ctx->GetStats()->sent_bytes;
    size_t r0 = ctx->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    // without wrap
    oup[rank] = prot.Compute(inp[rank], NdArrayRef(), meta);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = ctx->GetStats()->sent_bytes;
    size_t r1 = ctx->GetStats()->sent_actions;

    std::string exact_str = exact ? "exact" : "approx";

    SPDLOG_INFO(
        "Rank {}, [{} TR without wrap, {} bits to {} bits with {} samples], "
        "sent {} bits per "
        "element. Actions total {}, elapsed total time: {} ms.",
        rank, exact_str, meta.src_width, meta.dst_width, numel,
        (b1 - b0) * 8. / numel, (r1 - r0) * 1.0, pack_time);
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  const auto field_0 = oup[0].eltype().as<Ring2k>()->field();
  const auto field_1 = oup[1].eltype().as<Ring2k>()->field();
  EXPECT_EQ(field_0, dst_field);
  EXPECT_EQ(field_1, dst_field);
  EXPECT_EQ(oup[0].fxp_bits(), n);
  EXPECT_EQ(oup[1].fxp_bits(), n);

  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(got, n);

  // value check
  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> exp_(x);

    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dt = ring2k_t;
      NdArrayView<dt> got_(got);

      auto mask = makeMask<dt>(n);

      for (int64_t i = 0; i < shape.numel(); i++) {
        auto exp_v = static_cast<dt>((exp_[i] >> (m - n)) & mask);
        auto got_v = got_[i];

        if (exact) {
          EXPECT_EQ(exp_v, got_v);
        } else {
          // 1-bit approx, exp - got = 0 or 1
          EXPECT_TRUE(safe_check(exp_v, got_v, n));
        }
      }
    });
  });
}

TEST_P(TRGeneralTest, WorkWithWrap) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  int64_t m;
  int64_t n;
  std::tie(m, n) = std::get<0>(GetParam());
  const auto src_field = FixGetProperFiled(m);
  const auto dst_field = FixGetProperFiled(n);

  bool exact = std::get<1>(GetParam());

  // if wrap supplied, then must be exact
  if (!exact) {
    return;
  }

  NdArrayRef x = ring_rand(src_field, shape);
  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_sub(x, inp[0]);

  ring_reduce_(x, m);
  ring_reduce_(inp[0], m);
  ring_reduce_(inp[1], m);

  const auto kTR_bits = m - n;
  // compute wrap
  NdArrayRef wrap_exp = ring_zeros(src_field, shape);
  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> inp0_(inp[0]);
    NdArrayView<st> inp1_(inp[1]);
    NdArrayView<st> wrap_(wrap_exp);
    const auto mask = makeMask<st>(kTR_bits);

    pforeach(0, shape.numel(), [&](int64_t idx) {
      wrap_[idx] =
          static_cast<st>(((inp0_[idx] & mask) + (inp1_[idx] & mask)) > mask);
    });
  });

  NdArrayRef wrap[2];
  wrap[0] = ring_randbit(src_field, shape);
  wrap[1] = ring_xor(wrap_exp, wrap[0]);
  ring_reduce_(wrap[0], 1);
  ring_reduce_(wrap[1], 1);
  wrap[0] = wrap[0].as(makeType<BShrTy>(src_field, 1));
  wrap[1] = wrap[1].as(makeType<BShrTy>(src_field, 1));

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn, ot_kind);
    RingTruncateAndReduceProtocol prot(base);
    RingTruncateAndReduceProtocol::Meta meta;

    meta.exact = exact;
    meta.src_ring = src_field;
    meta.dst_ring = dst_field;
    meta.src_width = m;
    meta.dst_width = n;

    size_t b0 = ctx->GetStats()->sent_bytes;
    size_t r0 = ctx->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    // with wrap
    oup[rank] = prot.Compute(inp[rank], wrap[rank], meta);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = ctx->GetStats()->sent_bytes;
    size_t r1 = ctx->GetStats()->sent_actions;

    std::string exact_str = exact ? "exact" : "approx";

    SPDLOG_INFO(
        "Rank {}, [{} TR with wrap, {} bits to {} bits with {} samples], sent "
        "{} bits per "
        "element. Actions total {}, elapsed total time: {} ms.",
        rank, exact_str, meta.src_width, meta.dst_width, numel,
        (b1 - b0) * 8. / numel, (r1 - r0) * 1.0, pack_time);
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  const auto field_0 = oup[0].eltype().as<Ring2k>()->field();
  const auto field_1 = oup[1].eltype().as<Ring2k>()->field();
  EXPECT_EQ(field_0, dst_field);
  EXPECT_EQ(field_1, dst_field);
  EXPECT_EQ(oup[0].fxp_bits(), n);
  EXPECT_EQ(oup[1].fxp_bits(), n);

  auto got = ring_add(oup[0], oup[1]);
  MaskItInplace(got, n);

  // value check
  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> exp_(x);

    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dt = ring2k_t;
      NdArrayView<dt> got_(got);

      auto mask = makeMask<dt>(n);

      for (int64_t i = 0; i < shape.numel(); i++) {
        auto exp_v = static_cast<dt>((exp_[i] >> (m - n)) & mask);
        auto got_v = got_[i];

        if (exact) {
          EXPECT_EQ(exp_v, got_v);
        } else {
          // 1-bit approx, exp - got = 0 or 1
          EXPECT_TRUE(safe_check(exp_v, got_v, n));
        }
      }
    });
  });
}

}  // namespace spu::mpc::cheetah