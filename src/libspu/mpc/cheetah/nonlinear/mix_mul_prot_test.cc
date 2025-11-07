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

#include "libspu/mpc/cheetah/nonlinear/mix_mul_prot.h"

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

namespace {

// yacl_ss runs faster in small dataset
// yacl_ferret & emp_ferret runs faster in large dataset (and smaller comm.)
// emp_ferret seems to be the fastest
const auto ot_kind = CheetahOtKind::YACL_Softspoken;
// const auto ot_kind = CheetahOtKind::YACL_Ferret;
// const auto ot_kind = CheetahOtKind::EMP_Ferret;

const std::vector<std::tuple<int64_t, int64_t, int64_t>> bw_values = {
    std::make_tuple(8, 8, 16),     // full bw mul
    std::make_tuple(16, 16, 32),   //
    std::make_tuple(32, 32, 64),   //
    std::make_tuple(64, 64, 128),  //
    std::make_tuple(13, 8, 21),    //
    std::make_tuple(8, 13, 21),    // duplicate test for switch
                                   //
    std::make_tuple(10, 24, 34),   // partial bw mul
    std::make_tuple(13, 13, 20),   //
    std::make_tuple(24, 9, 32),    //
    std::make_tuple(12, 27, 27),   //
    std::make_tuple(64, 32, 64),   //
};

const std::vector<SignType> sign_values = {
    SignType::Unknown, SignType::Positive, SignType::Negative};

std::string getSignName(SignType sign) {
  if (sign == SignType::Unknown) {
    return "Unknown";
  } else if (sign == SignType::Positive) {
    return "Positive";
  } else if (sign == SignType::Negative) {
    return "Negative";
  } else {
    SPU_THROW("should not be here.");
  }
}

template <typename T>
typename std::make_unsigned<T>::type makeMask(size_t bw) {
  using U = typename std::make_unsigned<T>::type;
  if (bw == sizeof(U) * 8) {
    return static_cast<U>(-1);
  }
  return (static_cast<U>(1) << bw) - 1;
}

// view [0, 2^k) as [-2^k/2, 2^k/2)
// positive part: [0, 2^{k-1}) => 0,1,2,...2^{k-1}-1
// negative part: [2^{k-1}, 2^k) => -2^{k-1}, -2^{k-1}+1, ..., -1
template <typename U>
auto ToSignType(U x, size_t width) {
  using S = typename std::make_signed<U>::type;
  if (sizeof(U) * 8 == width) {
    return static_cast<S>(x);
  }

  U half = static_cast<U>(1) << (width - 1);
  if (x >= half) {
    U upper = static_cast<U>(1) << width;
    x -= upper;
  }
  return static_cast<S>(x);
}

NdArrayRef UnsignedMixMul(const NdArrayRef& x, const NdArrayRef& y,
                          FieldType field, int64_t bw) {
  SPU_ENFORCE(x.shape() == y.shape());
  SPU_ENFORCE(field >= x.eltype().as<Ring2k>()->field());
  SPU_ENFORCE(field >= y.eltype().as<Ring2k>()->field());

  auto out = ring_zeros(field, x.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    NdArrayView<u2k> out_(out);
    const auto msk = makeMask<ring2k_t>(bw);

    DISPATCH_ALL_FIELDS(x.eltype().as<Ring2k>()->field(), [&]() {
      using x_u2k = std::make_unsigned<ring2k_t>::type;
      NdArrayView<x_u2k> x_(x);

      DISPATCH_ALL_FIELDS(y.eltype().as<Ring2k>()->field(), [&]() {
        using y_u2k = std::make_unsigned<ring2k_t>::type;
        NdArrayView<y_u2k> y_(y);

        pforeach(0, x.numel(), [&](int64_t idx) {
          out_[idx] = static_cast<u2k>(x_[idx]) * static_cast<u2k>(y_[idx]);
          out_[idx] &= msk;
        });
      });
    });
  });

  return out;
}

NdArrayRef SignedMixMul(const NdArrayRef& x, const NdArrayRef& y,
                        FieldType field, int64_t bw) {
  SPU_ENFORCE(x.shape() == y.shape());
  SPU_ENFORCE(field >= x.eltype().as<Ring2k>()->field());
  SPU_ENFORCE(field >= y.eltype().as<Ring2k>()->field());

  auto out = ring_zeros(field, x.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    using s2k = std::make_signed_t<ring2k_t>;
    NdArrayView<s2k> out_(out);
    const auto msk = makeMask<ring2k_t>(bw);

    DISPATCH_ALL_FIELDS(x.eltype().as<Ring2k>()->field(), [&]() {
      using x_s2k = std::make_signed_t<ring2k_t>;
      NdArrayView<x_s2k> x_(x);

      DISPATCH_ALL_FIELDS(y.eltype().as<Ring2k>()->field(), [&]() {
        using y_s2k = std::make_signed_t<ring2k_t>;
        NdArrayView<y_s2k> y_(y);

        pforeach(0, x.numel(), [&](int64_t idx) {
          //? Must first compute the real value, then do the type cast!
          out_[idx] = static_cast<s2k>(ToSignType(x_[idx], x.fxp_bits())) *
                      static_cast<s2k>(ToSignType(y_[idx], y.fxp_bits()));
          out_[idx] &= msk;
        });
      });
    });
  });

  return out;
}

[[maybe_unused]] std::tuple<NdArrayRef, NdArrayRef, NdArrayRef> makeInput(
    size_t bits, SignType sign, FieldType field, const Shape& shape) {
  auto input = ring_rand(field, shape);
  // auto input = ring_iota(field, shape, 7);
  ring_reduce_(input, bits);
  const auto numel = shape.numel();

  if (sign != SignType::Unknown) {
    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> _input(input);
      if (sign == SignType::Positive) {
        // 0000 0111 ... 1111
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bits - 1)) - 1;
        pforeach(0, numel, [&](int64_t i) {  //
          _input[i] &= mask;
        });

      } else {
        // 0000 1000 ... 0000
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bits - 1));
        pforeach(0, numel, [&](int64_t i) {  //
          _input[i] |= mask;
        });
      }
    });
  }

  auto shr1 = ring_rand(field, shape);
  ring_reduce_(shr1, bits);

  auto shr2 = ring_sub(input, shr1);
  ring_reduce_(shr2, bits);

  EXPECT_TRUE(ring_all_equal(input, ring_reduce(ring_add(shr1, shr2), bits)));

  return std::make_tuple(input, shr1, shr2);
}

[[maybe_unused]] std::tuple<NdArrayRef, NdArrayRef, NdArrayRef> makeFixInput(
    size_t bits, FieldType field, const std::vector<uint64_t>& data) {
  // x, x0, x1
  SPU_ENFORCE(data.size() == 3);
  Shape shape = {1};

  auto x = ring_ones(field, shape);
  ring_mul_(x, data[0]);
  ring_reduce_(x, bits);

  auto x0 = ring_ones(field, shape);
  ring_mul_(x0, data[1]);
  ring_reduce_(x0, bits);

  auto x1 = ring_ones(field, shape);
  ring_mul_(x1, data[2]);
  ring_reduce_(x1, bits);

  return std::make_tuple(x, x0, x1);
}

}  // namespace

class CrossMulFullTest
    : public ::testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 13;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CrossMulFullTest, testing::ValuesIn(bw_values),
    [](const testing::TestParamInfo<CrossMulFullTest::ParamType>& p) {
      return fmt::format("{}x{}_{}", std::get<0>(p.param), std::get<1>(p.param),
                         std::get<2>(p.param));
    });

TEST_P(CrossMulFullTest, Work) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  const auto m = std::get<0>(GetParam());
  const auto n = std::get<1>(GetParam());
  const auto l = std::get<2>(GetParam());

  NdArrayRef inp[2];
  NdArrayRef out[2];

  inp[0] = ring_rand(FixGetProperFiled(m), shape);
  ring_reduce_(inp[0], m);
  inp[0].set_fxp_bits(m);
  inp[1] = ring_rand(FixGetProperFiled(n), shape);
  ring_reduce_(inp[1], n);
  inp[1].set_fxp_bits(n);

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn, ot_kind);

    MixMulProtocol prot(base);
    MixMulProtocol::Meta meta;

    meta.bw_x = m;
    meta.bw_y = n;
    meta.bw_out = l;

    meta.field_x = FixGetProperFiled(m);
    meta.field_y = FixGetProperFiled(n);
    meta.field_out = FixGetProperFiled(l);

    size_t b0 = ctx->GetStats()->sent_bytes;
    size_t r0 = ctx->GetStats()->sent_actions;

    yacl::ElapsedTimer pack_timer;
    // CrossMul only deals with unsigned mul
    out[rank] = prot.CrossMul(inp[rank], meta);
    double pack_time = pack_timer.CountMs() * 1.0;

    size_t b1 = ctx->GetStats()->sent_bytes;
    size_t r1 = ctx->GetStats()->sent_actions;

    SPDLOG_INFO(
        "[CrossMul {}x{} = {} with {} samples]: rank {}, sent {} bits per "
        "element. Actions total {}, elapsed total time: {} ms.",
        m, n, l, kBenchSize, rank, (b1 - b0) * 8. / shape.numel(),
        (r1 - r0) * 1.0, pack_time);
  });

  // shape and type check
  EXPECT_EQ(out[0].shape(), inp[0].shape());
  EXPECT_EQ(out[1].shape(), inp[1].shape());
  EXPECT_EQ(out[0].eltype().as<Ring2k>()->field(), FixGetProperFiled(l));
  EXPECT_EQ(out[1].eltype().as<Ring2k>()->field(), FixGetProperFiled(l));
  EXPECT_EQ(out[0].fxp_bits(), l);
  EXPECT_EQ(out[1].fxp_bits(), l);

  // values check
  auto expected = UnsignedMixMul(inp[0], inp[1], FixGetProperFiled(l), l);
  auto got = ring_add(out[0], out[1]);
  ring_reduce_(got, l);

  DISPATCH_ALL_FIELDS(FixGetProperFiled(l), [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    NdArrayView<u2k> expected_(expected);
    NdArrayView<u2k> got_(got);
    for (int64_t idx = 0; idx < expected.numel(); idx++) {  //
      EXPECT_EQ(expected_[idx], got_[idx]);
    }
  });
}

class ComputeWrapTest
    : public ::testing::TestWithParam<std::tuple<size_t, SignType>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 13;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ComputeWrapTest,
    testing::Combine(testing::Values(6, 8, 13, 16, 23, 32, 64, 128),
                     testing::ValuesIn(sign_values)),
    [](const testing::TestParamInfo<ComputeWrapTest::ParamType>& p) {
      return fmt::format("{}x{}", getSignName(std::get<1>(p.param)),
                         std::get<0>(p.param));
    });

TEST_P(ComputeWrapTest, Work) {
  const SignType sign = std::get<1>(GetParam());
  const size_t bits = std::get<0>(GetParam());
  const auto field = FixGetProperFiled(bits);

  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  NdArrayRef input;
  NdArrayRef inp[2];
  std::tie(input, inp[0], inp[1]) = makeInput(bits, sign, field, shape);

  NdArrayRef out[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn, ot_kind);

    MixMulProtocol prot(base);
    MixMulProtocol::WrapMeta meta;
    meta.sign = sign;
    meta.src_ring = field;
    meta.src_width = bits;

    size_t b0 = ctx->GetStats()->sent_bytes;
    size_t r0 = ctx->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    out[rank] = prot.ComputeWrap(inp[rank], meta);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = ctx->GetStats()->sent_bytes;
    size_t r1 = ctx->GetStats()->sent_actions;

    SPDLOG_INFO(
        "Rank {}, [ComputeWrap, sign {}, {} bits, with {} samples], sent {} "
        "bits per "
        "element. Actions total {}, elapsed total time: {} ms.",
        rank, getSignName(sign), bits, kBenchSize,
        (b1 - b0) * 8. / shape.numel(), (r1 - r0) * 1.0, pack_time);
  });

  // shape and type check
  EXPECT_EQ(out[0].shape(), inp[0].shape());
  EXPECT_EQ(out[1].shape(), inp[1].shape());
  // wrap always get FM8 and 1 bit.
  EXPECT_EQ(out[0].eltype().as<Ring2k>()->field(), FM8);
  EXPECT_EQ(out[1].eltype().as<Ring2k>()->field(), FM8);

  // debug part
  // ring_print(input, "input");
  // ring_print(inp[0], "inp0");
  // ring_print(inp[1], "inp1");
  // ring_print(out[0], "out0");
  // ring_print(out[1], "out1");

  // value check
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _input(input);
    NdArrayView<ring2k_t> _inp0(inp[0]);
    NdArrayView<ring2k_t> _inp1(inp[1]);
    NdArrayView<uint8_t> _out0(out[0]);
    NdArrayView<uint8_t> _out1(out[1]);

    const auto msk = makeMask<ring2k_t>(bits);

    for (int64_t i = 0; i < kBenchSize; ++i) {
      // check sign of input
      if (sign == SignType::Positive) {  //
        EXPECT_EQ(_input[i] >> (bits - 1), 0);
      } else if (sign == SignType::Negative) {  //
        EXPECT_EQ(_input[i] >> (bits - 1), 1);
      }

      // check of wrap
      auto s = (_inp0[i] + _inp1[i]) & msk;
      auto expect = (s < _inp0[i]) ? 1 : 0;
      auto got = (_out0[i] ^ _out1[i]) & 1;
      EXPECT_EQ(expect, got);
    }
  });
}

class FixMixedMulTest
    : public ::testing::TestWithParam<std::tuple<
          std::tuple<int64_t, int64_t, int64_t>, SignType, SignType, bool>> {
 public:
  // shape: (kBenchRow, kBenchRow)
  static constexpr int64_t kBenchSize = 1LL << 12;
  // static constexpr int64_t kBenchSize = 1LL << 20;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, FixMixedMulTest,
    testing::Combine(testing::ValuesIn(bw_values),    // (bw_x, bw_y, bw_out)
                     testing::ValuesIn(sign_values),  // sign_x
                     testing::ValuesIn(sign_values),  // sign_y
                     testing::Values(false, true)     // signed_arith
                     ));

TEST_P(FixMixedMulTest, Work) {
  int64_t m;
  int64_t n;
  int64_t l;

  std::tie(m, n, l) = std::get<0>(GetParam());
  const auto field_x = FixGetProperFiled(m);
  const auto field_y = FixGetProperFiled(n);
  const auto field_out = FixGetProperFiled(l);

  const SignType sign_x = std::get<1>(GetParam());
  const SignType sign_y = std::get<2>(GetParam());
  const bool signed_arith = std::get<3>(GetParam());

  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};
  // Shape shape = {1};

  const auto numel = shape.numel();

  NdArrayRef x;
  NdArrayRef xshr[2];
  NdArrayRef y;
  NdArrayRef yshr[2];
  std::tie(x, xshr[0], xshr[1]) = makeInput(m, sign_x, field_x, shape);
  std::tie(y, yshr[0], yshr[1]) = makeInput(n, sign_y, field_y, shape);

  x.set_fxp_bits(m);
  y.set_fxp_bits(n);
  xshr[0].set_fxp_bits(m);
  xshr[1].set_fxp_bits(m);
  yshr[0].set_fxp_bits(n);
  yshr[1].set_fxp_bits(n);

  NdArrayRef out[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn, ot_kind);

    MixMulProtocol prot(base);
    MixMulProtocol::Meta meta;

    meta.signed_arith = signed_arith;
    meta.sign_x = sign_x;
    meta.sign_y = sign_y;
    meta.bw_x = m;
    meta.bw_y = n;
    meta.bw_out = l;
    meta.field_x = field_x;
    meta.field_y = field_y;
    meta.field_out = field_out;

    size_t b0 = ctx->GetStats()->sent_bytes;
    size_t r0 = ctx->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    out[rank] = prot.Compute(xshr[rank], yshr[rank], meta);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = ctx->GetStats()->sent_bytes;
    size_t r1 = ctx->GetStats()->sent_actions;

    const std::string mul_type =
        signed_arith ? "SignedMixMul" : "UnsignedMixMul";

    SPDLOG_INFO(
        "Rank {}, [{}, sign {}x{}, {}x{}={} bits, with {} "
        "samples], sent {} bits per element. Actions total {}, elapsed total "
        "time: {} ms.",
        rank, mul_type, getSignName(sign_x), getSignName(sign_y), m, n, l,
        numel, (b1 - b0) * 8. / shape.numel(), (r1 - r0) * 1.0, pack_time);
  });

  // shape and type check
  EXPECT_EQ(out[0].shape(), shape);
  EXPECT_EQ(out[1].shape(), shape);

  EXPECT_EQ(out[0].fxp_bits(), l);
  EXPECT_EQ(out[1].fxp_bits(), l);

  // debug part
  // ring_print(x, "x");
  // ring_print(y, "y");

  // ring_print(xshr[0], "xshr[0]");
  // ring_print(xshr[1], "xshr[1]");
  // ring_print(yshr[0], "yshr[0]");
  // ring_print(yshr[1], "yshr[1]");

  // value check
  NdArrayRef expected;
  if (signed_arith) {
    expected = SignedMixMul(x, y, field_out, l);
  } else {
    expected = UnsignedMixMul(x, y, field_out, l);
  }
  auto got = ring_add(out[0], out[1]);
  ring_reduce_(got, l);

  // ring_print(expected, "expected");
  // ring_print(got, "got");

  DISPATCH_ALL_FIELDS(field_out, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    using s2k = std::make_signed<ring2k_t>::type;

    NdArrayView<u2k> _expected_u(expected);
    NdArrayView<u2k> _got_u(got);
    NdArrayView<s2k> _expected_s(expected);
    NdArrayView<s2k> _got_s(got);

    for (int64_t idx = 0; idx < numel; idx++) {  //
      if (signed_arith) {
        EXPECT_EQ(_expected_s[idx], _got_s[idx]);

      } else {
        EXPECT_EQ(_expected_u[idx], _got_u[idx]);
      }
    }
  });
}

}  // namespace spu::mpc::cheetah