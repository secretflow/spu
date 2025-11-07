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

#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class TruncateProtTest : public ::testing::TestWithParam<
                             std::tuple<FieldType, bool, bool, std::string>> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, TruncateProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(true, false),  // signed, unsigned
                     testing::Values(true, false),  // exact, prob
                     testing::Values("Unknown", "Zero", "One")),
    [](const testing::TestParamInfo<TruncateProtTest::ParamType> &p) {
      return fmt::format("{}{}{}MSB{}", std::get<0>(p.param),
                         std::get<1>(p.param) ? "Signed" : "Unsigned",
                         std::get<2>(p.param) ? "Exact" : "Prob",
                         std::get<3>(p.param));
    });

template <typename T>
bool SignBit(T x) {
  using uT = typename std::make_unsigned<T>::type;
  return (static_cast<uT>(x) >> (8 * sizeof(T) - 1)) & 1;
}

TEST_P(TruncateProtTest, Basic) {
  size_t kWorldSize = 2;
  int64_t n = 4096;
  size_t shift = 12;
  FieldType field = std::get<0>(GetParam());
  bool signed_arith = std::get<1>(GetParam());
  bool exact = std::get<2>(GetParam());
  std::string msb = std::get<3>(GetParam());
  SignType sign;

  NdArrayRef inp[2];
  inp[0] = ring_rand(field, {n});

  if (msb == "Unknown") {
    inp[1] = ring_rand(field, {n});
    sign = SignType::Unknown;
  } else {
    auto msg = ring_rand(field, {n});
    DISPATCH_ALL_FIELDS(field, [&]() {
      auto xmsg = NdArrayView<ring2k_t>(msg);
      size_t bw = SizeOf(field) * 8;
      if (msb == "Zero") {
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1)) - 1;
        pforeach(0, msg.numel(), [&](int64_t i) { xmsg[i] &= mask; });

        sign = SignType::Positive;
      } else {
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1));
        pforeach(0, msg.numel(), [&](int64_t i) { xmsg[i] |= mask; });
        sign = SignType::Negative;
      }
    });

    inp[1] = ring_sub(msg, inp[0]);
  }

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    TruncateProtocol trunc_prot(base);
    TruncateProtocol::Meta meta;
    meta.sign = sign;
    meta.signed_arith = signed_arith;
    meta.shift_bits = shift;
    meta.use_heuristic = false;
    meta.exact = exact;

    [[maybe_unused]] auto b0 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s0 = ctx->GetStats()->sent_actions.load();

    oup[rank] = trunc_prot.Compute(inp[rank], meta);

    [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

    SPDLOG_INFO("Truncate {} bits share by {} bits {} bits each #sent {}",
                SizeOf(field) * 8, meta.shift_bits,
                (b1 - b0) * 8. / inp[0].numel(), (s1 - s0));
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using signed_t = std::make_signed<ring2k_t>::type;
    using usigned_t = std::make_unsigned<ring2k_t>::type;

    if (signed_arith) {
      auto xout0 = NdArrayView<signed_t>(oup[0]);
      auto xout1 = NdArrayView<signed_t>(oup[1]);
      auto xinp0 = absl::MakeSpan(&inp[0].at<signed_t>(0), inp[0].numel());
      auto xinp1 = absl::MakeSpan(&inp[1].at<signed_t>(0), inp[1].numel());

      for (int64_t i = 0; i < n; ++i) {
        signed_t in = xinp0[i] + xinp1[i];
        signed_t expected = in >> shift;
        if (sign != SignType::Unknown) {
          ASSERT_EQ(SignBit<signed_t>(in), sign == SignType::Negative);
        }
        signed_t got = xout0[i] + xout1[i];
        EXPECT_NEAR(expected, got, exact ? 0 : 1);
      }
    } else {
      auto xout0 = NdArrayView<usigned_t>(oup[0]);
      auto xout1 = NdArrayView<usigned_t>(oup[1]);
      auto xinp0 = absl::MakeSpan(&inp[0].at<usigned_t>(0), inp[0].numel());
      auto xinp1 = absl::MakeSpan(&inp[1].at<usigned_t>(0), inp[1].numel());

      for (int64_t i = 0; i < n; ++i) {
        usigned_t in = xinp0[i] + xinp1[i];
        usigned_t expected = (in) >> shift;
        if (sign != SignType::Unknown) {
          ASSERT_EQ(SignBit<usigned_t>(in), sign == SignType::Negative);
        }
        usigned_t got = xout0[i] + xout1[i];
        ASSERT_NEAR(expected, got, exact ? 0 : 1);
      }
    }
  });
}

TEST_P(TruncateProtTest, Heuristic) {
  size_t kWorldSize = 2;
  int64_t n = 4096;
  size_t shift = 13;
  FieldType field = std::get<0>(GetParam());
  bool signed_arith = std::get<1>(GetParam());
  bool exact = std::get<2>(GetParam());
  std::string msb = std::get<3>(GetParam());
  if (not signed_arith or msb != "Unknown") {
    return;
  }

  NdArrayRef inp[2];
  inp[0] = ring_rand(field, {n});

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto msg = ring_rand(field, {n});
    ring_rshift_(msg,
                 {static_cast<int64_t>(TruncateProtocol::kHeuristicBound)});
    NdArrayView<ring2k_t> xmsg(msg);
    for (int64_t i = 0; i < n; i += 2) {
      // 50% percent negative
      xmsg[i] = -xmsg[i];
    }
    // inp[0] + inp[1] = msg
    inp[1] = ring_sub(msg, inp[0]);
  });

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    TruncateProtocol trunc_prot(base);
    TruncateProtocol::Meta meta;
    meta.sign = SignType::Unknown;
    meta.signed_arith = true;
    meta.shift_bits = shift;
    meta.use_heuristic = true;
    meta.exact = exact;
    oup[rank] = trunc_prot.Compute(inp[rank], meta);
  });

  [[maybe_unused]] int count_zero = 0;
  [[maybe_unused]] int count_pos = 0;
  [[maybe_unused]] int count_neg = 0;
  DISPATCH_ALL_FIELDS(field, [&]() {
    using signed_t = std::make_signed<ring2k_t>::type;

    auto xout0 = NdArrayView<signed_t>(oup[0]);
    auto xout1 = NdArrayView<signed_t>(oup[1]);
    auto xinp0 = NdArrayView<signed_t>(inp[0]);
    auto xinp1 = NdArrayView<signed_t>(inp[1]);

    for (int64_t i = 0; i < n; ++i) {
      signed_t in = xinp0[i] + xinp1[i];
      signed_t expected = in >> shift;
      signed_t got = xout0[i] + xout1[i];

      EXPECT_NEAR(expected, got, exact ? 0 : 1);
      if (expected == got) {
        count_zero += 1;
      } else if (expected < got) {
        count_pos += 1;
      } else {
        count_neg += 1;
      }
    }
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

[[maybe_unused]] std::tuple<NdArrayRef, NdArrayRef, NdArrayRef> makeInput(
    size_t bits, SignType sign, FieldType field, const Shape &shape,
    bool heuristic = false) {
  if (heuristic) {
    SPU_ENFORCE(sign == SignType::Unknown);
    SPU_ENFORCE(bits == 8 * SizeOf(field));
  }
  auto input = ring_rand(field, shape);

  if (heuristic) {
    ring_rshift_(input,
                 {static_cast<int64_t>(TruncateProtocol::kHeuristicBound)});
    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> input_(input);
      for (int64_t i = 0; i < shape.numel(); i += 2) {
        // 50% percent negative
        input_[i] = -input_[i];
      }
    });
  }

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

  input.set_fxp_bits(bits);
  shr1.set_fxp_bits(bits);
  shr2.set_fxp_bits(bits);

  return std::make_tuple(input, shr1, shr2);
}

template <typename T>
bool safe_check(const T exp, const T got, int64_t bw) {
  using UT = std::make_unsigned_t<T>;
  if (exp == got) {
    return true;
  }

  // diff = 1
  if (exp > got && exp - got <= 1) {
    return true;
  }
  if (got > exp && got - exp <= 1) {
    return true;
  }

  // edge case, when exp/got = 0
  if (exp == 0 && static_cast<UT>(got) == makeMask<T>(bw)) {
    return true;
  }

  if (got == 0 && static_cast<UT>(exp) == makeMask<T>(bw)) {
    return true;
  }

  SPDLOG_INFO("exp: {}, got: {}, bw: {}", exp, got, bw);
  return false;
}

template <typename T, typename U>
bool safe_check(const T exp, const U got, int64_t bw) {
  return safe_check<T>(exp, static_cast<T>(got), bw);
}
}  // namespace

class TruncateGeneralTest
    : public ::testing::TestWithParam<std::tuple<std::tuple<int64_t, int64_t>,
                                                 bool, bool, SignType, bool>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, TruncateGeneralTest,
    testing::Combine(testing::ValuesIn(bw_values),
                     testing::Values(true, false),  // signed, unsigned
                     testing::Values(true, false),  // exact, prob
                     testing::ValuesIn(sign_values),
                     testing::Values(true, false)),  // heuristic
    [](const testing::TestParamInfo<TruncateGeneralTest::ParamType> &p) {
      return fmt::format("{}to{}{}{}MSB{}x{}",
                         std::get<0>(std::get<0>(p.param)),
                         std::get<1>(std::get<0>(p.param)),
                         std::get<1>(p.param) ? "Signed" : "Unsigned",
                         std::get<2>(p.param) ? "Exact" : "Prob",
                         getSignName(std::get<3>(p.param)),
                         std::get<4>(p.param) ? "Heuristic" : "NoHeuristic");
    });

TEST_P(TruncateGeneralTest, Work) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  int64_t m;
  int64_t n;
  std::tie(m, n) = std::get<0>(GetParam());
  const auto src_field = FixGetProperFiled(m);
  const auto shift = m - n;

  const auto signed_arith = std::get<1>(GetParam());
  const auto exact = std::get<2>(GetParam());
  const auto sign = std::get<3>(GetParam());
  const auto heuristic = std::get<4>(GetParam());

  if (heuristic) {
    if (sign != SignType::Unknown || !signed_arith) {
      return;
    }
    if (m != static_cast<int64_t>(SizeOf(src_field)) * 8) {
      return;
    }
  }

  NdArrayRef x;
  NdArrayRef xshr[2];
  std::tie(x, xshr[0], xshr[1]) =
      makeInput(m, sign, src_field, shape, heuristic);
  SPU_ENFORCE(x.fxp_bits() == m);
  SPU_ENFORCE(xshr[0].fxp_bits() == m);
  SPU_ENFORCE(xshr[1].fxp_bits() == m);

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn, ot_kind);

    TruncateProtocol trunc_prot(base);
    TruncateProtocol::Meta meta;

    meta.sign = sign;
    meta.signed_arith = signed_arith;
    meta.shift_bits = shift;
    meta.use_heuristic = heuristic;
    meta.exact = exact;

    size_t b0 = ctx->GetStats()->sent_bytes;
    size_t r0 = ctx->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    oup[rank] = trunc_prot.Compute(xshr[rank], meta);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = ctx->GetStats()->sent_bytes;
    size_t r1 = ctx->GetStats()->sent_actions;

    std::string exact_str = exact ? "exact" : "approx";
    std::string signed_str = signed_arith ? "Signed" : "Unsigned";

    SPDLOG_INFO(
        "Rank {}, [{} Truncate {} bits to {} bits with {} {} samples, sign "
        "{}], sent "
        "{} bits per element. Actions total {}, elapsed total time: {} ms.",
        rank, exact_str, m, n, signed_str, numel, getSignName(sign),
        (b1 - b0) * 8. / numel, (r1 - r0) * 1.0, pack_time);
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  const auto field_0 = oup[0].eltype().as<Ring2k>()->field();
  const auto field_1 = oup[1].eltype().as<Ring2k>()->field();
  // truncate keeps the same field
  EXPECT_EQ(field_0, src_field);
  EXPECT_EQ(field_1, src_field);
  EXPECT_EQ(oup[0].fxp_bits(), m);
  EXPECT_EQ(oup[1].fxp_bits(), m);

  auto got = ring_add(oup[0], oup[1]);
  ring_reduce_(got, m);

  // debug
  // ring_print(xshr[0], "xshr[0]");
  // ring_print(xshr[1], "xshr[1]");
  // ring_print(x, "x");
  // ring_print(got, "got");

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using usigned_t = std::make_unsigned<ring2k_t>::type;

    NdArrayView<usigned_t> _x(x);
    NdArrayView<usigned_t> _got(got);

    const auto msk0 = makeMask<usigned_t>(m);
    const auto msk1 = makeMask<usigned_t>(n);
    const auto delta = msk0 - msk1;
    for (int64_t i = 0; i < numel; ++i) {
      auto expected = (_x[i] >> shift) & msk0;
      if (signed_arith && (((_x[i] >> (m - 1)) & 1) == 1)) {
        expected += delta;
        expected &= msk0;
      }
      auto got = _got[i] & msk0;

      if (exact) {
        EXPECT_EQ(expected, got);
      } else {
        EXPECT_TRUE(safe_check(expected, got, m));
      }
    }
  });
}

}  // namespace spu::mpc::cheetah
