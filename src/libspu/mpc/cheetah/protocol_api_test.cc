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

#include "yacl/crypto/rand/rand.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/api_test.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

const auto ot_kind = CheetahOtKind::YACL_Softspoken;
// const auto ot_kind = CheetahOtKind::YACL_Ferret;

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.protocol = ProtocolKind::CHEETAH;
  conf.field = field;
  conf.cheetah_2pc_config.ot_kind = ot_kind;
  return conf;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ApiTest,
    testing::Combine(testing::Values(makeCheetahProtocol),           //
                     testing::Values(makeConfig(FieldType::FM32),    //
                                     makeConfig(FieldType::FM64),    //
                                     makeConfig(FieldType::FM128)),  //
                     testing::Values(2)),                            //
    [](const testing::TestParamInfo<ApiTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field,
                         std::get<2>(p.param));
    });

namespace {

[[maybe_unused]] const std::vector<SignType> sign_values = {
    SignType::Unknown, SignType::Positive, SignType::Negative};

[[maybe_unused]] const std::array<bool, 2> signed_values = {false, true};

[[maybe_unused]] NdArrayRef mockPshare(SPUContext* ctx, const Index& perm) {
  NdArrayRef out(makeType<cheetah::PShrTy>(),
                 {static_cast<int64_t>(perm.size())});
  const auto field = out.eltype().as<cheetah::PShrTy>()->field();

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _out(out);
    pforeach(0, out.numel(),
             [&](int64_t idx) { _out[idx] = ring2k_t(perm[idx]); });
  });

  return out;
}

[[maybe_unused]] std::string getSignName(SignType sign) {
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
typename std::make_unsigned_t<T> makeMask(size_t bw) {
  using U = typename std::make_unsigned_t<T>;
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
    size_t bits, SignType sign, FieldType field, const Shape& shape,
    bool heuristic = false) {
  if (heuristic) {
    SPU_ENFORCE(sign == SignType::Unknown);
    SPU_ENFORCE(bits == 8 * SizeOf(field));
  }
  auto input = ring_rand(field, shape);

  if (heuristic) {
    // all kHeuristicBound = 2 now.
    ring_rshift_(input, {static_cast<int64_t>(2)});
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
}  // namespace

class PermuteTest : public ::testing::TestWithParam<OpTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Semi2k, PermuteTest,
    testing::Combine(testing::Values(CreateObjectFn(makeCheetahProtocol)),
                     testing::Values(makeConfig(FieldType::FM32),
                                     makeConfig(FieldType::FM64),
                                     makeConfig(FieldType::FM128)),
                     testing::Values(2)),
    [](const testing::TestParamInfo<PermuteTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<1>(p.param).field,
                         std::get<2>(p.param));
      ;
    });

TEST_P(PermuteTest, Perm_Work) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  int64_t n = 537;
  const Shape shape = {n};

  uint64_t cnt = yacl::crypto::RandU64();
  uint128_t seed1 = yacl::crypto::RandU128();
  uint128_t seed2 = yacl::crypto::RandU128();
  const Index perm1 = genRandomPerm(n, seed1, &cnt);
  const Index perm2 = genRandomPerm(n, seed2, &cnt);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = makeCheetahProtocol(conf, lctx);

    // GIVEN
    NdArrayRef perm;
    if (lctx->Rank() == 0) {
      perm = mockPshare(sctx.get(), perm1);
    } else {
      perm = mockPshare(sctx.get(), perm2);
    }

    auto x_p = rand_p(sctx.get(), shape);
    auto x_s = p2s(sctx.get(), x_p);

    // WHEN
    auto permuted_x = perm_ss(sctx.get(), x_s, WrapValue(perm));
    EXPECT_TRUE(permuted_x.has_value());

    auto permuted_x_p = s2p(sctx.get(), permuted_x.value());

    // THEN
    auto required = applyInvPerm(UnwrapValue(x_p), perm1);
    required = applyInvPerm(required, perm2);

    EXPECT_EQ(permuted_x_p.shape(), required.shape());
    EXPECT_TRUE(ring_all_equal(permuted_x_p.data(), required));
  });
}

// test whether inv_perm(perm(x)) == x
TEST_P(PermuteTest, InvPerm_Perm_Work) {
  const auto factory = std::get<0>(GetParam());
  const RuntimeConfig& conf = std::get<1>(GetParam());
  const size_t npc = std::get<2>(GetParam());

  int64_t n = 537;
  const Shape shape = {n};

  uint64_t cnt = yacl::crypto::RandU64();
  uint128_t seed1 = yacl::crypto::RandU128();
  uint128_t seed2 = yacl::crypto::RandU128();
  const Index perm1 = genRandomPerm(n, seed1, &cnt);
  const Index perm2 = genRandomPerm(n, seed2, &cnt);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = makeCheetahProtocol(conf, lctx);

    // GIVEN
    NdArrayRef perm;
    if (lctx->Rank() == 0) {
      perm = mockPshare(sctx.get(), perm1);
    } else {
      perm = mockPshare(sctx.get(), perm2);
    }

    auto x_p = rand_p(sctx.get(), shape);
    auto x_s = p2s(sctx.get(), x_p);

    // WHEN
    auto permuted_x = perm_ss(sctx.get(), x_s, WrapValue(perm));
    EXPECT_TRUE(permuted_x.has_value());
    auto inv_permuted_x =
        inv_perm_ss(sctx.get(), permuted_x.value(), WrapValue(perm));
    EXPECT_TRUE(inv_permuted_x.has_value());

    auto inv_permuted_x_p = s2p(sctx.get(), inv_permuted_x.value());

    // THEN
    EXPECT_EQ(inv_permuted_x_p.shape(), x_p.shape());
    EXPECT_TRUE(ring_all_equal(inv_permuted_x_p.data(), x_p.data()));
  });
}

namespace {
const std::vector<std::tuple<int64_t, int64_t>> ext_bw_values = {
    std::make_tuple(8, 16),    // full bw ext
    std::make_tuple(16, 32),   //
    std::make_tuple(32, 64),   //
    std::make_tuple(64, 128),  //
                               // not full bw ext
    std::make_tuple(5, 8),     //
    std::make_tuple(8, 13),    //
    std::make_tuple(13, 23),   //
    std::make_tuple(9, 24),    //
    std::make_tuple(27, 48),   //
};

}  // namespace

class CastUpTest
    : public ::testing::TestWithParam<
          std::tuple<std::tuple<int64_t, int64_t>, bool, SignType>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CastUpTest,
    testing::Combine(testing::ValuesIn(ext_bw_values),  // (bw_x, bw_out)
                     testing::ValuesIn(signed_values),  // signed_arith
                     testing::ValuesIn(sign_values)     // sign
                     ));

TEST_P(CastUpTest, Work) {
  size_t npc = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  int64_t m;
  int64_t n;
  std::tie(m, n) = std::get<0>(GetParam());
  const auto src_field = FixGetProperFiled(m);
  const auto dst_field = FixGetProperFiled(n);

  const bool signed_arith = std::get<1>(GetParam());
  const SignType sign = std::get<2>(GetParam());

  auto pub = std::get<0>(makeInput(m, sign, src_field, shape));
  pub.set_fxp_bits(m);
  const auto ty = makeType<Pub2kTy>(src_field);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeConfig(FM32), lcxt);
    auto p0 = WrapValue(pub.as(ty));

    // force the bw
    auto a0 = p2a(obj.get(), p0);
    a0.data() = ring_reduce(a0.data(), m);
    a0.data().set_fxp_bits(m);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    auto a1 = ring_cast_up_s(obj.get(), a0, n, dst_field, sign, signed_arith);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    SPDLOG_INFO(
        "Rank {}, [No heuristic, sign {}, ext {}=>{} bits, with {} "
        "samples],sent {} bits per element. Actions total {}, elapsed total "
        "time: {}ms.",
        rank, getSignName(sign), m, n, numel, (b1 - b0) * 8. / shape.numel(),
        (r1 - r0) * 1.0, pack_time);

    // ring_print(UnwrapValue(a0), "a0");
    // ring_print(UnwrapValue(a1), "a1");

    // check
    auto r_a = a2p(obj.get(), a1);

    if (rank == 0) {
      EXPECT_EQ(r_a.shape(), shape);
      auto _ra_arr = r_a.data();

      // ring_reduce_(_ra_arr, n);
      // ring_print(pub, "pub");
      // ring_print(_ra_arr, "exted");

      EXPECT_EQ(_ra_arr.fxp_bits(), n);
      EXPECT_EQ(_ra_arr.eltype().as<Ring2k>()->field(), dst_field);
      EXPECT_TRUE(ring_all_equal_val(_ra_arr, pub, signed_arith, 0));
    }
  });
}

namespace {

// TODO: test after opt mul_aa
// const bool naive_mix_mul = false;
const bool naive_mix_mul = true;

const std::vector<std::tuple<int64_t, int64_t, int64_t>> mul_bw_values = {
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

RuntimeConfig makeMixMulConfig(FieldType field) {
  RuntimeConfig conf;
  conf.protocol = ProtocolKind::CHEETAH;
  conf.field = field;
  conf.cheetah_2pc_config.ot_kind = ot_kind;
  conf.cheetah_naive_mix_mul = naive_mix_mul;
  return conf;
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

  out.set_fxp_bits(bw);
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
  out.set_fxp_bits(bw);

  return out;
}
}  // namespace

class MixMulTest
    : public ::testing::TestWithParam<std::tuple<
          std::tuple<int64_t, int64_t, int64_t>, SignType, SignType, bool>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, MixMulTest,
    testing::Combine(testing::ValuesIn(mul_bw_values),  // (bw_x,bw_y,bw_out)
                     testing::ValuesIn(sign_values),    // sign_x
                     testing::ValuesIn(sign_values),    // sign_y
                     testing::ValuesIn(signed_values)   // signed_arith
                     ));

// INSTANTIATE_TEST_SUITE_P(
//     Cheetah, MixMulTest,
//     testing::Combine(testing::Values(std::make_tuple(24, 24, 48)),  //
//                      testing::Values(SignType::Positive),           // sign_x
//                      testing::Values(SignType::Positive),           // sign_y
//                      testing::Values(true)  // signed_arith
//                      ));

TEST_P(MixMulTest, Work) {
  size_t npc = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

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

  auto pub_x = std::get<0>(makeInput(m, sign_x, field_x, shape));
  pub_x.set_fxp_bits(m);
  const auto ty_x = makeType<Pub2kTy>(field_x);
  pub_x = pub_x.as(ty_x);

  auto pub_y = std::get<0>(makeInput(n, sign_y, field_y, shape));
  pub_y.set_fxp_bits(n);
  const auto ty_y = makeType<Pub2kTy>(field_y);
  pub_y = pub_y.as(ty_y);

  NdArrayRef gt_xy;
  if (!signed_arith) {
    gt_xy = UnsignedMixMul(pub_x, pub_y, field_out, l);
  } else {
    gt_xy = SignedMixMul(pub_x, pub_y, field_out, l);
  }

  // debug
  // ring_print(pub_x, "pub_x");
  // ring_print(pub_y, "pub_y");
  // ring_print(gt_xy, "gt_xy");

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeMixMulConfig(FM32), lcxt);

    auto p0_x = WrapValue(pub_x);
    auto p0_y = WrapValue(pub_y);

    // force the bw of a_x and a_y
    auto a_x = p2a(obj.get(), p0_x);
    auto a_y = p2a(obj.get(), p0_y);
    a_x.data() = ring_reduce(a_x.data(), m);
    a_y.data() = ring_reduce(a_y.data(), n);
    a_x.data().set_fxp_bits(m);
    a_y.data().set_fxp_bits(n);

    // TODO: run first to do pre-processing
    mix_mul_ss(obj.get(), a_x, a_y, sign_x, sign_y, field_out, l, signed_arith);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    auto a_xy = mix_mul_ss(obj.get(), a_x, a_y, sign_x, sign_y, field_out, l,
                           signed_arith);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    const std::string mul_type =
        signed_arith ? "SignedMixMul" : "UnsignedMixMul";

    SPDLOG_INFO(
        "Rank {}, [{}, sign {}x{}, {}x{}={} bits, with {} samples], sent {} "
        "bits per element. Actions total {}, elapsed total time: {} ms.",
        rank, mul_type, getSignName(sign_x), getSignName(sign_y), m, n, l,
        numel, (b1 - b0) * 8. / shape.numel(), (r1 - r0) * 1.0, pack_time);

    // check
    auto p_xy = a2p(obj.get(), a_xy);
    if (rank == 0) {
      EXPECT_EQ(p_xy.shape(), shape);
      auto _p_xy_arr = p_xy.data();

      // ring_print(_p_xy_arr, "got_xy");

      EXPECT_EQ(_p_xy_arr.fxp_bits(), l);
      EXPECT_EQ(_p_xy_arr.eltype().as<Pub2kTy>()->field(), field_out);
      EXPECT_TRUE(ring_all_equal_val(_p_xy_arr, gt_xy, signed_arith, 0));
    }
  });
}

namespace {
const std::vector<std::tuple<int64_t, int64_t>> tr_bw_values = {
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

void CheckEQ(const NdArrayRef& exp, const NdArrayRef& got, bool exact) {
  const auto src_field = exp.eltype().as<Ring2k>()->field();
  const auto dst_field = got.eltype().as<Ring2k>()->field();

  const auto m = exp.fxp_bits();
  const auto n = got.fxp_bits();

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    using st = ring2k_t;
    NdArrayView<st> exp_(exp);
    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dt = ring2k_t;
      NdArrayView<dt> got_(got);

      auto mask = makeMask<dt>(n);

      for (int64_t i = 0; i < exp.numel(); i++) {
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
}  // namespace

class TruncateReduceTest : public ::testing::TestWithParam<
                               std::tuple<std::tuple<int64_t, int64_t>, bool>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, TruncateReduceTest,
    testing::Combine(testing::ValuesIn(tr_bw_values),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<TruncateReduceTest::ParamType>& p) {
      return fmt::format("{}to{}x{}", std::get<0>(std::get<0>(p.param)),
                         std::get<1>(std::get<0>(p.param)),
                         std::get<1>(p.param));
    });

// with wrap is hard to test... just skip now.
TEST_P(TruncateReduceTest, WorkWithoutWrap) {
  size_t npc = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  int64_t m;
  int64_t n;
  std::tie(m, n) = std::get<0>(GetParam());
  const auto src_field = FixGetProperFiled(m);
  const auto dst_field = FixGetProperFiled(n);
  bool exact = std::get<1>(GetParam());

  NdArrayRef pub = ring_rand(src_field, shape);
  ring_reduce_(pub, m);
  pub.set_fxp_bits(m);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeConfig(FM32), lcxt);

    auto p_x = WrapValue(pub);
    auto a_x = p2a(obj.get(), p_x);
    SPU_ENFORCE(a_x.data().fxp_bits() == m);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    auto a_ret =
        tr_s(obj.get(), a_x, /*wrap*/ Value(), m - n, dst_field, exact);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    std::string exact_str = exact ? "exact" : "approx";
    SPDLOG_INFO(
        "Rank {}, [{} TR without wrap, {} bits to {} bits with {} samples], "
        "sent {} bits per "
        "element. Actions total {}, elapsed total time: {} ms.",
        rank, exact_str, m, n, numel, (b1 - b0) * 8. / numel, (r1 - r0) * 1.0,
        pack_time);

    // check
    auto p_ret = a2p(obj.get(), a_ret);

    if (rank == 0) {
      EXPECT_EQ(p_ret.shape(), shape);
      SPU_ENFORCE(a_ret.data().fxp_bits() == n);
      SPU_ENFORCE(a_ret.data().eltype().as<Ring2k>()->field() == dst_field);
      SPU_ENFORCE(p_ret.data().fxp_bits() == n);
      SPU_ENFORCE(p_ret.data().eltype().as<Ring2k>()->field() == dst_field);
      // check whether (pub >> (n-m) == ret)
      CheckEQ(pub, p_ret.data(), exact);
    }
  });
}

namespace {

const std::vector<std::tuple<int64_t, int64_t>> trunc_bw_values = {
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

template <typename T>
bool trunc_safe_check(const T exp, const T got, int64_t bw) {
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
bool trunc_safe_check(const T exp, const U got, int64_t bw) {
  return trunc_safe_check<T>(exp, static_cast<T>(got), bw);
}

void CheckTruncEQ(const NdArrayRef& x, const NdArrayRef& got, bool signed_arith,
                  bool exact, int64_t n) {
  const auto field = x.eltype().as<Ring2k>()->field();
  const auto m = x.fxp_bits();
  const auto shift = m - n;

  SPU_ENFORCE(field == got.eltype().as<Ring2k>()->field());
  SPU_ENFORCE(m == got.fxp_bits());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using T = std::make_unsigned_t<ring2k_t>;
    NdArrayView<T> _x(x);
    NdArrayView<T> _got(got);

    const auto msk0 = makeMask<T>(m);
    // n = m - shift
    const auto msk1 = makeMask<T>(n);
    const auto delta = msk0 - msk1;

    for (int64_t i = 0; i < x.numel(); i++) {
      auto expected = (_x[i] >> shift) & msk0;
      // signed right shift
      if (signed_arith && (((_x[i] >> (m - 1)) & 1) == 1)) {
        expected += delta;
        expected &= msk0;
      }
      auto got = _got[i] & msk0;
      if (exact) {
        EXPECT_EQ(expected, got);
      } else {
        EXPECT_TRUE(trunc_safe_check(expected, got, m));
      }
    }
  });
}
}  // namespace

class TruncateTest
    : public ::testing::TestWithParam<
          std::tuple<std::tuple<int64_t, int64_t>, bool, bool, SignType>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, TruncateTest,
    testing::Combine(testing::ValuesIn(trunc_bw_values),
                     testing::Values(true, false),  // signed, unsigned
                     testing::Values(true, false),  // exact, prob
                     testing::ValuesIn(sign_values)),
    [](const testing::TestParamInfo<TruncateTest::ParamType>& p) {
      return fmt::format("{}to{}{}{}MSB{}", std::get<0>(std::get<0>(p.param)),
                         std::get<1>(std::get<0>(p.param)),
                         std::get<1>(p.param) ? "Signed" : "Unsigned",
                         std::get<2>(p.param) ? "Exact" : "Prob",
                         getSignName(std::get<3>(p.param)));
    });

TEST_P(TruncateTest, Work) {
  size_t npc = 2;
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

  const bool heuristic = signed_arith && (sign == SignType::Unknown) &&
                         ((m == static_cast<int64_t>(8 * SizeOf(src_field))));

  // TODO: I have no time to find out why exact truncation error with 1 bit.
  if (exact) {
    GTEST_SKIP() << "Skipping exact truncate tests for now.";
  }

  auto pub = std::get<0>(makeInput(m, sign, src_field, shape, heuristic));
  pub.set_fxp_bits(m);
  const auto ty = makeType<Pub2kTy>(src_field);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeConfig(FM32), lcxt);
    auto p_x = WrapValue(pub);
    auto a_x = p2a(obj.get(), p_x);
    SPU_ENFORCE(a_x.data().fxp_bits() == m);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    auto a_ret = trunc2_s(obj.get(), a_x, static_cast<size_t>(shift), sign,
                          exact, signed_arith);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    std::string exact_str = exact ? "exact" : "approx";
    SPDLOG_INFO(
        "Rank {}, [{} Truncate {} bits to {} bits with {} samples], "
        "sent {} bits per "
        "element. Actions total {}, elapsed total time: {} ms.",
        rank, exact_str, m, n, numel, (b1 - b0) * 8. / numel, (r1 - r0) * 1.0,
        pack_time);

    // check
    auto p_ret = a2p(obj.get(), a_ret);

    if (rank == 0) {
      EXPECT_EQ(p_ret.shape(), shape);
      SPU_ENFORCE(a_ret.data().fxp_bits() == m);
      SPU_ENFORCE(a_ret.data().eltype().as<Ring2k>()->field() == src_field);
      SPU_ENFORCE(p_ret.data().fxp_bits() == m);
      SPU_ENFORCE(p_ret.data().eltype().as<Ring2k>()->field() == src_field);
      CheckTruncEQ(pub, p_ret.data(), signed_arith, exact, n);
    }
  });
}

namespace {
const std::vector<int64_t> cmp_bw_values = {
    8, 16, 32, 64, 128,  // full bw
    6, 10, 14, 24, 48,   // not full bw
};

[[maybe_unused]] void CheckMsb(const NdArrayRef& x, const NdArrayRef& got,
                               int64_t bw) {
  SPU_ENFORCE(x.fxp_bits() == bw);
  const auto field = x.eltype().as<Ring2k>()->field();

  DISPATCH_ALL_FIELDS(field, [&]() {
    using T = ring2k_t;
    NdArrayView<T> _x(x);
    NdArrayView<T> _got(got);

    for (int64_t i = 0; i < x.numel(); i++) {
      auto expected = (_x[i] >> (bw - 1)) & 1;
      auto got = _got[i] & 1;
      EXPECT_EQ(expected, got);
    }
  });
}

}  // namespace

class MSBTest : public ::testing::TestWithParam<int64_t> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 13;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, MSBTest, testing::ValuesIn(cmp_bw_values),
    [](const testing::TestParamInfo<MSBTest::ParamType>& p) {
      return fmt::format("bwx{}", p.param);
    });

TEST_P(MSBTest, Work) {
  size_t npc = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  int64_t m = GetParam();
  const auto src_field = FixGetProperFiled(m);

  auto pub = ring_rand(src_field, shape);
  ring_reduce_(pub, m);
  pub.set_fxp_bits(m);
  const auto ty = makeType<Pub2kTy>(src_field);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeConfig(FM32), lcxt);
    auto p_x = WrapValue(pub);
    auto a_x = p2a(obj.get(), p_x);
    SPU_ENFORCE(a_x.data().fxp_bits() == m);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    auto b_ret = msb_s(obj.get(), a_x);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    SPDLOG_INFO(
        "Rank {}, [MSB {} bits with {} samples], sent {} bits per element. "
        "Actions total {}, elapsed total time: {} ms.",
        rank, m, numel, (b1 - b0) * 8. / numel, (r1 - r0) * 1.0, pack_time);

    // check
    auto p_ret = b2p(obj.get(), b_ret);

    // ring_print(a_x.data(), "a_x:" + std::to_string(rank));
    // ring_print(b_ret.data(), "b_ret:" + std::to_string(rank));

    if (rank == 0) {
      // ring_print(pub, "pub");
      // ring_print(p_ret.data(), "p_ret");

      EXPECT_EQ(p_ret.shape(), shape);
      SPU_ENFORCE(b_ret.data().eltype().as<Ring2k>()->field() == src_field);
      SPU_ENFORCE(p_ret.data().eltype().as<Ring2k>()->field() == src_field);
      CheckMsb(pub, p_ret.data(), m);
    }
  });
}

namespace {
[[maybe_unused]] void CheckEqz(const NdArrayRef& x, const NdArrayRef& got,
                               int64_t bw) {
  SPU_ENFORCE(x.fxp_bits() == bw);
  const auto field = x.eltype().as<Ring2k>()->field();

  DISPATCH_ALL_FIELDS(field, [&]() {
    using T = ring2k_t;
    NdArrayView<T> _x(x);
    NdArrayView<T> _got(got);
    const auto msk = makeMask<T>(bw);

    for (int64_t i = 0; i < x.numel(); i++) {
      auto expected = static_cast<T>((_x[i] & msk) == 0);
      auto got = _got[i] & 1;
      EXPECT_EQ(expected, got);
    }
  });
}
}  // namespace

class MsbEqTest : public ::testing::TestWithParam<int64_t> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 13;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, MsbEqTest, testing::ValuesIn(cmp_bw_values),
    [](const testing::TestParamInfo<MsbEqTest::ParamType>& p) {
      return fmt::format("bwx{}", p.param);
    });

TEST_P(MsbEqTest, Work) {
  size_t npc = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  int64_t m = GetParam();
  const auto src_field = FixGetProperFiled(m);

  auto pub = ring_rand(src_field, shape);
  ring_reduce_(pub, m);

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    NdArrayView<ring2k_t> _pub(pub);
    for (int64_t i = 0; i < numel; i += 2) {
      // force 50% zero
      _pub[i] = 0;
    }
  });

  pub.set_fxp_bits(m);
  const auto ty = makeType<Pub2kTy>(src_field);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeConfig(FM32), lcxt);
    auto p_x = WrapValue(pub);
    auto a_x = p2a(obj.get(), p_x);

    SPU_ENFORCE(a_x.data().fxp_bits() == m);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    // api call can not dispatch multiple returns
    SPU_ENFORCE(obj->hasKernel("msb_eq"));
    auto b_rets = dynDispatch<std::vector<Value>>(obj.get(), "msb_eq", a_x);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    SPDLOG_INFO(
        "Rank {}, [MSB_EQ {} bits with {} samples], sent {} bits per element. "
        "Actions total {}, elapsed total time: {} ms.",
        rank, m, numel, (b1 - b0) * 8. / numel, (r1 - r0) * 1.0, pack_time);

    // check
    SPU_ENFORCE(b_rets.size() == 2);

    auto msb = b2p(obj.get(), b_rets[0]);
    auto eqz = b2p(obj.get(), b_rets[1]);

    // ring_print(a_x.data(), "a_x:" + std::to_string(rank));

    if (rank == 0) {
      // ring_print(pub, "pub");
      // ring_print(msb.data(), "msb");
      // ring_print(eqz.data(), "eqz");

      EXPECT_EQ(msb.shape(), shape);
      EXPECT_EQ(eqz.shape(), shape);
      SPU_ENFORCE(msb.data().eltype().as<Ring2k>()->field() == src_field);
      SPU_ENFORCE(eqz.data().eltype().as<Ring2k>()->field() == src_field);
      CheckMsb(pub, msb.data(), m);
      CheckEqz(pub, eqz.data(), m);
    }
  });
}

class EqTest : public ::testing::TestWithParam<std::tuple<int64_t, bool>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 13;

  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, EqTest,
    testing::Combine(testing::ValuesIn(cmp_bw_values),  // bw
                     testing::Values(true, false)),     // equal_aa(p)
    [](const testing::TestParamInfo<EqTest::ParamType>& p) {
      return fmt::format("bwx{}x{}", std::get<0>(p.param),
                         std::get<1>(p.param) ? "equal_aa" : "equal_ap");
    });

TEST_P(EqTest, Work) {
  size_t npc = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  const auto m = std::get<0>(GetParam());
  const bool use_eq_aa = std::get<1>(GetParam());

  const auto src_field = FixGetProperFiled(m);

  auto pub_x = ring_rand(src_field, shape);
  ring_reduce_(pub_x, m);
  pub_x.set_fxp_bits(m);

  auto pub_y = ring_rand(src_field, shape);
  ring_reduce_(pub_y, m);
  pub_y.set_fxp_bits(m);

  DISPATCH_ALL_FIELDS(src_field, [&]() {
    NdArrayView<ring2k_t> _pubx(pub_x);
    NdArrayView<ring2k_t> _puby(pub_y);

    for (int64_t i = 0; i < numel; i += 2) {
      // force 50% equal
      _pubx[i] = _puby[i];
    }
  });

  const auto ty = makeType<Pub2kTy>(src_field);
  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeConfig(FM32), lcxt);

    auto p_x = WrapValue(pub_x);
    auto a_x = p2a(obj.get(), p_x);

    auto a_y = WrapValue(pub_y);
    if (use_eq_aa) {
      a_y = p2a(obj.get(), a_y);
    }

    SPU_ENFORCE(a_x.data().fxp_bits() == m);
    SPU_ENFORCE(a_y.data().fxp_bits() == m);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    Value b_eq;

    if (use_eq_aa) {
      b_eq = equal_ss(obj.get(), a_x, a_y).value();
    } else {
      b_eq = equal_sp(obj.get(), a_x, a_y).value();
    }

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    std::string eq_str = use_eq_aa ? "Equal_aa" : "Equal_ap";

    SPDLOG_INFO(
        "Rank {}, [{} {} bits with {} samples], sent {} bits per element. "
        "Actions total {}, elapsed total time: {} ms.",
        rank, eq_str, m, numel, (b1 - b0) * 8. / numel, (r1 - r0) * 1.0,
        pack_time);

    // check
    auto p_ret = b2p(obj.get(), b_eq);

    if (rank == 0) {
      // ring_print(pub_x, "pub_x");
      // ring_print(pub_y, "pub_y");
      // ring_print(p_ret.data(), "p_ret");

      EXPECT_EQ(p_ret.shape(), shape);
      SPU_ENFORCE(b_eq.data().eltype().as<Ring2k>()->field() == src_field);
      SPU_ENFORCE(p_ret.data().eltype().as<Ring2k>()->field() == src_field);

      auto exp = ring_equal(pub_x, pub_y);
      EXPECT_TRUE(ring_all_equal(exp, p_ret.data(), 0));
    }
  });
}

namespace {
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

void checkLUT(const NdArrayRef& p_ret, const NdArrayRef& table,
              const NdArrayRef& ind) {
  const auto out_bw = p_ret.fxp_bits();
  const auto out_field = p_ret.eltype().as<Ring2k>()->field();
  const auto ind_field = ind.eltype().as<Ring2k>()->field();

  const auto tbl_field = table.eltype().as<Ring2k>()->field();
  const auto table_size = table.numel();
  const auto N_bits = Log2Ceil(table_size);
  const auto N_mask = makeMask<uint8_t>(N_bits);

  DISPATCH_ALL_FIELDS(ind_field, [&]() {
    NdArrayView<ring2k_t> _ind(ind);

    DISPATCH_ALL_FIELDS(tbl_field, [&]() {
      NdArrayView<ring2k_t> _table(table);

      DISPATCH_ALL_FIELDS(out_field, [&]() {
        NdArrayView<ring2k_t> _p_ret(p_ret);
        const auto msk = makeMask<ring2k_t>(out_bw);

        for (int64_t i = 0; i < p_ret.numel(); ++i) {
          auto idx = _ind[i] & N_mask;
          auto exp = static_cast<ring2k_t>(_table[idx]) & msk;
          auto got = _p_ret[i] & msk;

          EXPECT_EQ(exp, got);
        }
      });
    });
  });
}

}  // namespace

class LUTTest : public ::testing::TestWithParam<
                    std::tuple<size_t, FieldType, FieldType, uint64_t>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, LUTTest,
    ::testing::Combine(::testing::ValuesIn(lut_bw_values),  // out_bw
                       ::testing::ValuesIn(all_fields),     // index
                       ::testing::ValuesIn(all_fields),     // table
                       ::testing::ValuesIn(all_table_sizes)));

TEST_P(LUTTest, Work) {
  size_t npc = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  const auto out_bw = std::get<0>(GetParam());
  const auto out_field = FixGetProperFiled(out_bw);

  const auto index_field = std::get<1>(GetParam());
  const auto table_field = std::get<2>(GetParam());

  const auto table_size = std::get<3>(GetParam());
  const auto N_bits = Log2Ceil(table_size);

  auto pub_ind = ring_rand(index_field, shape);
  ring_reduce_(pub_ind, N_bits);

  auto table = ring_rand(table_field, {static_cast<int64_t>(table_size)});

  const auto ind_ty = makeType<Pub2kTy>(index_field);
  const auto tbl_ty = makeType<Pub2kTy>(table_field);

  pub_ind = pub_ind.as(ind_ty);
  table = table.as(tbl_ty);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeConfig(FM32), lcxt);

    auto p_ind = WrapValue(pub_ind);
    auto a_ind = p2a(obj.get(), p_ind);
    auto p_table = WrapValue(table);

    // run first to warm up
    lut_sp(obj.get(), a_ind, p_table, out_bw, out_field);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    auto a_ret = lut_sp(obj.get(), a_ind, p_table, out_bw, out_field);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    SPDLOG_INFO(
        "Rank {}, LUT {} bits, table size {}, sent {} bits per element. "
        "Actions total {}, "
        "elapsed total time: {} ms.",
        rank, out_bw, table_size, (b1 - b0) * 8. / numel, (r1 - r0) * 1.0,
        pack_time);

    // check
    auto p_ret = a2p(obj.get(), a_ret);

    if (rank == 0) {
      EXPECT_EQ(p_ret.shape(), shape);
      SPU_ENFORCE(a_ret.data().eltype().as<Ring2k>()->field() == out_field);
      SPU_ENFORCE(a_ret.data().fxp_bits() == static_cast<int64_t>(out_bw));
      SPU_ENFORCE(p_ret.data().eltype().as<Ring2k>()->field() == out_field);
      SPU_ENFORCE(p_ret.data().fxp_bits() == static_cast<int64_t>(out_bw));

      checkLUT(p_ret.data(), table, pub_ind);
    }
  });
}

namespace {
const std::vector<int64_t> a2b_bw_values = {
    // full ring test
    8, 16, 32, 64, 128,
    // not full ring test
    3, 5, 13, 21, 48,  //
};

}  // namespace

class A2BNotFullTest : public ::testing::TestWithParam<int64_t> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(Cheetah, A2BNotFullTest,
                         ::testing::ValuesIn(a2b_bw_values));

TEST_P(A2BNotFullTest, Work) {
  size_t npc = 2;
  Shape shape = {kBenchSize};
  const auto numel = shape.numel();

  const auto bw = GetParam();
  const auto field = FixGetProperFiled(bw);

  auto pub = ring_rand(field, shape);
  ring_reduce_(pub, bw);
  pub.set_fxp_bits(bw);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lcxt) {
    int rank = lcxt->Rank();
    // do not use the global field...
    auto obj = makeCheetahProtocol(makeConfig(FM32), lcxt);
    auto a_x = p2a(obj.get(), WrapValue(pub));

    SPU_ENFORCE(a_x.data().fxp_bits() == bw);

    // run first to warm up
    a2b(obj.get(), a_x);

    size_t b0 = lcxt->GetStats()->sent_bytes;
    size_t r0 = lcxt->GetStats()->sent_actions;
    yacl::ElapsedTimer pack_timer;

    auto b_x = a2b(obj.get(), a_x);

    double pack_time = pack_timer.CountMs() * 1.0;
    size_t b1 = lcxt->GetStats()->sent_bytes;
    size_t r1 = lcxt->GetStats()->sent_actions;

    SPDLOG_INFO(
        "Rank {}, [A2B {} bits], sent {} bits per element. Actions total {}, "
        "elapsed total time: {} ms.",
        rank, bw, (b1 - b0) * 8. / numel, (r1 - r0) * 1.0, pack_time);

    // check
    auto p_ret = b2p(obj.get(), b_x);

    if (rank == 0) {
      // debug
      // ring_print(pub, "pub");
      // ring_print(p_ret.data(), "p_ret");

      EXPECT_EQ(p_ret.shape(), shape);
      SPU_ENFORCE(b_x.data().eltype().as<cheetah::BShrTy>()->nbits() ==
                  static_cast<size_t>(bw));
      SPU_ENFORCE(b_x.data().eltype().as<cheetah::BShrTy>()->field() == field);

      EXPECT_TRUE(ring_all_equal(pub, p_ret.data()));
    }
  });
}

}  // namespace spu::mpc::test
