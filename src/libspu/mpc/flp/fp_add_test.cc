#include "libspu/mpc/flp/fp_add.h"

#include <random>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::flp {
namespace {

const int kP = 8;
const int kQ = 23;

std::unique_ptr<SPUContext> MakeCheetahCtx(
    const std::shared_ptr<yacl::link::Context>& lctx) {
  RuntimeConfig conf;
  conf.protocol = ProtocolKind::CHEETAH;
  conf.field = FM64;
  return makeCheetahProtocol(conf, lctx);
}

Value MakePublic(SPUContext* ctx, int64_t val, const Shape& shape,
                 int64_t fxp_bits = 0) {
  auto v = make_p(ctx, static_cast<uint128_t>(static_cast<uint64_t>(val)),
                  shape, FM64);
  if (fxp_bits > 0) {
    v.data().set_fxp_bits(fxp_bits);
  }
  return v;
}

Value MakeSecret(SPUContext* ctx, int64_t val, const Shape& shape,
                 int64_t fxp_bits = 0) {
  return p2s(ctx, MakePublic(ctx, val, shape, fxp_bits));
}

SharedFloat MakeSharedFloat(SPUContext* ctx, int64_t z, int64_t s, int64_t e,
                            int64_t m, const Shape& shape) {
  auto z_s = MakeSecret(ctx, z, shape);
  auto s_s = MakeSecret(ctx, s, shape);
  auto e_s = MakeSecret(ctx, e, shape);
  auto m_s = MakeSecret(ctx, m, shape);
  return SharedFloat(z_s, s_s, e_s, m_s, kP, kQ);
}

void CheckValue(const Value& got, const Value& expected) {
  EXPECT_EQ(got.shape(), expected.shape());
  EXPECT_TRUE(ring_all_equal(got.data(), expected.data()));
}

void CheckSharedFloat(SPUContext* ctx, const SharedFloat& got, int64_t z,
                      int64_t s, int64_t e, int64_t m, const Shape& shape) {
  CheckValue(s2p(ctx, got.z), MakePublic(ctx, z, shape));
  CheckValue(s2p(ctx, got.s), MakePublic(ctx, s, shape));
  CheckValue(s2p(ctx, got.e), MakePublic(ctx, e, shape));
  CheckValue(s2p(ctx, got.m), MakePublic(ctx, m, shape));
}

// Plaintext reference that mirrors FPAdd algorithm step by step.
struct PlainFloat {
  int64_t z;
  int64_t s;
  int64_t e;
  int64_t m;
};

int FindHighestBit(uint64_t val) {
  if (val == 0) return 0;
  int k = 63;
  while (k >= 0 && ((val >> k) & 1) == 0) --k;
  return k;
}

PlainFloat FPAddPlainRef(const PlainFloat& a, const PlainFloat& b) {
  const int q = kQ;
  const int p = kP;

  // Steps 1-3: determine larger (by exponent, then mantissa)
  bool swap = false;
  if (a.e < b.e) {
    swap = true;
  } else if (a.e == b.e && a.m < b.m) {
    swap = true;
  }

  const PlainFloat& large_src = swap ? b : a;
  const PlainFloat& small_src = swap ? a : b;
  PlainFloat large = large_src;
  PlainFloat small = small_src;

  // Step 4: d = large.e - small.e
  int d = large.e - small.e;

  // Step 5-6: if d > q+1, return large
  if (d > q + 1) {
    return large;
  }

  // Step 8: m_large_shifted = large.m << d
  uint64_t m_large_shifted = static_cast<uint64_t>(large.m) << d;

  // Step 9: small.m used directly (already in full ring)
  uint64_t m_small_extended = static_cast<uint64_t>(small.m);

  // Step 10: conditional negation if signs differ
  bool sign_diff = (large.s != small.s);
  int64_t m_small_signed =
      sign_diff ? -static_cast<int64_t>(m_small_extended)
                : static_cast<int64_t>(m_small_extended);

  // Step 11: m_sum = m_large_shifted + m_small_signed, width 2q+3
  int64_t m_sum_int = static_cast<int64_t>(m_large_shifted) + m_small_signed;

  // Step 12: e_result = small.e
  int e_result = small.e;

  // Determine result sign: large dominates
  int s_result = large.s;
  int z_result = 0;

  if (m_sum_int < 0) {
    m_sum_int = -m_sum_int;
    s_result = small.s;
  }

  uint64_t m_sum = static_cast<uint64_t>(m_sum_int);

  if (m_sum == 0) {
    z_result = 1;
    return PlainFloat{z_result, s_result, static_cast<int64_t>(1 - (1LL << (p - 1))), 0};
  }

  // Step 13-15: MSNZB + normalize
  int k = FindHighestBit(m_sum);
  int shift = 2 * q + 1 - k;
  uint64_t m_norm;
  if (shift > 0) {
    m_norm = m_sum << shift;
  } else if (shift < 0) {
    m_norm = m_sum >> (-shift);
  } else {
    m_norm = m_sum;
  }
  int e_norm = e_result + k - q;

  // Step 16: Round (Q = 2q+1 bits -> q bits), always round-to-nearest
  const int Q = 2 * q + 1;
  const int s_round = Q - q;
  const uint64_t threshold =
      (uint64_t{1} << (Q + 1)) - (uint64_t{1} << (Q - q - 1));
  const uint64_t round_bias = uint64_t{1} << (s_round - 1);

  uint64_t m_final = (m_norm + round_bias) >> s_round;
  int e_final = e_norm;
  if (m_norm >= threshold) {
    e_final = e_norm + 1;
  }

  // Step 19: FPCheck
  const int64_t overflow_thresh = (1LL << (p - 1)) - 1;
  const int64_t max_e = 1LL << (p - 1);
  const int64_t min_e = 1 - (1LL << (p - 1));
  const uint64_t max_m = 1ULL << q;
  const int64_t underflow_thresh = 2 - (1LL << (p - 1));

  bool cond_ov = (e_final > overflow_thresh);
  bool cond_un = (z_result == 1) || (e_final < underflow_thresh);

  if (cond_ov) {
    e_final = max_e;
    m_final = max_m;
  }
  if (cond_un) {
    z_result = 1;
    e_final = min_e;
    m_final = 0;
  }

  return PlainFloat{z_result, s_result, e_final, static_cast<int64_t>(m_final)};
}

}  // namespace

// 1.0 + 1.0 = 2.0  (same exponent, same sign)
TEST(FPAddTest, OneAndOne) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;  // mantissa representing 1.0

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    auto lhs = MakeSharedFloat(ctx.get(), 0, 0, 0, one_m, shape);
    auto rhs = MakeSharedFloat(ctx.get(), 0, 0, 0, one_m, shape);

    auto out = FPAdd(ctx.get(), lhs, rhs);

    // 1.0 + 1.0 = 2.0  ->  m unchanged (one_m), e = 1
    CheckSharedFloat(ctx.get(), out, 0, 0, 1, one_m, shape);
  });
}

// 1.0 * 2^3 + 1.0 * 2^0 = 9.0 * 2^0  (different exponents)
TEST(FPAddTest, DifferentExponents) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    auto lhs = MakeSharedFloat(ctx.get(), 0, 0, 3, one_m, shape);  // 1.0 * 2^3 = 8
    auto rhs = MakeSharedFloat(ctx.get(), 0, 0, 0, one_m, shape);  // 1.0 * 2^0 = 1

    auto out = FPAdd(ctx.get(), lhs, rhs);

    // 8 + 1 = 9.  Plaintext reference:
    auto expected = FPAddPlainRef(
        PlainFloat{0, 0, 3, one_m}, PlainFloat{0, 0, 0, one_m});
    CheckSharedFloat(ctx.get(), out, expected.z, expected.s, expected.e,
                     expected.m, shape);
  });
}

// Same magnitude, opposite signs: 2.0 + (-2.0) = 0
TEST(FPAddTest, OppositeSignsCancel) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;
  const int64_t min_e = 1 - (1LL << (kP - 1));

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    auto lhs = MakeSharedFloat(ctx.get(), 0, 0, 1, one_m, shape);  // 2.0
    auto rhs = MakeSharedFloat(ctx.get(), 0, 1, 1, one_m, shape);  // -2.0

    auto out = FPAdd(ctx.get(), lhs, rhs);

    // 2.0 + (-2.0) = 0  =>  z=1, m=0, e=min_e
    CheckSharedFloat(ctx.get(), out, 1, 0, min_e, 0, shape);
  });
}

// Exponent difference > q+1: small number is negligible
TEST(FPAddTest, LargeExpGap) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    // large: e = 100, small: e = 0, d = 100 > q+1 = 24
    auto large = MakeSharedFloat(ctx.get(), 0, 0, 100, one_m, shape);
    auto small = MakeSharedFloat(ctx.get(), 0, 0, 0, one_m, shape);

    auto out = FPAdd(ctx.get(), large, small);

    // Small is negligible -> result = large
    CheckSharedFloat(ctx.get(), out, 0, 0, 100, one_m, shape);
  });
}

// Zero input: 0 + x = x
TEST(FPAddTest, ZeroPlusValue) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    // zero: FPCheck will clamp z=1, e=min_e, m=0
    auto zero_input = MakeSharedFloat(ctx.get(), 1, 0, -127, 0, shape);
    auto value = MakeSharedFloat(ctx.get(), 0, 0, 5, one_m, shape);

    auto out = FPAdd(ctx.get(), zero_input, value);

    // zero + value = value (z=0, s=0, e=5, m=one_m)
    CheckSharedFloat(ctx.get(), out, 0, 0, 5, one_m, shape);
  });
}

// Different signs: 3.0 + (-1.0) = 2.0
TEST(FPAddTest, PositivePlusNegative) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;
  const int64_t one_half_m = one_m + (1LL << (kQ - 1));  // 1.5

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    // 1.5 * 2^1 = 3.0, positive
    auto lhs = MakeSharedFloat(ctx.get(), 0, 0, 1, one_half_m, shape);
    // 1.0 * 2^0 = 1.0, negative
    auto rhs = MakeSharedFloat(ctx.get(), 0, 1, 0, one_m, shape);

    auto out = FPAdd(ctx.get(), lhs, rhs);

    auto expected = FPAddPlainRef(
        PlainFloat{0, 0, 1, one_half_m}, PlainFloat{0, 1, 0, one_m});
    CheckSharedFloat(ctx.get(), out, expected.z, expected.s, expected.e,
                     expected.m, shape);
  });
}

// Swap case: smaller exponent first -> swap logic tested
TEST(FPAddTest, SwapOrder) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    // lhs (e=0) < rhs (e=3) -> swap should put rhs as large
    auto lhs = MakeSharedFloat(ctx.get(), 0, 0, 0, one_m, shape);
    auto rhs = MakeSharedFloat(ctx.get(), 0, 0, 3, one_m, shape);

    auto out1 = FPAdd(ctx.get(), lhs, rhs);
    auto out2 = FPAdd(ctx.get(), rhs, lhs);

    // FPAdd(lhs, rhs) == FPAdd(rhs, lhs)  (commutative)
    auto expected = FPAddPlainRef(
        PlainFloat{0, 0, 0, one_m}, PlainFloat{0, 0, 3, one_m});

    CheckSharedFloat(ctx.get(), out1, expected.z, expected.s, expected.e,
                     expected.m, shape);
    CheckSharedFloat(ctx.get(), out2, expected.z, expected.s, expected.e,
                     expected.m, shape);
  });
}

// Random values compared against plaintext reference
TEST(FPAddTest, RandomNormalValues) {
  const int npc = 2;
  const Shape shape = {1};

  constexpr int kNumCases = 80;
  std::mt19937_64 rng(202606051);

  std::uniform_int_distribution<int64_t> sign_dist(0, 1);
  std::uniform_int_distribution<int64_t> exp_dist(-5, 5);
  std::uniform_int_distribution<int64_t> mantissa_dist(1LL << kQ,
                                                       (1LL << (kQ + 1)) - 1);

  std::vector<std::tuple<PlainFloat, PlainFloat>> cases;
  cases.reserve(kNumCases);

  for (int i = 0; i < kNumCases; ++i) {
    cases.push_back({
        PlainFloat{0, sign_dist(rng), exp_dist(rng), mantissa_dist(rng)},
        PlainFloat{0, sign_dist(rng), exp_dist(rng), mantissa_dist(rng)},
    });
  }

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    for (const auto& [a_plain, b_plain] : cases) {
      auto expected = FPAddPlainRef(a_plain, b_plain);

      auto a = MakeSharedFloat(ctx.get(), a_plain.z, a_plain.s, a_plain.e,
                               a_plain.m, shape);
      auto b = MakeSharedFloat(ctx.get(), b_plain.z, b_plain.s, b_plain.e,
                               b_plain.m, shape);

      auto out = FPAdd(ctx.get(), a, b);

      CheckSharedFloat(ctx.get(), out, expected.z, expected.s, expected.e,
                       expected.m, shape);
    }
  });
}

}  // namespace spu::mpc::flp

