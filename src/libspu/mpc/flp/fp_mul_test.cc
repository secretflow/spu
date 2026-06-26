#include "libspu/mpc/flp/fp_mul.h"

#include <random>
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

  // Mantissa is q + 1 bits.
  auto m_s = MakeSecret(ctx, m, shape, kQ + 1);

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

}  // namespace

struct PlainFloat {
  int64_t z;
  int64_t s;
  int64_t e;
  int64_t m;
};

uint64_t MaskBits(size_t bits) {
  if (bits == 64) {
    return static_cast<uint64_t>(-1);
  }
  return (uint64_t{1} << bits) - 1;
}

uint64_t RNTEPlain(uint128_t x, size_t shift_bits, size_t out_bits) {
  uint128_t rounded = (x + (uint128_t{1} << (shift_bits - 1))) >> shift_bits;
  return static_cast<uint64_t>(rounded) & MaskBits(out_bits);
}

PlainFloat FPMulPlainRef(const PlainFloat& lhs, const PlainFloat& rhs) {
  const int q = kQ;

  PlainFloat out;
  out.z = lhs.z | rhs.z;
  out.s = lhs.s ^ rhs.s;

  int64_t e = lhs.e + rhs.e;
  uint128_t m_prod =
      static_cast<uint128_t>(lhs.m) * static_cast<uint128_t>(rhs.m);

  // Match current fp_mul.cc:
  // c = 1{m_prod < 2^{2q+1} - 2^{q-1}}
  const uint128_t threshold =
      (uint128_t{1} << (2 * q + 1)) - (uint128_t{1} << (q - 1));

  uint64_t m;
  if (m_prod < threshold) {
    // choose RNTE(m, q), exponent unchanged
    m = RNTEPlain(m_prod, static_cast<size_t>(q), static_cast<size_t>(q + 1));
  } else {
    // choose RNTE(m, q + 1), exponent + 1
    m = RNTEPlain(m_prod, static_cast<size_t>(q + 1),
                  static_cast<size_t>(q + 1));
    e += 1;
  }

  out.e = e;
  out.m = static_cast<int64_t>(m);

  // Match FPCheck.
  const int64_t overflow_thresh = (1LL << (kP - 1)) - 1;
  const int64_t max_e = 1LL << (kP - 1);
  const int64_t min_e = 1 - (1LL << (kP - 1));
  const uint64_t max_m = 1ULL << kQ;

  const bool cond_ov = out.e > overflow_thresh;
  const bool cond_un = (out.z == 1) || (out.e < (2 - (1LL << (kP - 1))));

  if (cond_ov) {
    out.e = max_e;
    out.m = static_cast<int64_t>(max_m);
  }

  if (cond_un) {
    out.z = 1;
    out.e = min_e;
    out.m = 0;
  }

  return out;
}

TEST(FPMulTest, OneTimesOne) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    auto lhs = MakeSharedFloat(ctx.get(), /*z=*/0, /*s=*/0, /*e=*/0,
                               /*m=*/one_m, shape);
    auto rhs = MakeSharedFloat(ctx.get(), /*z=*/0, /*s=*/0, /*e=*/0,
                               /*m=*/one_m, shape);

    auto out = FPMul(ctx.get(), lhs, rhs);

    CheckSharedFloat(ctx.get(), out, /*z=*/0, /*s=*/0, /*e=*/0,
                     /*m=*/one_m, shape);
  });
}

TEST(FPMulTest, ExponentAdd) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    auto lhs = MakeSharedFloat(ctx.get(), /*z=*/0, /*s=*/0, /*e=*/2,
                               /*m=*/one_m, shape);
    auto rhs = MakeSharedFloat(ctx.get(), /*z=*/0, /*s=*/0, /*e=*/3,
                               /*m=*/one_m, shape);

    auto out = FPMul(ctx.get(), lhs, rhs);

    CheckSharedFloat(ctx.get(), out, /*z=*/0, /*s=*/0, /*e=*/5,
                     /*m=*/one_m, shape);
  });
}

TEST(FPMulTest, SignXor) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    auto lhs = MakeSharedFloat(ctx.get(), /*z=*/0, /*s=*/0, /*e=*/0,
                               /*m=*/one_m, shape);
    auto rhs = MakeSharedFloat(ctx.get(), /*z=*/0, /*s=*/1, /*e=*/0,
                               /*m=*/one_m, shape);

    auto out = FPMul(ctx.get(), lhs, rhs);

    CheckSharedFloat(ctx.get(), out, /*z=*/0, /*s=*/1, /*e=*/0,
                     /*m=*/one_m, shape);
  });
}

TEST(FPMulTest, ZeroFlag) {
  const int npc = 2;
  const Shape shape = {1};
  const int64_t one_m = 1LL << kQ;
  const int64_t min_e = 1 - (1LL << (kP - 1));

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    auto lhs = MakeSharedFloat(ctx.get(), /*z=*/1, /*s=*/0, /*e=*/0,
                               /*m=*/one_m, shape);
    auto rhs = MakeSharedFloat(ctx.get(), /*z=*/0, /*s=*/0, /*e=*/0,
                               /*m=*/one_m, shape);

    auto out = FPMul(ctx.get(), lhs, rhs);

    // FPCheck should clamp zero result to z=1, e=min_e, m=0.
    CheckSharedFloat(ctx.get(), out, /*z=*/1, /*s=*/0, /*e=*/min_e,
                     /*m=*/0, shape);
  });
}

TEST(FPMulTest, RandomNormalValues) {
  const int npc = 2;
  const Shape shape = {1};

  constexpr int kNumCases = 160;
  std::mt19937_64 rng(202605529);

  std::uniform_int_distribution<int64_t> sign_dist(0, 1);
  std::uniform_int_distribution<int64_t> exp_dist(-10, 10);
  std::uniform_int_distribution<int64_t> mantissa_dist(1LL << kQ,
                                                       (1LL << (kQ + 1)) - 1);

  std::vector<PlainFloat> lhs_cases;
  std::vector<PlainFloat> rhs_cases;

  lhs_cases.reserve(kNumCases);
  rhs_cases.reserve(kNumCases);

  for (int i = 0; i < kNumCases; ++i) {
    lhs_cases.push_back(PlainFloat{
        /*z=*/0,
        /*s=*/sign_dist(rng),
        /*e=*/exp_dist(rng),
        /*m=*/mantissa_dist(rng),
    });

    rhs_cases.push_back(PlainFloat{
        /*z=*/0,
        /*s=*/sign_dist(rng),
        /*e=*/exp_dist(rng),
        /*m=*/mantissa_dist(rng),
    });
  }

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    for (int i = 0; i < kNumCases; ++i) {
      const auto& lhs_plain = lhs_cases[i];
      const auto& rhs_plain = rhs_cases[i];
      auto expected = FPMulPlainRef(lhs_plain, rhs_plain);

      auto lhs = MakeSharedFloat(ctx.get(), lhs_plain.z, lhs_plain.s,
                                 lhs_plain.e, lhs_plain.m, shape);
      auto rhs = MakeSharedFloat(ctx.get(), rhs_plain.z, rhs_plain.s,
                                 rhs_plain.e, rhs_plain.m, shape);

      auto out = FPMul(ctx.get(), lhs, rhs);

      CheckSharedFloat(ctx.get(), out, expected.z, expected.s, expected.e,
                       expected.m, shape);
    }
  });
}

}  // namespace spu::mpc::flp

// namespace spu::mpc::flp