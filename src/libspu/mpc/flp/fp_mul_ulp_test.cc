#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/flp/fp_mul.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::flp {
namespace {

const int kP = 8;
const int kQ = 23;
const int kNumSamples = 1000;

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

  // Mantissa uses q + 1 bits because it explicitly stores the leading 1.
  auto m_s = MakeSecret(ctx, m, shape, kQ + 1);

  return SharedFloat(z_s, s_s, e_s, m_s, kP, kQ);
}

double FloatToDouble(int64_t z, int64_t s, int64_t e, int64_t m) {
  if (z != 0) {
    return 0.0;
  }

  const double sign = (s != 0) ? -1.0 : 1.0;
  return sign * static_cast<double>(m) *
         std::ldexp(1.0, static_cast<int>(e - kQ));
}

// ULP spacing around exact_val for the current q-bit fraction format.
// Current format has q + 1 effective mantissa bits.
double ULP(double exact_val) {
  int exp = 0;
  std::frexp(std::abs(exact_val), &exp);
  return std::ldexp(1.0, exp - 1 - kQ);
}

struct ULPStats {
  int count_zero = 0;   // ULP == 0
  int count_half = 0;   // 0 < ULP <= 0.5
  int count_one = 0;    // 0.5 < ULP <= 1
  int count_above = 0;  // ULP > 1

  double max_ulp = 0.0;
  double sum_ulp = 0.0;
  int total = 0;
};

// Plain tuple: (z, s, e, m)
using PlainTuple = std::tuple<int64_t, int64_t, int64_t, int64_t>;

struct TestCase {
  PlainTuple a;
  PlainTuple b;
};

std::vector<TestCase> GenerateSamples() {
  std::mt19937_64 rng(20250603);

  const int64_t one_m = 1LL << kQ;
  const int64_t max_m = (1LL << (kQ + 1)) - 1;

  std::uniform_int_distribution<int64_t> s_dist(0, 1);
  std::uniform_int_distribution<int64_t> e_dist(-10, 10);

  const int samples_per_layer = kNumSamples / 4;

  std::vector<TestCase> cases;
  cases.reserve(kNumSamples);

  // Layer 1:
  // Low mantissa product.
  // Usually chooses RNTE(m_prod, q), exponent unchanged.
  {
    std::uniform_int_distribution<int64_t> m_dist(one_m, one_m + (one_m >> 2));

    for (int i = 0; i < samples_per_layer; ++i) {
      cases.push_back({
          {0, s_dist(rng), e_dist(rng), m_dist(rng)},
          {0, s_dist(rng), e_dist(rng), m_dist(rng)},
      });
    }
  }

  // Layer 2:
  // High mantissa product.
  // Usually chooses RNTE(m_prod, q + 1), exponent + 1.
  {
    std::uniform_int_distribution<int64_t> m_dist(one_m + (one_m >> 1), max_m);

    for (int i = 0; i < samples_per_layer; ++i) {
      cases.push_back({
          {0, s_dist(rng), e_dist(rng), m_dist(rng)},
          {0, s_dist(rng), e_dist(rng), m_dist(rng)},
      });
    }
  }

  // Layer 3:
  // Fully random normal mantissas.
  {
    std::uniform_int_distribution<int64_t> m_dist(one_m, max_m);

    for (int i = 0; i < samples_per_layer; ++i) {
      cases.push_back({
          {0, s_dist(rng), e_dist(rng), m_dist(rng)},
          {0, s_dist(rng), e_dist(rng), m_dist(rng)},
      });
    }
  }

  // Layer 4:
  // Identity cases: x * 1.0.
  {
    std::uniform_int_distribution<int64_t> m_dist(one_m, max_m);

    for (int i = 0; i < samples_per_layer; ++i) {
      cases.push_back({
          {0, s_dist(rng), e_dist(rng), m_dist(rng)},
          {0, 0, 0, one_m},
      });
    }
  }

  return cases;
}

TEST(FPMulULPTest, RandomNormalSamples) {
  const int npc = 2;
  const Shape shape = {1};

  const auto cases = GenerateSamples();

  std::vector<ULPStats> per_party(npc);

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);
    const int rank = lctx->Rank();

    for (const auto& c : cases) {
      auto [az, as, ae, am] = c.a;
      auto [bz, bs, be, bm] = c.b;

      auto lhs = MakeSharedFloat(ctx.get(), az, as, ae, am, shape);
      auto rhs = MakeSharedFloat(ctx.get(), bz, bs, be, bm, shape);

      auto out = FPMul(ctx.get(), lhs, rhs);

      auto rz = s2p(ctx.get(), out.z);
      auto rs = s2p(ctx.get(), out.s);
      auto re = s2p(ctx.get(), out.e);
      auto rm = s2p(ctx.get(), out.m);

      const auto got_z =
          static_cast<int64_t>(NdArrayView<const uint64_t>(rz.data())[0]);
      const auto got_s =
          static_cast<int64_t>(NdArrayView<const uint64_t>(rs.data())[0]);
      const auto got_e =
          static_cast<int64_t>(NdArrayView<const uint64_t>(re.data())[0]);
      const auto got_m =
          static_cast<int64_t>(NdArrayView<const uint64_t>(rm.data())[0]);

      const double v_a = FloatToDouble(az, as, ae, am);
      const double v_b = FloatToDouble(bz, bs, be, bm);

      // For multiplication, double is enough here:
      // 24-bit mantissa * 24-bit mantissa produces at most 48 exact bits,
      // while double has 53 mantissa bits.
      const double v_exact = v_a * v_b;

      if (v_exact == 0.0) {
        per_party[rank].count_zero++;
        per_party[rank].total++;
        continue;
      }

      const double v_got = FloatToDouble(got_z, got_s, got_e, got_m);
      const double ulp_val = ULP(v_exact);
      const double ulp_err = std::abs(v_got - v_exact) / ulp_val;

      per_party[rank].sum_ulp += ulp_err;
      per_party[rank].max_ulp = std::max(per_party[rank].max_ulp, ulp_err);
      per_party[rank].total++;

      if (ulp_err == 0.0) {
        per_party[rank].count_zero++;
      } else if (ulp_err <= 0.5) {
        per_party[rank].count_half++;
      } else if (ulp_err <= 1.0) {
        per_party[rank].count_one++;
      } else {
        per_party[rank].count_above++;
      }
    }
  });

  const auto& stats = per_party[0];

  std::cout << "--- FPMul ULP Error Test (" << stats.total << " samples) ---"
            << std::endl;
  std::cout << "  ULP = 0:         " << stats.count_zero << " ("
            << (100.0 * stats.count_zero / stats.total) << "%)" << std::endl;
  std::cout << "  ULP in (0, 0.5]: " << stats.count_half << " ("
            << (100.0 * stats.count_half / stats.total) << "%)" << std::endl;
  std::cout << "  ULP in (0.5, 1]: " << stats.count_one << " ("
            << (100.0 * stats.count_one / stats.total) << "%)" << std::endl;
  std::cout << "  ULP > 1:         " << stats.count_above << " ("
            << (100.0 * stats.count_above / stats.total) << "%)" << std::endl;
  std::cout << "  Max ULP:         " << stats.max_ulp << std::endl;
  std::cout << "  Mean ULP:        " << (stats.sum_ulp / stats.total)
            << std::endl;

  EXPECT_EQ(stats.count_above, 0)
      << stats.count_above << " samples exceed 1 ULP";
}

}  // namespace
}  // namespace spu::mpc::flp