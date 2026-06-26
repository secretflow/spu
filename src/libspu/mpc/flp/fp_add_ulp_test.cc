#include "libspu/mpc/flp/fp_add.h"

#include <cmath>
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
const int kNumSamples = 10000;
const double kULPTarget = 0.5;

std::unique_ptr<SPUContext> MakeCheetahCtx(
    const std::shared_ptr<yacl::link::Context>& lctx) {
  RuntimeConfig conf;
  conf.protocol = ProtocolKind::CHEETAH;
  conf.field = FM64;
  return makeCheetahProtocol(conf, lctx);
}

Value MakePublic(SPUContext* ctx, int64_t val, const Shape& shape) {
  return make_p(ctx, static_cast<uint128_t>(static_cast<uint64_t>(val)), shape,
                FM64);
}

SharedFloat MakeSharedFloat(SPUContext* ctx, int64_t z, int64_t s, int64_t e,
                            int64_t m, const Shape& shape) {
  auto z_s = p2s(ctx, MakePublic(ctx, z, shape));
  auto s_s = p2s(ctx, MakePublic(ctx, s, shape));
  auto e_s = p2s(ctx, MakePublic(ctx, e, shape));
  auto m_s = p2s(ctx, MakePublic(ctx, m, shape));
  return SharedFloat(z_s, s_s, e_s, m_s, kP, kQ);
}

// Plaintext float to double
double FloatToDouble(int64_t z, int64_t s, int64_t e, int64_t m) {
  if (z != 0) return 0.0;
  double sign = (s != 0) ? -1.0 : 1.0;
  return sign * static_cast<double>(m) * std::ldexp(1.0, static_cast<int>(e - kQ));
}

// ULP spacing at a given value magnitude
double ULP(double exact_val) {
  int exp;
  std::frexp(std::abs(exact_val), &exp);
  return std::ldexp(1.0, exp - 1 - kQ);
}

struct ULPStats {
  int count_zero = 0;    // ULP == 0
  int count_half = 0;    // 0 < ULP <= 0.5
  int count_one = 0;     // 0.5 < ULP <= 1
  int count_above = 0;   // ULP > 1
  double max_ulp = 0.0;
  double sum_ulp = 0.0;
  int total = 0;
};

// Input specification: (z, s, e, m)
using PlainTuple = std::tuple<int64_t, int64_t, int64_t, int64_t>;

struct TestCase {
  PlainTuple a;
  PlainTuple b;
};

std::vector<TestCase> GenerateSamples() {
  std::mt19937_64 rng(42);
  const int64_t min_m = 1LL << kQ;          // 2^23
  const int64_t max_m = (1LL << (kQ + 1)) - 1;  // 2^24 - 1
  std::uniform_int_distribution<int64_t> m_dist(min_m, max_m);
  std::uniform_int_distribution<int64_t> s_dist(0, 1);

  const int samples_per_layer = kNumSamples / 5;
  std::vector<TestCase> cases;

  // Layer 1: d=0, same sign
  {
    std::uniform_int_distribution<int64_t> e_dist(-5, 5);
    for (int i = 0; i < samples_per_layer; ++i) {
      int64_t sig = s_dist(rng);
      int64_t e = e_dist(rng);
      cases.push_back({{0, sig, e, m_dist(rng)}, {0, sig, e, m_dist(rng)}});
    }
  }

  // Layer 2: 0 < d <= q+1, same sign
  {
    std::uniform_int_distribution<int64_t> e1_dist(-5, 5);
    for (int i = 0; i < samples_per_layer; ++i) {
      int64_t sig = s_dist(rng);
      int64_t e1 = e1_dist(rng);
      int64_t e2 = e1_dist(rng);
      int64_t d = std::abs(e1 - e2);
      if (d == 0 || d > kQ + 1) {
        --i;
        continue;
      }
      cases.push_back({{0, sig, e1, m_dist(rng)}, {0, sig, e2, m_dist(rng)}});
    }
  }

  // Layer 3: 0 < d <= q+1, different signs
  {
    std::uniform_int_distribution<int64_t> e_dist(-5, 5);
    for (int i = 0; i < samples_per_layer; ++i) {
      int64_t e1 = e_dist(rng);
      int64_t e2 = e_dist(rng);
      int64_t d = std::abs(e1 - e2);
      if (d == 0 || d > kQ + 1) {
        --i;
        continue;
      }
      cases.push_back({{0, 0, e1, m_dist(rng)}, {0, 1, e2, m_dist(rng)}});
    }
  }

  // Layer 4: d > q+1 (early return)
  {
    for (int i = 0; i < samples_per_layer; ++i) {
      int64_t sig = s_dist(rng);
      cases.push_back(
          {{0, sig, 100, m_dist(rng)}, {0, sig, 0, m_dist(rng)}});
    }
  }

  // Layer 5: boundary exponents
  {
    const int64_t overflow_thresh = (1LL << (kP - 1)) - 1;  // 127
    const int64_t underflow_thresh = 2 - (1LL << (kP - 1));  // -126
    std::uniform_int_distribution<int64_t> e_dist(
        underflow_thresh, overflow_thresh);
    for (int i = 0; i < samples_per_layer; ++i) {
      int64_t sig = s_dist(rng);
      cases.push_back(
          {{0, sig, e_dist(rng), m_dist(rng)}, {0, sig, e_dist(rng), m_dist(rng)}});
    }
  }

  return cases;
}

TEST(FPAddULPTest, RandomSamples) {
  const int npc = 2;
  const Shape shape = {1};
  auto cases = GenerateSamples();

  ULPStats stats;
  std::vector<ULPStats> per_party(2);  // both parties should agree

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);
    const int rank = lctx->Rank();

    for (const auto& c : cases) {
      auto [az, as, ae, am] = c.a;
      auto [bz, bs, be, bm] = c.b;

      // MPC computation
      auto lhs = MakeSharedFloat(ctx.get(), az, as, ae, am, shape);
      auto rhs = MakeSharedFloat(ctx.get(), bz, bs, be, bm, shape);
      auto out = FPAdd(ctx.get(), lhs, rhs);

      // Reveal
      auto rz = s2p(ctx.get(), out.z);
      auto rs = s2p(ctx.get(), out.s);
      auto re = s2p(ctx.get(), out.e);
      auto rm = s2p(ctx.get(), out.m);

      int64_t got_z = static_cast<int64_t>(NdArrayView<const uint64_t>(rz.data())[0]);
      int64_t got_s = static_cast<int64_t>(NdArrayView<const uint64_t>(rs.data())[0]);
      int64_t got_e = static_cast<int64_t>(NdArrayView<const uint64_t>(re.data())[0]);
      int64_t got_m = static_cast<int64_t>(NdArrayView<const uint64_t>(rm.data())[0]);

      // Exact reference: double has 53-bit mantissa > our 2q+3=49 bits
      double v_a = FloatToDouble(az, as, ae, am);
      double v_b = FloatToDouble(bz, bs, be, bm);
      double v_exact = v_a + v_b;

      // Handle exact zero
      if (v_exact == 0.0) {
        per_party[rank].count_zero++;
        per_party[rank].total++;
        continue;
      }

      double v_got = FloatToDouble(got_z, got_s, got_e, got_m);
      double ulp_val = ULP(v_exact);
      double ulp_err = std::abs(v_got - v_exact) / ulp_val;

      per_party[rank].sum_ulp += ulp_err;
      per_party[rank].total++;
      per_party[rank].max_ulp = std::max(per_party[rank].max_ulp, ulp_err);

      if (ulp_err == 0.0)
        per_party[rank].count_zero++;
      else if (ulp_err <= 0.5)
        per_party[rank].count_half++;
      else if (ulp_err <= 1.0)
        per_party[rank].count_one++;
      else
        per_party[rank].count_above++;
    }
  });

  stats = per_party[0];  // both parties should agree

  std::cout << "--- FPAdd ULP Error Test (" << stats.total
            << " samples) ---" << std::endl;
  std::cout << "  ULP = 0:       " << stats.count_zero << " ("
            << (100.0 * stats.count_zero / stats.total) << "%)" << std::endl;
  std::cout << "  ULP in (0, 0.5]: " << stats.count_half << " ("
            << (100.0 * stats.count_half / stats.total) << "%)" << std::endl;
  std::cout << "  ULP in (0.5, 1]: " << stats.count_one << " ("
            << (100.0 * stats.count_one / stats.total) << "%)" << std::endl;
  std::cout << "  ULP > 1:       " << stats.count_above << " ("
            << (100.0 * stats.count_above / stats.total) << "%)" << std::endl;
  std::cout << "  Max ULP:  " << stats.max_ulp << std::endl;
  std::cout << "  Mean ULP: " << (stats.sum_ulp / stats.total) << std::endl;

  EXPECT_EQ(stats.count_above, 0)
      << stats.count_above << " samples exceed 1 ULP";
}

}  // namespace
}  // namespace spu::mpc::flp
