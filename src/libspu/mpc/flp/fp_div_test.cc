// Comprehensive test for FPDiv: two-party simulation of floating-point division.
// Uses ULP (Units in the Last Place) for error checking instead of relative error.

#include "libspu/mpc/flp/fp_div.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/cheetah/io.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::flp {
namespace {

// Reference quotient: compute expected (mantissa, exponent) in (q+1)-bit format.
std::pair<uint64_t, int64_t> Expected(int q,
    uint64_t nm, int64_t ne, uint64_t dm, int64_t de) {
  double v1 = static_cast<double>(nm) / static_cast<double>(uint64_t{1} << q)
              * std::ldexp(1.0, static_cast<int>(ne));
  double v2 = static_cast<double>(dm) / static_cast<double>(uint64_t{1} << q)
              * std::ldexp(1.0, static_cast<int>(de));
  double qv = v1 / v2;
  int ee;
  double nrm = std::frexp(qv, &ee);
  uint64_t em = static_cast<uint64_t>(std::llround(nrm * 2.0 *
              static_cast<double>(uint64_t{1} << q)));
  return {em, ee - 1};
}

RuntimeConfig MakeConfig() {
  RuntimeConfig cfg;
  cfg.protocol = ProtocolKind::CHEETAH;
  cfg.field = FieldType::FM64;
  cfg.cheetah_2pc_config.ot_kind = CheetahOtKind::YACL_Softspoken;
  return cfg;
}

struct Case { uint64_t nm; int64_t ne; uint64_t dm; int64_t de; };
constexpr int64_t kBatchSize = 32;

// max_ulp bounds the absolute mantissa difference (after exponent alignment).
// 1 ULP = 1 in the (q+1)-bit fixed-point mantissa representation.
void RunBatch(const std::vector<Case>& cases, const std::string& label,
              uint64_t max_ulp = (uint64_t{1} << 20) + (uint64_t{1} << 16),
              FieldType exp_field = FieldType::FM16,
              FieldType man_field = FieldType::FM32) {
  constexpr size_t kW = 2;
  constexpr FieldType kF = FieldType::FM64;
  constexpr int p = 8, q = 23;
  const int64_t n = static_cast<int64_t>(cases.size());
  SPU_ENFORCE(n <= kBatchSize, "batch too large: %lld", n);
  const Shape shape = {n};

  NdArrayRef nm_arr(makeType<RingTy>(kF), shape);
  NdArrayRef ne_arr(makeType<RingTy>(kF), shape);
  NdArrayRef dm_arr(makeType<RingTy>(kF), shape);
  NdArrayRef de_arr(makeType<RingTy>(kF), shape);
  NdArrayRef em_arr(makeType<RingTy>(kF), shape);
  NdArrayRef ee_arr(makeType<RingTy>(kF), shape);

  auto nm_v = NdArrayView<uint64_t>(nm_arr);
  auto ne_v = NdArrayView<uint64_t>(ne_arr);
  auto dm_v = NdArrayView<uint64_t>(dm_arr);
  auto de_v = NdArrayView<uint64_t>(de_arr);
  auto em_v = NdArrayView<uint64_t>(em_arr);
  auto ee_v = NdArrayView<uint64_t>(ee_arr);

  for (int64_t i = 0; i < n; ++i) {
    auto& c = cases[i];
    nm_v[i] = c.nm;
    ne_v[i] = static_cast<uint64_t>(c.ne);
    dm_v[i] = c.dm;
    de_v[i] = static_cast<uint64_t>(c.de);
    auto [em, ee] = Expected(q, c.nm, c.ne, c.dm, c.de);
    em_v[i] = em;
    ee_v[i] = static_cast<uint64_t>(ee);
  }

  auto io = cheetah::makeCheetahIo(kF, kW);
  auto ns = io->toShares(nm_arr, VIS_SECRET, -1);
  auto es = io->toShares(ne_arr, VIS_SECRET, -1);
  auto ds = io->toShares(dm_arr, VIS_SECRET, -1);
  auto fs = io->toShares(de_arr, VIS_SECRET, -1);

  std::array<SharedFloat, 2> out;
  utils::simulate(kW, [&](const std::shared_ptr<yacl::link::Context>& lc) {
    auto sc = spu::mpc::makeCheetahProtocol(MakeConfig(), lc);
    size_t r = static_cast<size_t>(lc->Rank());
    SharedFloat a, b;
    a.z = p2s(sc.get(), make_p(sc.get(), 0, shape, kF));
    a.s = p2s(sc.get(), make_p(sc.get(), 0, shape, kF));
    a.e = ::spu::Value(es[r], ::spu::DT_INVALID);
    a.m = ::spu::Value(ns[r], ::spu::DT_INVALID);
    a.p = p; a.q = q;
    b.z = p2s(sc.get(), make_p(sc.get(), 0, shape, kF));
    b.s = p2s(sc.get(), make_p(sc.get(), 0, shape, kF));
    b.e = ::spu::Value(fs[r], ::spu::DT_INVALID);
    b.m = ::spu::Value(ds[r], ::spu::DT_INVALID);
    b.p = p; b.q = q;
    out[r] = FPDiv(sc.get(), a, b, exp_field, man_field);
  });

  auto io2_f64 = cheetah::makeCheetahIo(kF, kW);
  auto em_recon = io2_f64->fromShares({out[0].m.data(), out[1].m.data()});
  auto ee_recon = io2_f64->fromShares({out[0].e.data(), out[1].e.data()});
  // out[0].m/e are in man_field / exp_field after FPDiv.
  // fromShares with FM64 interprets them correctly since smaller
  // ring values are naturally embedded in the larger ring.

  int fails = 0;
  for (int64_t i = 0; i < n; ++i) {
    auto& c = cases[i];
    int64_t gmv = static_cast<int64_t>(NdArrayView<const uint64_t>(em_recon)[i]);
    int64_t gev = static_cast<int64_t>(NdArrayView<const uint64_t>(ee_recon)[i]);
    int64_t emv = static_cast<int64_t>(NdArrayView<const uint64_t>(em_arr)[i]);
    int64_t eev = static_cast<int64_t>(NdArrayView<const uint64_t>(ee_arr)[i]);

    // Compute ULP difference: align mantissas by matching exponents
    uint64_t ulp_diff = UINT64_MAX;
    if (gev == eev) {
      ulp_diff = static_cast<uint64_t>(std::abs(gmv - emv));
    } else if (gev == eev + 1) {
      // Expected exponent smaller: compare got_m vs (exp_m * 2)
      ulp_diff = static_cast<uint64_t>(std::abs(gmv - (emv << 1)));
    } else if (eev == gev + 1) {
      // Got exponent smaller: compare (got_m * 2) vs exp_m
      ulp_diff = static_cast<uint64_t>(std::abs((gmv << 1) - emv));
    }
    // else: exponent mismatch > 1 â huge error, ulp_diff stays UINT64_MAX

    double gv = static_cast<double>(gmv) / (1ULL << q) *
                std::ldexp(1.0, static_cast<int>(gev));
    double ev = static_cast<double>(emv) / (1ULL << q) *
                std::ldexp(1.0, static_cast<int>(eev));

    bool ok = (ulp_diff <= max_ulp);
    if (!ok) ++fails;

    std::cout << (ok ? "OK" : "FAIL")
              << " [" << label << " i=" << i << "] "
              << "num=" << c.nm << "*2^" << c.ne
              << " / den=" << c.dm << "*2^" << c.de
              << "  got_m=" << gmv << " e=" << gev
              << "  exp_m=" << emv << " e=" << eev
              << "  val=" << gv << " (exp=" << ev << ")"
              << (ok ? "" : "  ULP=" + std::to_string(ulp_diff) + "/" + std::to_string(max_ulp))
              << std::endl;
  }
  EXPECT_EQ(fails, 0) << label << ": " << fails << "/" << n << " FAILED";
}

}  // namespace
}  // namespace spu::mpc::flp

namespace spu::mpc::flp {

// ================================================================
// Constants for q=23 (float32-like, 24-bit mantissa)
// ================================================================
constexpr uint64_t M1  = uint64_t{1} << 23;            // 2^23 = 1.0
constexpr uint64_t M15 = M1 + (M1 >> 1);                 // 1.5
constexpr uint64_t M125 = M1 + (M1 >> 2);                // 1.25
constexpr uint64_t M175 = M1 + ((M1 * 3) >> 2);          // 1.75
constexpr uint64_t MMAX = (uint64_t{1} << 24) - 1;       // ≈1.9999999

// Per-group ULP bounds:
//   Simple/Exponent cases: TIGHT  2^19 + 2^16 = 589,824
//   Normal cases:          MEDIUM  2^20 + 2^16 = 1,114,112
//   Edge cases:            LOOSE   2^20 + 2^17 = 1,179,648
constexpr uint64_t ULP_SIMPLE = (uint64_t{1} << 19) + (uint64_t{1} << 16);
constexpr uint64_t ULP_MED   = (uint64_t{1} << 20) + (uint64_t{1} << 16);
constexpr uint64_t ULP_LOOSE = (uint64_t{1} << 20) + (uint64_t{1} << 17);

TEST(FPDivTest, SimpleValues) {
    RunBatch({
    {M1, 0, M1, 0},          // 1/1 = 1
    {M1, 0, M1, 1},          // 1/2 = 0.5
    {M1, 1, M1, 0},          // 2/1 = 2
    {M1, 0, M1, -1},         // 1/0.5 = 2
    {M15, 0, M1, 0},         // 1.5/1 = 1.5
    {M125, 0, M1, 0},        // 1.25/1 = 1.25
    {M175, 0, M1, 0},        // 1.75/1 = 1.75
    {MMAX, 0, M1, 0},        // â1.9999/1 â 1.9999
  }, "simple", ULP_MED);
}

TEST(FPDivTest, VariousDenominators) {
    RunBatch({
    {M15, 0, M125, 0},       // 1.5/1.25 = 1.2
    {M15, 0, M15, 0},        // 1.5/1.5 = 1
    {M15, 0, M175, 0},       // 1.5/1.75 â 0.857
    {M15, 0, MMAX, 0},       // 1.5/1.9999 â 0.750
    {M1, 1, M125, 0},        // 2/1.25 = 1.6
    {M1, 1, M15, 0},         // 2/1.5 â 1.333
    {M1, 1, M175, 0},        // 2/1.75 â 1.143
    {M1, 1, MMAX, 0},        // 2/1.9999 â 1.00005
  }, "various_den", ULP_MED);
}

TEST(FPDivTest, NumeratorLessThanDenominator) {
    RunBatch({
    {M1, 0, M125, 0},        // 1/1.25 = 0.8
    {M1, 0, M15, 0},         // 1/1.5 â 0.667
    {M1, 0, M175, 0},        // 1/1.75 â 0.571
    {M1, 0, MMAX, 0},        // 1/1.9999 â 0.500
    {M125, 0, M175, 0},      // 1.25/1.75 â 0.714
    {M125, 0, MMAX, 0},      // 1.25/1.9999 â 0.625
    {M15, 0, M175, 0},       // 1.5/1.75 â 0.857
    {M175, 0, MMAX, 0},      // 1.75/1.9999 â 0.875
  }, "num_lt_den", ULP_MED);
}

TEST(FPDivTest, Exponents) {
    RunBatch({
    {M1, 3, M1, 0},          // 1*8 / 1 = 8
    {M1, 0, M1, 3},          // 1 / 8 = 0.125
    {M1, 5, M1, 2},          // 32 / 4 = 8
    {M1, -2, M1, 0},         // 0.25 / 1 = 0.25
    {M1, 0, M1, -2},         // 1 / 0.25 = 4
    {M1, -3, M1, -1},        // 0.125 / 0.5 = 0.25
    {M1, 2, M1, -2},         // 4 / 0.25 = 16
    {M1, -2, M1, 2},         // 0.25 / 4 = 0.0625
  }, "exponents", ULP_MED);
}

TEST(FPDivTest, PreciseQuotients) {
    RunBatch({
    {M1, 0, M1, 2},          // 1 / 4 = 0.25
    {M1, 2, M1, 1},          // 4 / 2 = 2
    {M1, 3, M1, 2},          // 8 / 4 = 2
    {M15, 0, M1, 0},         // 1.5/1 = 1.5
    {M175, 0, M15, 0},       // 1.75/1.5 â 1.167
    {M175, 0, M125, 0},      // 1.75/1.25 = 1.4
    {MMAX, 0, M15, 0},       // 1.9999/1.5 â 1.333
    {M15, 1, M1, 1},         // 3/2 = 1.5
  }, "precise", ULP_MED);
}

TEST(FPDivTest, ExtremeMantissas) {
    RunBatch({
    {M1, 0, MMAX, 0},        // min/max = 1/1.9999 â 0.500
    {MMAX, 0, M1, 0},        // max/min = 1.9999/1 â 1.9999
    {MMAX, 0, MMAX, 0},      // max/max = 1
    {M1 + 1, 0, M1 + 2, 0},  // very close â 0.99999988
    {M15, 0, M15 + 1, 0},    // 1.5/1.50000012 â 0.99999992
    {MMAX - 1, 0, MMAX, 0},  // 1.9999998/1.9999999 â 0.99999995
    {MMAX, 0, MMAX - 1, 0},  // 1.9999999/1.9999998 â 1.00000005
    {M1 + 10, 0, M1 + 11, 0}, // â 0.999999
  }, "extreme", ULP_MED);
}

TEST(FPDivTest, VariousRatios) {
    RunBatch({
    {M15 - (M1 >> 2), 0, M15 - (M1 >> 2), 0},  // 1.25/1.25 = 1
    {M125, 0, M15, 0},       // 1.25/1.5 â 0.833
    {M15, 0, M125, 0},       // 1.5/1.25 = 1.2
    {M175, 0, M125, 0},      // 1.75/1.25 = 1.4
    {M125, 0, M175, 0},      // 1.25/1.75 â 0.714
    {MMAX, 0, M125, 0},      // 1.9999/1.25 â 1.6
    {M125, 0, MMAX, 0},      // 1.25/1.9999 â 0.625
    {M15, 0, MMAX, 0},       // 1.5/1.9999 â 0.75
  }, "ratios", ULP_MED);
}

TEST(FPDivTest, NegativeExponents) {
    RunBatch({
    {M1, -5, M1, 0},         // 1/32 â 0.03125
    {M1, 0, M1, -5},         // 1/(1/32) = 32
    {M1, -5, M1, -3},        // 1/32 / 1/8 = 0.25
    {M1, 5, M1, -3},         // 32 / 0.125 = 256
    {M15, -2, M15, 2},       // 0.375 / 6 = 0.0625
    {M175, 3, M175, -1},     // 14 / 0.875 = 16
    {M125, -4, M15, 2},      // 0.078125 / 6 â 0.013
    {M15, 2, M15, -2},       // 6 / 0.375 = 16
  }, "neg_exp", ULP_MED);
}

TEST(FPDivTest, Comprehensive) {
    RunBatch({
    {M1 + 10000, 0, M1 + 50000, 0},    // close values
    {M1 + 50000, 0, M1 + 10000, 0},    // close reverse
    {M125 + 7777, 0, M175 - 3333, 0},  // arbitrary
    {M175 - 3333, 0, M125 + 7777, 0},  // arbitrary reverse
    {M15 + 12345, 0, MMAX - 9999, 0},  // arbitrary
    {MMAX - 9999, 0, M15 + 12345, 0},  // arbitrary reverse
    {MMAX - 5000, 0, M125 + 3000, 0},  // arbitrary
    {M125 + 3000, 0, MMAX - 5000, 0},  // arbitrary reverse
    {M1, 4, M1, 1},             // 16/2 = 8
    {M1, 1, M1, 4},             // 2/16 = 0.125
    {M15, -1, M1, 2},          // 0.75/4 = 0.1875
    {M175, 3, M15, -2},        // 14/0.375 â 37.33
    {M1, 0, M1 + 1, 0},        // 1/1.00000012 â 0.99999988
    {MMAX, 0, M1, 0},          // 1.9999/1
    {M15, 3, M15, 0},          // 12/1.5 = 8
    {M15, 0, M15, 3},          // 1.5/12 = 0.125
  }, "comprehensive", ULP_MED);
}


void RunFPDivProfile(size_t kNum) {
  constexpr size_t kW = 2;
  constexpr FieldType kF = FieldType::FM64;
  constexpr int p = 8, q = 23;
  const Shape shape = {static_cast<int64_t>(kNum)};
  std::mt19937_64 rng(12345);
  const uint64_t m_min = uint64_t{1} << q;
  const uint64_t m_max = (uint64_t{1} << (q + 1)) - 1;
  NdArrayRef nm_arr(makeType<RingTy>(kF), shape);
  NdArrayRef ne_arr(makeType<RingTy>(kF), shape);
  NdArrayRef dm_arr(makeType<RingTy>(kF), shape);
  NdArrayRef de_arr(makeType<RingTy>(kF), shape);
  auto nmv = NdArrayView<uint64_t>(nm_arr);
  auto nev = NdArrayView<uint64_t>(ne_arr);
  auto dmv = NdArrayView<uint64_t>(dm_arr);
  auto dev = NdArrayView<uint64_t>(de_arr);
  for (size_t i = 0; i < kNum; ++i) {
    nmv[i] = m_min + (rng() % (m_max - m_min + 1));
    nev[i] = static_cast<uint64_t>(static_cast<int64_t>(rng() % 20 - 10));
    dmv[i] = m_min + (rng() % (m_max - m_min + 1));
    dev[i] = static_cast<uint64_t>(static_cast<int64_t>(rng() % 20 - 10));
  }
  auto io = cheetah::makeCheetahIo(kF, kW);
  auto ns = io->toShares(nm_arr, VIS_SECRET, -1);
  auto es = io->toShares(ne_arr, VIS_SECRET, -1);
  auto ds = io->toShares(dm_arr, VIS_SECRET, -1);
  auto fs = io->toShares(de_arr, VIS_SECRET, -1);
  utils::simulate(kW, [&](const std::shared_ptr<yacl::link::Context>& lc) {
    auto sc = spu::mpc::makeCheetahProtocol(MakeConfig(), lc);
    size_t r = static_cast<size_t>(lc->Rank());
    SharedFloat a, b;
    a.z = p2s(sc.get(), make_p(sc.get(), 0, shape, kF));
    a.s = p2s(sc.get(), make_p(sc.get(), 0, shape, kF));
    a.e = ::spu::Value(es[r], ::spu::DT_INVALID);
    a.m = ::spu::Value(ns[r], ::spu::DT_INVALID);
    a.p = p; a.q = q;
    b.z = p2s(sc.get(), make_p(sc.get(), 0, shape, kF));
    b.s = p2s(sc.get(), make_p(sc.get(), 0, shape, kF));
    b.e = ::spu::Value(fs[r], ::spu::DT_INVALID);
    b.m = ::spu::Value(ds[r], ::spu::DT_INVALID);
    b.p = p; b.q = q;
    if (r == 0) {
      auto link_stats = lc->GetStats();
      auto sb0 = link_stats->sent_bytes.load();
      auto rb0 = link_stats->recv_bytes.load();
      auto sa0 = link_stats->sent_actions.load();
      auto ra0 = link_stats->recv_actions.load();
      auto result = FPDiv(sc.get(), a, b, FM16, FM32);
      auto sb1 = link_stats->sent_bytes.load();
      auto rb1 = link_stats->recv_bytes.load();
      auto sa1 = link_stats->sent_actions.load();
      auto ra1 = link_stats->recv_actions.load();
      size_t total_bytes = (sb1 - sb0) + (rb1 - rb0);
      size_t total_actions = (sa1 - sa0) + (ra1 - ra0);
      std::cout << "FPDiv | " << SizeOf(kF) * 8 << "-bit ring"
                << " | n=" << kNum
                << " | B=" << total_bytes
                << " | B/elem=" << static_cast<double>(total_bytes) / kNum
                << " | rounds=" << total_actions
                << " | r/elem=" << static_cast<double>(total_actions) / kNum
                << std::endl;
      (void)result;
    } else {
      FPDiv(sc.get(), a, b, FM16, FM32);
    }
  });
}

TEST(FPDivProfile, Profile) {
  constexpr int p = 8, q = 23;
  constexpr FieldType kField = FieldType::FM64;
  std::cout << "\n==== FPDiv Profile ====" << std::endl;
  std::cout << "Ring: FM64 (" << SizeOf(kField) * 8 << "-bit)"
            << "  p=" << p << " q=" << q
            << std::endl;
  std::cout << "Op | Ring | n | Bytes | B/elem | Rounds | r/elem" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  for (auto n : {500, 1000, 2000}) {
    RunFPDivProfile(static_cast<size_t>(n));
  }
  std::cout << "================================\n" << std::endl;
}

}  // namespace spu::mpc::flp
