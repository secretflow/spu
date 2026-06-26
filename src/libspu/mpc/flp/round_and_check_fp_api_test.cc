// Functional test for RoundMantissaAndCheckSharedApi using two-party simulate.
// Implements ΠRound&Check from BEACON [Rathee et al., S&P 2024, Figure 9].
// Uses (p, Q, q) = (8, 15, 7) matching the BF16 setting from the paper.
// Verification: reconstructed mantissa and exponent must match plaintext reference.

#include "libspu/mpc/flp/round_and_check_fp_api.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/cheetah/io.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::flp {
namespace {

RuntimeConfig MakeConfig() {
  RuntimeConfig conf;
  conf.protocol = ProtocolKind::CHEETAH;
  conf.field = FieldType::FM64;
  conf.cheetah_2pc_config.ot_kind = CheetahOtKind::YACL_Softspoken;
  return conf;
}

// Reference threshold: 2^{Q+1} - 2^{Q-q-1}
uint64_t RefThreshold(int Q, int q) {
  return (uint64_t{1} << (Q + 1)) - (uint64_t{1} << (Q - q - 1));
}

// Paper's ΠRN: (x + 2^{s-1}) >> s   (round-to-nearest, ties-up)
uint64_t RefRN(uint64_t x, int s) {
  return (x + (uint64_t{1} << (s - 1))) >> s;
}

// Paper's ΠRound&Check reference (plaintext).
//   Input:  mantissa m (Q+1 bits), exponent e
//   Output: mantissa out_m (q+1 bits), exponent out_e
//
//   c = (m < 2^{Q+1} - 2^{Q-q-1})   // 1=no overflow, 0=overflow
//   m_c = RN(m) = (m + 2^{s-1}) >> s
//   out_m = c ? m_c : (m_c >> 1)     // m_c >> 1 = 2^q when overflow
//   out_e = c ? e   : e + 1
void RefRoundAndCheck(int Q, int q, uint64_t m, uint64_t e,
                      uint64_t& out_m, uint64_t& out_e) {
  const int s = Q - q;
  const uint64_t threshold = RefThreshold(Q, q);
  const uint64_t m_c = RefRN(m, s);       // RN result
  const uint64_t m_renorm = m_c >> 1;     // = 2^q when overflow

  // c = (m < threshold) → 1=no overflow, 0=overflow
  if (m < threshold) {
    out_m = m_c;       // use RN result
    out_e = e;         // exponent unchanged
  } else {
    out_m = m_renorm;  // renormalized to 2^q
    out_e = e + 1;     // exponent incremented
  }
}

}  // namespace

TEST(RoundAndCheckFpApiTest, MatchesPaperProtocol) {
  // BF16 settings from BEACON paper: p=8 exponent bits, q=7 mantissa bits.
  // Q=15 gives higher intermediate precision (16-bit mantissa).
  constexpr int kQ = 15;
  constexpr int kq = 7;
  constexpr size_t kWorldSize = 2;
  constexpr FieldType kField = FieldType::FM64;

  const uint64_t threshold = RefThreshold(kQ, kq);

  // ---- Test vectors: specific boundary cases ----
  std::vector<uint64_t> mans;
  std::vector<uint64_t> exps;

  // Edge cases around the normalized range and threshold
  const uint64_t test_m_vec[] = {
      0,
      1,
      (uint64_t{1} << kQ) - 1,   // 2^Q - 1
      (uint64_t{1} << kQ),       // 2^Q (min normalized)
      threshold - 2,
      threshold - 1,             // just below threshold
      threshold,                 // at threshold (overflow)
      threshold + 1,
      (uint64_t{1} << (kQ + 1)) - 1  // 2^{Q+1} - 1 (max)
  };
  const uint64_t test_e_vec[] = {0, 1, 5, 10};

  for (auto m : test_m_vec) {
    for (auto e : test_e_vec) {
      mans.push_back(m);
      exps.push_back(e);
    }
  }

  // ---- Random test cases ----
  std::mt19937_64 rng(42);  // deterministic seed
  constexpr int kNumRandom = 200;
  for (int i = 0; i < kNumRandom; ++i) {
    // Generates values across the full [0, 2^{Q+1}) range
    uint64_t m = rng() % (uint64_t{1} << (kQ + 1));
    uint64_t e = rng() % 20;
    mans.push_back(m);
    exps.push_back(e);
  }

  const int64_t numel = static_cast<int64_t>(mans.size());
  const Shape shape = {numel};

  // ---- Reference outputs (plaintext) ----
  NdArrayRef ref_m(makeType<RingTy>(kField), shape);
  NdArrayRef ref_e(makeType<RingTy>(kField), shape);
  {
    auto rvm = NdArrayView<uint64_t>(ref_m);
    auto rve = NdArrayView<uint64_t>(ref_e);
    for (int64_t i = 0; i < numel; ++i) {
      RefRoundAndCheck(kQ, kq, mans[static_cast<size_t>(i)],
                       exps[static_cast<size_t>(i)], rvm[i], rve[i]);
    }
  }

  // ---- MPC input ----
  NdArrayRef msg_m(makeType<RingTy>(kField), shape);
  NdArrayRef msg_e(makeType<RingTy>(kField), shape);
  {
    auto vm = NdArrayView<uint64_t>(msg_m);
    auto ve = NdArrayView<uint64_t>(msg_e);
    for (int64_t i = 0; i < numel; ++i) {
      vm[i] = mans[static_cast<size_t>(i)];
      ve[i] = exps[static_cast<size_t>(i)];
    }
  }

  // Generate input shares ONCE outside simulate
  auto io = cheetah::makeCheetahIo(kField, kWorldSize);
  auto m_shares = io->toShares(msg_m, VIS_SECRET, -1);
  auto e_shares = io->toShares(msg_e, VIS_SECRET, -1);

  std::array<SharedFloat, 2> out_shares;
  utils::simulate(kWorldSize,
                  [&](const std::shared_ptr<yacl::link::Context>& lctx) {
                    auto sctx = spu::mpc::makeCheetahProtocol(MakeConfig(), lctx);
                    const size_t rank = static_cast<size_t>(lctx->Rank());

                    SharedFloat in;
                    in.m = ::spu::Value(m_shares[rank], ::spu::DT_INVALID);
                    in.e = ::spu::Value(e_shares[rank], ::spu::DT_INVALID);
                    in.z = make_p(sctx.get(), 0, shape, kField);
                    in.s = make_p(sctx.get(), 0, shape, kField);
                    in.p = 8;
                    in.q = kq;

                    out_shares[rank] =
                        RoundMantissaAndCheckSharedApi(sctx.get(), in, kQ, kq);
                  });

  // Reconstruct outputs from both parties' shares
  auto io2 = cheetah::makeCheetahIo(kField, kWorldSize);
  NdArrayRef got_m =
      io2->fromShares({out_shares[0].m.data(), out_shares[1].m.data()});
  NdArrayRef got_e =
      io2->fromShares({out_shares[0].e.data(), out_shares[1].e.data()});

  auto gmv = NdArrayView<const uint64_t>(got_m);
  auto gev = NdArrayView<const uint64_t>(got_e);
  auto rmv = NdArrayView<const uint64_t>(ref_m);
  auto rev = NdArrayView<const uint64_t>(ref_e);

  // ---- Verify results ----
  int fails = 0;
  int ulp_fails = 0;
  double max_ulp = 0.0;

  for (int64_t i = 0; i < numel; ++i) {
    const uint64_t in_m = mans[static_cast<size_t>(i)];
    const int64_t in_e = static_cast<int64_t>(exps[static_cast<size_t>(i)]);
    const uint64_t gm = gmv[i];
    const int64_t ge = static_cast<int64_t>(gev[i]);
    const uint64_t rm = rmv[i];
    const int64_t re = static_cast<int64_t>(rev[i]);

    const bool match_m = (gm == rm);
    const bool match_e = (ge == re);

    // ULP computation:
    //   exact_val = in_m * 2^{in_e - Q}
    //   approx_val = gm * 2^{ge - q}
    //   ulp = 2^{ge - q}
    //   error_ulp = |exact_val - approx_val| / ulp
    const double exact =
        static_cast<double>(in_m) * std::ldexp(1.0, static_cast<int>(in_e) - kQ);
    const double approx =
        static_cast<double>(gm) * std::ldexp(1.0, static_cast<int>(ge) - kq);
    const double ulp_val =
        std::ldexp(1.0, static_cast<int>(ge) - kq);
    const double error_ulp = std::abs(exact - approx) / ulp_val;
    max_ulp = std::max(max_ulp, error_ulp);

    if (error_ulp >= 0.5) ++ulp_fails;

    const bool ok = match_m && match_e;
    if (!ok) ++fails;

    std::cout << (ok ? "OK" : "FAIL") << " [i=" << i << "] "
              << "in_m=" << in_m << " in_e=" << in_e << "  "
              << "got_m=" << gm << " got_e=" << ge << "  "
              << "ref_m=" << rm << " ref_e=" << re << "  "
              << "ULP=" << error_ulp << std::endl;
  }

  std::cout << "\n==== Summary ====" << std::endl;
  std::cout << "Total cases: " << numel << std::endl;
  std::cout << "Mantissa/exponent mismatches: " << fails << std::endl;
  std::cout << "ULP >= 0.5 failures: " << ulp_fails << std::endl;
  std::cout << "Max ULP error: " << max_ulp << std::endl;

  EXPECT_EQ(fails, 0);
  EXPECT_LT(max_ulp, 0.5);
}



void RunRoundCheckProfile(int Q, int q, FieldType field, size_t kNum) {
  constexpr size_t kW = 2;
  const Shape shape = {static_cast<int64_t>(kNum)};
  std::mt19937_64 rng(12345);
  NdArrayRef msg_m(makeType<RingTy>(field), shape);
  NdArrayRef msg_e(makeType<RingTy>(field), shape);
  auto vm = NdArrayView<uint64_t>(msg_m);
  auto ve = NdArrayView<uint64_t>(msg_e);
  for (size_t i = 0; i < kNum; ++i) {
    vm[i] = rng() % (uint64_t{1} << (Q + 1));
    ve[i] = rng() % 20;
  }
  auto io = cheetah::makeCheetahIo(field, kW);
  auto m_shares = io->toShares(msg_m, VIS_SECRET, -1);
  auto e_shares = io->toShares(msg_e, VIS_SECRET, -1);
  utils::simulate(kW, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto sctx = spu::mpc::makeCheetahProtocol(MakeConfig(), lctx);
    size_t rank = static_cast<size_t>(lctx->Rank());
    SharedFloat in;
    in.m = ::spu::Value(m_shares[rank], ::spu::DT_INVALID);
    in.e = ::spu::Value(e_shares[rank], ::spu::DT_INVALID);
    in.z = make_p(sctx.get(), 0, shape, field);
    in.s = make_p(sctx.get(), 0, shape, field);
    in.p = 8;
    in.q = q;
    if (rank == 0) {
      auto link_stats = lctx->GetStats();
      auto sb0 = link_stats->sent_bytes.load();
      auto rb0 = link_stats->recv_bytes.load();
      auto sa0 = link_stats->sent_actions.load();
      auto ra0 = link_stats->recv_actions.load();
      auto result = RoundMantissaAndCheckSharedApi(sctx.get(), in, Q, q);
      auto sb1 = link_stats->sent_bytes.load();
      auto rb1 = link_stats->recv_bytes.load();
      auto sa1 = link_stats->sent_actions.load();
      auto ra1 = link_stats->recv_actions.load();
      size_t total_bytes = (sb1 - sb0) + (rb1 - rb0);
      size_t total_actions = (sa1 - sa0) + (ra1 - ra0);
      std::cout << "Round&Check | " << SizeOf(field) * 8 << "-bit ring"
                << " | n=" << kNum
                << " | B=" << total_bytes
                << " | B/elem=" << static_cast<double>(total_bytes) / kNum
                << " | rounds=" << total_actions
                << " | r/elem=" << static_cast<double>(total_actions) / kNum
                << std::endl;
      (void)result;
    } else {
      RoundMantissaAndCheckSharedApi(sctx.get(), in, Q, q);
    }
  });
}

TEST(RoundAndCheckFpApiProfile, Profile) {
  constexpr int kQ = 15;
  constexpr int kq = 7;
  constexpr FieldType kField = FieldType::FM64;
  std::cout << "\n==== Round&Check Profile ====" << std::endl;
  std::cout << "Ring: FM64 (" << SizeOf(kField) * 8 << "-bit)"
            << "  Q=" << kQ << " q=" << kq << " s=" << (kQ - kq)
            << std::endl;
  std::cout << "Op | Ring | n | Bytes | B/elem | Rounds | r/elem" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  for (auto n : {500, 1000, 2000}) {
    RunRoundCheckProfile(kQ, kq, kField, static_cast<size_t>(n));
  }
  std::cout << "================================\n" << std::endl;
}

}  // namespace spu::mpc::flp
