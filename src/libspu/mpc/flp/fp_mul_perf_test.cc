#include <cstdint>
#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/flp/fp_mul.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::flp {
namespace {

const int kP = 8;
const int kQ = 23;
const int kIterations = 2000;
const int64_t kBatchSize = 128;

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

}  // namespace

TEST(FPMulPerfTest, Run2000) {
  const int npc = 2;
  const Shape shape = {kBatchSize};

  const int64_t one_m = 1LL << kQ;

  // 1.5 and 1.25 in current SharedFloat format:
  // value = m * 2^(e - q)
  const int64_t m_a = one_m + (one_m >> 1);  // 1.5
  const int64_t m_b = one_m + (one_m >> 2);  // 1.25

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);
    const int rank = lctx->Rank();

    auto lhs = MakeSharedFloat(ctx.get(), 0, 0, 0, m_a, shape);
    auto rhs = MakeSharedFloat(ctx.get(), 0, 0, 0, m_b, shape);

    // Warm-up: initialize OT connections and avoid counting first-call
    // overhead.
    FPMul(ctx.get(), lhs, rhs);

    size_t b0 = lctx->GetStats()->sent_bytes;
    size_t r0 = lctx->GetStats()->sent_actions;

    yacl::ElapsedTimer timer;
    for (int i = 0; i < kIterations; ++i) {
      FPMul(ctx.get(), lhs, rhs);
    }
    double elapsed_ms = timer.CountMs();

    size_t b1 = lctx->GetStats()->sent_bytes;
    size_t r1 = lctx->GetStats()->sent_actions;

    if (rank == 0) {
      std::cout << "--- FPMul perf over " << kIterations << " iterations ---"
                << std::endl;
      std::cout << "  batch size:     " << kBatchSize << std::endl;
      std::cout << "  total time:     " << elapsed_ms << " ms" << std::endl;
      std::cout << "  avg time:       " << (elapsed_ms / kIterations)
                << " ms/call" << std::endl;
      std::cout << "  avg time/elem:  "
                << (elapsed_ms / kIterations / kBatchSize) << " ms/elem"
                << std::endl;
      std::cout << "  total sent:     " << ((b1 - b0) / 1024.0) << " KiB"
                << std::endl;
      std::cout << "  avg sent:       " << ((b1 - b0) * 1.0 / kIterations)
                << " bytes/call" << std::endl;
      std::cout << "  avg sent/elem:  "
                << ((b1 - b0) * 1.0 / kIterations / kBatchSize) << " bytes/elem"
                << std::endl;
      std::cout << "  avg actions:    " << ((r1 - r0) * 1.0 / kIterations)
                << " per call" << std::endl;
    }
  });
}

}  // namespace spu::mpc::flp
