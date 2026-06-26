#include "libspu/mpc/flp/fp_check.h"

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::flp {
namespace {

const int kP = 8;
const int kQ = 23;
const int kIterations = 1000;

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

}  // namespace

TEST(FPCheckPerfTest, Run1000) {
  const int npc = 2;
  const Shape shape = {128};

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);
    const int rank = lctx->Rank();

    // Build input once
    auto z_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto s_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto e_s = p2s(ctx.get(), MakePublic(ctx.get(), 130, shape));
    auto m_s = p2s(ctx.get(), MakePublic(ctx.get(), 42, shape));
    SharedFloat sf(z_s, s_s, e_s, m_s, kP, kQ);

    // Warm-up: 1 call to init OT connections
    FPCheck(ctx.get(), sf);

    // Measure
    size_t b0 = lctx->GetStats()->sent_bytes;
    size_t r0 = lctx->GetStats()->sent_actions;

    yacl::ElapsedTimer timer;
    for (int i = 0; i < kIterations; ++i) {
      FPCheck(ctx.get(), sf);
    }
    double elapsed_ms = timer.CountMs();

    size_t b1 = lctx->GetStats()->sent_bytes;
    size_t r1 = lctx->GetStats()->sent_actions;

    if (rank == 0) {
      std::cout << "--- FPCheck perf over " << kIterations
                << " iterations ---" << std::endl;
      std::cout << "  total time: " << elapsed_ms << " ms" << std::endl;
      std::cout << "  avg time:   " << (elapsed_ms / kIterations) << " ms/call"
                << std::endl;
      std::cout << "  total sent: " << ((b1 - b0) / 1024.0) << " KiB"
                << std::endl;
      std::cout << "  avg sent:   " << ((b1 - b0) * 1.0 / kIterations)
                << " bytes/call" << std::endl;
      std::cout << "  avg actions: " << ((r1 - r0) * 1.0 / kIterations)
                << " per call" << std::endl;
    }
  });
}

}  // namespace spu::mpc::flp
