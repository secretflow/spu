#include "libspu/mpc/flp/fp_check.h"

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

// Create a public constant in FM64, handling negative values.
Value MakePublic(SPUContext* ctx, int64_t val, const Shape& shape) {
  return make_p(ctx, static_cast<uint128_t>(static_cast<uint64_t>(val)), shape,
                FM64);
}

void CheckValue(const Value& got, const Value& expected) {
  EXPECT_EQ(got.shape(), expected.shape());
  EXPECT_TRUE(ring_all_equal(got.data(), expected.data()));
}

}  // namespace

TEST(FPCheckTest, NormalValue) {
  const int npc = 2;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    Shape shape = {1};

    auto z_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto s_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto e_s = p2s(ctx.get(), MakePublic(ctx.get(), 5, shape));
    auto m_s = p2s(ctx.get(), MakePublic(ctx.get(), 100, shape));

    SharedFloat sf(z_s, s_s, e_s, m_s, kP, kQ);
    auto result = FPCheck(ctx.get(), sf);

    CheckValue(s2p(ctx.get(), result.z), MakePublic(ctx.get(), 0, shape));
    CheckValue(s2p(ctx.get(), result.s), MakePublic(ctx.get(), 0, shape));
    CheckValue(s2p(ctx.get(), result.e), MakePublic(ctx.get(), 5, shape));
    CheckValue(s2p(ctx.get(), result.m), MakePublic(ctx.get(), 100, shape));
  });
}

TEST(FPCheckTest, Overflow) {
  const int npc = 2;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    Shape shape = {1};

    auto z_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto s_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto e_s = p2s(ctx.get(), MakePublic(ctx.get(), 130, shape));
    auto m_s = p2s(ctx.get(), MakePublic(ctx.get(), 42, shape));

    SharedFloat sf(z_s, s_s, e_s, m_s, kP, kQ);
    auto result = FPCheck(ctx.get(), sf);

    const int64_t max_e = 1LL << (kP - 1);
    const uint64_t max_m = 1ULL << kQ;

    CheckValue(s2p(ctx.get(), result.z), MakePublic(ctx.get(), 0, shape));
    CheckValue(s2p(ctx.get(), result.s), MakePublic(ctx.get(), 0, shape));
    CheckValue(s2p(ctx.get(), result.e), MakePublic(ctx.get(), max_e, shape));
    CheckValue(s2p(ctx.get(), result.m),
               MakePublic(ctx.get(), static_cast<int64_t>(max_m), shape));
  });
}

TEST(FPCheckTest, Underflow) {
  const int npc = 2;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    Shape shape = {1};

    auto z_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto s_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto e_s = p2s(ctx.get(), MakePublic(ctx.get(), -130, shape));
    auto m_s = p2s(ctx.get(), MakePublic(ctx.get(), 42, shape));

    SharedFloat sf(z_s, s_s, e_s, m_s, kP, kQ);
    auto result = FPCheck(ctx.get(), sf);

    const int64_t min_e = 1 - (1LL << (kP - 1));

    CheckValue(s2p(ctx.get(), result.z), MakePublic(ctx.get(), 1, shape));
    CheckValue(s2p(ctx.get(), result.s), MakePublic(ctx.get(), 0, shape));
    CheckValue(s2p(ctx.get(), result.e), MakePublic(ctx.get(), min_e, shape));
    CheckValue(s2p(ctx.get(), result.m), MakePublic(ctx.get(), 0, shape));
  });
}

TEST(FPCheckTest, ZeroFlag) {
  const int npc = 2;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = MakeCheetahCtx(lctx);

    Shape shape = {1};

    auto z_s = p2s(ctx.get(), MakePublic(ctx.get(), 1, shape));
    auto s_s = p2s(ctx.get(), MakePublic(ctx.get(), 0, shape));
    auto e_s = p2s(ctx.get(), MakePublic(ctx.get(), 100, shape));
    auto m_s = p2s(ctx.get(), MakePublic(ctx.get(), 999, shape));

    SharedFloat sf(z_s, s_s, e_s, m_s, kP, kQ);
    auto result = FPCheck(ctx.get(), sf);

    const int64_t min_e = 1 - (1LL << (kP - 1));

    CheckValue(s2p(ctx.get(), result.z), MakePublic(ctx.get(), 1, shape));
    CheckValue(s2p(ctx.get(), result.s), MakePublic(ctx.get(), 0, shape));
    CheckValue(s2p(ctx.get(), result.e), MakePublic(ctx.get(), min_e, shape));
    CheckValue(s2p(ctx.get(), result.m), MakePublic(ctx.get(), 0, shape));
  });
}

}  // namespace spu::mpc::flp