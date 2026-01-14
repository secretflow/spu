#include "libspu/kernel/hal/logstar.h"

#include "gtest/gtest.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {

class BrentKungTest : public ::testing::Test {};
class ExtractOrderedTest : public ::testing::Test {};
namespace {
SPUContext makeSPUContextWithProfile(
    ProtocolKind prot_kind, FieldType field,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  RuntimeConfig cfg;
  cfg.protocol = prot_kind;
  cfg.field = field;
  cfg.enable_action_trace = false;

  if (lctx->Rank() == 0) {
    cfg.enable_hal_profile = true;
    cfg.enable_pphlo_profile = true;
  }
  return test::makeSPUContext(cfg, lctx);
}
}  // namespace

TEST_F(BrentKungTest, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
    int64_t n = 8;
    int64_t block_size = 2;
    xt::xarray<float> x = {5, 6, 2, 2, 10, 10, 4, 3, 1, 1, 2, 2, 5, 5, 3, 2};
    xt::xarray<float> valids = {1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1};
    xt::xarray<float> g = {0, 1, 1, 0, 1, 1, 1, 0};
    xt::xarray<float> x_out_expected = {{5, 6}, {5, 6}, {5, 6}, {4, 3},
                                        {4, 3}, {4, 3}, {4, 3}, {3, 2}};
    xt::xarray<float> valid_out_expected = {{1, 1}, {1, 1}, {1, 1}, {0, 1},
                                            {0, 1}, {0, 1}, {0, 1}, {1, 1}};

    x.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
    valids.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto v_s = test::makeValue(&ctx, valids, VIS_SECRET);
    g.reshape({static_cast<size_t>(n), 1});
    auto g_s = test::makeValue(&ctx, g, VIS_SECRET);
    setupTrace(&ctx, ctx.config());

    auto [x_out_s, valid_out_s] = duplicate_brent_kung(&ctx, x_s, v_s, g_s);

    test::printProfileData(&ctx);
    auto x_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, x_out_s));
    auto valid_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, valid_out_s));
    if (lctx->Rank() == 0) {
      std::cout << "x_out_expected:\n" << x_out_expected << std::endl;
      std::cout << "x_out_opened (Actual):\n" << x_out_opened << std::endl;
      std::cout << "valid_out_expected:\n" << valid_out_expected << std::endl;
      std::cout << "valid_out_opened (Actual):\n"
                << valid_out_opened << std::endl;
    }
    EXPECT_TRUE(xt::allclose(x_out_opened, x_out_expected));
    EXPECT_EQ(valid_out_opened.shape()[0], n);
    EXPECT_EQ(valid_out_opened.shape()[1], block_size);
    EXPECT_TRUE(xt::allclose(valid_out_opened, valid_out_expected));
  });
}

TEST_F(BrentKungTest, LargeScaleRealNumbers) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
    const int64_t n = 1000000;
    const int64_t block_size = 2;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist_x(0, 1000);
    std::uniform_int_distribution<int> dist_binary(0, 1);  // 改为 int 类型

    xt::xarray<float> x = xt::zeros<float>({n * block_size});
    xt::xarray<float> valids = xt::zeros<float>({n * block_size});
    xt::xarray<float> g = xt::zeros<float>({n});

    for (auto& v : x) v = dist_x(rng);
    for (auto& v : valids) v = static_cast<float>(dist_binary(rng));
    g[0] = 0.0F;
    for (int64_t i = 1; i < n; ++i) {
      g[i] = static_cast<float>(dist_binary(rng));
    }

    x.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
    valids.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto v_s = test::makeValue(&ctx, valids, VIS_SECRET);
    g.reshape({static_cast<size_t>(n), 1});
    auto g_s = test::makeValue(&ctx, g, VIS_SECRET);
    setupTrace(&ctx, ctx.config());

    auto [x_out_s, valid_out_s] = duplicate_brent_kung(&ctx, x_s, v_s, g_s);

    test::printProfileData(&ctx);
    if (lctx->Rank() == 0) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "duplicate_brent_kung Large Scale Test (Real numbers):"
                << std::endl;
      std::cout << "  - Input Shape : " << n << " blocks of size " << block_size
                << std::endl;
      std::cout << "  - Protocol    : " << protocol << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    auto x_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, x_out_s));
    auto valid_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, valid_out_s));

    // verifiy correctness
    if (lctx->Rank() == 0) {
      const float atol = 0.01;
      for (int64_t i = 0; i < n; ++i) {
        if (g(i, 0) == 0) {
          for (int64_t b = 0; b < block_size; ++b) {
            EXPECT_NEAR(x_out_opened(i, b), x(i, b), atol)
                << "x reset failed at i=" << i << ", b=" << b
                << ", diff=" << std::abs(x_out_opened(i, b) - x(i, b));
            EXPECT_NEAR(valid_out_opened(i, b), valids(i, b), atol)
                << "valids reset failed at i=" << i << ", b=" << b;
          }
        } else {
          for (int64_t b = 0; b < block_size; ++b) {
            EXPECT_NEAR(x_out_opened(i, b), x_out_opened(i - 1, b), atol)
                << "x propagation failed at i=" << i << ", b=" << b
                << ", diff = "
                << std::abs(x_out_opened(i, b) - x_out_opened(i - 1, b));
            EXPECT_NEAR(valid_out_opened(i, b), valid_out_opened(i - 1, b),
                        atol)
                << "valids propagation failed at i=" << i << ", b=" << b;
          }
        }
      }
    }
  });
}

TEST_F(BrentKungTest, LargeScaleIntergers) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
    const int64_t n = 1000000;
    const int64_t block_size = 2;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_x(0, 1000);
    std::uniform_int_distribution<int> dist_binary(0, 1);

    xt::xarray<int> x = xt::zeros<int>({n * block_size});
    xt::xarray<int> valids = xt::zeros<int>({n * block_size});
    xt::xarray<int> g = xt::zeros<int>({n});

    for (auto& v : x) v = dist_x(rng);
    for (auto& v : valids) v = dist_binary(rng);
    g[0] = 0;
    for (int64_t i = 1; i < n; ++i) {
      g[i] = dist_binary(rng);
    }

    x.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
    valids.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto v_s = test::makeValue(&ctx, valids, VIS_SECRET);
    g.reshape({static_cast<size_t>(n), 1});
    auto g_s = test::makeValue(&ctx, g, VIS_SECRET);
    setupTrace(&ctx, ctx.config());

    auto [x_out_s, valid_out_s] = duplicate_brent_kung(&ctx, x_s, v_s, g_s);

    test::printProfileData(&ctx);
    if (lctx->Rank() == 0) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "duplicate_brent_kung Large Scale Test (Intergers):"
                << std::endl;
      std::cout << "  - Input Shape : " << n << " blocks of size " << block_size
                << std::endl;
      std::cout << "  - Protocol    : " << protocol << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    auto x_out_opened =
        hal::dump_public_as<int>(&ctx, hal::reveal(&ctx, x_out_s));
    auto valid_out_opened =
        hal::dump_public_as<int>(&ctx, hal::reveal(&ctx, valid_out_s));

    // verifiy correctness
    if (lctx->Rank() == 0) {
      const float atol = 0.01;
      for (int64_t i = 0; i < n; ++i) {
        if (g(i, 0) == 0) {
          for (int64_t b = 0; b < block_size; ++b) {
            EXPECT_NEAR(x_out_opened(i, b), x(i, b), atol)
                << "x reset failed at i=" << i << ", b=" << b
                << ", diff=" << std::abs(x_out_opened(i, b) - x(i, b));
            EXPECT_NEAR(valid_out_opened(i, b), valids(i, b), atol)
                << "valids reset failed at i=" << i << ", b=" << b;
          }
        } else {
          for (int64_t b = 0; b < block_size; ++b) {
            EXPECT_NEAR(x_out_opened(i, b), x_out_opened(i - 1, b), atol)
                << "x propagation failed at i=" << i << ", b=" << b
                << ", diff = "
                << std::abs(x_out_opened(i, b) - x_out_opened(i - 1, b));
            EXPECT_NEAR(valid_out_opened(i, b), valid_out_opened(i - 1, b),
                        atol)
                << "valids propagation failed at i=" << i << ", b=" << b;
          }
        }
      }
    }
  });
}

}  // namespace spu::kernel::hal