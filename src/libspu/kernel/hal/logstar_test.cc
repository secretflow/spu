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
      std::cout << "x:\n" << x << std::endl;
      std::cout << "valid:\n" << valids << std::endl;
      std::cout << "g:\n" << g << std::endl;
      std::cout << "x_out:\n" << x_out_opened << std::endl;
      std::cout << "valid_out:\n" << valid_out_opened << std::endl;
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

TEST_F(ExtractOrderedTest, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

        int64_t num_arrays = 2;
        int64_t n = 6;
        xt::xarray<int64_t> x = {{0, 1, 2, 3, 4, 5}, {10, 11, 12, 13, 14, 15}};
        xt::xarray<int64_t> valids = {0, 1, 0, 1, 1, 0};
        xt::xarray<int64_t> x_out_expected = {{1, 3, 4}, {11, 13, 14}};

        auto x_in = test::makeValue(&ctx, x, VIS_SECRET);
        valids.reshape({1, static_cast<size_t>(n)});
        auto valids_in = test::makeValue(&ctx, valids, VIS_SECRET);

        auto res = extract_ordered(&ctx, x_in, valids_in);
        auto& y = res.first;
        auto valid_count = res.second;

        EXPECT_EQ(valid_count, x_out_expected.shape()[1]);
        if (lctx->Rank() == 0) {
          std::cout << "x: " << x << std::endl;
          std::cout << "valids: " << valids << std::endl;
        }

        for (int64_t i = 0; i < num_arrays; ++i) {
          auto y_revealed = hal::reveal(&ctx, y[i]);
          auto y_row = hal::dump_public_as<int64_t>(&ctx, y_revealed);

          xt::xarray<int64_t> y_valid_part;
          if (y_row.dimension() == 2) {
            y_valid_part = xt::view(y_row, 0, xt::range(0, valid_count));
          } else {
            y_valid_part = xt::view(y_row, xt::range(0, valid_count));
          }
          auto y_expected_row = xt::row(x_out_expected, i);
          EXPECT_TRUE(xt::allclose(y_valid_part, y_expected_row));

          if (lctx->Rank() == 0) {
            std::cout << "y[" << i << "]: " << y_valid_part << std::endl;
          }
        }
      });
}

TEST_F(ExtractOrderedTest, LargeScale) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
    const int64_t num_arrays = 2;
    const int64_t n = 1000000;
    std::vector<std::vector<int64_t>> xs(num_arrays, std::vector<int64_t>(n));
    for (int64_t r = 0; r < num_arrays; ++r) {
      int64_t base = r * 100000;
      for (int64_t i = 0; i < n; ++i) {
        xs[r][i] = base + i;
      }
    }

    // valid bits
    std::vector<int64_t> valids(n);
    for (int64_t i = 0; i < n; ++i) {
      valids[i] = (i % 5 == 0) ? 1 : 0;
    }

    std::vector<int64_t> x_combined;
    x_combined.reserve(num_arrays * n);
    for (int64_t r = 0; r < num_arrays; ++r) {
      x_combined.insert(x_combined.end(), xs[r].begin(), xs[r].end());
    }
    xt::xarray<int64_t> x_arr = xt::adapt(x_combined);
    x_arr.reshape({static_cast<size_t>(num_arrays), static_cast<size_t>(n)});
    auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);

    xt::xarray<int64_t> f_arr = xt::adapt(valids);
    f_arr.reshape({1, static_cast<size_t>(n)});
    auto f_in = test::makeValue(&ctx, f_arr, VIS_SECRET);

    // Ground Truth
    std::vector<std::vector<int64_t>> exps(num_arrays);
    for (int64_t i = 0; i < n; ++i) {
      if (valids[i]) {
        for (int64_t r = 0; r < num_arrays; ++r) {
          exps[r].push_back(xs[r][i]);
        }
      }
    }
    size_t valid_count_expected = exps.empty() ? 0 : exps[0].size();

    std::vector<int64_t> y_combined;
    y_combined.reserve(num_arrays * valid_count_expected);
    for (int64_t r = 0; r < num_arrays; ++r) {
      y_combined.insert(y_combined.end(), exps[r].begin(), exps[r].end());
    }
    xt::xarray<int64_t> x_out_expected = xt::adapt(y_combined);
    x_out_expected.reshape({static_cast<size_t>(num_arrays),
                            static_cast<size_t>(valid_count_expected)});

    setupTrace(&ctx, ctx.config());
    auto res = extract_ordered(&ctx, x_in, f_in);
    test::printProfileData(&ctx);

    auto& y = res.first;
    auto valid_count = res.second;
    EXPECT_EQ(valid_count, static_cast<int64_t>(valid_count_expected));

    for (int64_t i = 0; i < num_arrays; ++i) {
      auto y_revealed = hal::reveal(&ctx, y[i]);
      auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

      xt::xarray<int64_t> y_valid_part;
      if (y_vec.dimension() == 2) {
        y_valid_part = xt::view(y_vec, 0, xt::range(0, valid_count));
      } else {
        y_valid_part = xt::view(y_vec, xt::range(0, valid_count));
      }

      auto y_expected_row = xt::row(x_out_expected, i);
      EXPECT_TRUE(xt::allclose(y_valid_part, y_expected_row));
    }
  });
}

TEST(LogstarTest, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
        xt::xarray<float> x = {1, 3, 20, 40, 55};
        xt::xarray<float> y = {2, 50, 60, 70, 80};
        if (lctx->Rank() == 0) {
          std::cout << "x = \n" << x << std::endl;
          std::cout << "y = \n" << y << std::endl;
        }
        auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
        auto y_s = test::makeValue(&ctx, y, VIS_SECRET);
        setupTrace(&ctx, ctx.config());

        // Merge
        logstar(&ctx, x_s, y_s);

        // test::printProfileData(&ctx);
      });
}

TEST(LogstarRecursiveTest, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
        xt::xarray<float> x = {{{1, 0}, {3, 1}, {20, 1}, {40, 1}, {55, 1}},
                               {{2, 1}, {3, 1}, {4, 1}, {5, 1}, {55, 1}}};
        xt::xarray<float> y = {{{2, 1}, {50, 1}, {60, 1}, {70, 1}, {80, 1}},
                               {{3, 1}, {4, 1}, {5, 1}, {6, 1}, {88, 1}}};
        if (lctx->Rank() == 0) {
          std::cout << "x = \n" << x << std::endl;
          std::cout << "y = \n" << y << std::endl;
        }
        auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
        auto y_s = test::makeValue(&ctx, y, VIS_SECRET);
        setupTrace(&ctx, ctx.config());

        // Merge
        LogstarRecursive(&ctx, x_s, y_s);

        // test::printProfileData(&ctx);
      });
}

}  // namespace spu::kernel::hal