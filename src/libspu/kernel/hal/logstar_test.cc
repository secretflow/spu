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
    const int64_t batch_size = 2;
    const int64_t n = 8;
    const int64_t block_size = 2;

    xt::xarray<float> x = {{5, 6, 2, 2, 10, 10, 4, 3, 1, 1, 2, 2, 5, 5, 3, 2},
                           {1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1}};
    xt::xarray<float> g = {{0, 1, 1, 0, 1, 1, 1, 0}, {0, 1, 1, 0, 1, 1, 1, 0}};

    xt::xarray<float> x_out_expected = {
        {5, 6, 5, 6, 5, 6, 4, 3, 4, 3, 4, 3, 4, 3, 3, 2},
        {1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1}};

    x.reshape({static_cast<size_t>(batch_size), static_cast<size_t>(n),
               static_cast<size_t>(block_size), 1});
    auto x_s = test::makeValue(&ctx, x, VIS_SECRET);

    g.reshape({static_cast<size_t>(batch_size), static_cast<size_t>(n)});
    auto g_s = test::makeValue(&ctx, g, VIS_SECRET);

    setupTrace(&ctx, ctx.config());

    auto x_out_s = duplicate_brent_kung(&ctx, x_s, g_s);

    test::printProfileData(&ctx);
    auto x_out_opened =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, x_out_s));

    x.reshape(
        {static_cast<size_t>(batch_size), static_cast<size_t>(n * block_size)});
    x_out_opened.reshape(
        {static_cast<size_t>(batch_size), static_cast<size_t>(n * block_size)});
    x_out_expected.reshape(
        {static_cast<size_t>(batch_size), static_cast<size_t>(n * block_size)});
    if (lctx->Rank() == 0) {
      std::cout << "x:\n" << x << std::endl;
      std::cout << "g:\n" << g << std::endl;
      std::cout << "x_out:\n" << x_out_opened << std::endl;
    }
    if (lctx->Rank() == 0) {
      EXPECT_TRUE(xt::allclose(x_out_opened, x_out_expected));
    }
  });
}

TEST_F(BrentKungTest, LargeScaleInputs) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);
        const int64_t batch_size = 1;
        const int64_t n = 100000;
        const int64_t block_size = 2;
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist_x(0, 1000);
        std::uniform_int_distribution<int> dist_binary(0, 1);

        xt::xarray<float> x = xt::zeros<float>({batch_size * n * block_size});
        xt::xarray<float> g = xt::zeros<float>({batch_size * n});

        for (auto& v : x) v = dist_x(rng);
        g[0] = 0.0F;
        for (int64_t i = 1; i < n; ++i) {
          g[i] = static_cast<float>(dist_binary(rng));
        }

        x.reshape({static_cast<size_t>(batch_size), static_cast<size_t>(n),
                   static_cast<size_t>(block_size), 1});
        auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
        g.reshape({static_cast<size_t>(batch_size), static_cast<size_t>(n)});
        auto g_s = test::makeValue(&ctx, g, VIS_SECRET);

        setupTrace(&ctx, ctx.config());
        auto x_out_s = duplicate_brent_kung(&ctx, x_s, g_s);

        test::printProfileData(&ctx);
        if (lctx->Rank() == 0) {
          std::cout << "\n========================================"
                    << std::endl;
          std::cout << "duplicate_brent_kung Large Scale Test (Real numbers):"
                    << std::endl;
          std::cout << "  - Input Shape : " << n << " blocks of size "
                    << block_size << std::endl;
          std::cout << "  - Protocol    : " << protocol << std::endl;
          std::cout << "========================================\n"
                    << std::endl;
        }

        auto x_out_opened =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, x_out_s));

        x_out_opened.reshape(
            {static_cast<size_t>(n), static_cast<size_t>(block_size)});
        x.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
        g.reshape({static_cast<size_t>(n), 1});

        if (lctx->Rank() == 0) {
          const float atol = 0.01;
          for (int64_t i = 0; i < n; ++i) {
            if (g(i, 0) == 0) {
              for (int64_t b = 0; b < block_size; ++b) {
                EXPECT_NEAR(x_out_opened(i, b), x(i, b), atol)
                    << "x reset failed at i=" << i << ", b=" << b
                    << ", diff=" << std::abs(x_out_opened(i, b) - x(i, b));
              }
            } else {
              for (int64_t b = 0; b < block_size; ++b) {
                EXPECT_NEAR(x_out_opened(i, b), x_out_opened(i - 1, b), atol)
                    << "x propagation failed at i=" << i << ", b=" << b
                    << ", diff = "
                    << std::abs(x_out_opened(i, b) - x_out_opened(i - 1, b));
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

TEST_F(ExtractOrderedTest, LargeScaleInputs) {
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
        xt::xarray<float> x = {1, 3, 20, 20, 55};
        xt::xarray<float> y = {2, 20, 20, 70, 80};
        if (lctx->Rank() == 0) {
          std::cout << "x = \n" << x << std::endl;
          std::cout << "y = \n" << y << std::endl;
        }
        auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
        auto y_s = test::makeValue(&ctx, y, VIS_SECRET);
        setupTrace(&ctx, ctx.config());

        // Merge
        auto merged = logstar(&ctx, hal::SortDirection::Ascending, x_s, y_s);

        auto revealed =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, merged));
        if (ctx.lctx()->Rank() == 0) {
          std::cout << "merged: " << revealed << std::endl;
        }

        // test::printProfileData(&ctx);
      });
}

TEST(LogstarTest, LargeScaleInputs) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  const int n = 515;
  std::mt19937 gen(1139316);
  // std::mt19937 gen(6486);
  // auto seed = std::random_device{}();
  // std::cout << "seed:" << seed << std::endl;
  // std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(0, 100);
  // std::uniform_real_distribution<float> dist(0.0F, 10.0F);

  std::vector<float> x_vec(n);
  std::vector<float> y_vec(n);

  for (int i = 0; i < n; ++i) {
    x_vec[i] = static_cast<float>(dist(gen));
    y_vec[i] = static_cast<float>(dist(gen));
  }

  std::sort(x_vec.begin(), x_vec.end());
  std::sort(y_vec.begin(), y_vec.end());

  std::vector<float> expected_vec;
  expected_vec.reserve(2 * n);
  expected_vec.insert(expected_vec.end(), x_vec.begin(), x_vec.end());
  expected_vec.insert(expected_vec.end(), y_vec.begin(), y_vec.end());
  std::sort(expected_vec.begin(), expected_vec.end());
  xt::xarray<float> expected = xt::adapt(expected_vec);

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx = makeSPUContextWithProfile(protocol, field, lctx);

        xt::xarray<float> x = xt::adapt(x_vec);
        xt::xarray<float> y = xt::adapt(y_vec);

        if (lctx->Rank() == 0) {
          std::cout << "=========================================="
                    << std::endl;
          std::cout << "Testing Large Scale Random Merge..." << std::endl;
          std::cout << "Input sizes: nx = " << n << ", ny = " << n << std::endl;
          std::cout << "Total elements to merge: " << 2 * n << std::endl;
        }

        auto x_s = test::makeValue(&ctx, x, VIS_SECRET);
        auto y_s = test::makeValue(&ctx, y, VIS_SECRET);

        setupTrace(&ctx, ctx.config());
        auto merged = logstar(&ctx, hal::SortDirection::Ascending, x_s, y_s);
        test::printProfileData(&ctx);

        auto revealed =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, merged));

        if (lctx->Rank() == 0) {
          std::cout << "Verifying correctness..." << std::endl;

          bool is_match = true;
          int error_count = 0;

          for (size_t i = 0; i < 2 * n; ++i) {
            if (std::abs(revealed(i) - expected(i)) > 1e-2) {
              if (error_count == 0) {
                std::cout << "\n❌ [ERROR] First mismatch found at index " << i
                          << "!" << std::endl;
                std::cout << "SPU returned: " << revealed(i)
                          << " | Expected: " << expected(i) << std::endl;

                int start = std::max(0, static_cast<int>(i) - 5);
                int end = std::min(2 * n, static_cast<int>(i) + 5);

                std::cout << "--- Context SPU --- : ";
                for (int j = start; j < end; ++j)
                  std::cout << revealed(j) << ", ";
                std::cout << "\n--- Context Exp --- : ";
                for (int j = start; j < end; ++j)
                  std::cout << expected(j) << ", ";
                std::cout << std::endl;
              }
              is_match = false;
              error_count++;
              if (error_count >= 1) break;
            }
          }

          EXPECT_TRUE(is_match);
        }
      });
}

}  // namespace spu::kernel::hal