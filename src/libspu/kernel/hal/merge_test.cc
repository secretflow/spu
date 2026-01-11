#include "libspu/kernel/hal/merge.h"

#include <algorithm>
#include <random>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>

#include "gtest/gtest.h"
#include "magic_enum.hpp"

#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"
namespace spu::kernel::hal {

TEST(OddEvenMergeTest, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>
                                    &lctx) {
    SPUContext ctx = test::makeSPUContext(protocol, field, lctx);
    xt::xarray<float> keys = {{1, 3, 20}, {2, 50, 60}, {1, 3, 4}, {2, 5, 60}};
    if (lctx->Rank() == 0) {
      std::cout << "keys = \n" << keys << std::endl;
    }
    xt::xarray<float> res_expected = {{1, 2, 3, 20, 50, 60},
                                      {1, 2, 3, 4, 5, 60}};
    Value keys_s = test::makeValue(&ctx, keys, VIS_SECRET);

    // Merge
    std::vector<spu::Value> res_s =
        merge(&ctx, {keys_s}, {}, 1, false, hal::SortDirection::Ascending,
              Visibility::VIS_SECRET);

    EXPECT_EQ(res_s.size(), 1);
    auto res = hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[0]));

    if (lctx->Rank() == 0) {
      LOG(INFO) << "res_expected = \n" << res_expected;
      LOG(INFO) << "res  = \n" << res;
    }
    EXPECT_TRUE(xt::allclose(res, res_expected, 0.01, 0.001))
        << "expected\n"
        << res_expected << "\nvs got\n"
        << res;
  });
}

TEST(OddEvenMergeTest, LargeScaleRealNumbers) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  const int num_groups = 1;
  int input_size = 524288;
  int total_size = input_size * 2;
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> size_dist(1, input_size);
  std::uniform_real_distribution<float> key_dist(0.0, 1000.0);
  xt::xarray<float> keys = xt::zeros<float>({num_groups * 2, input_size});
  for (int group = 0; group < num_groups; ++group) {
    std::vector<float> keys_l(input_size);
    std::vector<float> keys_r(input_size);

    for (int i = 0; i < input_size; ++i) {
      keys_l[i] = key_dist(rng);
    }
    for (int i = 0; i < input_size; ++i) {
      keys_r[i] = key_dist(rng);
    }

    std::sort(keys_l.begin(), keys_l.end());
    std::sort(keys_r.begin(), keys_r.end());

    for (int i = 0; i < input_size; ++i) {
      keys(group, i) = keys_l[i];
    }
    for (int i = 0; i < input_size; ++i) {
      keys(group + 1, i) = keys_r[i];
    }
  }

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);
        Value keys_s = test::makeValue(&ctx, keys, VIS_SECRET);

        auto stats = lctx->GetStats();
        size_t start_bytes = stats->sent_bytes;
        size_t start_actions = stats->sent_actions;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Merge
        std::vector<spu::Value> res_s =
            merge(&ctx, {keys_s}, {}, 1, false, hal::SortDirection::Ascending,
                  Visibility::VIS_SECRET);

        auto end_time = std::chrono::high_resolution_clock::now();
        stats = lctx->GetStats();
        size_t end_bytes = stats->sent_bytes;
        size_t end_actions = stats->sent_actions;

        if (lctx->Rank() == 0) {
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_time - start_time)
                              .count();
          auto comm_bytes = end_bytes - start_bytes;
          auto comm_rounds = end_actions - start_actions;
          double comm_mb = static_cast<double>(comm_bytes) / 1024.0 / 1024.0;

          std::cout << "\n========================================"
                    << std::endl;
          std::cout << "Merge Large Scale Test:" << std::endl;
          std::cout << "  - Input Shape : (" << num_groups << ", " << input_size
                    << ") + (" << num_groups << ", " << input_size << ")"
                    << std::endl;
          std::cout << "  - Total Elements: " << num_groups * total_size
                    << std::endl;
          std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
          std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes ("
                    << comm_mb << " MB)" << std::endl;
          std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
                    << std::endl;
          std::cout << "========================================\n"
                    << std::endl;
        }

        ASSERT_EQ(res_s.size(), 1);
        auto res =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[0]));

        for (int group = 0; group < num_groups; ++group) {
          for (int col = 0; col < total_size - 1; ++col) {
            EXPECT_LE(res(group, col), res(group, col + 1))
                << "group " << group << " not sorted at position " << col;
          }
        }
      });
}

TEST(OddEvenMerge_WithPayload_Test, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  xt::xarray<float> keys = {
      {1, 3, 100}, {2, 4, 200}, {10, 30, 50}, {20, 40, 60}};
  xt::xarray<float> payloads = {{1, 1, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 0}};
  xt::xarray<float> expected_res_x = {{1, 2, 3, 4, 100, 200},
                                      {10, 20, 30, 40, 50, 60}};
  xt::xarray<float> expected_res_payload = {{1, 1, 1, 1, 0, 0},
                                            {1, 1, 0, 1, 1, 0}};

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>
                                    &lctx) {
    SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

    // 4. 输入转密文
    Value keys_s = test::makeValue(&ctx, keys, VIS_SECRET);
    Value payloads_s = test::makeValue(&ctx, payloads, VIS_SECRET);

    // ==========================================
    // 5. 记录开始时的统计指标
    // ==========================================
    auto stats = lctx->GetStats();
    size_t start_bytes = stats->sent_bytes;
    size_t start_actions = stats->sent_actions;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 6. 执行 Merge
    std::vector<spu::Value> res_s =
        merge(&ctx, {keys_s}, {payloads_s},
              1,      // sort_dim
              false,  // is_stable
              hal::SortDirection::Ascending, Visibility::VIS_SECRET);

    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();

    // ==========================================
    // 7. 计算并打印增量指标 (只在 Rank 0 打印)
    // ==========================================
    stats = lctx->GetStats();  // 刷新统计
    size_t end_bytes = stats->sent_bytes;
    size_t end_actions = stats->sent_actions;

    if (lctx->Rank() == 0) {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_time - start_time)
                          .count();
      auto comm_bytes = end_bytes - start_bytes;
      auto comm_rounds = end_actions - start_actions;
      double comm_mb = static_cast<double>(comm_bytes) / 1024.0 / 1024.0;

      std::cout << "\n========================================" << std::endl;
      std::cout << "Merge Protocol Execution Stats:" << std::endl;
      std::cout << "  - Input Shape : (2, " << keys.shape(1) << ") x 2 arrays "
                << std::endl;
      std::cout << "  - Protocol    : " << protocol << std::endl;
      std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
      std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes (" << comm_mb
                << " MB)" << std::endl;
      std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
                << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    // verefiy correctness
    ASSERT_EQ(res_s.size(), 2);

    if (lctx->Rank() == 0) {
      std::cout << "keys = \n" << keys << std::endl;
      std::cout << "payloads = \n" << payloads << std::endl;
    }

    auto res_x = hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[0]));
    auto res_payload =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[1]));

    if (lctx->Rank() == 0) {
      std::cout << "Data Verification Details:" << std::endl;
      std::cout << "----------------------------------------" << std::endl;

      std::cout << "[Values Comparison]" << std::endl;
      std::cout << "Expected:\n" << expected_res_x << std::endl;
      std::cout << "Actual  :\n" << res_x << std::endl;

      std::cout << "\n[Payloads Comparison]" << std::endl;
      std::cout << "Expected:\n" << expected_res_payload << std::endl;
      std::cout << "Actual  :\n" << res_payload << std::endl;
      std::cout << "----------------------------------------" << std::endl;
    }

    EXPECT_TRUE(xt::allclose(res_x, expected_res_x, 0.001, 0.001))
        << "Values sorting failed!";

    EXPECT_TRUE(xt::allclose(res_payload, expected_res_payload, 0.001, 0.001))
        << "Payloads tracking failed!";
  });
}

TEST(OddEvenMerge_WithPayload_Test, LargeScaleRealNumbers) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  const int num_groups = 1;
  int input_size = 524288;
  int total_size = input_size * 2;
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> size_dist(1, input_size);
  std::uniform_real_distribution<float> key_dist(0.0, 1000.0);
  std::uniform_int_distribution<int> payload_dist(0, 1);
  xt::xarray<float> keys = xt::zeros<float>({num_groups * 2, input_size});
  xt::xarray<float> payloads = xt::zeros<float>({num_groups * 2, input_size});

  // prepare sorted inputs and their payloads
  for (int group = 0; group < num_groups; ++group) {
    std::vector<float> keys_l(input_size);
    std::vector<float> keys_r(input_size);

    for (int i = 0; i < input_size; ++i) {
      keys_l[i] = key_dist(rng);
    }
    for (int i = 0; i < input_size; ++i) {
      keys_r[i] = key_dist(rng);
    }

    std::sort(keys_l.begin(), keys_l.end());
    std::sort(keys_r.begin(), keys_r.end());

    for (int i = 0; i < input_size; ++i) {
      keys(group, i) = keys_l[i];
      payloads(group, i) = static_cast<float>(payload_dist(rng));
    }
    for (int i = 0; i < input_size; ++i) {
      keys(group + 1, i) = keys_r[i];
      payloads(group + 1, i) = static_cast<float>(payload_dist(rng));
    }
  }

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>
                                    &lctx) {
    SPUContext ctx = test::makeSPUContext(protocol, field, lctx);
    Value keys_s = test::makeValue(&ctx, keys, VIS_SECRET);
    Value payloads_s = test::makeValue(&ctx, payloads, VIS_SECRET);

    auto stats = lctx->GetStats();
    size_t start_bytes = stats->sent_bytes;
    size_t start_actions = stats->sent_actions;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Merge
    std::vector<spu::Value> res_s =
        merge(&ctx, {keys_s}, {payloads_s},
              1,      // sort_dim
              false,  // is_stable
              hal::SortDirection::Ascending, Visibility::VIS_SECRET);

    auto end_time = std::chrono::high_resolution_clock::now();
    size_t end_bytes = stats->sent_bytes;
    size_t end_actions = stats->sent_actions;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    size_t comm_bytes = end_bytes - start_bytes;
    size_t comm_actions = end_actions - start_actions;

    ASSERT_EQ(res_s.size(), 2);
    auto res_x = hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[0]));
    auto res_payload =
        hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[1]));

    if (lctx->Rank() == 0) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Large Scale Test Performance Metrics:" << std::endl;
      std::cout << "========================================" << std::endl;
      std::cout << "Input Size: " << num_groups << " x " << input_size << " + "
                << num_groups << " x " << input_size << std::endl;
      std::cout << "Total Elements: " << num_groups * total_size << std::endl;
      std::cout << "Execution Time: " << duration << " ms" << std::endl;
      std::cout << "Communication: " << comm_bytes << " bytes ("
                << static_cast<double>(comm_bytes) / 1024.0 / 1024.0 << " MB)"
                << std::endl;
      std::cout << "Actions: " << comm_actions << std::endl;
      std::cout << "Avg bytes/element: "
                << static_cast<double>(comm_bytes) / (num_groups * total_size)
                << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    // verify correctness
    for (int group = 0; group < num_groups; ++group) {
      // is res_x sorted?
      for (int col = 0; col < total_size - 1; ++col) {
        EXPECT_LE(res_x(group, col), res_x(group, col + 1))
            << "Row " << group << " is not sorted at position " << col
            << ", values: " << res_x(group, col) << " > "
            << res_x(group, col + 1);
      }

      // is the sum of payloads remained? (we use this verify method
      // because Merge functuin is not a stable sort)
      float expected_payload_count = 0.0F;
      float actual_payload_count = 0.0F;
      for (int col = 0; col < total_size; ++col) {
        expected_payload_count += payloads(group, col);
      }
      for (int col = 0; col < total_size; ++col) {
        actual_payload_count += res_payload(group, col);
      }
      EXPECT_NEAR(actual_payload_count, expected_payload_count, 0.1F)
          << "Row " << group << " payload bits count mismatch! Expected: "
          << expected_payload_count << ", Got: " << actual_payload_count;

      // is the sum of x remained?
      double expected_sum = 0.0;
      double actual_sum = 0.0;
      for (int col = 0; col < total_size; ++col) {
        expected_sum += static_cast<double>(keys(group, col));
      }
      for (int col = 0; col < total_size; ++col) {
        actual_sum += static_cast<double>(res_x(group, col));
      }
      double relative_error =
          std::abs(actual_sum - expected_sum) / expected_sum;
      EXPECT_LT(relative_error, 0.001)
          << "Row " << group
          << " values sum mismatch! Expected: " << expected_sum
          << ", Got: " << actual_sum << ", Relative error: " << relative_error;
    }
  });
}

}  // namespace spu::kernel::hal
