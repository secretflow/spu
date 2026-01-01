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

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);
        xt::xarray<float> x1 = {{1, 3, 20}, {1, 3, 4}};
        xt::xarray<float> x2 = {{2, 50, 60}, {2, 5, 60}};
        if (lctx->Rank() == 0) {
          std::cout << "x1 = \n" << x1 << std::endl;
          std::cout << "x2 = \n" << x2 << std::endl;
        }
        xt::xarray<float> res_expected = {{1, 2, 3, 20, 50, 60},
                                          {1, 2, 3, 4, 5, 60}};
        Value x1_s = test::makeValue(&ctx, x1, VIS_SECRET);
        Value x2_s = test::makeValue(&ctx, x2, VIS_SECRET);

        // Merge
        std::vector<spu::Value> res_s = merge(
            &ctx, {x1_s, x2_s}, 1, false,
            [&](absl::Span<const spu::Value> inputs) {
              return hal::less(&ctx, inputs[0], inputs[1]);
            },
            Visibility::VIS_SECRET);

        EXPECT_EQ(res_s.size(), 1);
        auto res =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[0]));

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

  const int num_rows = 1;
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> size_dist(1, 524288);
  int input1_size = size_dist(rng);
  int input2_size = size_dist(rng);
  // int input1_size = 500000;
  // int input2_size = 500000;
  int total_size = input1_size + input2_size;

  std::uniform_real_distribution<float> value_dist(0.0, 1000.0);

  xt::xarray<float> x1 = xt::zeros<float>({num_rows, input1_size});
  xt::xarray<float> x2 = xt::zeros<float>({num_rows, input2_size});

  for (int row = 0; row < num_rows; ++row) {
    std::vector<float> temp_x1(input1_size);
    std::vector<float> temp_x2(input2_size);

    for (int i = 0; i < input1_size; ++i) {
      temp_x1[i] = value_dist(rng);
    }
    for (int i = 0; i < input2_size; ++i) {
      temp_x2[i] = value_dist(rng);
    }

    std::sort(temp_x1.begin(), temp_x1.end());
    std::sort(temp_x2.begin(), temp_x2.end());

    for (int i = 0; i < input1_size; ++i) {
      x1(row, i) = temp_x1[i];
    }
    for (int i = 0; i < input2_size; ++i) {
      x2(row, i) = temp_x2[i];
    }
  }

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);
        Value x1_s = test::makeValue(&ctx, x1, VIS_SECRET);
        Value x2_s = test::makeValue(&ctx, x2, VIS_SECRET);

        auto stats = lctx->GetStats();
        size_t start_bytes = stats->sent_bytes;
        size_t start_actions = stats->sent_actions;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Merge
        std::vector<spu::Value> res_s = merge(
            &ctx, {x1_s, x2_s}, 1, false,
            [&](absl::Span<const spu::Value> inputs) {
              return hal::less(&ctx, inputs[0], inputs[1]);
            },
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
          std::cout << "  - Input Shape : (" << num_rows << ", " << input1_size
                    << ") + (" << num_rows << ", " << input2_size << ")"
                    << std::endl;
          std::cout << "  - Total Elements: " << num_rows * total_size
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

        for (int row = 0; row < num_rows; ++row) {
          for (int col = 0; col < total_size - 1; ++col) {
            EXPECT_LE(res(row, col), res(row, col + 1))
                << "Row " << row << " not sorted at position " << col;
          }
        }
      });
}

TEST(OddEvenMerge_WithPayload_Test, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  xt::xarray<float> x1 = {{1, 3, 100}, {10, 30, 50}};
  xt::xarray<float> x2 = {{2, 4, 200}, {20, 40, 60}};
  xt::xarray<float> p1 = {{1, 1, 0}, {1, 0, 1}};
  xt::xarray<float> p2 = {{1, 1, 0}, {1, 1, 0}};
  xt::xarray<float> expected_res_x = {{1, 2, 3, 4, 100, 200},
                                      {10, 20, 30, 40, 50, 60}};
  xt::xarray<float> expected_res_payload = {{1, 1, 1, 1, 0, 0},
                                            {1, 1, 0, 1, 1, 0}};

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>
                                    &lctx) {
    SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

    // 4. 输入转密文
    Value x1_s = test::makeValue(&ctx, x1, VIS_SECRET);
    Value x2_s = test::makeValue(&ctx, x2, VIS_SECRET);
    Value p1_s = test::makeValue(&ctx, p1, VIS_SECRET);
    Value p2_s = test::makeValue(&ctx, p2, VIS_SECRET);

    std::vector<spu::Value> inputs_x = {x1_s, x2_s};
    std::vector<spu::Value> inputs_payload = {p1_s, p2_s};

    // ==========================================
    // 5. 记录开始时的统计指标
    // ==========================================
    auto stats = lctx->GetStats();
    size_t start_bytes = stats->sent_bytes;
    size_t start_actions = stats->sent_actions;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 6. 执行 Merge
    std::vector<spu::Value> res_s =
        merge_with_payloads(&ctx, inputs_x, inputs_payload,
                            1,      // sort_dim
                            false,  // is_stable
                            [&](absl::Span<const spu::Value> vals) {
                              return hal::less(&ctx, vals[0], vals[1]);
                            });

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
      std::cout << "  - Input Shape : (2, " << x1.shape(1) << ") x 2 arrays "
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
      std::cout << "x1 = \n" << x1 << std::endl;
      std::cout << "x2 = \n" << x2 << std::endl;
      std::cout << "p1 = \n" << p1 << std::endl;
      std::cout << "p2 = \n" << p2 << std::endl;
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
  const int num_rows = 1;  // number of arrays
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> size_dist(1, 524288);
  // sizes of arrays
  int input1_size = size_dist(rng);
  int input2_size = size_dist(rng);
  const int total_size = input1_size + input2_size;

  std::uniform_real_distribution<float> value_dist(0.0, 1000.0);
  std::uniform_int_distribution<int> payload_dist(0, 1);

  xt::xarray<float> x1 = xt::zeros<float>({num_rows, input1_size});
  xt::xarray<float> x2 = xt::zeros<float>({num_rows, input2_size});
  xt::xarray<float> p1 = xt::zeros<float>({num_rows, input1_size});
  xt::xarray<float> p2 = xt::zeros<float>({num_rows, input2_size});

  // prepare sorted inputs and their payloads
  for (int row = 0; row < num_rows; ++row) {
    std::vector<float> temp_x1(input1_size);
    std::vector<float> temp_x2(input2_size);

    for (int i = 0; i < input1_size; ++i) {
      temp_x1[i] = value_dist(rng);
      p1(row, i) = static_cast<float>(payload_dist(rng));
    }
    for (int i = 0; i < input2_size; ++i) {
      temp_x2[i] = value_dist(rng);
      p2(row, i) = static_cast<float>(payload_dist(rng));
    }

    std::sort(temp_x1.begin(), temp_x1.end());
    std::sort(temp_x2.begin(), temp_x2.end());

    for (int i = 0; i < input1_size; ++i) {
      x1(row, i) = temp_x1[i];
    }
    for (int i = 0; i < input2_size; ++i) {
      x2(row, i) = temp_x2[i];
    }
  }

  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);
        Value x1_s = test::makeValue(&ctx, x1, VIS_SECRET);
        Value x2_s = test::makeValue(&ctx, x2, VIS_SECRET);
        Value p1_s = test::makeValue(&ctx, p1, VIS_SECRET);
        Value p2_s = test::makeValue(&ctx, p2, VIS_SECRET);

        std::vector<spu::Value> inputs = {x1_s, x2_s};
        std::vector<spu::Value> inputs_payload = {p1_s, p2_s};

        auto stats = lctx->GetStats();
        size_t start_bytes = stats->sent_bytes;
        size_t start_actions = stats->sent_actions;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Merge
        std::vector<spu::Value> res_s =
            merge_with_payloads(&ctx, inputs, inputs_payload,
                                1,      // sort_dim
                                false,  // is_stable
                                [&](absl::Span<const spu::Value> vals) {
                                  return hal::less(&ctx, vals[0], vals[1]);
                                });

        auto end_time = std::chrono::high_resolution_clock::now();
        size_t end_bytes = stats->sent_bytes;
        size_t end_actions = stats->sent_actions;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time)
                            .count();
        size_t comm_bytes = end_bytes - start_bytes;
        size_t comm_actions = end_actions - start_actions;

        ASSERT_EQ(res_s.size(), 2);
        auto res_x =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[0]));
        auto res_payload =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, res_s[1]));

        if (lctx->Rank() == 0) {
          std::cout << "\n========================================"
                    << std::endl;
          std::cout << "Large Scale Test Performance Metrics:" << std::endl;
          std::cout << "========================================" << std::endl;
          std::cout << "Input Size: " << num_rows << " x " << input1_size
                    << " + " << num_rows << " x " << input2_size << std::endl;
          std::cout << "Total Elements: " << num_rows * total_size << std::endl;
          std::cout << "Execution Time: " << duration << " ms" << std::endl;
          std::cout << "Communication: " << comm_bytes << " bytes ("
                    << static_cast<double>(comm_bytes) / 1024.0 / 1024.0
                    << " MB)" << std::endl;
          std::cout << "Actions: " << comm_actions << std::endl;
          std::cout << "Avg bytes/element: "
                    << static_cast<double>(comm_bytes) / (num_rows * total_size)
                    << std::endl;
          std::cout << "========================================\n"
                    << std::endl;
        }

        // verify correctness
        for (int row = 0; row < num_rows; ++row) {
          // is res_x sorted?
          for (int col = 0; col < total_size - 1; ++col) {
            EXPECT_LE(res_x(row, col), res_x(row, col + 1))
                << "Row " << row << " is not sorted at position " << col
                << ", values: " << res_x(row, col) << " > "
                << res_x(row, col + 1);
          }

          // is the sum of  payloads remained? (we use this verify method
          // because Merge functuin is not a stable sort)
          float expected_payload_count = 0.0F;
          float actual_payload_count = 0.0F;
          for (int col = 0; col < input1_size; ++col) {
            expected_payload_count += p1(row, col);
          }
          for (int col = 0; col < input2_size; ++col) {
            expected_payload_count += p2(row, col);
          }
          for (int col = 0; col < total_size; ++col) {
            actual_payload_count += res_payload(row, col);
          }
          EXPECT_NEAR(actual_payload_count, expected_payload_count, 0.1F)
              << "Row " << row << " payload bits count mismatch! Expected: "
              << expected_payload_count << ", Got: " << actual_payload_count;

          // is the sum of x remained?
          double expected_sum = 0.0;
          double actual_sum = 0.0;
          for (int col = 0; col < input1_size; ++col) {
            expected_sum += static_cast<double>(x1(row, col));
          }
          for (int col = 0; col < input2_size; ++col) {
            expected_sum += static_cast<double>(x2(row, col));
          }
          for (int col = 0; col < total_size; ++col) {
            actual_sum += static_cast<double>(res_x(row, col));
          }
          double relative_error =
              std::abs(actual_sum - expected_sum) / expected_sum;
          EXPECT_LT(relative_error, 0.001)
              << "Row " << row
              << " values sum mismatch! Expected: " << expected_sum
              << ", Got: " << actual_sum
              << ", Relative error: " << relative_error;
        }
      });
}

}  // namespace spu::kernel::hal
