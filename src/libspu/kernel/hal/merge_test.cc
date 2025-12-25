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
// to print method name
std::ostream &operator<<(std::ostream &os,
                         spu::RuntimeConfig::SortMethod method) {
  os << magic_enum::enum_name(method);
  return os;
}
namespace spu::kernel::hal {

TEST(OddEvenMergeTest, WithoutValidBits) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  // 准备 Ground Truth 数据
  // xt::xarray<float> x1 = {{1, 3, 5, 20}, {1, 3, 5, 25}};
  // xt::xarray<float> x2 = {{2, 4, 6, 10}, {2, 4, 6, 15}};
  // xt::xarray<float> sorted_x = {{1, 2, 3, 4, 5, 6, 10, 20},
  //                               {1, 2, 3, 4, 5, 6, 15, 25}};
  xt::xarray<float> x1 = {{1, 3, 20}, {1, 3, 4}};
  xt::xarray<float> x2 = {{2, 5}, {2, 5}};
  xt::xarray<float> sorted_x = {{1, 2, 3, 5, 20}, {1, 2, 3, 4, 5}};

  // 启动 MPC 模拟环境
  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        // 1. 创建绑定了网络上下文的 SPU Context
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

        // 2. 准备密文输入
        Value x1_v = test::makeValue(&ctx, x1, VIS_SECRET);
        Value x2_v = test::makeValue(&ctx, x2, VIS_SECRET);

        // ==========================================
        // 3. 记录开始时的统计指标
        // ==========================================
        auto stats = lctx->GetStats();
        size_t start_bytes = stats->sent_bytes;
        size_t start_actions = stats->sent_actions;

        // 4. 执行 Merge 函数
        // 注意：确保你的 merge 函数里已经包含了之前讨论的 vector 转换逻辑
        std::vector<spu::Value> rets = merge(
            &ctx, {x1_v, x2_v}, 1, false,
            [&](absl::Span<const spu::Value> inputs) {
              return hal::less(&ctx, inputs[0], inputs[1]);
            },
            Visibility::VIS_SECRET);

        // ==========================================
        // 5. 计算并打印增量指标
        // ==========================================
        size_t end_bytes = stats->sent_bytes;
        size_t end_actions = stats->sent_actions;

        // 让 Rank 0 (通常是主机) 负责打印，避免日志刷屏
        if (lctx->Rank() == 0) {
          auto comm_bytes = end_bytes - start_bytes;
          auto comm_rounds = end_actions - start_actions;

          SPDLOG_INFO("========================================");
          SPDLOG_INFO("Merge Protocol Execution Stats:");
          SPDLOG_INFO("  - Protocol: {}", protocol);
          SPDLOG_INFO("  - Parties : {}", npc);
          SPDLOG_INFO("  - Comm Bytes : {} bytes", comm_bytes);
          SPDLOG_INFO("  - Comm Rounds: {} actions", comm_rounds);
          SPDLOG_INFO("========================================");
        }

        // 6. 验证结果正确性
        EXPECT_EQ(rets.size(), 1);

        auto revealed = hal::reveal(&ctx, rets[0]);
        auto sorted_x_hat = hal::dump_public_as<float>(&ctx, revealed);

        // 在 Rank 0 打印结果对比
        if (lctx->Rank() == 0) {
          LOG(INFO) << "sorted_x expected = \n" << sorted_x;
          LOG(INFO) << "sorted_x_hat got  = \n" << sorted_x_hat;
        }

        EXPECT_TRUE(xt::allclose(sorted_x, sorted_x_hat, 0.01, 0.001))
            << "expected\n"
            << sorted_x << "\nvs got\n"
            << sorted_x_hat;
      });
}

TEST(OddEvenMergeTest, WithoutValidBitsLargeScale) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  const int num_rows = 1;
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> size_dist(1, 524288);
  int input1_size = size_dist(rng);
  int input2_size = size_dist(rng);
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

        Value x1_sec = test::makeValue(&ctx, x1, VIS_SECRET);
        Value x2_sec = test::makeValue(&ctx, x2, VIS_SECRET);

        auto stats = lctx->GetStats();
        size_t start_bytes = stats->sent_bytes;
        size_t start_actions = stats->sent_actions;
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<spu::Value> rets = merge(
            &ctx, {x1_sec, x2_sec}, 1, false,
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

        ASSERT_EQ(rets.size(), 1);
        auto res_pub =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, rets[0]));

        for (int row = 0; row < num_rows; ++row) {
          for (int col = 0; col < total_size - 1; ++col) {
            EXPECT_LE(res_pub(row, col), res_pub(row, col + 1))
                << "Row " << row << " not sorted at position " << col;
          }
        }
      });
}

TEST(OddEvenMergeTest, WithPayload) {
  // 1. 基础配置
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  xt::xarray<float> x1 = {{1, 3, 100}, {10, 30, 50}};
  xt::xarray<float> x2 = {{2, 4, 200}, {20, 40, 60}};

  // Valid 标记 (1=有效, 0=哑元)
  xt::xarray<float> v1 = {{1, 1, 0}, {1, 0, 1}};
  xt::xarray<float> v2 = {{1, 1, 0}, {1, 1, 0}};

  // 预期排序后的 Values (升序, Shape 2x8)
  xt::xarray<float> expected_values = {{1, 2, 3, 4, 100, 200},
                                       {10, 20, 30, 40, 50, 60}};

  // 预期排序后的 Valids
  // Row 0: 有效值归并在前，哑元在后
  // Row 1: 全是有效值
  xt::xarray<float> expected_valids = {{1, 1, 1, 1, 0, 0}, {1, 1, 0, 1, 1, 0}};

  // 3. 启动模拟环境
  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

        // 4. 输入转密文
        Value x1_sec = test::makeValue(&ctx, x1, VIS_SECRET);
        Value x2_sec = test::makeValue(&ctx, x2, VIS_SECRET);
        Value v1_sec = test::makeValue(&ctx, v1, VIS_SECRET);
        Value v2_sec = test::makeValue(&ctx, v2, VIS_SECRET);

        std::vector<spu::Value> inputs = {x1_sec, x2_sec};
        std::vector<spu::Value> inputs_valid = {v1_sec, v2_sec};

        // ==========================================
        // 5. 记录开始时的统计指标
        // ==========================================
        auto stats = lctx->GetStats();
        size_t start_bytes = stats->sent_bytes;
        size_t start_actions = stats->sent_actions;
        auto start_time = std::chrono::high_resolution_clock::now();

        // 6. 执行 Merge
        std::vector<spu::Value> results =
            merge_with_payloads(&ctx, inputs, inputs_valid,
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

          std::cout << "\n========================================"
                    << std::endl;
          std::cout << "Merge Protocol Execution Stats:" << std::endl;
          std::cout << "  - Input Shape : (2, " << x1.shape(1)
                    << ") x 2 arrays " << std::endl;
          std::cout << "  - Protocol    : " << protocol << std::endl;
          std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
          std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes ("
                    << comm_mb << " MB)" << std::endl;
          std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
                    << std::endl;
          std::cout << "========================================\n"
                    << std::endl;
        }

        // 8. 验证结果并打印对比
        ASSERT_EQ(results.size(), 2);

        auto res_values_pub =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, results[0]));
        auto res_valids_pub =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, results[1]));

        if (lctx->Rank() == 0) {
          std::cout << "Data Verification Details:" << std::endl;
          std::cout << "----------------------------------------" << std::endl;

          std::cout << "[Values Comparison]" << std::endl;
          std::cout << "Expected:\n" << expected_values << std::endl;
          std::cout << "Actual  :\n" << res_values_pub << std::endl;

          std::cout << "\n[Valids Comparison]" << std::endl;
          std::cout << "Expected:\n" << expected_valids << std::endl;
          std::cout << "Actual  :\n" << res_valids_pub << std::endl;
          std::cout << "----------------------------------------" << std::endl;
        }

        EXPECT_TRUE(xt::allclose(res_values_pub, expected_values, 0.001, 0.001))
            << "Values sorting failed!";

        EXPECT_TRUE(xt::allclose(res_valids_pub, expected_valids, 0.001, 0.001))
            << "Valids tracking failed!";
      });
}

// 大规模数据测试，支持多维输入数组
TEST(OddEvenMergeTest, WithValidBitsLargeScale) {
  // 1. 基础配置
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  // 2. 准备大规模测试数据
  const int num_rows = 1;
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> size_dist(1, 524288);
  int input1_size = size_dist(rng);
  int input2_size = size_dist(rng);
  const int total_size = input1_size + input2_size;

  // 使用随机数生成器创建测试数据
  std::uniform_real_distribution<float> value_dist(0.0, 1000.0);
  std::uniform_int_distribution<int> valid_dist(0, 1);

  // 创建数组
  xt::xarray<float> x1 = xt::zeros<float>({num_rows, input1_size});
  xt::xarray<float> x2 = xt::zeros<float>({num_rows, input2_size});
  xt::xarray<float> v1 = xt::zeros<float>({num_rows, input1_size});
  xt::xarray<float> v2 = xt::zeros<float>({num_rows, input2_size});

  // 填充数据并排序
  for (int row = 0; row < num_rows; ++row) {
    // 生成随机数据
    std::vector<float> temp_x1(input1_size);
    std::vector<float> temp_x2(input2_size);

    for (int i = 0; i < input1_size; ++i) {
      temp_x1[i] = value_dist(rng);
      v1(row, i) = static_cast<float>(valid_dist(rng));
    }
    for (int i = 0; i < input2_size; ++i) {
      temp_x2[i] = value_dist(rng);
      v2(row, i) = static_cast<float>(valid_dist(rng));
    }

    // 对每个输入数组进行排序（升序）
    std::sort(temp_x1.begin(), temp_x1.end());
    std::sort(temp_x2.begin(), temp_x2.end());

    // 写回到 xtensor 数组
    for (int i = 0; i < input1_size; ++i) {
      x1(row, i) = temp_x1[i];
    }
    for (int i = 0; i < input2_size; ++i) {
      x2(row, i) = temp_x2[i];
    }
  }

  // 3. 启动MPC模拟环境
  mpc::utils::simulate(
      npc, [&](const std::shared_ptr<yacl::link::Context> &lctx) {
        SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

        // 4. 输入转密文
        Value x1_sec = test::makeValue(&ctx, x1, VIS_SECRET);
        Value x2_sec = test::makeValue(&ctx, x2, VIS_SECRET);
        Value v1_sec = test::makeValue(&ctx, v1, VIS_SECRET);
        Value v2_sec = test::makeValue(&ctx, v2, VIS_SECRET);

        std::vector<spu::Value> inputs = {x1_sec, x2_sec};
        std::vector<spu::Value> inputs_valid = {v1_sec, v2_sec};

        // 5. 记录性能指标
        auto stats = lctx->GetStats();
        size_t start_bytes = stats->sent_bytes;
        size_t start_actions = stats->sent_actions;
        auto start_time = std::chrono::high_resolution_clock::now();

        // 6. 执行merge_with_valids
        std::vector<spu::Value> results =
            merge_with_payloads(&ctx, inputs, inputs_valid,
                                1,      // sort_dim
                                false,  // is_stable
                                [&](absl::Span<const spu::Value> vals) {
                                  return hal::less(&ctx, vals[0], vals[1]);
                                });

        // 7. 记录结束时间并计算开销
        auto end_time = std::chrono::high_resolution_clock::now();
        size_t end_bytes = stats->sent_bytes;
        size_t end_actions = stats->sent_actions;

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time)
                            .count();
        size_t comm_bytes = end_bytes - start_bytes;
        size_t comm_actions = end_actions - start_actions;

        // 8. 验证结果
        ASSERT_EQ(results.size(), 2);

        auto res_values_pub =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, results[0]));
        auto res_valids_pub =
            hal::dump_public_as<float>(&ctx, hal::reveal(&ctx, results[1]));

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
                    << (double)comm_bytes / 1024.0 / 1024.0 << " MB)"
                    << std::endl;
          std::cout << "Actions: " << comm_actions << std::endl;
          std::cout << "Avg bytes/element: "
                    << (double)comm_bytes / (num_rows * total_size)
                    << std::endl;
          std::cout << "========================================\n"
                    << std::endl;
        }

        // 9. 验证归并正确性
        for (int row = 0; row < num_rows; ++row) {
          // 检查输出是否已排序（升序）
          for (int col = 0; col < total_size - 1; ++col) {
            EXPECT_LE(res_values_pub(row, col), res_values_pub(row, col + 1))
                << "Row " << row << " is not sorted at position " << col
                << ", values: " << res_values_pub(row, col) << " > "
                << res_values_pub(row, col + 1);
          }

          // 检查每行的有效位总数是否正确
          float expected_valid_count = 0.0f;
          float actual_valid_count = 0.0f;
          for (int col = 0; col < input1_size; ++col) {
            expected_valid_count += v1(row, col);
          }
          for (int col = 0; col < input2_size; ++col) {
            expected_valid_count += v2(row, col);
          }
          for (int col = 0; col < total_size; ++col) {
            actual_valid_count += res_valids_pub(row, col);
          }
          EXPECT_NEAR(actual_valid_count, expected_valid_count, 0.1f)
              << "Row " << row << " valid bits count mismatch! Expected: "
              << expected_valid_count << ", Got: " << actual_valid_count;

          // 检查值的总和是否保持不变（使用相对误差）
          double expected_sum = 0.0;
          double actual_sum = 0.0;
          for (int col = 0; col < input1_size; ++col) {
            expected_sum += static_cast<double>(x1(row, col));
          }
          for (int col = 0; col < input2_size; ++col) {
            expected_sum += static_cast<double>(x2(row, col));
          }
          for (int col = 0; col < total_size; ++col) {
            actual_sum += static_cast<double>(res_values_pub(row, col));
          }
          double relative_error =
              std::abs(actual_sum - expected_sum) / expected_sum;
          EXPECT_LT(relative_error, 0.001)  // 允许 0.1% 的相对误差
              << "Row " << row
              << " values sum mismatch! Expected: " << expected_sum
              << ", Got: " << actual_sum
              << ", Relative error: " << relative_error;
        }
      });
}

}  // namespace spu::kernel::hal
