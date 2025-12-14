#include "libspu/kernel/hlo/Logstar.h"

#include "gtest/gtest.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hlo {

class BrentKungTest : public ::testing::Test {};
class ExtractOrderedTest : public ::testing::Test {};

//-----------------------------------------------------------------------------------
//                            AggregateBrentKung Tests
//-----------------------------------------------------------------------------------
// // TEST the AggregateBrentKung with valid bits for BasicCorrectness
// TEST_F(BrentKungTest, BasicCorrectness) {
//   const size_t npc = 2;
//   const auto protocol = ProtocolKind::SEMI2K;
//   const auto field = FieldType::FM64;

//   mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
//     SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

//     // 准备输入数据
//     int64_t n = 8;
//     int64_t block_size = 2;

//     std::vector<int64_t> x_data = {5, 6, 2, 2, 10, 10, 4, 3,
//                                    1, 1, 2, 2, 5,  5,  3, 2};
    
//     std::vector<int64_t> valid_data = {1, 1, 0, 0, 1, 1, 0, 1,
//                                        0, 0, 1, 1, 0, 0, 1, 1};

//     std::vector<int64_t> g_data = {0, 1, 1, 0, 1, 1, 1, 0};

//     xt::xarray<int64_t> y_expected = {{5, 6}, {5, 6}, {5, 6}, {4, 3},
//                                       {4, 3}, {4, 3}, {4, 3}, {3, 2}};
    
//     xt::xarray<int64_t> valid_expected = {{1, 1}, {1, 1}, {1, 1}, {0, 1},
//                                           {0, 1}, {0, 1}, {0, 1}, {1, 1}};

//     // 构造 SPU 输入
//     xt::xarray<int64_t> x_arr = xt::adapt(x_data);
//     x_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
//     auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);

//     xt::xarray<int64_t> v_arr = xt::adapt(valid_data);
//     v_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
//     auto v_in = test::makeValue(&ctx, v_arr, VIS_SECRET);

//     xt::xarray<int64_t> g_arr = xt::adapt(g_data);
//     g_arr.reshape({static_cast<size_t>(n), 1});
//     auto g_in = test::makeValue(&ctx, g_arr, VIS_SECRET);

//     // 记录开始时的统计指标
//     auto stats = lctx->GetStats();
//     size_t start_bytes = stats->sent_bytes;
//     size_t start_actions = stats->sent_actions;
//     auto start_time = std::chrono::high_resolution_clock::now();

//     // 执行AggregateBrentKung
//     auto [y_out, valid_out] = AggregateBrentKung(&ctx, x_in, v_in, g_in);

//     // 记录结束时间
//     auto end_time = std::chrono::high_resolution_clock::now();

//     // 计算并打印增量指标
//     stats = lctx->GetStats();  // 刷新统计
//     size_t end_bytes = stats->sent_bytes;
//     size_t end_actions = stats->sent_actions;
//     if (lctx->Rank() == 0) {
//       auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
//                           end_time - start_time)
//                           .count();
//       auto comm_bytes = end_bytes - start_bytes;
//       auto comm_rounds = end_actions - start_actions;
//       double comm_mb = static_cast<double>(comm_bytes) / 1024.0 / 1024.0;

//       std::cout << "\n========================================"
//                 << std::endl;
//       std::cout << "AggregateBrentKung Protocol Execution Stats:" << std::endl;
//       std::cout << "  - Input Shape : " << n << " blocks of size " << block_size
//                 << std::endl;
//       std::cout << "  - Protocol    : " << protocol << std::endl;
//       std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
//       std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes ("
//                 << comm_mb << " MB)" << std::endl;
//       std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
//                 << std::endl;
//       std::cout << "========================================\n"
//                 << std::endl;
//     }

//     // 验证 Y
//     auto y_revealed = hal::reveal(&ctx, y_out);
//     auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

//     // 验证 Valid
//     auto v_revealed = hal::reveal(&ctx, valid_out);
//     auto v_vec = hal::dump_public_as<int64_t>(&ctx, v_revealed);
    
//     // =========== [新增/修改部分开始] ===========
//     // 只在 Rank 0 打印，避免多线程输出混乱
//     if (lctx->Rank() == 0) {
//         std::cout << "y_expected:\n" << y_expected << std::endl;
//         std::cout << "y_vec (Actual):\n" << y_vec << std::endl;
        
//         std::cout << "valid_expected:\n" << valid_expected << std::endl;
//         std::cout << "v_vec (Actual):\n" << v_vec << std::endl;

//         // 获取通信统计
//         auto stats = lctx->GetStats();
//         std::cout << "Communication Traffic (Sent Bytes): " << stats->sent_bytes << std::endl;
//         std::cout << "Communication Rounds (Sent Actions): " << stats->sent_actions << std::endl;
//     }
//     // =========== [新增/修改部分结束] ===========

//     EXPECT_TRUE(xt::allclose(y_vec, y_expected));
//     EXPECT_EQ(v_vec.shape()[0], n);
//     EXPECT_EQ(v_vec.shape()[1], block_size);
//     EXPECT_TRUE(xt::allclose(v_vec, valid_expected));
//   });
// }

// TEST the AggregateBrentKung with valid bits for LargeScaleData
TEST_F(BrentKungTest, LargeScaleData) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

    // 配置大;
    const int64_t n = 1000000;  // 可调整为 256, 512, 1024, 4096 等
    const int64_t block_size = 3;

    // 准备随机数生成器
    std::mt19937 rng(42);  // 固定种子
    std::uniform_int_distribution<int64_t> dist_x(0, 100);
    std::uniform_int_distribution<int64_t> dist_g(0, 1);

    // 生成随机输入数据
    std::vector<int64_t> x_data(n * block_size);
    std::vector<int64_t> valid_data(n * block_size);
    std::vector<int64_t> g_data(n);

    for (auto& v : x_data) v = dist_x(rng);
    for (auto& v : valid_data) v = dist_g(rng);
    for (int64_t i = 0; i < n; ++i) {
      g_data[i] = dist_g(rng);
      // 强制第 g 为 0，确保初始状态明确
      if (i == 0) g_data[i] = 0;
    }

    // 构造 SPU 输入
    xt::xarray<int64_t> x_arr = xt::adapt(x_data);
    x_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);

    xt::xarray<int64_t> v_arr = xt::adapt(valid_data);
    v_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto v_in = test::makeValue(&ctx, v_arr, VIS_SECRET);

    xt::xarray<int64_t> g_arr = xt::adapt(g_data);
    g_arr.reshape({static_cast<size_t>(n), 1});
    auto g_in = test::makeValue(&ctx, g_arr, VIS_SECRET);

    // 记录开始时的统计指标
    auto stats = lctx->GetStats();
    size_t start_bytes = stats->sent_bytes;
    size_t start_actions = stats->sent_actions;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行 AggregateBrentKung
    auto [y_out, valid_out] = AggregateBrentKung(&ctx, x_in, v_in, g_in);

    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算并打印增量指标
    stats = lctx->GetStats();
    size_t end_bytes = stats->sent_bytes;
    size_t end_actions = stats->sent_actions;
    
    if (lctx->Rank() == 0) {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_time - start_time).count();
      auto comm_bytes = end_bytes - start_bytes;
      auto comm_rounds = end_actions - start_actions;
      double comm_mb = static_cast<double>(comm_bytes) / 1024.0 / 1024.0;

      std::cout << "\n========================================"
                << std::endl;
      std::cout << "AggregateBrentKung Large Scale Test:" << std::endl;
      std::cout << "  - Input Shape : " << n << " blocks of size " << block_size
                << std::endl;
      std::cout << "  - Protocol    : " << protocol << std::endl;
      std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
      std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes ("
                << comm_mb << " MB)" << std::endl;
      std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
                << std::endl;
      std::cout << "========================================\n"
                << std::endl;
    }

    // 验证结果
    auto y_revealed = hal::reveal(&ctx, y_out);
    auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

    auto v_revealed = hal::reveal(&ctx, valid_out);
    auto v_vec = hal::dump_public_as<int64_t>(&ctx, v_revealed);

    // 验证形状
    EXPECT_EQ(y_vec.shape()[0], n);
    EXPECT_EQ(y_vec.shape()[1], block_size);
    EXPECT_EQ(v_vec.shape()[0], n);
    EXPECT_EQ(v_vec.shape()[1], block_size);

    // 验证基本属性: 第一行应该等于输入（因为 g[0] = 0）
    for (int64_t b = 0; b < block_size; ++b) {
      EXPECT_EQ(y_vec(0, b), x_arr(0, b));
      EXPECT_EQ(v_vec(0, b), v_arr(0, b));
    }

    // g=1，值 验证传
    for (int64_t i = 1; i < n; ++i) {
      if (g_data[i] == 1 && g_data[i-1] == 1) {
        // 如果连续两个都是 1，那么 y[i] 应该等于 y[i-1]
        for (int64_t b = 0; b < block_size; ++b) {
          EXPECT_EQ(y_vec(i, b), y_vec(i-1, b)) 
            << "Propagation failed at i=" << i << ", b=" << b;
          EXPECT_EQ(v_vec(i, b), v_vec(i-1, b))
            << "Valid propagation failed at i=" << i << ", b=" << b;
        }
      }
    }

    if (lctx->Rank() == 0) {
      std::cout << "Large scale test passed successfully!" << std::endl;
    }
  });
}


// // // TEST the AggregateBrentKung without valid bits for BasicCorrectness
// TEST_F(BrentKungTest, Debug1024) {
//   const size_t npc = 2;
//   const auto protocol = ProtocolKind::SEMI2K;
//   const auto field = FieldType::FM64;

//   mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
//     SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

//     const int64_t n = 262144;
//     const int64_t block_size = 4;

//     std::mt19937 rng(42);
//     std::uniform_int_distribution<int64_t> dist_x(0, 100);
//     std::uniform_int_distribution<int64_t> dist_g(0, 1);

//     std::vector<int64_t> x_data(n * block_size);
//     std::vector<int64_t> valid_data(n * block_size);
//     std::vector<int64_t> g_data(n);

//     for (auto& v : x_data) v = dist_x(rng);
//     for (auto& v : valid_data) v = dist_g(rng);
//     for (int64_t i = 0; i < n; ++i) {
//       g_data[i] = dist_g(rng);
//       if (i == 0) g_data[i] = 0;
//     }

//     // 计算 ground truth
//     xt::xarray<int64_t> y_expected = xt::zeros<int64_t>({(size_t)n, (size_t)block_size});
//     xt::xarray<int64_t> valid_expected = xt::zeros<int64_t>({(size_t)n, (size_t)block_size});

//     const int64_t* x_ptr = x_data.data();
//     const int64_t* v_ptr = valid_data.data();

//     for (int64_t b = 0; b < block_size; ++b) {
//       y_expected(0, b) = x_ptr[b];
//       valid_expected(0, b) = v_ptr[b];
//     }

//     for (int64_t i = 1; i < n; ++i) {
//       if (g_data[i] == 0) {
//         for (int64_t b = 0; b < block_size; ++b) {
//           y_expected(i, b) = x_ptr[i * block_size + b];
//           valid_expected(i, b) = v_ptr[i * block_size + b];
//         }
//       } else {
//         for (int64_t b = 0; b < block_size; ++b) {
//           y_expected(i, b) = y_expected(i - 1, b);
//           valid_expected(i, b) = valid_expected(i - 1, b);
//         }
//       }
//     }

//     xt::xarray<int64_t> x_arr = xt::adapt(x_data);
//     x_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
//     auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);

//     xt::xarray<int64_t> v_arr = xt::adapt(valid_data);
//     v_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
//     auto v_in = test::makeValue(&ctx, v_arr, VIS_SECRET);

//     xt::xarray<int64_t> g_arr = xt::adapt(g_data);
//     g_arr.reshape({static_cast<size_t>(n), 1});
//     auto g_in = test::makeValue(&ctx, g_arr, VIS_SECRET);

//     auto [y_out, valid_out] = AggregateBrentKung(&ctx, x_in, v_in, g_in);

//     auto y_revealed = hal::reveal(&ctx, y_out);
//     auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

//     auto v_revealed = hal::reveal(&ctx, valid_out);
//     auto v_vec = hal::dump_public_as<int64_t>(&ctx, v_revealed);

//     if (lctx->Rank() == 0) {
//       bool y_match = xt::allclose(y_vec, y_expected);
//       bool v_match = xt::allclose(v_vec, valid_expected);
      
//       std::cout << "\nn=1024, block_size=4 Test:" << std::endl;
//       std::cout << "Y matches ground truth: " << y_match << std::endl;
//       std::cout << "V matches ground truth: " << v_match << std::endl;
      
//       if (!y_match || !v_match) {
//         int mismatch_count_y = 0, mismatch_count_v = 0;
//         for (int64_t i = 0; i < n; ++i) {
//           for (int64_t b = 0; b < block_size; ++b) {
//             if (y_vec(i, b) != y_expected(i, b)) {
//               if (mismatch_count_y < 5) {  // 只打印前5个不匹配
//                 std::cout << "Y mismatch at [" << i << "," << b << "]: expected " 
//                           << y_expected(i, b) << ", got " << y_vec(i, b) << std::endl;
//               }
//               mismatch_count_y++;
//             }
//             if (v_vec(i, b) != valid_expected(i, b)) {
//               if (mismatch_count_v < 5) {
//                 std::cout << "V mismatch at [" << i << "," << b << "]: expected " 
//                           << valid_expected(i, b) << ", got " << v_vec(i, b) << std::endl;
//               }
//               mismatch_count_v++;
//             }
//           }
//         }
//         std::cout << "Total Y mismatches: " << mismatch_count_y << " / " << (n * block_size) << std::endl;
//         std::cout << "Total V mismatches: " << mismatch_count_v << " / " << (n * block_size) << std::endl;
//       }
//     }

//     EXPECT_TRUE(xt::allclose(y_vec, y_expected)) << "Y mismatch";
//     EXPECT_TRUE(xt::allclose(v_vec, valid_expected)) << "V mismatch";
//   });
// }

// // TEST the AggregateBrentKung without valid bits for LargeScaleBenchmark
// TEST_F(BrentKungTest, BasicCorrectnessWithoutValids) {
//   const size_t npc = 2;
//   const auto protocol = ProtocolKind::SEMI2K;
//   const auto field = FieldType::FM64;

//   mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
//                                     lctx) {
//     SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

//     // 准备输入数据
//     int64_t n = 8;
//     int64_t block_size = 2;

//     std::vector<int64_t> x_data = {5, 6, 2, 2, 10, 10, 4, 3,
//                                    1, 1, 2, 2, 5,  5,  3, 2};
//     std::vector<int64_t> g_data = {0, 1, 1, 0, 1, 1, 1, 0};
//     xt::xarray<int64_t> y_expected = {{5, 6}, {5, 6}, {5, 6}, {4, 3},
//                                       {4, 3}, {4, 3}, {4, 3}, {3, 2}};

//     xt::xarray<int64_t> x_arr = xt::adapt(x_data);
//     x_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
//     auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);

//     xt::xarray<int64_t> g_arr = xt::adapt(g_data);
//     g_arr.reshape({static_cast<size_t>(n), 1});
//     auto g_in = test::makeValue(&ctx, g_arr, VIS_SECRET);

//     auto stats = lctx->GetStats();
//     size_t start_bytes = stats->sent_bytes;
//     size_t start_actions = stats->sent_actions;

//     auto y_out = AggregateBrentKung_without_valids(&ctx, x_in, g_in);

//     size_t end_bytes = stats->sent_bytes;
//     size_t end_actions = stats->sent_actions;

//     if (lctx->Rank() == 0) {
//       auto comm_bytes = end_bytes - start_bytes;
//       auto comm_rounds = end_actions - start_actions;
//       std::cout << "\n========================================" << std::endl;
//       std::cout << "Benchmark Stats (N=" << n << "):" << std::endl;
//       std::cout << "  - Comm Volume: " << comm_bytes << " bytes" << std::endl;
//       std::cout << "  - Comm Rounds: " << comm_rounds << " actions"
//                 << std::endl;
//       std::cout << "========================================\n" << std::endl;
//     }
//     // ==============================================================

//     auto y_revealed = hal::reveal(&ctx, y_out);
//     auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

//     if (lctx->Rank() == 0) {
//       std::cout << "Result Y (Rank 0): ";
//       if (y_vec.size() > 20) {
//         std::cout << y_vec[0] << " ... " << y_vec[y_vec.size() - 1];
//       } else {
//         for (auto v : y_vec) std::cout << v << " ";
//       }
//       std::cout << std::endl;
//     }

//     // 检查正确性
//     EXPECT_EQ(y_vec.size(), n * block_size);
//     EXPECT_TRUE(xt::allclose(y_vec, y_expected));
//   });
// }

// // TEST the AggregateBrentKung without valid bits for LargeScaleBenchmark
// TEST_F(BrentKungTest, LargeScaleWithoutValids) {
//   const size_t npc = 2;
//   const auto protocol = ProtocolKind::SEMI2K;
//   const auto field = FieldType::FM64;

//   // ==========================================
//   // 1. 配置数据规模
//   // ==========================================
//   // 你可以修改这里，比如 1024, 4096, 65536 等
//   const int64_t n = 256;
//   const int64_t block_size = 2;  // 每个元素的大小

//   // 准备随机数生成器
//   std::mt19937 rng(std::random_device{}());
//   std::uniform_int_distribution<int64_t> dist_x(0, 100);  // x 的值域
//   std::uniform_int_distribution<int64_t> dist_g(0, 1);    // g 只能是 0 或 1

//   // ==========================================
//   // 2. 生成随机输入数据
//   // ==========================================
//   std::vector<int64_t> x_data(n * block_size);
//   std::vector<int64_t> g_data(n);

//   for (auto& v : x_data) v = dist_x(rng);
//   for (int64_t i = 0; i < n; ++i) {
//     g_data[i] = dist_g(rng);
//     // 强制第一个元素的 g 为 0，确保初始状态明确（可选，视具体算法要求）
//     if (i == 0) g_data[i] = 0;
//   }

//   // ==========================================
//   // 3. 计算 Ground Truth (CPU 本地逻辑)
//   // ==========================================
//   // 根据之前的观测，逻辑似乎是：
//   // result[i] = (g[i] == 0) ? x[i] : result[i-1]
//   xt::xarray<int64_t> y_expected =
//       xt::zeros<int64_t>({(size_t)n, (size_t)block_size});

//   // 将 x_data 映射为方便访问的视图
//   const int64_t* x_ptr = x_data.data();

//   for (int64_t i = 0; i < n; ++i) {
//     if (g_data[i] == 0) {
//       // Reset: 取当前的 x
//       for (int64_t b = 0; b < block_size; ++b) {
//         y_expected(i, b) = x_ptr[i * block_size + b];
//       }
//     } else {
//       // Propagate: 复制前一个 y (注意 i=0 时 g
//       // 必须为0，否则越界，前面已强制处理)
//       if (i > 0) {
//         for (int64_t b = 0; b < block_size; ++b) {
//           y_expected(i, b) = y_expected(i - 1, b);
//         }
//       } else {
//         // Fallback for i=0 if g=1 (though we forced g=0)
//         for (int64_t b = 0; b < block_size; ++b) {
//           y_expected(i, b) = x_ptr[i * block_size + b];
//         }
//       }
//     }
//   }

//   // ==========================================
//   // 4. 开始 SPU 模拟
//   // ==========================================
//   mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
//                                     lctx) {
//     SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

//     // 转换输入为 xtensor 并 reshape
//     xt::xarray<int64_t> x_arr = xt::adapt(x_data);
//     x_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});

//     xt::xarray<int64_t> g_arr = xt::adapt(g_data);
//     g_arr.reshape({static_cast<size_t>(n), 1});

//     // 制作密文 (VIS_SECRET)
//     auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);
//     auto g_in = test::makeValue(&ctx, g_arr, VIS_SECRET);

//     // 记录开始状态
//     auto stats = lctx->GetStats();
//     size_t start_bytes = stats->sent_bytes;
//     size_t start_actions = stats->sent_actions;

//     // ------------------------------------
//     // 执行核心算法
//     // ------------------------------------
//     auto y_out = AggregateBrentKung_without_valids(&ctx, x_in, g_in);
//     // ------------------------------------

//     // 记录结束状态
//     size_t end_bytes = stats->sent_bytes;
//     size_t end_actions = stats->sent_actions;

//     // 打印统计信息 (Rank 0)
//     if (lctx->Rank() == 0) {
//       auto comm_bytes = end_bytes - start_bytes;
//       auto comm_rounds = end_actions - start_actions;

//       double comm_mb = comm_bytes / 1024.0 / 1024.0;

//       std::cout << "\n========================================" << std::endl;
//       std::cout << "Benchmark Stats (N=" << n << "):" << std::endl;
//       std::cout << "  - Protocol   : " << protocol << std::endl;
//       std::cout << "  - Comm Volume: " << comm_bytes << " bytes (" << comm_mb
//                 << " MB)" << std::endl;
//       std::cout << "  - Comm Rounds: " << comm_rounds << " actions"
//                 << std::endl;
//       std::cout << "========================================\n" << std::endl;
//     }

//     // 验证结果
//     auto y_revealed = hal::reveal(&ctx, y_out);
//     auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

//     // 校验 (allclose 对于整数就是完全相等)
//     EXPECT_EQ(y_vec.shape(), y_expected.shape());
//     EXPECT_TRUE(xt::allclose(y_vec, y_expected))
//         << "SPU result mismatch with CPU ground truth!";
//   });
// }

//-----------------------------------------------------------------------------------
//                            ExtractOrder Tests
//-----------------------------------------------------------------------------------
// TEST_F(ExtractOrderedTest, BasicCorrectness) {
//   // 设置 MPC 环境参数：2方计算，SEMI2K 协议，64位环
//   const size_t npc = 2;
//   const auto protocol = ProtocolKind::SEMI2K;
//   const auto field = FieldType::FM64;

//   mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
//     SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

//     // -------------------------------------------------------
//     // 1. 准备输入数据 - 支持多个数组
//     // -------------------------------------------------------
//     int64_t num_arrays = 2;
//     int64_t n = 6;
//     xt::xarray<int64_t> x = {{0, 1, 2, 3, 4, 5}, {10, 11, 12, 13, 14, 15}};    // valid bits
//     xt::xarray<int64_t> valids = {0, 1, 0, 1, 1, 0};
    
//     // 预期输出（提取有效元素）
//     xt::xarray<int64_t> y_expected = {{1, 3, 4},
//                                       {11, 13, 14}};

//     auto x_in = test::makeValue(&ctx, x, VIS_SECRET);
//     valids.reshape({1, static_cast<size_t>(n)}); 
//     auto valids_in = test::makeValue(&ctx, valids, VIS_SECRET);

//     // -------------------------------------------------------
//     // 2. 执行协议并记录性能指标
//     // -------------------------------------------------------
    
//     // 记录开始状态
//     auto stats = lctx->GetStats();
//     size_t start_bytes = stats->sent_bytes;
//     size_t start_actions = stats->sent_actions;
//     auto start_time = std::chrono::high_resolution_clock::now();

//     // === 调用核心函数 ===
//     auto res = extract_ordered(&ctx, x_in, valids_in);
//     auto &y = res.first;
//     auto valid_count = res.second;
//     // ===================

//     // 记录结束时间
//     auto end_time = std::chrono::high_resolution_clock::now();

//     // -------------------------------------------------------
//     // 3. 打印统计信息
//     // -------------------------------------------------------
//     stats = lctx->GetStats();  // 刷新统计
//     size_t end_bytes = stats->sent_bytes;
//     size_t end_actions = stats->sent_actions;

//     if (lctx->Rank() == 0) {
//       auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
//                           end_time - start_time)
//                           .count();
//       auto comm_bytes = end_bytes - start_bytes;
//       auto comm_rounds = end_actions - start_actions;
//       double comm_mb = static_cast<double>(comm_bytes) / 1024.0 / 1024.0;

//       std::cout << "\n========================================"
//                 << std::endl;
//       std::cout << "ExtractOrdered Protocol Execution Stats:" << std::endl;
//       std::cout << "  - Input Shape : " << " num_arrays = " << num_arrays << ", n = " << n << std::endl;
//       std::cout << "  - Protocol    : " << protocol << std::endl;
//       std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
//       std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes ("
//                 << comm_mb << " MB)" << std::endl;
//       std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
//                 << std::endl;
//       std::cout << "========================================\n"
//                 << std::endl;
//     }

//     // -------------------------------------------------------
//     // 4. 验证结果
//     // -------------------------------------------------------
//     EXPECT_EQ(valid_count, y_expected.shape()[1]);

//     for (int64_t i = 0; i < num_arrays; ++i) {
//       auto y_revealed = hal::reveal(&ctx, y[i]);
//       auto y_row = hal::dump_public_as<int64_t>(&ctx, y_revealed);

//       // 截取有效部分 [0, valid_count)
//       // 注意：y_row 可能是 (1, n) 或 (n)，视具体实现而定，这里统一处理
//       xt::xarray<int64_t> y_valid_part;
//       if (y_row.dimension() == 2) {
//         y_valid_part = xt::view(y_row, 0, xt::range(0, valid_count));
//       } else {
//         y_valid_part = xt::view(y_row, xt::range(0, valid_count));
//       }
//       auto y_expected_row = xt::row(y_expected, i);
//       EXPECT_TRUE(xt::allclose(y_valid_part, y_expected_row));

//       if (lctx->Rank() == 0) {
//         std::cout << "Array " << i << " Result (Valid): " << y_valid_part
//                   << std::endl;
//         std::cout << "Array " << i << " Expected      : " << y_expected_row
//                   << std::endl << std::endl;
//       }
//     }

//   });
// }

// // 大规模输入测试：验证在 n=1024 情况下的正确性和稳定性
// TEST_F(ExtractOrderedTest, LargeScale) {
//   const size_t npc = 2;
//   const auto protocol = ProtocolKind::SEMI2K;
//   const auto field = FieldType::FM64;

//   mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
//     SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

//     // -------------------------------------------------------
//     // 1. 准备输入数据 - 支持多个数组
//     // -------------------------------------------------------
//     const int64_t num_arrays = 1;
//     const int64_t n = 1000000;
//     std::vector<std::vector<int64_t>> xs(num_arrays, std::vector<int64_t>(n));
//     for (int64_t r = 0; r < num_arrays; ++r) {
//       int64_t base = r * 100000;  // 不同行不同基数
//       for (int64_t i = 0; i < n; ++i) {
//         xs[r][i] = base + i;
//       }
//     }

//     // valid bits
//     std::vector<int64_t> valids(n);
//     for (int64_t i = 0; i < n; ++i) {
//       valids[i] = (i % 5 == 0) ? 1 : 0;
//     }

//     // 扁平化 xs 为 [num_arrays * n]，然后 reshape 为 [num_arrays, n]
//     std::vector<int64_t> x_combined;
//     x_combined.reserve(num_arrays * n);
//     for (int64_t r = 0; r < num_arrays; ++r) {
//       x_combined.insert(x_combined.end(), xs[r].begin(), xs[r].end());
//     }
//     xt::xarray<int64_t> x_arr = xt::adapt(x_combined);
//     x_arr.reshape({static_cast<size_t>(num_arrays), static_cast<size_t>(n)});
//     auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);

//     // valids 为 [1, n]
//     xt::xarray<int64_t> f_arr = xt::adapt(valids);
//     f_arr.reshape({1, static_cast<size_t>(n)});
//     auto f_in = test::makeValue(&ctx, f_arr, VIS_SECRET);

//     // 准备 Ground Truth
//     std::vector<std::vector<int64_t>> exps(num_arrays);
//     for (int64_t i = 0; i < n; ++i) {
//       if (valids[i]) {
//         for (int64_t r = 0; r < num_arrays; ++r) {
//           exps[r].push_back(xs[r][i]);
//         }
//       }
//     }
//     size_t valid_count_expected = exps.empty() ? 0 : exps[0].size();

//     // 扁平化预期并 reshape 为 [num_arrays, valid_count_expected]
//     std::vector<int64_t> y_combined;
//     y_combined.reserve(num_arrays * valid_count_expected);
//     for (int64_t r = 0; r < num_arrays; ++r) {
//       y_combined.insert(y_combined.end(), exps[r].begin(), exps[r].end());
//     }
//     xt::xarray<int64_t> y_expected = xt::adapt(y_combined);
//     y_expected.reshape({static_cast<size_t>(num_arrays),
//                         static_cast<size_t>(valid_count_expected)});

//     // -------------------------------------------------------
//     // 2. 执行协议并记录性能指标
//     // -------------------------------------------------------
//     auto stats = lctx->GetStats();
//     size_t start_bytes = stats->sent_bytes;
//     size_t start_actions = stats->sent_actions;
//     auto start_time = std::chrono::high_resolution_clock::now();

//     auto res = extract_ordered(&ctx, x_in, f_in);
//     auto &y = res.first;
//     auto valid_count = res.second;

//     // -------------------------------------------------------
//     // 3. 打印统计信息
//     // -------------------------------------------------------
//     auto end_time = std::chrono::high_resolution_clock::now();
//     stats = lctx->GetStats();  // 刷新统计
//     size_t end_bytes = stats->sent_bytes;
//     size_t end_actions = stats->sent_actions;
//     if (lctx->Rank() == 0) {
//       auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
//                           end_time - start_time)
//                           .count();
//       auto comm_bytes = end_bytes - start_bytes;
//       auto comm_rounds = end_actions - start_actions;
//       double comm_mb = static_cast<double>(comm_bytes) / 1024.0 / 1024.0;

//       std::cout << "\n========================================" << std::endl;
//       std::cout << "ExtractOrdered Protocol Execution Stats:" << std::endl;
//       std::cout << "  - Input Shape : " << " num_arrays = " << num_arrays << ", n = " << n << std::endl;
//       std::cout << "  - Protocol    : " << protocol << std::endl;
//       std::cout << "  - Time Cost   : " << duration << " ms" << std::endl;
//       std::cout << "  - Comm Bytes  : " << comm_bytes << " bytes ("
//                 << comm_mb << " MB)" << std::endl;
//       std::cout << "  - Comm Rounds : " << comm_rounds << " actions"
//                 << std::endl;
//       std::cout << "========================================\n" << std::endl;
//     }

//     EXPECT_EQ(valid_count, static_cast<int64_t>(valid_count_expected));

//     for (int64_t i = 0; i < num_arrays; ++i) {
//       auto y_revealed = hal::reveal(&ctx, y[i]);
//       auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

//       xt::xarray<int64_t> y_valid_part;
//       if (y_vec.dimension() == 2) {
//         y_valid_part = xt::view(y_vec, 0, xt::range(0, valid_count));
//       } else {
//         y_valid_part = xt::view(y_vec, xt::range(0, valid_count));
//       }

//       auto y_expected_row = xt::row(y_expected, i);

//       if (lctx->Rank() == 0) {
//         std::cout << "LargeScale Array " << i << " valid_count=" << valid_count << std::endl;
//       }

//       EXPECT_TRUE(xt::allclose(y_valid_part, y_expected_row));
//     }
//   });
// }

}  // namespace spu::kernel::hlo