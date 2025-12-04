#include "libspu/kernel/hal/brent_kung_network.h"

#include "gtest/gtest.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {

class BrentKungTest : public ::testing::Test {};

TEST_F(BrentKungTest, BasicCorrectness) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

    // 准备输入数据
    int64_t n = 8;
    int64_t block_size = 2;

    std::vector<int64_t> x_data = {5, 6, 2, 2, 10, 10, 4, 3,
                                   1, 1, 2, 2, 5,  5,  3, 2};
    std::vector<int64_t> g_data = {0, 1, 1, 0, 1, 1, 1, 0};
    xt::xarray<int64_t> y_expected = {{5, 6}, {5, 6}, {5, 6}, {4, 3},
                                      {4, 3}, {4, 3}, {4, 3}, {3, 2}};

    // ==============================================================
    // 修改点 1：将输入转为 Secret (密文)，否则通信量为 0
    // ==============================================================
    // 原代码: auto x_in = hal::constant(&ctx, x_view, spu::DT_I64, {n,
    // block_size}); 修改为: 使用 test::makeValue 创建 VIS_SECRET
    xt::xarray<int64_t> x_arr = xt::adapt(x_data);
    x_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});
    auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);

    xt::xarray<int64_t> g_arr = xt::adapt(g_data);
    g_arr.reshape({static_cast<size_t>(n), 1});
    auto g_in = test::makeValue(&ctx, g_arr, VIS_SECRET);
    // ==============================================================

    // ==============================================================
    // 修改点 2：记录开始时的统计值
    // ==============================================================
    auto stats = lctx->GetStats();
    size_t start_bytes = stats->sent_bytes;
    size_t start_actions = stats->sent_actions;

    // 执行算法
    auto y_out = AggregateBrentKung(&ctx, x_in, g_in);

    // ==============================================================
    // 修改点 3：计算并打印差值 (仅 Rank 0 打印)
    // ==============================================================
    size_t end_bytes = stats->sent_bytes;
    size_t end_actions = stats->sent_actions;

    if (lctx->Rank() == 0) {
      auto comm_bytes = end_bytes - start_bytes;
      auto comm_rounds = end_actions - start_actions;
      std::cout << "\n========================================" << std::endl;
      std::cout << "Benchmark Stats (N=" << n << "):" << std::endl;
      std::cout << "  - Comm Volume: " << comm_bytes << " bytes" << std::endl;
      std::cout << "  - Comm Rounds: " << comm_rounds << " actions"
                << std::endl;
      std::cout << "========================================\n" << std::endl;
    }
    // ==============================================================

    // 输出数据转回C++类型
    auto y_revealed = hal::reveal(&ctx, y_out);
    auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

    if (lctx->Rank() == 0) {
      std::cout << "Result Y (Rank 0): ";
      // 简单的打印逻辑，避免刷屏
      if (y_vec.size() > 20) {
        std::cout << y_vec[0] << " ... " << y_vec[y_vec.size() - 1];
      } else {
        for (auto v : y_vec) std::cout << v << " ";
      }
      std::cout << std::endl;
    }

    // 检查正确性
    EXPECT_EQ(y_vec.size(), n * block_size);
    EXPECT_TRUE(xt::allclose(y_vec, y_expected));
  });
}

TEST_F(BrentKungTest, LargeScaleBenchmark) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  // ==========================================
  // 1. 配置数据规模
  // ==========================================
  // 你可以修改这里，比如 1024, 4096, 65536 等
  const int64_t n = 256;
  const int64_t block_size = 2;  // 每个元素的大小

  // 准备随机数生成器
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int64_t> dist_x(0, 100);  // x 的值域
  std::uniform_int_distribution<int64_t> dist_g(0, 1);    // g 只能是 0 或 1

  // ==========================================
  // 2. 生成随机输入数据
  // ==========================================
  std::vector<int64_t> x_data(n * block_size);
  std::vector<int64_t> g_data(n);

  for (auto& v : x_data) v = dist_x(rng);
  for (int64_t i = 0; i < n; ++i) {
    g_data[i] = dist_g(rng);
    // 强制第一个元素的 g 为 0，确保初始状态明确（可选，视具体算法要求）
    if (i == 0) g_data[i] = 0;
  }

  // ==========================================
  // 3. 计算 Ground Truth (CPU 本地逻辑)
  // ==========================================
  // 根据之前的观测，逻辑似乎是：
  // result[i] = (g[i] == 0) ? x[i] : result[i-1]
  xt::xarray<int64_t> y_expected =
      xt::zeros<int64_t>({(size_t)n, (size_t)block_size});

  // 将 x_data 映射为方便访问的视图
  const int64_t* x_ptr = x_data.data();

  for (int64_t i = 0; i < n; ++i) {
    if (g_data[i] == 0) {
      // Reset: 取当前的 x
      for (int64_t b = 0; b < block_size; ++b) {
        y_expected(i, b) = x_ptr[i * block_size + b];
      }
    } else {
      // Propagate: 复制前一个 y (注意 i=0 时 g
      // 必须为0，否则越界，前面已强制处理)
      if (i > 0) {
        for (int64_t b = 0; b < block_size; ++b) {
          y_expected(i, b) = y_expected(i - 1, b);
        }
      } else {
        // Fallback for i=0 if g=1 (though we forced g=0)
        for (int64_t b = 0; b < block_size; ++b) {
          y_expected(i, b) = x_ptr[i * block_size + b];
        }
      }
    }
  }

  // ==========================================
  // 4. 开始 SPU 模拟
  // ==========================================
  mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>&
                                    lctx) {
    SPUContext ctx = test::makeSPUContext(protocol, field, lctx);

    // 转换输入为 xtensor 并 reshape
    xt::xarray<int64_t> x_arr = xt::adapt(x_data);
    x_arr.reshape({static_cast<size_t>(n), static_cast<size_t>(block_size)});

    xt::xarray<int64_t> g_arr = xt::adapt(g_data);
    g_arr.reshape({static_cast<size_t>(n), 1});

    // 制作密文 (VIS_SECRET)
    auto x_in = test::makeValue(&ctx, x_arr, VIS_SECRET);
    auto g_in = test::makeValue(&ctx, g_arr, VIS_SECRET);

    // 记录开始状态
    auto stats = lctx->GetStats();
    size_t start_bytes = stats->sent_bytes;
    size_t start_actions = stats->sent_actions;

    // ------------------------------------
    // 执行核心算法
    // ------------------------------------
    auto y_out = AggregateBrentKung(&ctx, x_in, g_in);
    // ------------------------------------

    // 记录结束状态
    size_t end_bytes = stats->sent_bytes;
    size_t end_actions = stats->sent_actions;

    // 打印统计信息 (Rank 0)
    if (lctx->Rank() == 0) {
      auto comm_bytes = end_bytes - start_bytes;
      auto comm_rounds = end_actions - start_actions;

      double comm_mb = comm_bytes / 1024.0 / 1024.0;

      std::cout << "\n========================================" << std::endl;
      std::cout << "Benchmark Stats (N=" << n << "):" << std::endl;
      std::cout << "  - Protocol   : " << protocol << std::endl;
      std::cout << "  - Comm Volume: " << comm_bytes << " bytes (" << comm_mb
                << " MB)" << std::endl;
      std::cout << "  - Comm Rounds: " << comm_rounds << " actions"
                << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    // 验证结果
    auto y_revealed = hal::reveal(&ctx, y_out);
    auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_revealed);

    // 校验 (allclose 对于整数就是完全相等)
    EXPECT_EQ(y_vec.shape(), y_expected.shape());
    EXPECT_TRUE(xt::allclose(y_vec, y_expected))
        << "SPU result mismatch with CPU ground truth!";
  });
}

}  // namespace spu::kernel::hal