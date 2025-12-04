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

TEST(SortTest, SimpleSimulatedWithMetrics) {
  const size_t npc = 2;
  const auto protocol = ProtocolKind::SEMI2K;
  const auto field = FieldType::FM64;

  // 准备 Ground Truth 数据
  xt::xarray<float> x1 = {{1, 3, 5, 20}, {1, 3, 5, 25}};
  xt::xarray<float> x2 = {{2, 4, 6, 10}, {2, 4, 6, 15}};
  xt::xarray<float> sorted_x = {{1, 2, 3, 4, 5, 6, 10, 20},
                                {1, 2, 3, 4, 5, 6, 15, 25}};

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

}  // namespace spu::kernel::hal
