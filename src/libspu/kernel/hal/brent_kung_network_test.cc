#include "libspu/kernel/hal/brent_kung_network.h"

#include "gtest/gtest.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {

class BrentKungTest : public ::testing::Test {
};

TEST_F(BrentKungTest, BasicCorrectness) {
    const size_t npc = 2;

    mpc::utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext ctx = test::makeSPUContext(ProtocolKind::SEMI2K, FieldType::FM64, lctx);
        
        // 准备输入数据
        int64_t n = 8;
        int64_t block_size = 2;

        std::vector<int64_t> x_data = {
            5, 6,  2, 2,  10, 10,  4, 3,
            1, 1,  2, 2,  5, 5,   3, 2
        };
        std::vector<int64_t> g_data = {0, 1, 1, 0, 1, 1, 1, 0};
        xt::xarray<int64_t> y_expected = {
            {5, 6},  {5, 6},  {5, 6},  {4, 3},
            {4, 3},  {4, 3},  {4, 3},   {3, 2}
        };

        // 把输入数据转换为spu内部对象x_in和g_in
        spu::PtBufferView x_view(
            x_data.data(), 
            spu::PT_I64, 
            {static_cast<int64_t>(x_data.size())}, 
            {1}
        );
        auto x_in = hal::constant(&ctx, x_view, spu::DT_I64, {n, block_size});
        
        spu::PtBufferView g_view(
            g_data.data(), 
            spu::PT_I64, 
            {static_cast<int64_t>(g_data.size())}, 
            {1}
        );
        auto g_in = hal::constant(&ctx, g_view, spu::DT_I64, {n, 1});

        // 执行算法
        auto y_out = AggregateBrentKung(&ctx, x_in, g_in);
        
        // 输出数据转回C++类型
        auto y_vec = hal::dump_public_as<int64_t>(&ctx, y_out);

        if (lctx->Rank() == 0) {
            std::cout << "Result Y (Rank 0): ";
            for (auto v : y_vec) std::cout << v << " ";
            std::cout << std::endl;
        }
        
        // 检查正确性
        EXPECT_EQ(y_vec.size(), n * block_size);
        EXPECT_TRUE(xt::allclose(y_vec, y_expected));
    });
}

} // namespace spu::kernel::hal