// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/kernel/hal/join.h"

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/soprf.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal {

TEST(JoinTest, Work) {
  FieldType field = FieldType::FM64;
  ProtocolKind prot = ProtocolKind::SEMI2K;
  size_t num_join_keys = 2;
  const size_t num_hash = 3;
  const double scale_factor = 1.5;

  const Shape shape_1 = {3, 8};
  const Shape shape_2 = {4, 9};

  xt::xarray<uint64_t> data_1 = {{1, 4, 3, 5, 2, 6, 7, 0},
                                 {1, 2, 3, 4, 5, 6, 5, 0},
                                 {11, 22, 33, 44, 55, 66, 77, 000}};
  xt::xarray<uint64_t> data_2 = {{3, 5, 7, 9, 1, 13, 4, 6, 0},
                                 {3, 4, 5, 6, 1, 8, 9, 10, 0},
                                 {33, 44, 55, 66, 77, 88, 99, 110, 121},
                                 {111, 222, 333, 444, 555, 666, 777, 888, 999}};

  mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        SPUContext sctx = test::makeSPUContext(prot, field, lctx);

        // auto table_1 = spu::kernel::hlo::Seal(
        //     &sctx, spu::kernel::hlo::Constant(&sctx, data_1, shape_1));
        // auto table_2 = spu::kernel::hlo::Seal(
        //     &sctx, spu::kernel::hlo::Constant(&sctx, data_2, shape_2));

        // 创建表格1的多个列（这里假设 data_1 的每一行是一个列）
        std::vector<Value> table1_columns;
        for (int64_t i = 0; i < shape_1[0]; ++i) {  // 遍历行数 = 列数
          // 提取第i列数据
          xt::xarray<uint64_t> col_data = xt::row(data_1, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table1_columns.push_back(col);
        }

        // 创建表格2的多个列
        std::vector<Value> table2_columns;
        for (int64_t i = 0; i < shape_2[0]; ++i) {
          xt::xarray<uint64_t> col_data = xt::row(data_2, i);
          Value col = test::makeValue(&sctx, col_data, VIS_SECRET);
          table2_columns.push_back(col);
        }

        // 转换为 Span
        absl::Span<const Value> table1_span =
            absl::MakeConstSpan(table1_columns);
        absl::Span<const Value> table2_span =
            absl::MakeConstSpan(table2_columns);

        // 调用 join_uu
        auto ret = join_uu(&sctx, table1_span, table2_span, num_join_keys,
                           num_hash, scale_factor);

        // 遍历ret的每一列并输出
        for (size_t i = 0; i < ret.size(); ++i) {
          auto ret_hat =
              hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret[i]));

          if (lctx->Rank() == 0) {
            std::cout << "Join output column " << i << ": " << ret_hat
                      << std::endl;
          }
        }

        // auto ret_hat =
        //     hal::dump_public_as<uint64_t>(&sctx, hal::reveal(&sctx, ret));

        // if (lctx->Rank() == 0) {
        //   std::cout << "Join output: " << ret_hat << std::endl;
        // }
      });
}

}  // namespace spu::kernel::hal
