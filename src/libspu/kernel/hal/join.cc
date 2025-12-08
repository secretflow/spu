// Copyright 2025 Ant Group Co., Ltd.
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

#include "yacl/utils/cuckoo_index.h"

#include "libspu/core/bit_utils.h"
#include "libspu/core/context.h"
#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/random.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/utils.h"
#include "libspu/kernel/hlo/permute.h"
#include "libspu/kernel/hlo/soprf.h"
#include "libspu/kernel/test_util.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/spu.h"

namespace spu::kernel::hal {

// Secure join_uu
// Ref:
// https://dl.acm.org/doi/abs/10.1145/3372297.3423358
std::vector<spu::Value> join_uu(SPUContext* ctx,
                                absl::Span<const spu::Value> table_1,
                                absl::Span<const spu::Value> table_2,
                                size_t num_join_keys, size_t num_hash,
                                double scale_factor) {
  // 定义join_uu函数，接受两个表和连接键的数量作为参数
  // 连接键默认放在表的前面
  // 不支持输入有空行

  SPU_TRACE_HAL_DISP(ctx, table_1.size(), table_2.size());
  SPU_ENFORCE(!table_1.empty(), "table_1 is empty");
  SPU_ENFORCE(!table_2.empty(), "table_2 is empty");
  SPU_ENFORCE(num_join_keys > 0, "num_join_keys must be greater than 0");
  SPU_ENFORCE(num_join_keys <= table_1.size(),
              "num_join_keys exceeds table_1 size");
  SPU_ENFORCE(num_join_keys <= table_2.size(),
              "num_join_keys exceeds table_2 size");

  // table_1的行数
  const int64_t n_1 = table_1[0].shape()[0];
  // table_2的行数
  const int64_t n_2 = table_2[0].shape()[0];

  // 生成join keys的SoPrf输出
  std::vector<spu::Value> join_keys;
  join_keys.reserve(num_join_keys);
  for (size_t i = 0; i < num_join_keys; ++i) {
    auto t1_reshaped = spu::kernel::hal::reshape(ctx, table_1[i], {1, n_1});
    auto t2_reshaped = spu::kernel::hal::reshape(ctx, table_2[i], {1, n_2});
    spu::Value key_i = hal::_concatenate(ctx, {t1_reshaped, t2_reshaped}, 1);
    join_keys.push_back(key_i);
  }
  spu::Value ret = spu::kernel::hlo::SoPrf(ctx, absl::MakeSpan(join_keys));

  // 将ret的前n_1行reveal给P_0，后n_2行reveal给P_1
  // 需要使用reveal_to，但是有问题
  auto e_1 = hal::dump_public_as<uint128_t>(
      ctx, hal::reveal(ctx, hal::slice(ctx, ret, {0, 0}, {1, n_1})));

  auto e_2 = hal::dump_public_as<uint128_t>(
      ctx, hal::reveal(ctx, hal::slice(ctx, ret, {0, n_1}, {1, n_1 + n_2})));

  // 布谷鸟哈希初始化
  //  输入的四个参数分别为：num_input, num_stash, num_hash, scale_factor
  yacl::CuckooIndex::Options opts = {static_cast<uint64_t>(n_2), 0, num_hash,
                                     scale_factor};
  yacl::CuckooIndex cuckoo_index(opts);
  const int64_t num_perm_0 = cuckoo_index.bins().size();

  // 定义由参与方1根据布谷鸟哈希生成的置换pi_1，置换的大小为num_perm_0
  std::vector<int64_t> pi_1(num_perm_0);

  // 定义由参与方0根据num_hash个hash函数生成的置换pi_0，每个置换的大小为n_1
  std::vector<std::vector<int64_t>> pi_0(num_hash, std::vector<int64_t>(n_1));

  if (ctx->lctx()->Rank() == 0) {
    for (int64_t i = 0; i < num_perm_0; ++i) {
      pi_1[i] = 0;
    }

    // 对e_1执行num_hash个哈希函数，输出置换pi_0
    for (size_t i = 0; i < num_hash; ++i) {
      for (int64_t j = 0; j < n_1; ++j) {
        yacl::CuckooIndex::HashRoom e_1_hash(e_1[j]);
        pi_0[i][j] = e_1_hash.GetHash(i) % num_perm_0;
      }
    }

  } else if (ctx->lctx()->Rank() == 1) {
    for (size_t i = 0; i < num_hash; ++i) {
      for (int64_t j = 0; j < n_1; ++j) {
        pi_0[i][j] = 0;
      }
    }

    // 对e_2执行布谷鸟哈希，输出置换pi_1
    cuckoo_index.Insert(e_2);

    // 定义置换pi_1满足pi_1(j)=i，其中e_2[i]=t[j]
    int64_t tmp = n_2;
    for (int64_t i = 0; i < num_perm_0; ++i) {
      const auto& bin = cuckoo_index.bins()[i];
      if (!bin.IsEmpty()) {
        pi_1[i] = bin.InputIdx();
      } else {
        pi_1[i] = tmp;
        tmp = tmp + 1;
      }
    }
  }

  // 将置换pi_1转换为spu::Value
  auto pi_1_c = hal::constant(ctx, pi_1, DT_I64);
  auto pi_1_v = hal::_p2v(ctx, pi_1_c, 1).setDtype(pi_1_c.dtype());

  // 将table_2每一列的大小从n_2扩充成num_perm_0
  std::vector<spu::Value> table_2_vec;
  table_2_vec.reserve(table_2.size() + 1);
  for (const auto& col : table_2) {
    auto pad_value = hal::seal(ctx, hal::constant(ctx, 0, col.dtype()));
    auto col_extended =
        hal::pad(ctx, col, pad_value, {0}, {num_perm_0 - n_2}, {0});
    table_2_vec.push_back(col_extended);
  }
  // 添加一个前n_2为1，后num_perm_0 - n_2为0的列，表示是否为填充行
  std::vector<uint8_t> indicator_data(num_perm_0, 0);
  for (int64_t i = 0; i < n_2; ++i) {
    indicator_data[i] = 1;
  }
  auto indicator_c = hal::constant(ctx, indicator_data, DT_U8);
  auto indicator_v = hal::_p2s(ctx, indicator_c).setDtype(indicator_c.dtype());
  table_2_vec.push_back(indicator_v);

  // 对表table_2_vec执行置换pi_1
  auto table_t_2 =
      hlo::GeneralPermute(ctx, absl::MakeSpan(table_2_vec), pi_1_v);

  // 用置换pi_0从table_t_2生成num_hash个table_t_2_i
  std::vector<spu::Value> table_t_1;
  table_t_1.reserve((table_2.size() + 1) * num_hash);

  for (size_t i = 0; i < num_hash; ++i) {
    auto pi_0_c = hal::constant(ctx, pi_0[i], DT_I64);
    auto pi_0_v = hal::_p2v(ctx, pi_0_c, 0).setDtype(pi_0_c.dtype());
    auto table_t_2_i =
        hlo::GeneralPermute(ctx, absl::MakeSpan(table_t_2), pi_0_v);
    table_t_1.insert(table_t_1.end(), table_t_2_i.begin(), table_t_2_i.end());
  }

  // 取出table_1的前num_join_keys列
  std::vector<spu::Value> table_1_keys;
  table_1_keys.reserve(num_join_keys);
  for (size_t i = 0; i < num_join_keys; ++i) {
    table_1_keys.push_back(table_1[i]);
  }
  auto table_1_key = hal::concatenate(ctx, table_1_keys, 0);

  // 对于table_t_1中的每个表table_t_2_i的前num_join_keys列，和table_1_keys进行比较，相等则输出1，否则输出0
  std::vector<spu::Value> join_result_cols;
  join_result_cols.reserve(num_hash);
  for (size_t i = 0; i < num_hash; ++i) {
    std::vector<spu::Value> table_t_2_i_keys;
    table_t_2_i_keys.reserve(num_join_keys);
    for (size_t j = 0; j < num_join_keys; ++j) {
      table_t_2_i_keys.push_back(
          table_t_1[i * (table_2.size() + 1) + j]);  //+1是因为多了一列indicator
    }
    auto table_t_2_i_key = hal::concatenate(ctx, table_t_2_i_keys, 0);

    spu::Value eq_result = hal::equal(ctx, table_1_key, table_t_2_i_key);

    // 对eq_result分成n_1行，一共num_join_keys列，然后这些列按行进行and
    spu::Value and_result = hal::slice(ctx, eq_result, {0}, {n_1});
    for (size_t j = 1; j < num_join_keys; ++j) {
      and_result = hal::bitwise_and(
          ctx, and_result,
          hal::slice(ctx, eq_result, {static_cast<int64_t>(j) * n_1},
                     {static_cast<int64_t>(j + 1) * n_1}));
    }
    and_result = hal::bitwise_and(
        ctx, and_result,
        hal::slice(ctx, table_t_1[i * (table_2.size() + 1) + table_2.size()],
                   {0}, {n_1}));  // 和indicator列进行and

    join_result_cols.push_back(and_result);
  }

  // 获得table_2的连接结果，匹配的行输出table_2中的值，不匹配的行输出0
  std::vector<spu::Value> table_2_result;
  table_2_result.reserve(table_2.size());
  for (size_t col_idx = 0; col_idx < table_2.size(); ++col_idx) {
    spu::Value col_result =
        hal::constant(ctx, 0, table_2[col_idx].dtype(), table_1[0].shape());
    spu::Value control_bit = hal::constant(ctx, 0, join_result_cols[0].dtype(),
                                           join_result_cols[0].shape());
    for (size_t hash_idx = 0; hash_idx < num_hash; ++hash_idx) {
      // 对control_bit进行not非运算
      control_bit = hal::bitwise_not(ctx, control_bit);
      // 将control_bit和join_result_cols[hash_idx]进行and运算
      control_bit =
          hal::bitwise_and(ctx, control_bit, join_result_cols[hash_idx]);
      // 将control_bit和table_t_1中对应的列进行乘法运算
      spu::Value table_t_2_i_col =
          table_t_1[hash_idx * (table_2.size() + 1) + col_idx];
      spu::Value mul_result = hal::mul(ctx, table_t_2_i_col, control_bit);
      col_result = hal::add(ctx, col_result, mul_result);
    }
    table_2_result.push_back(col_result);
  }

  // 对join_result_cols按位或，得到最终的join结果
  spu::Value join_result = join_result_cols[0];
  ;
  for (size_t i = 1; i < join_result_cols.size(); ++i) {
    join_result = hal::bitwise_or(ctx, join_result, join_result_cols[i]);
  }

  // 组合输出结果
  std::vector<spu::Value> join_results;
  join_results.reserve(table_1.size() + table_2.size() + 1);
  for (const auto& col : table_1) {
    join_results.push_back(col);
  }
  for (const auto& col : table_2_result) {
    join_results.push_back(col);
  }
  join_results.push_back(join_result);

  return join_results;
}

}  // namespace spu::kernel::hal