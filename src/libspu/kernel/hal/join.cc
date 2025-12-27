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

#include "libspu/core/context.h"
#include "libspu/core/trace.h"
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
#include "libspu/spu.h"

namespace spu::kernel::hal {

namespace {

spu::Value _cuckoo_hash_to_perm(SPUContext* ctx, const Value& e_1,
                                const Value& e_2, size_t num_hash,
                                size_t scale_factor, FieldType field) {
  // Input two private values e_1 and e_2, belonging to P0 and P1 respectively,
  // output the concatenated result of two permutations pi_0 and pi_1 based on
  // Cuckoo Hash.
  if (e_1.isPrivate() && e_2.isPrivate()) {
    SPU_ENFORCE(e_1.storage_type().as<Private>()->owner() == 0,
                "e_1 must be owned by P0");
    SPU_ENFORCE(e_2.storage_type().as<Private>()->owner() == 1,
                "e_2 must be owned by P1");
    return _cuckoo_hash_to_perm_v(ctx, e_1, e_2, num_hash, scale_factor, field);
  } else {
    SPU_THROW("e_1 and e_2 must be private values of P0 and P1 respectively");
  }
}

}  // namespace

// Secure join_uu
// Ref:
// https://dl.acm.org/doi/abs/10.1145/3372297.3423358
std::vector<spu::Value> join_uu(SPUContext* ctx,
                                absl::Span<const spu::Value> table_1,
                                absl::Span<const spu::Value> table_2,
                                size_t num_join_keys, size_t num_hash,
                                size_t scale_factor, FieldType field) {
  // Input: two tables, the number of join keys, the number of hashes for Cuckoo
  // Hash, scale factor for calculating the number of bins of Cuckoo Hash. The
  // join keys are placed in front of the table by default. Input with blank
  // lines is not supported.

  SPU_TRACE_HAL_DISP(ctx, table_1.size(), table_2.size());
  SPU_ENFORCE(!table_1.empty(), "table_1 is empty");
  SPU_ENFORCE(!table_2.empty(), "table_2 is empty");
  SPU_ENFORCE(num_join_keys > 0, "num_join_keys must be greater than 0");
  SPU_ENFORCE(num_join_keys <= table_1.size(),
              "num_join_keys exceeds table_1 size");
  SPU_ENFORCE(num_join_keys <= table_2.size(),
              "num_join_keys exceeds table_2 size");

  // Number of rows in table_1
  const int64_t n_1 = table_1[0].shape()[0];
  // Number of rows in table_2
  const int64_t n_2 = table_2[0].shape()[0];

  // Generate SoPrf output of join keys.
  std::vector<spu::Value> join_keys;
  join_keys.reserve(num_join_keys);
  for (size_t i = 0; i < num_join_keys; ++i) {
    auto t1_reshaped = spu::kernel::hal::reshape(ctx, table_1[i], {1, n_1});
    auto t2_reshaped = spu::kernel::hal::reshape(ctx, table_2[i], {1, n_2});
    spu::Value key_i = hal::_concatenate(ctx, {t1_reshaped, t2_reshaped}, 1);
    join_keys.push_back(key_i);
  }
  spu::Value ret = spu::kernel::hlo::SoPrf(ctx, absl::MakeSpan(join_keys));

  // Give the first n_1 line of ret reveal to P_0 and the last n_2 line reveal
  spu::Value e_1 =
      hal::reveal_to(ctx, hal::slice(ctx, ret, {0, 0}, {1, n_1}), 0);
  spu::Value e_2 =
      hal::reveal_to(ctx, hal::slice(ctx, ret, {0, n_1}, {1, n_1 + n_2}), 1);

  // 当field == FieldType::FM64 && num_join_keys ==
  // 1时，后面需要使用FM64，否则结果不对
  FieldType field_2 = FieldType::FM128;
  if (field == FieldType::FM64 && num_join_keys == 1) {
    field_2 = FieldType::FM64;
  }

  // compute cuckoo hash table size
  yacl::CuckooIndex::Options opts = {static_cast<uint64_t>(n_2), 0, num_hash,
                                     scale_factor / 10.0};
  yacl::CuckooIndex cuckoo_index(opts);
  const int64_t num_perm_1 = cuckoo_index.bins().size();

  //_cuckoo_hash_to_perm
  spu::Value perm_all =
      _cuckoo_hash_to_perm(ctx, e_1, e_2, num_hash, scale_factor, field_2);

  // Get permutation pi_1.
  auto pi_1_p = hal::reshape(
      ctx,
      hal::slice(ctx, perm_all, {static_cast<int64_t>(num_hash), 0},
                 {static_cast<int64_t>(num_hash + 1), num_perm_1}),
      {num_perm_1});
  auto pi_1_v = hal::_p2v(ctx, pi_1_p, 1).setDtype(pi_1_p.dtype());

  // Expand the size of each column of table_2 from n_2 to num_perm_1.
  std::vector<spu::Value> table_2_vec;
  table_2_vec.reserve(table_2.size() + 1);
  for (const auto& col : table_2) {
    auto pad_value = hal::seal(ctx, hal::constant(ctx, 0, col.dtype()));
    auto col_extended =
        hal::pad(ctx, col, pad_value, {0}, {num_perm_1 - n_2}, {0});
    table_2_vec.push_back(col_extended);
  }
  // Add a column with n_2 as 1 and num_perm_1-n_2 as 0 to indicate whether it
  // is a filled row.
  std::vector<uint8_t> indicator_data(num_perm_1, 0);
  for (int64_t i = 0; i < n_2; ++i) {
    indicator_data[i] = 1;
  }
  auto indicator_c = hal::constant(ctx, indicator_data, DT_U8);
  auto indicator_v = hal::_p2s(ctx, indicator_c).setDtype(indicator_c.dtype());
  table_2_vec.push_back(indicator_v);

  // Perform permutation pi_1 on table_2_vec.
  auto table_t_2 =
      hlo::GeneralPermute(ctx, absl::MakeSpan(table_2_vec), pi_1_v);

  // Generating num_hash table_t_2 _ i from table _ t _ 2 with permutation pi_0.
  std::vector<spu::Value> table_t_1;
  table_t_1.reserve((table_2.size() + 1) * num_hash);

  for (size_t i = 0; i < num_hash; ++i) {
    auto pi_0_p = hal::reshape(
        ctx,
        hal::slice(ctx, perm_all, {static_cast<int64_t>(i), 0},
                   {static_cast<int64_t>(i + 1), static_cast<int64_t>(n_1)}),
        {n_1});
    auto pi_0_v = hal::_p2v(ctx, pi_0_p, 0).setDtype(pi_0_p.dtype());
    auto table_t_2_i =
        hlo::GeneralPermute(ctx, absl::MakeSpan(table_t_2), pi_0_v);
    table_t_1.insert(table_t_1.end(), table_t_2_i.begin(), table_t_2_i.end());
  }

  // Take out the first num_join_keys column of table_1.
  std::vector<spu::Value> table_1_keys;
  table_1_keys.reserve(num_join_keys);
  for (size_t i = 0; i < num_join_keys; ++i) {
    table_1_keys.push_back(table_1[i]);
  }
  auto table_1_key = hal::concatenate(ctx, table_1_keys, 0);

  // Compare the first num_join_keys column of table_t_2_i in table_t_1 with
  // table_1_keys, and output 1 if they are equal, otherwise output 0.
  std::vector<spu::Value> join_result_cols;
  join_result_cols.reserve(num_hash);
  for (size_t i = 0; i < num_hash; ++i) {
    std::vector<spu::Value> table_t_2_i_keys;
    table_t_2_i_keys.reserve(num_join_keys);
    for (size_t j = 0; j < num_join_keys; ++j) {
      table_t_2_i_keys.push_back(
          table_t_1[(i * (table_2.size() + 1)) +
                    j]);  //+1 is because there is one more column indicator.
    }
    auto table_t_2_i_key = hal::concatenate(ctx, table_t_2_i_keys, 0);

    spu::Value eq_result = hal::equal(ctx, table_1_key, table_t_2_i_key);

    // The eq_result is divided into n_1 rows, with a total of num_join_keys
    // columns, and then these columns are AND by rows.
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
                   {0}, {n_1}));  // And the indicator column.

    join_result_cols.push_back(and_result);
  }

  // Get the join result of table_2, and output the values in table_2 for the
  // matched rows and 0 for the unmatched rows.
  std::vector<spu::Value> table_2_result;
  table_2_result.reserve(table_2.size());
  for (size_t col_idx = 0; col_idx < table_2.size(); ++col_idx) {
    spu::Value col_result =
        hal::constant(ctx, 0, table_2[col_idx].dtype(), table_1[0].shape());
    spu::Value control_bit = hal::constant(ctx, 0, join_result_cols[0].dtype(),
                                           join_result_cols[0].shape());
    for (size_t hash_idx = 0; hash_idx < num_hash; ++hash_idx) {
      // Not operation on control_bit.
      control_bit = hal::bitwise_not(ctx, control_bit);
      // And control_bit and join_result_cols[hash_idx].
      control_bit =
          hal::bitwise_and(ctx, control_bit, join_result_cols[hash_idx]);
      // Multiply the corresponding columns in control_bit and table_t_1.
      spu::Value table_t_2_i_col =
          table_t_1[(hash_idx * (table_2.size() + 1)) + col_idx];
      spu::Value mul_result = hal::mul(ctx, table_t_2_i_col, control_bit);
      col_result = hal::add(ctx, col_result, mul_result);
    }
    table_2_result.push_back(col_result);
  }

  // Bitwise OR the join_result_cols to get the final join result.
  spu::Value join_result = join_result_cols[0];
  ;
  for (size_t i = 1; i < join_result_cols.size(); ++i) {
    join_result = hal::bitwise_or(ctx, join_result, join_result_cols[i]);
  }

  // Combined output result
  std::vector<spu::Value> join_results;
  join_results.reserve(table_1.size() + table_2.size() + 1);
  for (const auto& col : table_1) {
    join_results.push_back(hal::mul(ctx, col, join_result));
  }
  for (const auto& col : table_2_result) {
    join_results.push_back(col);
  }
  join_results.push_back(join_result);

  return join_results;
}

}  // namespace spu::kernel::hal