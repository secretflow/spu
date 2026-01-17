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
#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/random.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/soprf.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/spu.h"

namespace spu::kernel::hal {

namespace {

std::vector<spu::Value> _cuckoo_hash_to_perm(SPUContext* ctx, const Value& e_1,
                                             const Value& e_2, size_t num_hash,
                                             double scale_factor,
                                             size_t num_join_keys) {
  // Input two private values e_1 and e_2, belonging to P1 and P2 respectively,
  // output num_hash +1 permutations perm_all based on Cuckoo Hash where the
  // first num_hash permutations are belonging to P1 and the last permutation
  // belonging to P2.
  if (e_1.isPrivate() && e_2.isPrivate()) {
    SPU_ENFORCE(e_1.storage_type().as<Private>()->owner() == 0,
                "e_1 must be owned by P0");
    SPU_ENFORCE(e_2.storage_type().as<Private>()->owner() == 1,
                "e_2 must be owned by P1");
    return _cuckoo_hash_to_perm_v(ctx, e_1, e_2, num_hash, scale_factor,
                                  num_join_keys);
  } else {
    SPU_THROW("e_1 and e_2 must be private values of P0 and P1 respectively");
  }
}

Value prefix_sum(SPUContext* ctx, const Value& x) {
  bool input_is_1d = (x.shape().ndim() == 1);
  int64_t original_n = x.numel();

  Value x_2d;
  if (input_is_1d) {
    // reshape 1D -> [1, n]
    x_2d = hal::reshape(ctx, x, {1, original_n});
  } else {
    SPU_ENFORCE(x.shape().ndim() == 2U && x.shape()[0] == 1,
                "x should be 1-row matrix or 1-d vector");
    x_2d = x;
  }

  const int64_t n = x_2d.numel();
  if (n == 0) {
    return x;
  }

  std::vector<Value> parts;
  parts.reserve(n);

  auto first = hal::slice(ctx, x_2d, {0, 0}, {1, 1});
  parts.push_back(first);

  for (int64_t i = 1; i < n; ++i) {
    auto current = hal::slice(ctx, x_2d, {0, i}, {1, i + 1});
    auto prev_sum = parts.back();
    auto sum = hal::add(ctx, prev_sum, current);
    parts.push_back(sum);
  }

  auto result_2d = hal::concatenate(ctx, parts, 1);

  if (input_is_1d) {
    return hal::reshape(ctx, result_2d, {original_n});
  }

  return result_2d;
}

template <typename Fn>
spu::Value associative_reduce(Fn&& fn, SPUContext* ctx, const Value& in) {
  SPU_ENFORCE(in.shape().ndim() >= 1U, "input should not be scalar");

  const Shape& shape = in.shape();
  const auto N = shape.back();

  // Compute output shape: remove the last dimension
  Shape out_shape;
  if (shape.ndim() == 1) {
    out_shape = {1};
  } else {
    out_shape = Shape(shape.begin(), shape.end() - 1);
  }

  // Edge case: if the last dimension has <= 1 element or tensor is empty
  if (N < 2 || shape.numel() == 0) {
    return in;
  }

  // Reshape to 2D {M, N} tensor for easier processing
  const auto M = shape.numel() / N;
  spu::Value current = hal::reshape(ctx, in, {M, N});

  int64_t len = N;

  // Find the largest power of 2 that is <= len
  int64_t lower_exp = 1;
  while ((lower_exp << 1) <= len) {
    lower_exp <<= 1;
  }

  // If len > lower_exp, pre-reduce the extra part to make len = lower_exp
  if (len != lower_exp) {
    // c = [lower_exp, len): the extra elements
    // b = [2*lower_exp - len, lower_exp): elements to pair with c
    // a = [0, 2*lower_exp - len): untouched elements
    auto c = hal::slice(ctx, current, {0, lower_exp}, {M, len}, {});
    auto b =
        hal::slice(ctx, current, {0, 2 * lower_exp - len}, {M, lower_exp}, {});
    auto t = fn(ctx, c, b);  // t = c ⊕ b

    auto a = hal::slice(ctx, current, {0, 0}, {M, 2 * lower_exp - len}, {});
    current = hal::concatenate(ctx, {a, t}, 1);
    len = lower_exp;
  }

  // Now len is a power of 2, do standard halving reduction
  while (len > 1) {
    const int64_t half = len / 2;
    auto lhs = hal::slice(ctx, current, {0, 0}, {M, half}, {});
    auto rhs = hal::slice(ctx, current, {0, half}, {M, 2 * half}, {});
    current = fn(ctx, lhs, rhs);
    len = half;
  }

  return hal::reshape(ctx, current, out_shape);
}

}  // namespace

// Secure join_uu
// Ref:
// https://dl.acm.org/doi/abs/10.1145/3372297.3423358
std::vector<spu::Value> join_uu(SPUContext* ctx,
                                absl::Span<const spu::Value> table_1,
                                absl::Span<const spu::Value> table_2,
                                size_t num_join_keys, size_t num_hash,
                                double scale_factor) {
  // Input: two tables, the number of join keys, the number of hashes for Cuckoo
  // Hash, scale factor for calculating the number of bins of Cuckoo Hash. The
  // join keys are placed in front of the table by default. Input with blank
  // lines is not supported.
  {
    SPU_TRACE_HAL_DISP(ctx, table_1.size(), table_2.size());
    SPU_ENFORCE(!table_1.empty(), "table_1 is empty");
    SPU_ENFORCE(!table_2.empty(), "table_2 is empty");
    SPU_ENFORCE(num_join_keys > 0, "num_join_keys must be greater than 0");
    SPU_ENFORCE(num_join_keys <= table_1.size(),
                "num_join_keys exceeds table_1 size");
    SPU_ENFORCE(num_join_keys <= table_2.size(),
                "num_join_keys exceeds table_2 size");

    // Check that each column in both tables is 1-d
    for (const auto& col : table_1) {
      SPU_ENFORCE(col.shape().ndim() == 1U,
                  "each column in table_1 must be 1-d");
    }
    for (const auto& col : table_2) {
      SPU_ENFORCE(col.shape().ndim() == 1U,
                  "each column in table_2 must be 1-d");
    }

    // Check that each column in both tables has the same number of rows
    for (const auto& col : table_1) {
      SPU_ENFORCE(col.shape()[0] == table_1[0].shape()[0],
                  "all columns in table_1 must have the same number of rows");
    }
    for (const auto& col : table_2) {
      SPU_ENFORCE(col.shape()[0] == table_2[0].shape()[0],
                  "all columns in table_2 must have the same number of rows");
    }

    // Check that at least one of the join keys in both tables is of secret type
    bool has_secret_key = false;
    for (size_t i = 0; i < num_join_keys; i++) {
      if (table_1[i].isSecret() || table_2[i].isSecret()) {
        has_secret_key = true;
        break;
      }
    }
    SPU_ENFORCE(has_secret_key,
                "at least one of the join keys must be of secret type");
  }

  // Compute table sizes and cmpare them. If table_1 is smaller, swap table_1
  // and table_2
  //  This is because in the join_uu algorithm, a large number of permutation
  //  operations are performed on table_2, so placing the smaller table in the
  //  table_2 position can reduce the overhead of permutation operations.
  {
    const int64_t table_1_size = table_1[0].shape()[0] * table_1.size();
    const int64_t table_2_size = table_2[0].shape()[0] * table_2.size();
    if (table_1_size < table_2_size) {
      return join_uu(ctx, table_2, table_1, num_join_keys, num_hash,
                     scale_factor);
    }
  }

  //  Number of rows in table_1
  const int64_t n_1 = table_1[0].shape()[0];
  // Number of rows in table_2
  const int64_t n_2 = table_2[0].shape()[0];

  // Generate SoPrf output of join keys.
  std::vector<spu::Value> join_keys;
  join_keys.reserve(num_join_keys);
  for (size_t i = 0; i < num_join_keys; ++i) {
    std::vector<spu::Value> to_concat;
    to_concat.push_back(table_1[i]);
    to_concat.push_back(table_2[i]);
    spu::Value key_i = hal::concatenate(ctx, to_concat, 0);
    join_keys.push_back(key_i);
  }
  spu::Value join_keys_soprf = hal::soprf(ctx, absl::MakeSpan(join_keys));

  // Give the first n_1 line of join_keys_soprf reveal to P_0 and the last n_2
  // line reveal to P_1
  spu::Value e_1 =
      hal::reveal_to(ctx, hal::slice(ctx, join_keys_soprf, {0}, {n_1}), 0);
  spu::Value e_2 = hal::reveal_to(
      ctx, hal::slice(ctx, join_keys_soprf, {n_1}, {n_1 + n_2}), 1);

  // compute cuckoo hash table size
  yacl::CuckooIndex::Options opts = {static_cast<uint64_t>(n_2), 0, num_hash,
                                     scale_factor};
  yacl::CuckooIndex cuckoo_index(opts);
  const int64_t cuckoo_hash_size = cuckoo_index.bins().size();

  //_cuckoo_hash_to_perm
  std::vector<spu::Value> perm_all = _cuckoo_hash_to_perm(
      ctx, e_1, e_2, num_hash, scale_factor, num_join_keys);

  // Generate permutation pi_2 based on the last row of perm_all.
  const auto& pi_2_v = perm_all.back();

  // Expand the size of each column of table_2 from n_2 to cuckoo_hash_size.
  std::vector<spu::Value> table_2_expand;
  table_2_expand.reserve(table_2.size() + 1);
  for (const auto& col : table_2) {
    std::vector<uint64_t> invalid_value(cuckoo_hash_size - n_2, INT64_MAX);
    auto pad_value =
        hal::seal(ctx, hal::constant(ctx, invalid_value, col.dtype()));
    auto col_extended = hal::concatenate(ctx, {col, pad_value}, 0);
    table_2_expand.push_back(col_extended);
  }

  // Add a column with n_2 as 1 and cuckoo_hash_size-n_2 as 0 to indicate
  // whether it is a filled row.
  std::vector<uint8_t> indicator_data(cuckoo_hash_size, 0);
  for (int64_t i = 0; i < n_2; ++i) {
    indicator_data[i] = 1;
  }
  auto indicator_c = hal::constant(ctx, indicator_data, DT_U8);
  auto indicator_s = hal::_p2s(ctx, indicator_c).setDtype(indicator_c.dtype());
  table_2_expand.push_back(indicator_s);

  // Perform permutation pi_2 on table_2_expand.
  std::vector<spu::Value> tbl2_perm_by_pi2;
  for (const auto& col : table_2_expand) {
    auto table_t_2_i = hal::apply_general_permute_1d(ctx, col, pi_2_v);
    tbl2_perm_by_pi2.push_back(table_t_2_i);
  }

  // Get each pi_1 corresponding to each hash function, and use it to permute
  //  tbl2_perm_by_pi2 to get the table_t_i_i corresponding to the hash
  //  function, then concatenate all table_t_i_i together to get
  //  tbl2_perm_by_pi1
  std::vector<spu::Value> tbl2_perm_by_pi1;
  tbl2_perm_by_pi1.reserve(num_hash * (table_2.size() + 1));
  for (size_t i = 0; i < num_hash; ++i) {
    // Generate permutation pi_1 based on the i-th row of perm_all.
    const spu::Value& pi_1_v = perm_all[i];
    std::vector<spu::Value> tbl2_perm_by_pi1_i;
    tbl2_perm_by_pi1_i.reserve(tbl2_perm_by_pi2.size());
    for (const auto& col : tbl2_perm_by_pi2) {
      tbl2_perm_by_pi1_i.push_back(
          hal::apply_general_permute_1d(ctx, col, pi_1_v));
    }
    tbl2_perm_by_pi1.insert(tbl2_perm_by_pi1.end(), tbl2_perm_by_pi1_i.begin(),
                            tbl2_perm_by_pi1_i.end());
  }

  // Take out the first num_join_keys column of table_1.
  std::vector<spu::Value> table_1_keys;
  table_1_keys.reserve(num_join_keys);
  for (size_t i = 0; i < num_join_keys; ++i) {
    table_1_keys.push_back(table_1[i]);
  }
  auto table_1_keys_concat = hal::concatenate(ctx, table_1_keys, 0);

  // Compare the first num_join_keys column of tbl2_perm_by_pi1_i in
  // tbl2_perm_by_pi1 with table_1_keys, and output 1 if they are equal,
  // otherwise output 0.
  std::vector<spu::Value> tbl2_flag_of_hashes;
  tbl2_flag_of_hashes.reserve(num_hash);
  size_t begin_id_of_hash_i = 0;
  for (size_t i = 0; i < num_hash; ++i) {
    begin_id_of_hash_i =
        i * (table_2.size() +
             1);  //+1 is because there is one more column indicator.
    std::vector<spu::Value> tbl2_perm_by_pi1_of_hash_i;
    tbl2_perm_by_pi1_of_hash_i.reserve(num_join_keys);
    for (size_t j = 0; j < num_join_keys; ++j) {
      tbl2_perm_by_pi1_of_hash_i.push_back(
          tbl2_perm_by_pi1[begin_id_of_hash_i + j]);
    }
    auto tbl2_perm_by_pi1_of_hash_i_concat =
        hal::concatenate(ctx, tbl2_perm_by_pi1_of_hash_i, 0);

    spu::Value eq_result =
        hal::equal(ctx, table_1_keys_concat, tbl2_perm_by_pi1_of_hash_i_concat);

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
        hal::slice(ctx, tbl2_perm_by_pi1[begin_id_of_hash_i + table_2.size()],
                   {0}, {n_1}));  // And the indicator column.

    tbl2_flag_of_hashes.push_back(and_result);
  }

  // Get the join result of table_2, and output the values in table_2 for the
  // matched rows and 0 for the unmatched rows.
  std::vector<spu::Value> table_2_result;
  table_2_result.reserve(table_2.size());
  for (size_t col_idx = num_join_keys; col_idx < table_2.size(); ++col_idx) {
    spu::Value col_result =
        hal::constant(ctx, 0, table_2[col_idx].dtype(), table_1[0].shape());
    spu::Value processed_mask = hal::constant(
        ctx, 0, tbl2_flag_of_hashes[0].dtype(), tbl2_flag_of_hashes[0].shape());
    for (size_t hash_idx = 0; hash_idx < num_hash; ++hash_idx) {
      // Get a mask for the first match only.
      spu::Value control_bit =
          hal::bitwise_and(ctx, tbl2_flag_of_hashes[hash_idx],
                           hal::bitwise_not(ctx, processed_mask));
      // Multiply the corresponding columns in control_bit and tbl2_perm_by_pi1.
      spu::Value col_temp =
          tbl2_perm_by_pi1[(hash_idx * (table_2.size() + 1)) + col_idx];
      spu::Value mul_result = hal::mul(ctx, col_temp, control_bit);
      col_result = hal::add(ctx, col_result, mul_result);
      // Update the processed mask
      processed_mask =
          hal::bitwise_or(ctx, processed_mask, tbl2_flag_of_hashes[hash_idx]);
    }
    table_2_result.push_back(col_result);
  }

  // Bitwise OR the tbl2_flag_of_hashes to get the final join result.
  spu::Value valid_flag = tbl2_flag_of_hashes[0];
  auto bitwise_or_reducer = [](SPUContext* ctx, const Value& a,
                               const Value& b) {
    return hal::bitwise_or(ctx, a, b);
  };

  std::vector<Value> tbl2_flag_of_hashes_2d;
  tbl2_flag_of_hashes_2d.reserve(tbl2_flag_of_hashes.size());
  for (const auto& col : tbl2_flag_of_hashes) {
    auto col_2d = hal::reshape(ctx, col, {1, col.numel()});
    tbl2_flag_of_hashes_2d.push_back(col_2d);
  }

  auto tbl2_flag_of_hashes_2d_concat =
      hal::concatenate(ctx, tbl2_flag_of_hashes_2d, 0);
  auto transposed = hal::transpose(ctx, tbl2_flag_of_hashes_2d_concat);
  valid_flag = associative_reduce(bitwise_or_reducer, ctx, transposed);

  // Combined output result
  std::vector<spu::Value> join_results;
  join_results.reserve(table_1.size() + table_2.size() + 1);
  join_results.push_back(valid_flag);
  for (const auto& col : table_1) {
    join_results.push_back(col);
  }
  for (const auto& col : table_2_result) {
    join_results.push_back(col);
  }
  return join_results;
}

// Secure join_un
// Ref:
// https://eprint.iacr.org/2024/141.pdf
std::vector<spu::Value> join_un(SPUContext* ctx,
                                absl::Span<const spu::Value> table_1,
                                absl::Span<const spu::Value> table_2,
                                size_t num_join_keys) {
  // Input: two tables and the number of join keys.
  // The join keys are placed in front of the table by default.
  // Input with blank lines is not supported.
  // The join keys in table_1 are unique keys, while those in table_2 can be
  // duplicate.
  {
    SPU_TRACE_HAL_DISP(ctx, table_1.size(), table_2.size());
    SPU_ENFORCE(!table_1.empty(), "table_1 is empty");
    SPU_ENFORCE(!table_2.empty(), "table_2 is empty");
    SPU_ENFORCE(num_join_keys > 0, "num_join_keys must be greater than 0");
    SPU_ENFORCE(num_join_keys <= table_1.size(),
                "num_join_keys exceeds table_1 size");
    SPU_ENFORCE(num_join_keys <= table_2.size(),
                "num_join_keys exceeds table_2 size");

    // Check that each column in both tables is 1-d
    for (const auto& col : table_1) {
      SPU_ENFORCE(col.shape().ndim() == 1U,
                  "each column in table_1 must be 1-d");
    }
    for (const auto& col : table_2) {
      SPU_ENFORCE(col.shape().ndim() == 1U,
                  "each column in table_2 must be 1-d");
    }

    // Check that each column in both tables has the same number of rows
    for (const auto& col : table_1) {
      SPU_ENFORCE(col.shape()[0] == table_1[0].shape()[0],
                  "all columns in table_1 must have the same number of rows");
    }
    for (const auto& col : table_2) {
      SPU_ENFORCE(col.shape()[0] == table_2[0].shape()[0],
                  "all columns in table_2 must have the same number of rows");
    }

    // Check that at least one of the join keys in both tables is of secret type
    bool has_secret_key = false;
    for (size_t i = 0; i < num_join_keys; i++) {
      if (table_1[i].isSecret() || table_2[i].isSecret()) {
        has_secret_key = true;
        break;
      }
    }
    SPU_ENFORCE(has_secret_key,
                "at least one of the join keys must be of secret type");
  }

  // Number of rows in table_1
  const int64_t n_1 = table_1[0].shape()[0];
  // Number of rows in table_2
  const int64_t n_2 = table_2[0].shape()[0];

  // Generate num_join_keys random values
  std::vector<Value> rand_values;
  rand_values.reserve(num_join_keys);
  for (size_t j = 0; j < num_join_keys; j++) {
    auto rand_scalar = hal::random(ctx, VIS_PUBLIC, DT_I64, {1});
    rand_values.push_back(hal::broadcast_to(ctx, rand_scalar, {n_1 + n_2}));
  }

  // Define mapped_join_keys of length n_1 + n_2
  Value mapped_join_keys;

  // Map the concatenated join keys from table_1 and table_2
  mapped_join_keys = hal::mul(
      ctx, hal::concatenate(ctx, {table_1[0], table_2[0]}, 0), rand_values[0]);
  for (size_t j = 1; j < num_join_keys; j++) {
    mapped_join_keys = hal::add(
        ctx, mapped_join_keys,
        hal::mul(ctx, hal::concatenate(ctx, {table_1[j], table_2[j]}, 0),
                 rand_values[j]));
  }

  // Concatenate mapped_join_keys, and the first n_1 values of mapped_join_keys
  Value mapped_join_keys_concat = hal::concatenate(
      ctx, {mapped_join_keys, hal::slice(ctx, mapped_join_keys, {0}, {n_1})},
      0);

  // Sort mapped_join_keys_concat to get permutation perm
  std::vector<spu::Value> keys_vec{mapped_join_keys_concat};
  auto perm = hal::gen_inv_perm_1d(ctx, absl::MakeSpan(keys_vec),
                                   hal::SortDirection::Ascending, 1, -1);

  // Extract columns from table_1 starting from num_join_keys and create a
  // column of all ones, store in table_1_payloads_and_1
  std::vector<spu::Value> table_1_payloads_and_1;
  for (size_t i = num_join_keys; i < table_1.size(); i++) {
    table_1_payloads_and_1.push_back(table_1[i]);
  }

  auto ones_column =
      hal::seal(ctx, hal::constant(ctx, 1, table_1_payloads_and_1[0].dtype(),
                                   table_1_payloads_and_1[0].shape()));
  table_1_payloads_and_1.push_back(ones_column);

  // Concatenate table_1_payloads_and_1 with n_2 zeros, then concatenate with
  // its negation
  std::vector<spu::Value> table_1_payloads_processed;
  for (size_t i = 0; i < table_1_payloads_and_1.size(); i++) {
    auto zeros_column = hal::seal(
        ctx, hal::constant(ctx, 0, table_1_payloads_and_1[i].dtype(), {n_2}));

    // Extend the column to length [n_1 + n_2]
    auto col_extended =
        hal::concatenate(ctx, {table_1_payloads_and_1[i], zeros_column}, 0);
    // Create negation of table_1_payloads_and_1[i]
    auto zero_broadcasted = hal::seal(
        ctx, hal::constant(ctx, 0, table_1_payloads_and_1[i].dtype(), {n_1}));
    auto col_negated =
        hal::sub(ctx, zero_broadcasted, table_1_payloads_and_1[i]);

    // Concatenate: extended column + negation
    auto col_concat = hal::concatenate(ctx, {col_extended, col_negated}, 0);

    table_1_payloads_processed.push_back(col_concat);
  }

  // Apply inverse permutation perm to table_1_payloads
  std::vector<spu::Value> table_1_payloads_after_perm =
      hal::apply_inv_permute_1d(ctx, absl::MakeSpan(table_1_payloads_processed),
                                perm);

  std::vector<spu::Value> table_1_payloads_after_scan;
  table_1_payloads_after_scan.reserve(n_1 - num_join_keys);
  // Compute prefix sum of table_1_payloads_after_perm
  for (const auto& i : table_1_payloads_after_perm) {
    auto i_prefix_sum = prefix_sum(ctx, i);
    table_1_payloads_after_scan.push_back(i_prefix_sum);
  }

  // Apply permutation perm to table_1_payloads_after_scan
  std::vector<spu::Value> table_1_payloads_after_inv_perm =
      hal::apply_permute_1d(ctx, absl::MakeSpan(table_1_payloads_after_scan),
                            perm);

  // Slice table_1_payloads_after_inv_perm to get rows from n_1 to n_1 + n_2
  std::vector<spu::Value> table_1_payloads_result;
  table_1_payloads_result.reserve(table_1_payloads_after_inv_perm.size());
  for (const auto& i : table_1_payloads_after_inv_perm) {
    auto i_sliced = hal::slice(ctx, i, {n_1}, {n_1 + n_2});
    table_1_payloads_result.push_back(i_sliced);
  }

  // Define output as table_2 and table_1_payloads_result
  std::vector<Value> join_results;
  join_results.reserve(table_2.size() + table_1_payloads_result.size());
  join_results.push_back(table_1_payloads_result.back());  // control bit column
  for (const auto& col : table_2) {
    join_results.push_back(col);
  }
  // push back all but the last column of table_1_payloads_result
  for (size_t i = 0; i < table_1_payloads_result.size() - 1; i++) {
    join_results.push_back(table_1_payloads_result[i]);
  }

  return join_results;
}

}  // namespace spu::kernel::hal