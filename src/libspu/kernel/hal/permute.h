// Copyright 2023 Ant Group Co., Ltd.
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

#pragma once

#include "absl/types/span.h"

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

struct TopKConfig {
  bool value_only;  // only return values
  bool confusion;   // add random noise to hide the data-dependant pattern
  int64_t k_lo;     // the `k_lo`-th largest element order is guaranteed
  int64_t k_hi;     // returning the largest `k_hi` values (or with indices)
};

using SimpleCompFn = std::function<spu::Value(SPUContext *, const spu::Value &,
                                              const spu::Value &)>;

using CompFn = std::function<spu::Value(absl::Span<const spu::Value>)>;

using Permute1dFn =
    std::function<std::vector<spu::Value>(absl::Span<const spu::Value>)>;

// sort direction for sorters without comparators
enum class SortDirection {
  Ascending,
  Descending,
};

// general sort1d with comparator
std::vector<spu::Value> sort1d(SPUContext *ctx,
                               absl::Span<spu::Value const> inputs,
                               const CompFn &cmp, Visibility comparator_ret_vis,
                               bool is_stable);

// simple sort1d.
//
// Inputs:
//  - inputs: a vector of 1-D operands to be sorted
//  - direction: sorting order
//  - num_keys: the number of operands to treat as keys (count from index 0)
//  - valid_bits: indicates the numeric range of keys for performance hint
std::vector<spu::Value> simple_sort1d(SPUContext *ctx,
                                      absl::Span<spu::Value const> inputs,
                                      SortDirection direction, int64_t num_keys,
                                      int64_t valid_bits);

// transform n-d permute to 1-d permute and applying permute function to each
// 1-d array
std::vector<spu::Value> permute(SPUContext *ctx,
                                absl::Span<const spu::Value> inputs,
                                int64_t permute_dim,
                                const Permute1dFn &permute_fn);

// general topk1d
// Inputs:
//  -inputs: an 1-D operand to search top k elements
//  -scalar_cmp: comparison function for single value
//  -config: topk config
std::vector<Value> topk_1d(SPUContext *ctx, const spu::Value &input,
                           const SimpleCompFn &scalar_cmp,
                           const TopKConfig &config);

// For each input x, we get y = perm^{-1} (x), i.e. y[i] = x[perm^{-1}(i)]
std::vector<spu::Value> apply_inv_permute_1d(
    SPUContext *ctx, absl::Span<const spu::Value> inputs,
    const spu::Value &perm);

// For each input x, we get y = perm(x), i.e. y[i] = x[perm(i)]
std::vector<spu::Value> apply_permute_1d(SPUContext *ctx,
                                         absl::Span<const spu::Value> inputs,
                                         const spu::Value &perm);

}  // namespace spu::kernel::hal

namespace spu::kernel::hal::internal{
std::vector<spu::Value> apply_inv_perm(SPUContext *ctx,
                                       absl::Span<spu::Value const> inputs,
                                       const spu::Value &perm);

}