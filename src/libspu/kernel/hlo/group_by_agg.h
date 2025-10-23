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

#pragma once

#include "libspu/kernel/hal/group_by_agg.h"

namespace spu::kernel::hlo {

enum class GroupByAggMode {
  // Ref: https://ieeexplore.ieee.org/document/9835540
  // Scape: Scalable Collaborative Analytics System on Private Database with
  // Malicious Security
  //
  // Using prefix-sum network
  PrefixSumMode,
  // Ref: https://eprint.iacr.org/2024/141
  // Secure Statistical Analysis on Multiple Datasets: Join and Group-By
  // Only use sort + permute
  DirectMode,
  // Automatically choose the best mode (heuristic)
  AutoMode,
};

enum class OutputFormat {
  // sort by keys
  GroupedOrder,
  // sort by keys, but only one of each key kept
  OutputOrder,
};

struct GroupByAggOptions {
  // if true, then the valid bits also cover the bits of payloads
  bool valid_bits_include_payloads = false;
  // if true, fill the rest of keys or payloads with dummy values
  // the dummy value will be set to 1000 0000 ... 0000 which is not used under
  // the encoding scheme of SPU
  bool key_fill_with_dummy = false;
  bool payload_fill_with_dummy = false;
  // if true, only output the aggregated payloads
  // you can set this to true when you can generate the keys locally
  bool drop_keys = false;
  GroupByAggMode mode = GroupByAggMode::AutoMode;
  OutputFormat output_format = OutputFormat::OutputOrder;
};

enum class AggFunc {
  Sum,
  Count,
  Avg,
  Max,
  Min,
  Percentile,
};

// TODO(zjj): add unsigned hint after implementing unsigned optimization of sort
// Note: we only support 1d keys and payloads now.
std::vector<Value> groupby_agg(SPUContext* ctx,
                               absl::Span<spu::Value const> keys,
                               absl::Span<spu::Value const> payloads,
                               AggFunc agg_func, absl::Span<int64_t> valid_bits,
                               const GroupByAggOptions& options);

}  // namespace spu::kernel::hlo