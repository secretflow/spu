// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/sort.h"

#include "libspu/kernel/hal/permute.h"

namespace spu::kernel::hlo {

std::vector<spu::Value> Sort(SPUContext *ctx,
                             absl::Span<const spu::Value> inputs,
                             int64_t sort_dim, bool is_stable,
                             const hal::CompFn &comparator_body,
                             Visibility comparator_ret_vis) {
  auto sort_fn = [&](absl::Span<const spu::Value> input) {
    return hal::sort1d(ctx, input, comparator_body, comparator_ret_vis,
                       is_stable);
  };
  return hal::permute(ctx, inputs, sort_dim, sort_fn);
}

std::vector<spu::Value> SimpleSort(SPUContext *ctx,
                                   absl::Span<const spu::Value> inputs,
                                   int64_t sort_dim,
                                   hal::SortDirection direction,
                                   int64_t num_keys, int64_t valid_bits,
                                   bool is_stable) {
  auto sort_fn = [&](absl::Span<const spu::Value> input) {
    return hal::simple_sort1d(ctx, input, direction, num_keys, valid_bits,
                              is_stable);
  };
  return hal::permute(ctx, inputs, sort_dim, sort_fn);
}

}  // namespace spu::kernel::hlo
