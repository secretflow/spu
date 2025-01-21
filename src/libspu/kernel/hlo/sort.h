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

#pragma once

#include "libspu/kernel/hal/permute.h"

namespace spu::kernel::hlo {

// This API corresponds to XLA's Sort
std::vector<spu::Value> Sort(SPUContext* ctx,
                             absl::Span<const spu::Value> inputs,
                             int64_t sort_dim, bool is_stable,
                             const hal::CompFn& comparator_body,
                             Visibility comparator_ret_vis);

// This API is a simplified sort operation which only supports
// ascending/descending order. The num_keys parameter indicates the number of
// operands to treat as sort keys. The valid_bits parameter indicates the
// numeric range of keys for performance hint. Currently, for SPU lowering path
// of SimpleSort, the num_keys is always 1.
std::vector<spu::Value> SimpleSort(SPUContext* ctx,
                                   absl::Span<const spu::Value> inputs,
                                   int64_t sort_dim,
                                   hal::SortDirection direction,
                                   int64_t num_keys = 1,
                                   int64_t valid_bits = -1);

}  // namespace spu::kernel::hlo
