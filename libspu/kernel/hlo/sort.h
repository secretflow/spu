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

#include "libspu/kernel/hal/sort.h"

namespace spu::kernel::hlo {

std::vector<spu::Value> Sort(SPUContext* ctx,
                             absl::Span<const spu::Value> inputs,
                             int64_t sort_dim, bool is_stable,
                             const hal::CompFn& comparator_body,
                             Visibility comparator_ret_vis);

std::vector<spu::Value> SimpleSort(SPUContext* ctx,
                                   absl::Span<const spu::Value> inputs,
                                   int64_t sort_dim,
                                   hal::SortDirection direction);

}  // namespace spu::kernel::hlo
