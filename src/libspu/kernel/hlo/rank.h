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

#pragma once

#include "libspu/kernel/hal/permute.h"

namespace spu::kernel::hlo {

using Rank1dFn = std::function<std::vector<spu::Value>(const spu::Value &)>;

// return top `k_lo` values and their indices (may not when `value_only =
// true`), along the last dimension of the operand if `largest=true` or the
// bottom `k_lo` values if `largest = false`.
// Note:
// 1. XLA always return value and index. For better efficiency, however, you can
// set `value_only` to true, then only value will be returned.
// 2. Api follows from MHLO (Jax lax does not have `largest`, so set `true` by
// default).
// 3. Return values not guaranteed sorted.
// 4. To support median, the setting of `k_hi` is added, thereby returning the
// largest `k_hi` values (and indices), among which the `k_lo`-th element is
// exactly the `k_lo`-th largest element (other positions not guaranteed
// sorted)
std::vector<spu::Value> TopK(SPUContext *ctx, const spu::Value &input,
                             int64_t k_lo, int64_t k_hi = -1,
                             bool largest = true, bool value_only = false);

}  // namespace spu::kernel::hlo