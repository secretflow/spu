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

#include "absl/types/span.h"

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

// A quick path for groupby sum, when:
//   1. all keys are private and have the same owner
//
// We take an example of single key groupby sum to illustrate the idea:
//  keys:    [1, 3, 1, 3, 2] , payload: [2, 4, 3, 5, 1]
//  result:  [1, 2, 3, x, x] , payload: [5, 1, 9, x, x] (x means don't care)
//
// Steps:
// 1. Compute k_g = [1,1,2,3,3] , the group marks e = [0,1,1,0,1] (locally)
// 2. Compute v_g = [2,3,1,4,5] (need one `inv_perm_xv`, x relying on the
// visibility of payloads)
// 3. Compute w_g = prefix_sum(v_g) = [2,5,6,10,15] (locally)
// 4. Sort e descending (locally) and permute w_g accordingly (need one
// `inv_perm_xv`), i.e. y = [5,6,15,x,x]
// 5. Compute s =  y - right_shift(y)  = y - [0,5,6,15,x] = [5,1,9,x,x]
//
// Note: As the keys are private, the caller can extract the output keys and
// valid groupby payloads by itself
//
// Warning: we skip the sanity checks here, which should be done by the caller
// if the visibility requirements are not met, the performance may be degraded.
std::vector<Value> private_groupby_sum_1d(
    SPUContext *ctx, absl::Span<spu::Value const> keys,
    absl::Span<spu::Value const> payloads);

// A quick path for groupby avg, nearly the same as groupby sum, except the last
// step:
//   instead of directly returning the sum results, we need to divide the sum
//   results with the group sizes (which can be computed with group marks)

// Note: As the keys are private, the caller can extract the output keys and
// valid groupby payloads by itself
//
// Warning:
// 1. we skip the sanity checks here, which should be done by the caller
// if the visibility requirements are not met, the performance may be degraded.
// 2. if unsafe_drop_rest is set to true, then THE UNIQUE COUNT OF KEYS must be
// revealed to all parties, but the performance may be much better if there are
// many duplicated keys.
std::vector<Value> private_groupby_avg_1d(SPUContext *ctx,
                                          absl::Span<spu::Value const> keys,
                                          absl::Span<spu::Value const> payloads,
                                          bool unsafe_drop_rest = false);
}  // namespace spu::kernel::hal
