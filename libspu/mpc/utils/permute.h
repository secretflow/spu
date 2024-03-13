// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc {

constexpr char kPermModule[] = "Permute";

using PermVector = std::vector<int64_t>;

PermVector genRandomPerm(size_t size, uint64_t seed);

PermVector genInversePerm(absl::Span<const int64_t> pv);

// generate permutation vector that can make x ordered
PermVector genPermBySort(const NdArrayRef& x);

// reorder 1-d tensor element by applying inverse permutation.
// ret = ApplyInvPerm(x, pv) -> ret[pv[i]] = x[i]
NdArrayRef applyInvPerm(const NdArrayRef& x, absl::Span<const int64_t> pv);

// reorder 1-d tensor element by applying permutation.
// ret = ApplyPerm(x, pv) -> ret[i] = x[pv[i]]
NdArrayRef applyPerm(const NdArrayRef& x, absl::Span<const int64_t> pv);

// get a permutation vector from a ring
PermVector ring2pv(const NdArrayRef& x);

}  // namespace spu::mpc