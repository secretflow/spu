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

#include "libspu/core/memref.h"

namespace spu::mpc {

MemRef genInversePerm(const MemRef& perm);

// generate permutation vector that can make x ordered
Index genPermBySort(const MemRef& x);

// reorder 1-d memref element by applying inverse permutation.
// ret = ApplyInvPerm(x, pv) -> ret[pv[i]] = x[i]
MemRef applyInvPerm(const MemRef& x, absl::Span<const int64_t> pv);
MemRef applyInvPerm(const MemRef& x, const MemRef& pv);

// reorder 1-d memref element by applying permutation.
// ret = ApplyPerm(x, pv) -> ret[i] = x[pv[i]]
MemRef applyPerm(const MemRef& x, absl::Span<const int64_t> pv);
MemRef applyPerm(const MemRef& x, const MemRef& pv);

// get a permutation vector from a ring
Index ring2pv(const MemRef& x);

Index genRandomPerm(size_t numel, uint128_t seed, uint64_t* ctr);
}  // namespace spu::mpc