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

NdArrayRef genInversePerm(const NdArrayRef& perm);

// generate permutation vector that can make x ordered
Index genPermBySort(const NdArrayRef& x);

// reorder 1-d tensor element by applying inverse permutation.
// ret = ApplyInvPerm(x, pv) -> ret[pv[i]] = x[i]
NdArrayRef applyInvPerm(const NdArrayRef& x, absl::Span<const int64_t> pv);
NdArrayRef applyInvPerm(const NdArrayRef& x, const NdArrayRef& pv);

// reorder 1-d tensor element by applying permutation.
// ret = ApplyPerm(x, pv) -> ret[i] = x[pv[i]]
NdArrayRef applyPerm(const NdArrayRef& x, absl::Span<const int64_t> pv);
NdArrayRef applyPerm(const NdArrayRef& x, const NdArrayRef& pv);

// pv can not be the bijection.
// the shape of x and pv can be different.
// e.g.
// \pi=(3,2,3,6,3,6), X = (4,1,8,2,7,9,5,5)
// then \pi(X) = (2,8,2,5,2,5)
NdArrayRef generalApplyPerm(const NdArrayRef& x, const NdArrayRef& pv);

// perm: [n] -> [m], po: [m] -> [m] (a bijection)
// got pr: [n] -> [m]
// s.t. perm = po ∘ pr
NdArrayRef solvePerm(const NdArrayRef& perm, absl::Span<const int64_t> po);

// get a permutation vector from a ring
Index ring2pv(const NdArrayRef& x);

Index genRandomPerm(size_t numel, uint128_t seed, uint64_t* ctr);
}  // namespace spu::mpc