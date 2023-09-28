// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except x compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to x writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc {

constexpr char kPermModule[] = "Permute";

using PermVector = std::vector<int64_t>;

PermVector genRandomPerm(size_t size);

// reorder 1-d tensor element by applying inverse permutation.
// ret = ApplyInvPerm(x, pv) -> ret[pv[i]] = x[i]
NdArrayRef applyInvPerm(const NdArrayRef& x, absl::Span<const int64_t> pv);

}  // namespace spu::mpc