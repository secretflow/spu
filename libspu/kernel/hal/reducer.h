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

#include <vector>

#include "absl/types/span.h"

#include "libspu/core/memref.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

using BatchedMemRefBinaryFn = std::function<std::vector<spu::MemRef>(
    absl::Span<spu::MemRef const> lhs, absl::Span<spu::MemRef const> rhs)>;

using BroadcastCallbackFcn = std::function<void(const Shape &)>;

std::vector<spu::MemRef> Reduce(SPUContext *ctx,
                                absl::Span<const spu::MemRef> inputs,
                                absl::Span<const spu::MemRef> init_values,
                                const Axes &dims_to_reduce,
                                const BatchedMemRefBinaryFn &reducer,
                                const BroadcastCallbackFcn &bcaster,
                                bool ignore_init_values = false);

}  // namespace spu::kernel::hal
