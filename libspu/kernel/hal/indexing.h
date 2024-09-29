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

#include "libspu/core/memref.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

MemRef DynamicUpdateSlice(SPUContext *ctx, const MemRef &operand,
                          const MemRef &update,
                          absl::Span<const MemRef> start_indices,
                          bool prefer_in_place);

MemRef DynamicSlice(SPUContext *ctx, const MemRef &operand,
                    const Sizes &slice_size,
                    absl::Span<const MemRef> start_indices);

}  // namespace spu::kernel::hal
