// Copyright 2021 Ant Group Co., Ltd.
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

#include "libspu/core/context.h"
#include "libspu/core/memref.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/shape.h"

namespace spu::kernel::hal {

// Returns a SPU value from given C/C++ buffer.
//
// The result visibility is public.
MemRef constant(SPUContext* ctx, PtBufferView init, const Shape& shape = {});

// Returns a SPU zero value, which is equal to
//  constant(ctx, 0, dtype, shape);
//
// The result visibility is public.
MemRef zeros(SPUContext* ctx, PtType pt_type, const Shape& shape = {});

// Returns a one-dimensional value.
//
// The result visibility is public.
MemRef iota(SPUContext* ctx, PtType pt_type, int64_t numel);

}  // namespace spu::kernel::hal
