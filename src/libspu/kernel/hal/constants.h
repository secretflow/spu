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
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/shape.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {

// Returns a SPU value from given C/C++ buffer.
//
// The result visibility is public.
Value constant(SPUContext* ctx, PtBufferView init, DataType dtype,
               const Shape& shape = {},
               FieldType perm_field = FieldType::FT_INVALID);

// Returns a SPU zero value, which is equal to
//  constant(ctx, 0, dtype, shape);
//
// The result visibility is public.
Value zeros(SPUContext* ctx, DataType dtype, const Shape& shape = {});

// Returns a one-dimensional value.
//
// The result visibility is public.
Value iota(SPUContext* ctx, DataType dtype, int64_t numel,
           FieldType perm_field = FieldType::FT_INVALID);

// Returns the SPU epsilon, the positive distance between two fixed point value.
//
// The result visibility is public.
Value epsilon(SPUContext* ctx, DataType dtype, const Shape& shape = {});

}  // namespace spu::kernel::hal
