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
#include "libspu/core/shape_util.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {

// Returns a SPU value from given C/C++ buffer.
//
// The result visibility is public.
Value constant(SPUContext* ctx, PtBufferView init, DataType dtype,
               ShapeView shape = {});

// Returns a SPU zero value, which is equal to
//  constant(ctx, 0, dtype, shape);
//
// The result visibility is public.
Value zeros(SPUContext* ctx, DataType dtype, ShapeView shape = {});

// Returns a one-dimentional value.
//
// The result visibility is public.
Value iota(SPUContext* ctx, DataType dtype, int64_t numel);

// Returns the SPU epsilon, the positive distance between two fixed point value.
//
// The result visibility is public.
Value epsilon(SPUContext* ctx, DataType dtype,
              absl::Span<const int64_t> shape = {});

}  // namespace spu::kernel::hal
