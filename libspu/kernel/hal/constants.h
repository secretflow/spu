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

#include "libspu/core/pt_buffer_view.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hal {

// TODO: move this to shape_utils.
using ShapeView = absl::Span<int64_t const>;

// Returns a SPU value from given C/C++ buffer.
//
// The result visibility is public.
Value constant(HalContext* ctx, PtBufferView init, DataType dtype,
               ShapeView shape = {});

// Returns a SPU zero value, which is equal to
//  constant(ctx, 0, dtype, shape);
//
// The result visibility is public.
Value zeros(HalContext* ctx, DataType dtype, ShapeView shape = {});

// Returns a one-dimentional value.
//
// The result visibility is public.
Value iota(HalContext* ctx, DataType dtype, int64_t numel);

// Returns the SPU epsilon, the positive distance between two fixed point value.
//
// The result visibility is public.
Value epsilon(HalContext* ctx, absl::Span<const int64_t> shape = {});

}  // namespace spu::kernel::hal
