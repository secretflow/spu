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

#include "libspu/kernel/hal/shape_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/shape_util.h"
#include "libspu/core/vectorize.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::kernel::hal {

Value transpose(HalContext* ctx, const Value& in,
                absl::Span<const int64_t> permutation) {
  SPU_TRACE_HAL_DISP(ctx, in);

  // compact clone is a rather expensive memory operation.
  // To prevent transposed value being cloned multiple times in later ops, clone
  // the value here.
  return Value(in.data().transpose(permutation), in.dtype()).clone();
}

Value slice(HalContext* ctx, const Value& in,
            absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> end_indices,
            absl::Span<const int64_t> strides) {
  SPU_TRACE_HAL_DISP(ctx, in, start_indices, end_indices, strides);

  return Value(in.data().slice(start_indices, end_indices, strides),
               in.dtype());
}

Value slice_scalar_at(HalContext* ctx, const Value& input,
                      absl::Span<const int64_t> indices) {
  return Value(input.data().slice_scalar_at(indices), input.dtype());
}

Value update_slice(HalContext* ctx, const Value& in, const Value& update,
                   absl::Span<const int64_t> start_indices) {
  auto ret = in.clone();
  auto u = stype_cast(ctx, update, ret.storage_type());
  ret.data().update_slice(u.data(), start_indices);
  return ret;
}

Value reshape(HalContext* ctx, const Value& in,
              absl::Span<const int64_t> to_shape) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return Value(in.data().reshape(to_shape), in.dtype());
}

Value broadcast_to(HalContext* ctx, const Value& in,
                   absl::Span<const int64_t> to_shape,
                   absl::Span<const int64_t> in_dims) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return Value(in.data().broadcast_to(to_shape, in_dims), in.dtype());
}

Value reverse(HalContext* ctx, const Value& in,
              absl::Span<const int64_t> dimensions) {
  SPU_TRACE_HAL_DISP(ctx, in, dimensions);

  return Value(in.data().reverse(dimensions), in.dtype());
}

Value expand(HalContext* ctx, const Value& in,
             absl::Span<const int64_t> to_shape) {
  return Value(in.data().expand(to_shape), in.dtype());
}

}  // namespace spu::kernel::hal
