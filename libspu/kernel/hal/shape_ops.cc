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

#include "libspu/core/context.h"
#include "libspu/core/trace.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hal {

Value transpose(SPUContext* ctx, const Value& in, const Axes& permutation) {
  SPU_TRACE_HAL_DISP(ctx, in, permutation);

  return _transpose(ctx, in, permutation);
}

Value slice(SPUContext* ctx, const Value& in, const Index& start_indices,
            const Index& end_indices, const Strides& strides) {
  SPU_TRACE_HAL_DISP(ctx, in, start_indices, end_indices, strides);

  return _extract_slice(ctx, in, start_indices, end_indices, strides);
}

Value slice_scalar_at(SPUContext*, const Value& input, const Index& indices) {
  return Value(input.data().slice_scalar_at(indices), input.dtype());
}

Value update_slice(SPUContext* ctx, const Value& in, const Value& update,
                   const Index& start_indices) {
  SPU_TRACE_HAL_DISP(ctx, in, start_indices);

  if (in.storage_type() != update.storage_type()) {
    auto u =
        _cast_type(ctx, update, in.storage_type()).setDtype(update.dtype());

    return update_slice(ctx, in, u, start_indices);
  }

  return _update_slice(ctx, in, update, start_indices).setDtype(in.dtype());
}

Value reshape(SPUContext* ctx, const Value& in, const Shape& to_shape) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return _reshape(ctx, in, to_shape).setDtype(in.dtype());
}

Value broadcast_to(SPUContext* ctx, const Value& in, const Shape& to_shape,
                   const Axes& in_dims) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return _broadcast(ctx, in, to_shape, in_dims).setDtype(in.dtype());
}

Value reverse(SPUContext* ctx, const Value& in, const Axes& dimensions) {
  SPU_TRACE_HAL_DISP(ctx, in, dimensions);

  return _reverse(ctx, in, dimensions);
}

Value expand(SPUContext* ctx, const Value& in, const Shape& to_shape) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return _fill(ctx, in, to_shape);
}

Value pad(SPUContext* ctx, const Value& in, const Value& padding_value,
          const Sizes& edge_padding_low, const Sizes& edge_padding_high,
          const Sizes& interior_padding) {
  if (in.storage_type() != padding_value.storage_type()) {
    auto ct =
        _common_type(ctx, in.storage_type(), padding_value.storage_type());
    auto normalized_in = _cast_type(ctx, in, ct).setDtype(in.dtype());
    auto normalized_padding_value =
        _cast_type(ctx, padding_value, ct).setDtype(padding_value.dtype());
    return pad(ctx, normalized_in, normalized_padding_value, edge_padding_low,
               edge_padding_high, interior_padding);
  }

  return _pad(ctx, in, padding_value, edge_padding_low, edge_padding_high,
              interior_padding);
}

Value concatenate(SPUContext* ctx, const std::vector<Value>& values,
                  int64_t axis) {
  SPU_TRACE_HAL_DISP(ctx, values, axis);
  SPU_ENFORCE(!values.empty(), "got={}", values.size());

  if (values.size() == 1) {
    // Nothing to concat
    return values.front();
  }

  bool all_same_dtype = std::all_of(
      values.begin() + 1, values.end(),
      [&](const Value& v) { return v.dtype() == values.begin()->dtype(); });
  SPU_ENFORCE(all_same_dtype, "not all element has same dtype");

  bool all_same_stype =
      std::all_of(values.begin() + 1, values.end(), [&](const Value& v) {
        return v.storage_type() == values.begin()->storage_type();
      });

  if (!all_same_stype) {
    Type common_type = values[0].storage_type();
    for (size_t idx = 1; idx < values.size(); idx++) {
      common_type = _common_type(ctx, common_type, values[idx].storage_type());
    }

    std::vector<Value> common_values;
    std::transform(values.cbegin(), values.cend(),
                   std::back_inserter(common_values), [&](const Value& x) {
                     return _cast_type(ctx, x, common_type).setDtype(x.dtype());
                   });

    return concatenate(ctx, common_values, axis);
  }

  SPU_ENFORCE(all_same_stype);

  return _concatenate(ctx, values, axis);
}

}  // namespace spu::kernel::hal
