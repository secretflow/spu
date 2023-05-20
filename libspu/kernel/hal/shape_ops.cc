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
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::kernel::hal {

Value transpose(SPUContext* ctx, const Value& in,
                absl::Span<const int64_t> permutation) {
  SPU_TRACE_HAL_DISP(ctx, in);

  // compact clone is a rather expensive memory operation.
  // To prevent transposed value being cloned multiple times in later ops, clone
  // the value here.
  return Value(in.data().transpose(permutation), in.dtype()).clone();
}

Value slice(SPUContext* ctx, const Value& in,
            absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> end_indices,
            absl::Span<const int64_t> strides) {
  SPU_TRACE_HAL_DISP(ctx, in, start_indices, end_indices, strides);

  return Value(in.data().slice(start_indices, end_indices, strides),
               in.dtype());
}

Value slice_scalar_at(SPUContext* ctx, const Value& input,
                      absl::Span<const int64_t> indices) {
  return Value(input.data().slice_scalar_at(indices), input.dtype());
}

Value update_slice(SPUContext* ctx, const Value& in, const Value& update,
                   absl::Span<const int64_t> start_indices) {
  auto ret = in.clone();
  auto u = stype_cast(ctx, update, ret.storage_type());
  ret.data().update_slice(u.data(), start_indices);
  return ret;
}

Value reshape(SPUContext* ctx, const Value& in,
              absl::Span<const int64_t> to_shape) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return Value(in.data().reshape(to_shape), in.dtype());
}

Value broadcast_to(SPUContext* ctx, const Value& in,
                   absl::Span<const int64_t> to_shape,
                   absl::Span<const int64_t> in_dims) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return Value(in.data().broadcast_to(to_shape, in_dims), in.dtype());
}

Value reverse(SPUContext* ctx, const Value& in,
              absl::Span<const int64_t> dimensions) {
  SPU_TRACE_HAL_DISP(ctx, in, dimensions);

  return Value(in.data().reverse(dimensions), in.dtype());
}

Value expand(SPUContext* ctx, const Value& in,
             absl::Span<const int64_t> to_shape) {
  return Value(in.data().expand(to_shape), in.dtype());
}

Value pad(SPUContext* ctx, const Value& in, const Value& padding_value,
          absl::Span<const int64_t> edge_padding_low,
          absl::Span<const int64_t> edge_padding_high,
          absl::Span<const int64_t> interior_padding) {
  if (in.storage_type() != padding_value.storage_type()) {
    auto ct =
        _common_type(ctx, in.storage_type(), padding_value.storage_type());
    auto normalized_in = _cast_type(ctx, in, ct).setDtype(in.dtype());
    auto normalized_padding_value =
        _cast_type(ctx, padding_value, ct).setDtype(padding_value.dtype());
    return pad(ctx, normalized_in, normalized_padding_value, edge_padding_low,
               edge_padding_high, interior_padding);
  }

  return Value(in.data().pad(padding_value.data(), edge_padding_low,
                             edge_padding_high, interior_padding),
               in.dtype());
}

Value concatenate(SPUContext* ctx, absl::Span<const Value> values,
                  const size_t& axis) {
  SPU_TRACE_HAL_DISP(ctx, axis);
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

  std::vector<NdArrayRef> array(values.size() - 1);
  for (int64_t idx = 1; idx < static_cast<int64_t>(values.size()); ++idx) {
    array[idx - 1] = values[idx].data();
  }

  return Value(values[0].data().concatenate(array, axis), values[0].dtype());
}

}  // namespace spu::kernel::hal
