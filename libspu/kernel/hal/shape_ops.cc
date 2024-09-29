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

MemRef transpose(SPUContext* ctx, const MemRef& in, const Axes& permutation) {
  SPU_TRACE_HAL_DISP(ctx, in, permutation);

  return _transpose(ctx, in, permutation);
}

MemRef slice(SPUContext* ctx, const MemRef& in, const Index& offsets,
             const Shape& sizes, const Strides& strides) {
  SPU_TRACE_HAL_DISP(ctx, in, offsets, sizes, strides);

  return _extract_slice(ctx, in, offsets, sizes, strides);
}

MemRef slice_scalar_at(SPUContext*, const MemRef& input, const Index& indices) {
  return input.slice_scalar_at(indices);
}

MemRef insert_slice(SPUContext* ctx, const MemRef& in, const MemRef& update,
                    const Index& offsets, const Strides& strides,
                    bool prefer_in_place) {
  SPU_TRACE_HAL_DISP(ctx, in, offsets, strides);

  if (in.eltype() != update.eltype()) {
    auto ct = _common_type(ctx, update.eltype(), in.eltype());
    auto i = _cast_type(ctx, in, ct);
    auto u = _cast_type(ctx, update, ct);

    return insert_slice(ctx, i, u, offsets, strides, prefer_in_place);
  }

  return _insert_slice(ctx, in, update, offsets, strides, prefer_in_place);
}

MemRef reshape(SPUContext* ctx, const MemRef& in, const Shape& to_shape) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return _reshape(ctx, in, to_shape);
}

MemRef broadcast_to(SPUContext* ctx, const MemRef& in, const Shape& to_shape,
                    const Axes& in_dims) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return _broadcast(ctx, in, to_shape, in_dims);
}

MemRef reverse(SPUContext* ctx, const MemRef& in, const Axes& dimensions) {
  SPU_TRACE_HAL_DISP(ctx, in, dimensions);

  return _reverse(ctx, in, dimensions);
}

MemRef expand(SPUContext* ctx, const MemRef& in, const Shape& to_shape) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return _fill(ctx, in, to_shape);
}

MemRef pad(SPUContext* ctx, const MemRef& in, const MemRef& padding_value,
           const Sizes& edge_padding_low, const Sizes& edge_padding_high,
           const Sizes& interior_padding) {
  if (in.eltype() != padding_value.eltype()) {
    auto ct = _common_type(ctx, in.eltype(), padding_value.eltype());
    auto normalized_in = _cast_type(ctx, in, ct);
    auto normalized_padding_value = _cast_type(ctx, padding_value, ct);
    return pad(ctx, normalized_in, normalized_padding_value, edge_padding_low,
               edge_padding_high, interior_padding);
  }

  if (!interior_padding.empty()) {
    // Deduce padded shape first
    const auto& input_shape = in.shape();
    Shape out_shape;
    for (size_t i = 0; i < input_shape.size(); i++) {
      out_shape.emplace_back(edge_padding_low[i] + edge_padding_high[i] +
                             interior_padding[i] * (input_shape[i] - 1) +
                             input_shape[i]);
    }

    auto filled = _fill(ctx, padding_value, out_shape);
    Strides insert_strides;
    std::for_each(interior_padding.begin(), interior_padding.end(),
                  [&insert_strides](int64_t idx) {
                    insert_strides.emplace_back(idx + 1);
                  });
    return insert_slice(ctx, filled, in, Index(edge_padding_low),
                        insert_strides, true);
  } else {
    return _pad(ctx, in, padding_value, edge_padding_low, edge_padding_high);
  }
}

MemRef concatenate(SPUContext* ctx, const std::vector<MemRef>& values,
                   int64_t axis) {
  SPU_TRACE_HAL_DISP(ctx, values, axis);
  SPU_ENFORCE(!values.empty(), "got={}", values.size());

  if (values.size() == 1) {
    // Nothing to concat
    return values.front();
  }

  bool all_same_stype = std::all_of(
      values.begin() + 1, values.end(),
      [&](const MemRef& v) { return v.eltype() == values.begin()->eltype(); });

  if (!all_same_stype) {
    Type common_type = values[0].eltype();
    for (size_t idx = 1; idx < values.size(); idx++) {
      common_type = _common_type(ctx, common_type, values[idx].eltype());
    }

    std::vector<MemRef> common_values;
    std::transform(
        values.cbegin(), values.cend(), std::back_inserter(common_values),
        [&](const MemRef& x) { return _cast_type(ctx, x, common_type); });

    return concatenate(ctx, common_values, axis);
  }

  SPU_ENFORCE(all_same_stype);

  return _concatenate(ctx, values, axis);
}

MemRef squeeze(SPUContext* ctx, const MemRef& in, int64_t dim) {
  SPU_ENFORCE(dim >= 0 && dim < in.shape().ndim(),
              "input shape {} and squeezing dim {} are mismatched", in.shape(),
              dim);
  Shape new_shape = in.shape();
  if (new_shape[dim] == 1) {
    new_shape.erase(new_shape.begin() + dim);
    return hal::reshape(ctx, in, new_shape);
  }
  return in;
}

MemRef unsqueeze(SPUContext* ctx, const MemRef& in, int64_t dim) {
  SPU_ENFORCE(dim >= 0 && dim <= in.shape().ndim(),
              "input shape {} and unsqueezing dim {} are mismatched",
              in.shape(), dim);
  Shape new_shape = in.shape();
  new_shape.insert(new_shape.begin() + dim, 1);
  return hal::reshape(ctx, in, new_shape);
}

}  // namespace spu::kernel::hal
