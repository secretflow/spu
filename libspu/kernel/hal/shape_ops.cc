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

#include "libspu/core/ndarray_ref.h"
#include "libspu/kernel/hal/prot_wrapper.h"

namespace spu::kernel::hal {
namespace {

// TODO: these code is copied from ring.cc, remove it when shape ops is lowered
// to mpc layer.
Type _common_type(SPUContext* ctx, const Type& a, const Type& b) {
  if (a.isa<Secret>() && b.isa<Secret>()) {
    return _common_type_s(ctx, a, b);
  } else if (a.isa<Secret>()) {
    return a;
  } else if (b.isa<Secret>()) {
    return b;
  } else {
    SPU_ENFORCE(a.isa<Public>() && b.isa<Public>());
    return a;
  }
}

Value _cast_type(SPUContext* ctx, const Value& x, const Type& to) {
  if (x.storage_type() == to) {
    return x;
  }
  if (x.isPublic() && to.isa<Public>()) {
    return x;
  } else if (x.isPublic() && to.isa<Secret>()) {
    // FIXME: casting to BShare semantic is wrong.
    return _p2s(ctx, x);
  } else if (x.isSecret() && to.isa<Secret>()) {
    return _cast_type_s(ctx, x, to);
  } else {
    SPU_THROW("show not be here x={}, to={}", x, to);
  }
}

}  // namespace

Value transpose(SPUContext* ctx, const Value& in, const Axes& permutation) {
  SPU_TRACE_HAL_DISP(ctx, in);

  Axes perm = permutation;
  if (perm.empty()) {
    // by default, transpose the data in reverse order.
    perm.resize(in.shape().size());
    std::iota(perm.rbegin(), perm.rend(), 0);
  }

  // sanity check.
  SPU_ENFORCE_EQ(perm.size(), in.shape().size());
  std::set<int64_t> uniq(perm.begin(), perm.end());
  SPU_ENFORCE_EQ(uniq.size(), perm.size(), "perm={} is not unique", perm);

  // fast path, if identity permutation, return it.
  Axes no_perm(in.shape().size());
  std::iota(no_perm.begin(), no_perm.end(), 0);
  if (perm == no_perm) {
    return in;
  }

  return Value(in.data().transpose(perm), in.dtype());
}

Value slice(SPUContext* ctx, const Value& in, const Index& start_indices,
            const Index& end_indices, const Strides& strides) {
  SPU_TRACE_HAL_DISP(ctx, in, start_indices, end_indices, strides);

  return Value(in.data().slice(start_indices, end_indices, strides),
               in.dtype());
}

Value slice_scalar_at(SPUContext*, const Value& input, const Index& indices) {
  return Value(input.data().slice_scalar_at(indices), input.dtype());
}

Value update_slice(SPUContext* ctx, const Value& in, const Value& update,
                   const Index& start_indices) {
  if (in.storage_type() != update.storage_type()) {
    auto u =
        _cast_type(ctx, update, in.storage_type()).setDtype(update.dtype());

    return update_slice(ctx, in, u, start_indices);
  }

  auto ret = in.clone();
  ret.data().update_slice(update.data(), start_indices);
  return ret;
}

Value reshape(SPUContext* ctx, const Value& in, const Shape& to_shape) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return Value(in.data().reshape(to_shape), in.dtype());
}

Value broadcast_to(SPUContext* ctx, const Value& in, const Shape& to_shape,
                   const Axes& in_dims) {
  SPU_TRACE_HAL_DISP(ctx, in, to_shape);

  return Value(in.data().broadcast_to(to_shape, in_dims), in.dtype());
}

Value reverse(SPUContext* ctx, const Value& in, const Axes& dimensions) {
  SPU_TRACE_HAL_DISP(ctx, in, dimensions);

  return Value(in.data().reverse(dimensions), in.dtype());
}

Value expand(SPUContext*, const Value& in, const Shape& to_shape) {
  return Value(in.data().expand(to_shape), in.dtype());
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

  return Value(in.data().pad(padding_value.data(), edge_padding_low,
                             edge_padding_high, interior_padding),
               in.dtype());
}

Value concatenate(SPUContext* ctx, absl::Span<const Value> values,
                  int64_t axis) {
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
