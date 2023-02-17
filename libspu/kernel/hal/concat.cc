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

#include "libspu/kernel/hal/concat.h"

#include <cstdint>
#include <numeric>

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/shape_util.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {
namespace {

void calcNextPtrs(std::vector<int64_t>& coord, int64_t& idim,
                  const std::vector<int64_t>& shape, const std::byte*& ptr_a,
                  const std::vector<int64_t>& strides_a, std::byte*& ptr_b,
                  const std::vector<int64_t>& strides_b) {
  for (idim = shape.size() - 1; idim >= 0; --idim) {
    if (++coord[idim] == shape[idim]) {
      // Once a dimension is done, just unwind by strides
      coord[idim] = 0;
      ptr_a -= (shape[idim] - 1) * strides_a[idim];
      ptr_b -= (shape[idim] - 1) * strides_b[idim];
    } else {
      ptr_a += strides_a[idim];
      ptr_b += strides_b[idim];
      break;
    }
  }
}

}  // namespace

Value concatenate(HalContext* ctx, absl::Span<const Value> values,
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

  std::vector<int64_t> result_shape = values.front().shape();
  for (size_t idx = 1; idx < values.size(); ++idx) {
    result_shape[axis] += values[idx].shape()[axis];
  }

  // Preallocate output buffer
  Value result({values.front().storage_type(), result_shape},
               values.front().dtype());
  auto elsize = result.elsize();

  // Generating slices
  std::vector<Value> result_slices(values.size());
  {
    std::vector<int64_t> start(result_shape.size(), 0);
    std::vector<int64_t> end = result_shape;
    std::vector<int64_t> strides(result_shape.size(), 1);
    for (size_t idx = 0; idx < values.size(); ++idx) {
      end[axis] = start[axis] + values[idx].shape()[axis];
      result_slices[idx] = slice(ctx, result, start, end, strides);
      std::swap(start[axis], end[axis]);
    }
  }

  for (size_t idx = 0; idx < values.size(); ++idx) {
    const auto* from_ptr =
        static_cast<const std::byte*>(values[idx].data().data());
    auto* to_ptr = static_cast<std::byte*>(result_slices[idx].data().data());

    std::vector<int64_t> from_strides = values[idx].strides();
    std::vector<int64_t> to_strides = result_slices[idx].strides();
    std::vector<int64_t> shape = values[idx].shape();

    // try optimize memcpy by make larger block size.
    auto compact_strides = makeCompactStrides(shape);
    int64_t blksize = elsize;
    int64_t ndims = shape.size() - 1;
    for (; ndims >= 0; ndims--) {
      if (from_strides[ndims] == to_strides[ndims] &&
          from_strides[ndims] == compact_strides[ndims]) {
        blksize *= shape[ndims];
        shape[ndims] = 1;
      } else {
        break;
      }
    }
    ndims++;
    shape.resize(ndims);
    from_strides.resize(ndims);
    to_strides.resize(ndims);

    // convert to byte based stride.
    for (size_t dim = 0; dim < from_strides.size(); dim++) {
      from_strides[dim] *= elsize;
      to_strides[dim] *= elsize;
    }

    int64_t idim = ndims - 1;
    std::vector<int64_t> indicies(ndims, 0);
    do {
      std::copy_n(from_ptr, blksize, to_ptr);
      calcNextPtrs(indicies, idim, shape, from_ptr, from_strides, to_ptr,
                   to_strides);
    } while (idim >= 0);
  }

  return result;
}

}  // namespace spu::kernel::hal
