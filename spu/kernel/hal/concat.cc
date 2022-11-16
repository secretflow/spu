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

#include "spu/kernel/hal/concat.h"

#include <cstdint>
#include <numeric>

#include "yasl/base/exception.h"

#include "spu/core/parallel_utils.h"
#include "spu/core/shape_util.h"
#include "spu/kernel/hal/ring.h"
#include "spu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {

Value concatenate(HalContext* ctx, absl::Span<const Value> values,
                  const size_t& axis) {
  SPU_TRACE_HAL(ctx, axis);
  YASL_ENFORCE(!values.empty(), "got={}", values.size());

  bool all_same_dtype = std::all_of(
      values.begin() + 1, values.end(),
      [&](const Value& v) { return v.dtype() == values.begin()->dtype(); });
  YASL_ENFORCE(all_same_dtype, "not all element has same dtype");

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
    std::transform(
        values.cbegin(), values.cend(), std::back_inserter(common_values),
        [&](const Value& x) { return _cast_type(ctx, x, common_type); });

    return concatenate(ctx, common_values, axis).setDtype(values[0].dtype());
  }

  YASL_ENFORCE(all_same_stype);

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

  auto next_two_iter_ =
      [&](std::vector<int64_t>& coord, int64_t& idim,
          absl::Span<const int64_t> shape, const std::byte*& ptr_a,
          absl::Span<const int64_t> strides_a, std::byte*& ptr_b,
          absl::Span<const int64_t> strides_b) {
        for (idim = shape.size() - 1; idim >= 0; --idim) {
          if (++coord[idim] == shape[idim]) {
            // Once a dimension is done, just unwind by strides
            coord[idim] = 0;
            ptr_a -= (shape[idim] - 1) * elsize * strides_a[idim];
            ptr_b -= (shape[idim] - 1) * elsize * strides_b[idim];
          } else {
            ptr_a += strides_a[idim] * elsize;
            ptr_b += strides_b[idim] * elsize;
            break;
          }
        }
      };

  // 5 here is just a magic number
  if (values.size() < 5) {
    // When there are just a few values to concat. Try to parallel on value
    // level...
    for (size_t idx = 0; idx < values.size(); ++idx) {
      auto g_size = std::max<int64_t>(
          (values[idx].numel() + getNumberOfProc()) / getNumberOfProc(), 2048);
      yasl::parallel_for(
          0, values[idx].numel(), g_size, [&](int64_t begin, int64_t end) {
            std::vector<int64_t> indicies =
                unflattenIndex(begin, values[idx].shape());
            int64_t idim = values[idx].shape().size() - 1;
            const auto* in_ptr = &values[idx].data().at(indicies);
            auto* to_ptr = &result_slices[idx].data().at(indicies);
            for (int64_t e_idx = begin; e_idx < end; ++e_idx) {
              std::memcpy(to_ptr, in_ptr, elsize);
              next_two_iter_(indicies, idim, values[idx].shape(), in_ptr,
                             values[idx].strides(), to_ptr,
                             result_slices[idx].strides());
            }
          });
    }
  } else {
    // When there are a lot of values to concat (usually during im2col, where
    // each value is just a window), try to parallel on inputs
    yasl::parallel_for(
        0, values.size(),
        (values.size() + getNumberOfProc()) / getNumberOfProc(),
        [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            std::vector<int64_t> indicies(values[idx].shape().size(), 0);
            const auto* from_ptr =
                static_cast<const std::byte*>(values[idx].data().data());
            auto* to_ptr =
                static_cast<std::byte*>(result_slices[idx].data().data());

            int64_t idim = values[idx].shape().size() - 1;
            do {
              std::memcpy(to_ptr, from_ptr, elsize);
              next_two_iter_(indicies, idim, values[idx].shape(), from_ptr,
                             values[idx].strides(), to_ptr,
                             result_slices[idx].strides());
            } while (idim >= 0);
          }
        });
  }

  return result;
}

}  // namespace spu::kernel::hal
