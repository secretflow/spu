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

#include "yasl/base/exception.h"

#include "spu/core/parallel_utils.h"
#include "spu/core/shape_util.h"
#include "spu/kernel/hal/ring.h"

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
  std::vector<int64_t> offsets(values.size(), 0);
  for (size_t idx = 1; idx < values.size(); ++idx) {
    offsets[idx] = result_shape[axis];
    result_shape[axis] += values[idx].shape()[axis];
  }

  // Preallocate output buffer
  Value result({values.front().storage_type(), result_shape},
               values.front().dtype());
  auto elsize = result.elsize();

  // TODO(xiaochen): This is still very inefficient, consider a better
  // implementation
  for (size_t idx = 0; idx < values.size(); ++idx) {
    yasl::parallel_for(0, values[idx].numel(), 2048,
                       [&](int64_t begin, int64_t end) {
                         std::vector<int64_t> from_indicies =
                             unflattenIndex(begin, values[idx].shape());
                         std::vector<int64_t> to_indicies;
                         for (int64_t e_idx = begin; e_idx < end; ++e_idx) {
                           to_indicies = from_indicies;
                           to_indicies[axis] += offsets[idx];
                           result.copyElementFrom(values[idx], from_indicies,
                                                  to_indicies, elsize);
                           bumpIndices<int64_t>(values[idx].shape(),
                                                absl::MakeSpan(from_indicies));
                         }
                       });
  }

  return result;
}

}  // namespace spu::kernel::hal
