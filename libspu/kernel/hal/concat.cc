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

namespace spu::kernel::hal {

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

  std::vector<NdArrayRef> array(values.size() - 1);
  for (int64_t idx = 1; idx < static_cast<int64_t>(values.size()); ++idx) {
    array[idx - 1] = values[idx].data();
  }

  return Value(values[0].data().concatenate(array, axis), values[0].dtype());
}

}  // namespace spu::kernel::hal
