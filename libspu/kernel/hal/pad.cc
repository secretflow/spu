// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/kernel/hal/pad.h"

#include <cstdint>
#include <numeric>

#include "libspu/core/prelude.h"
#include "libspu/core/shape_util.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {

namespace {

std::vector<int64_t> deducePadShape(
    absl::Span<const int64_t> input_shape,
    absl::Span<const int64_t> edge_padding_low,
    absl::Span<const int64_t> edge_padding_high,
    absl::Span<const int64_t> interior_padding) {
  std::vector<int64_t> dims;
  SPU_ENFORCE(edge_padding_low.size() == input_shape.size());
  SPU_ENFORCE(edge_padding_high.size() == input_shape.size());
  SPU_ENFORCE(interior_padding.size() == input_shape.size());
  for (size_t i = 0; i < input_shape.size(); i++) {
    dims.emplace_back(edge_padding_low[i] + edge_padding_high[i] +
                      interior_padding[i] * (input_shape[i] - 1) +
                      input_shape[i]);
  }

  return dims;
}

}  // namespace

Value pad(HalContext* ctx, const Value& in, const Value& padding_value,
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

  Value result = expand(ctx, padding_value,
                        deducePadShape(in.shape(), edge_padding_low,
                                       edge_padding_high, interior_padding));

  const auto& result_shape = result.shape();
  const auto& input_shape = in.shape();

  auto elsize = result.elsize();

  yacl::parallel_for(0, in.numel(), 1024, [&](int64_t begin, int64_t end) {
    std::vector<int64_t> unflatten = unflattenIndex(begin, input_shape);

    std::vector<int64_t> target_index(result_shape.size());
    for (int64_t idx = begin; idx < end; ++idx) {
      bool valid = true;
      for (size_t i = 0; i < unflatten.size(); ++i) {
        // Interior padding occurs logically before edge padding, so in the case
        // of negative edge padding elements are removed from the
        // interior-padded operand.
        target_index[i] =
            edge_padding_low[i] + unflatten[i] * (interior_padding[i] + 1);

        // Account for negative low and high padding: skip assignment if the
        // any target index is out of range.
        if (target_index[i] < 0 || target_index[i] >= result_shape[i]) {
          valid = false;
          break;
        }
      }
      if (valid) {
        result.copyElementFrom(in, unflatten, target_index, elsize);
      }
      bumpIndices<int64_t>(in.shape(), absl::MakeSpan(unflatten));
    }
  });

  return result;
}

}  // namespace spu::kernel::hal
