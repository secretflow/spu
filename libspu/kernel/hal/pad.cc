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

namespace spu::kernel::hal {

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

  return Value(in.data().pad(padding_value.data(), edge_padding_low,
                             edge_padding_high, interior_padding),
               in.dtype());
}

}  // namespace spu::kernel::hal
