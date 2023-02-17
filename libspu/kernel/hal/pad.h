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

#pragma once

#include "libspu/kernel/context.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hal {

//// the pad function
// @param in, the param
// @param padding_value, to fill in the added padding
// @param edge_padding_low, the amount of padding added at the
//        low-end (next to index 0) of each dimension
// @param edge_padding_high, the amount of padding added at the high-end
//        (next to the highest index) of each dimension
// @param interior_padding, the amount of padding added between any two elements
//        in each dimension
Value pad(HalContext* ctx, const Value& in, const Value& padding_value,
          absl::Span<const int64_t> edge_padding_low,
          absl::Span<const int64_t> edge_padding_high,
          absl::Span<const int64_t> interior_padding);

}  // namespace spu::kernel::hal
