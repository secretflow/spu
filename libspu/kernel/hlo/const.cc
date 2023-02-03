// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/kernel/hlo/const.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hlo {

spu::Value Constant(HalContext *ctx, PtBufferView view,
                    absl::Span<const int64_t> out_shape) {
  if (view.shape == out_shape) {
    return hal::constant(ctx, std::move(view));
  } else {
    auto s = hal::constant(ctx, view);
    return hal::broadcast_to(ctx, s, out_shape);
  }
}

spu::Value Epsilon(HalContext *ctx) { return hal::epsilon(ctx); }

}  // namespace spu::kernel::hlo