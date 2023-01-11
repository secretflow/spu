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

#pragma once

#include <algorithm>

#include "spu/core/xt_helper.h"
#include "spu/kernel/hlo/casting.h"
#include "spu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

spu::Value Constant(HalContext *ctx, PtBufferView view,
                    absl::Span<const int64_t> out_shape);

template <typename T>
spu::Value Iota(HalContext *ctx, int64_t numel, Visibility vis) {
  std::vector<T> tmp(numel);
  std::iota(tmp.begin(), tmp.end(), 0);
  auto c = Constant(ctx, tmp, {numel});
  if (vis == VIS_PUBLIC) {
    return c;
  } else {
    return Seal(ctx, c);
  }
}

spu::Value Epsilon(HalContext *ctx);

}  // namespace spu::kernel::hlo
