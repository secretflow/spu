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

#pragma once

#include <string>
#include <vector>

#include "xtensor/xarray.hpp"

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel {

int32_t getI32Value(SPUContext *ctx, const spu::Value &value);

xt::xarray<int64_t> getIndices(SPUContext *ctx, const spu::Value &value);

template <typename FnTy>
void forEachIndex(absl::Span<const int64_t> shape,
                  absl::Span<const int64_t> base,
                  absl::Span<const int64_t> count,
                  absl::Span<const int64_t> incr, FnTy &&visitor_function) {
  SPU_ENFORCE_EQ(shape.size(), base.size());
  SPU_ENFORCE_EQ(incr.size(), base.size());
  SPU_ENFORCE_EQ(count.size(), base.size());

  const auto rank = static_cast<int64_t>(shape.size());
  // Allows handling R0 arrays, such that the visitor function will be called
  // once with the proper empty indexes.
  int64_t n = rank;
  Index indexes(base.begin(), base.end());

  while (n >= 0) {
    visitor_function(indexes);
    // Increments dimensions in minor to major order.
    for (n = rank - 1; n >= 0; --n) {
      indexes[n] += incr[n];
      if (indexes[n] < base[n] + count[n]) {
        break;
      }
      indexes[n] = base[n];
    }
  }
}

template <typename FnType>
void forEachIndex(absl::Span<const int64_t> shape,
                  const FnType &visitor_function) {
  std::vector<int64_t> base(shape.size(), 0);
  std::vector<int64_t> incr(shape.size(), 1);
  return forEachIndex(shape, base,
                      /*count=*/shape, incr, visitor_function);
}

/// Expand the base according to window
//
// let base    = (B0, B1, ..., Bn)
//     window  = (W0, W1, ..., Wn)
//     stride  = (S0, S1, ..., Sn)
// return        (N0, N1, ..., Nn, W0, W1, ..., Wn) where
//     num_win = (N0, N1, ..., Nn), where Ni = (Bi-Wi)/Si+1
spu::Value expandWindow(SPUContext *ctx, const spu::Value &base,
                        const Shape &window_shape,
                        const Strides &window_strides,
                        absl::Span<const std::pair<int64_t, int64_t>> padding,
                        const spu::Value &init_val);

}  // namespace spu::kernel
