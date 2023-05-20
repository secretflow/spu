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

bool getBooleanValue(SPUContext *ctx, const spu::Value &value);

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
  std::vector<int64_t> indexes(base.begin(), base.end());

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

// return false if it's in padding or dilation area
inline bool getBaseIndexFromWindowIndex(
    absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const int64_t> window_dilation,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    absl::Span<const int64_t> base_shape,
    absl::Span<const int64_t> base_dilation,
    absl::Span<const int64_t> window_count_index,
    absl::Span<const int64_t> window_index, absl::Span<int64_t> base_index) {
  const int64_t ndim = base_shape.size();

  for (int64_t dim = 0; dim < ndim; ++dim) {
    // Padding is applied to the dilated base. Say that padding is 3 and
    // dilation is 2 for some dimension. After applying base dilation and
    // padding, the dimension looks like:
    // P P P E D D E D D ... E D D E P P P
    // where E are the elements and D are the holes. So, the elements are
    // located in indices: padding + k*base_dilation for k = {0, 1, 2, ...}.
    // We are accessing elements in the transformed base at indices:
    // window_count_index * stride + window_index * window_dilation.
    // Solving for k gives us
    // (win_count_i * stride + win_i * win_dilation - pad) / base_dilation
    // When this is a natural number, we index an original element.
    // Otherwise, we index a 0 (pad or hole), and we don't need to apply
    // the callback f.
    base_index[dim] = window_count_index[dim] * window_strides[dim] +
                      window_index[dim] * window_dilation[dim] -
                      window_padding[dim].first;
    if (base_index[dim] % base_dilation[dim] != 0) {
      // out of bound
      return true;
    }
    base_index[dim] /= base_dilation[dim];
    if (base_index[dim] < 0 || base_index[dim] >= base_shape[dim]) {
      // out of bound
      return true;
    }
  }
  return false;
}

inline void RunOnWindowIndex(
    absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const int64_t> window_dilation,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    absl::Span<const int64_t> base_shape,
    absl::Span<const int64_t> base_dilation,
    absl::Span<const int64_t> window_count_index,
    absl::Span<const int64_t> window_index,
    const std::function<void(absl::Span<const int64_t>)> &f) {
  std::vector<int64_t> base_index(base_shape.size());
  bool out_of_bound = getBaseIndexFromWindowIndex(
      window_shape, window_strides, window_dilation, window_padding, base_shape,
      base_dilation, window_count_index, window_index,
      absl::MakeSpan(base_index));
  if (!out_of_bound) {
    f(base_index);
  }
}

}  // namespace spu::kernel
