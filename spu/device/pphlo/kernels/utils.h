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

#include "spu/hal/value.h"

namespace spu {

class HalContext;

namespace device {

bool getConditionValue(HalContext *ctx, const hal::Value &value);

xt::xarray<int64_t> getIndicies(HalContext *ctx, const hal::Value &value);

template <typename FnTy>
void forEachIndex(absl::Span<const int64_t> shape,
                  absl::Span<const int64_t> base,
                  absl::Span<const int64_t> count,
                  absl::Span<const int64_t> incr, FnTy &&visitor_function) {
  YASL_ENFORCE_EQ(shape.size(), base.size());
  YASL_ENFORCE_EQ(incr.size(), base.size());
  YASL_ENFORCE_EQ(count.size(), base.size());

  const auto rank = static_cast<int64_t>(shape.size());
  // Allows handling R0 arrays, such that the visitor function will be called
  // once with the proper empty indexes.
  int64_t n = -1;
  std::vector<int64_t> indexes(base.begin(), base.end());

  while (n < rank) {
    visitor_function(indexes);
    // Increments dimensions in minor to major order.
    for (n = 0; n < rank; ++n) {
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
  std::vector<int64_t> base(shape.size());
  std::vector<int64_t> incr(shape.size(), 1);
  return forEachIndex(shape, base,
                      /*count=*/shape, incr, visitor_function);
}

inline void
RunOnWindowIndex(absl::Span<const int64_t> window_shape,
                 absl::Span<const int64_t> window_strides,
                 absl::Span<const int64_t> window_dilation,
                 absl::Span<const std::pair<int64_t, int64_t>> window_padding,
                 absl::Span<const int64_t> base_shape,
                 absl::Span<const int64_t> base_dilation,
                 absl::Span<const int64_t> window_count_index,
                 absl::Span<const int64_t> window_index,
                 const std::function<void(absl::Span<const int64_t>)> &f) {
  const int64_t rank = base_shape.size();
  std::vector<int64_t> base_index(rank);
  bool out_of_bound = false;
  for (int64_t i = 0; i < rank; ++i) {
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
    base_index[i] = window_count_index[i] * window_strides[i] +
                    window_index[i] * window_dilation[i] -
                    window_padding[i].first;
    if (base_index[i] % base_dilation[i] != 0) {
      out_of_bound = true;
      break;
    }
    base_index[i] /= base_dilation[i];
    if (base_index[i] < 0 || base_index[i] >= base_shape[i]) {
      out_of_bound = true;
      break;
    }
  }
  if (!out_of_bound) {
    f(base_index);
  }
}

} // namespace device
} // namespace spu
