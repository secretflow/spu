// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/kernel/hal/utils.h"

namespace spu::kernel::hal {

spu::MemRef expandWindow(SPUContext *ctx, const spu::MemRef &base,
                         const Shape &window_shape,
                         const Strides &window_strides) {
  const size_t ndim = base.shape().size();

  // sanity check.
  SPU_ENFORCE(ndim == window_shape.size());
  SPU_ENFORCE(ndim == window_strides.size());

  // let base    = (B0, B1, ..., Bn)
  //     window  = (W0, W1, ..., Wn)
  //     stride  = (S0, S1, ..., Sn)
  // return        (N0, N1, ..., Nn, W0, W1, ..., Wn) where
  //     num_win = (N0, N1, ..., Nn), where Ni = (Bi-Wi)/Si+1
  const Shape &B = base.shape();
  const Shape &W = window_shape;
  const Strides &S = window_strides;
  Shape N(ndim);
  for (size_t dim = 0; dim < ndim; dim++) {
    N[dim] = (B[dim] - W[dim]) / S[dim] + 1;
  }

  // sample all windows.
  std::vector<spu::MemRef> windows;
  {
    Index window_index(ndim, 0);
    do {
      // for each window_index, find the corresponding window.
      Index start(ndim);
      for (size_t dim = 0; dim < ndim; dim++) {
        start[dim] = window_index[dim] * S[dim];
      }
      auto window = hal::slice(ctx, base, start, W, {});

      windows.emplace_back(hal::unsqueeze(ctx, window));
    } while (bumpIndices(N, absl::MakeSpan(window_index)));
  }

  // concatenate windows.
  auto res = hal::concatenate(ctx, windows, 0);
  SPU_ENFORCE_EQ(static_cast<int64_t>(windows.size()), N.numel());

  // reshape to tiled layout.
  // res_shape = (N0, N1, ..., Nn, W0, W1, ..., Wn)
  Shape res_shape = N;
  res_shape.insert(res_shape.end(), W.begin(), W.end());
  return hal::reshape(ctx, res, res_shape);
}

spu::MemRef expandWindow(SPUContext *ctx, const spu::MemRef &base,
                         const Shape &window_shape,
                         const Strides &window_strides,
                         absl::Span<const std::pair<int64_t, int64_t>> padding,
                         const spu::MemRef &init_val) {
  // sanity check.
  const size_t ndim = base.shape().size();
  SPU_ENFORCE(ndim == padding.size());

  Sizes padding_lo(ndim);
  Sizes padding_hi(ndim);
  bool need_pad = false;
  for (size_t idx = 0; idx < padding.size(); idx++) {
    padding_lo[idx] = padding[idx].first;
    padding_hi[idx] = padding[idx].second;
    need_pad |= (padding[idx].first != 0 || padding[idx].second != 0);
  }
  if (need_pad) {
    MemRef padded = hal::pad(ctx, base, init_val, padding_lo, padding_hi);
    return expandWindow(ctx, padded, window_shape, window_strides);
  }

  return expandWindow(ctx, base, window_shape, window_strides);
}

}  // namespace spu::kernel::hal
