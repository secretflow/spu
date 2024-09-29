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

#pragma once

#include "libspu/core/context.h"
#include "libspu/core/memref.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {

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

// This is SPU's version of JAX's associative_scan
// See:
// https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html
//
// Refer to
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// for the detailed algorithm explanation
//
// fn: an associative binary Function
// in: a 1-d tensor
template <typename Fn>
MemRef associative_scan(Fn &&fn, SPUContext *ctx, const MemRef &in) {
  SPU_ENFORCE(in.shape().ndim() == 1U, "input should be 1d");
  const auto numel = in.numel();
  if (numel < 2) {
    return in;
  }

  // merge consecutive even/odd index elements
  auto reduced_elems = fn(ctx, hal::slice(ctx, in, {0}, {numel / 2}, {2}),
                          hal::slice(ctx, in, {1}, {numel / 2}, {2}));
  // process half elements recursively and get odd index elements
  auto odd_elems = associative_scan(fn, ctx, reduced_elems);

  // get even index elements
  MemRef even_elems;
  if (numel % 2 == 0) {
    even_elems =
        fn(ctx, hal::slice(ctx, odd_elems, {0}, {odd_elems.numel() - 1}, {1}),
           hal::slice(ctx, in, {2}, {odd_elems.numel() - 1}, {2}));
  } else {
    even_elems =
        fn(ctx, odd_elems, hal::slice(ctx, in, {2}, {odd_elems.numel()}, {2}));
  }
  // concat the 0th element
  auto final_even_elems =
      hal::concatenate(ctx, {hal::slice(ctx, in, {0}, {1}), even_elems}, 0);

  // concat even and odd elems interleavely

  auto zero = hal::constant(ctx, 0, {1});
  zero = hal::_ring_cast(ctx, zero, final_even_elems.eltype().semantic_type());

  auto pad_even =
      hal::pad(ctx, final_even_elems, zero, {0},
               {final_even_elems.numel() == odd_elems.numel() ? 1 : 0}, {1});
  auto pad_odd =
      hal::pad(ctx, odd_elems, zero, {1},
               {final_even_elems.numel() == odd_elems.numel() ? 0 : 1}, {1});

  auto ret = hal::_add(ctx, pad_even, pad_odd);
  return ret;
}

/// Expand the base according to window
//
// let base    = (B0, B1, ..., Bn)
//     window  = (W0, W1, ..., Wn)
//     stride  = (S0, S1, ..., Sn)
// return        (N0, N1, ..., Nn, W0, W1, ..., Wn) where
//     num_win = (N0, N1, ..., Nn), where Ni = (Bi-Wi)/Si+1
MemRef expandWindow(SPUContext *ctx, const MemRef &base,
                    const Shape &window_shape, const Strides &window_strides,
                    absl::Span<const std::pair<int64_t, int64_t>> padding,
                    const MemRef &init_val);

}  // namespace spu::kernel::hal