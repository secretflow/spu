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
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {

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
spu::Value associative_scan(Fn&& fn, SPUContext* ctx, const Value& in) {
  SPU_ENFORCE(in.shape().ndim() == 1U, "input should be 1d");
  const auto numel = in.numel();
  if (numel < 2) {
    return in;
  }

  // merge consecutive even/odd index elements
  auto reduced_elems = fn(ctx, hal::slice(ctx, in, {0}, {numel - 1}, {2}),
                          hal::slice(ctx, in, {1}, {numel}, {2}));
  // process half elements recursively and get odd index elements
  auto odd_elems = associative_scan(fn, ctx, reduced_elems);

  // get even index elements
  spu::Value even_elems;
  if (numel % 2 == 0) {
    even_elems =
        fn(ctx, hal::slice(ctx, odd_elems, {0}, {odd_elems.numel() - 1}, {1}),
           hal::slice(ctx, in, {2}, {numel}, {2}));
  } else {
    even_elems = fn(ctx, odd_elems, hal::slice(ctx, in, {2}, {numel}, {2}));
  }
  // concat the 0th element
  auto final_even_elems =
      hal::concatenate(ctx, {hal::slice(ctx, in, {0}, {1}), even_elems}, 0);

  // concat even and odd elems interleavely
  auto zero = hal::constant(ctx, 0U, in.dtype(), {1});
  auto pad_even =
      hal::pad(ctx, final_even_elems, zero, {0},
               {final_even_elems.numel() == odd_elems.numel() ? 1 : 0}, {1});
  auto pad_odd =
      hal::pad(ctx, odd_elems, zero, {1},
               {final_even_elems.numel() == odd_elems.numel() ? 0 : 1}, {1});

  auto ret = hal::_add(ctx, pad_even, pad_odd).setDtype(in.dtype());
  return ret;
}

//////////////////////////////////////////////////////////////////////////////
// Shape utils
//////////////////////////////////////////////////////////////////////////////

/// the squeeze function, i.e., removes dimensions of size 1 from the shape of a
/// tensor.
// @param in, the input
// @param dim, the dimension to be squeezed
Value squeeze(SPUContext* ctx, const Value& in, int64_t dim = 0);

/// the unsqueeze function, i.e., expands a tensor with a length 1 axis inserted
/// at index axis.
// @param in, the input
// @param dim, the dimension to be unsqueezed
Value unsqueeze(SPUContext* ctx, const Value& in, int64_t dim = 0);

}  // namespace spu::kernel::hal