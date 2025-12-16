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

#include <stack>

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/core/vectorize.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {

//////////////////////////////////////////////////////////////////////////////
// Shape utils
//////////////////////////////////////////////////////////////////////////////

/// the squeeze function, i.e., removes dimensions of size 1 from the shape of
/// a tensor.
// @param in, the input
// @param dim, the dimension to be squeezed
Value squeeze(SPUContext* ctx, const Value& in, int64_t dim = 0);

/// the unsqueeze function, i.e., expands a tensor with a length 1 axis
/// inserted at index axis.
// @param in, the input
// @param dim, the dimension to be unsqueezed
Value unsqueeze(SPUContext* ctx, const Value& in, int64_t dim = 0);

// This is SPU's version of JAX's associative_scan
// See:
// https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html
//
// Refer to
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// for the detailed algorithm explanation
//
// fn: an associative binary Function
// in: a tensor, scan the last axis
template <typename Fn>
spu::Value associative_scan(Fn&& fn, SPUContext* ctx, const Value& in) {
  SPU_ENFORCE(in.shape().ndim() >= 1U, "input should not be scalar");
  // First reshape to 2D {M, N} tensor, scan each N elements
  const Shape shape = in.shape();
  const auto N = shape.back();
  // in case some empty tensors
  if (N < 2 || shape.numel() == 0) {
    return in;
  }
  const auto M = shape.numel() / N;
  spu::Value in_2d = hal::reshape(ctx, in, {M, N});

  // merge consecutive even/odd index elements
  spu::Value odd_elems;
  std::vector<spu::Value> odd_vec;
  std::vector<spu::Value> even_vec;
  {
    for (int64_t i = 0; i < M; ++i) {
      odd_vec.push_back(hal::slice(ctx, in_2d, {i, 0}, {i + 1, N - 1}, {1, 2}));
      even_vec.push_back(hal::slice(ctx, in_2d, {i, 1}, {i + 1, N}, {1, 2}));
    }
    std::vector<spu::Value> reduced_elems_vec;
    vmap(odd_vec.begin(), odd_vec.end(), even_vec.begin(), even_vec.end(),
         std::back_inserter(reduced_elems_vec),
         [&](const spu::Value& odd, const spu::Value& even) {
           return fn(ctx, odd, even);
         });

    auto concat_reduced_elems = hal::concatenate(ctx, reduced_elems_vec, 0);

    // process half elements recursively and get odd index elements
    odd_elems = associative_scan(fn, ctx, concat_reduced_elems);
  }

  // get even index elements
  odd_vec.clear();
  even_vec.clear();
  spu::Value even_elems;
  {
    std::vector<spu::Value> even_elems_vec;
    for (int64_t i = 0; i < M; ++i) {
      if (N % 2 == 0) {
        odd_vec.push_back(hal::slice(ctx, odd_elems, {i, 0},
                                     {i + 1, odd_elems.shape().back() - 1},
                                     {1, 1}));
      } else {
        odd_vec.push_back(hal::slice(ctx, odd_elems, {i, 0},
                                     {i + 1, odd_elems.shape().back()}, {}));
      }
      even_vec.push_back(hal::slice(ctx, in_2d, {i, 2}, {i + 1, N}, {1, 2}));
    }
    vmap(odd_vec.begin(), odd_vec.end(), even_vec.begin(), even_vec.end(),
         std::back_inserter(even_elems_vec),
         [&](const spu::Value& odd, const spu::Value& even) {
           return fn(ctx, odd, even);
         });

    even_elems = hal::concatenate(ctx, even_elems_vec, 0);
  }
  // concat the 0th element
  auto final_even_elems = hal::concatenate(
      ctx, {hal::slice(ctx, in_2d, {0, 0}, {M, 1}), even_elems}, 1);

  // concat even and odd elems interleavely
  auto zero = hal::constant(ctx, 0U, in.dtype(), {1});
  auto pad_even = hal::pad(
      ctx, final_even_elems, zero, {0, 0},
      {0, final_even_elems.numel() == odd_elems.numel() ? 1 : 0}, {0, 1});
  auto pad_odd = hal::pad(
      ctx, odd_elems, zero, {0, 1},
      {0, final_even_elems.numel() == odd_elems.numel() ? 0 : 1}, {0, 1});

  auto ret = hal::_add(ctx, pad_even, pad_odd).setDtype(in.dtype());
  return hal::reshape(ctx, ret, in.shape());
}

// Optimized version of associative_reduce with guaranteed max lg(n) + 1
// reducer calls.
//
// Algorithm:
// 1. Find the largest power of 2 (lower_exp) that is <= n
// 2. If n > lower_exp, pre-reduce the "extra" elements to make length =
//    lower_exp
//    - c = n - lower_exp extra elements at the end
//    - b = c elements from the middle to pair with c
//    - a = 2*lower_exp - n elements at the beginning (untouched)
//    - Result: [a, b⊕c] has exactly lower_exp elements
// 3. Standard halving reduction on power-of-2 length
//
// Example for n = 7, lower_exp = 4:
//   Original: [a0, a1, a2, a3, a4, a5, a6]
//             |---|---|---|---|---|---|
//              a        b        c
//   Step 1:   [a0, a4⊕a1, a5⊕a2, a6⊕a3]  (length = 4)
//   Step 2-4: Standard halving reduction
//
// fn: an associative binary Function
// in: a tensor, reduce the last axis
template <typename Fn>
spu::Value associative_reduce(Fn&& fn, SPUContext* ctx, const Value& in) {
  SPU_ENFORCE(in.shape().ndim() >= 1U, "input should not be scalar");

  const Shape& shape = in.shape();
  const auto N = shape.back();

  // Compute output shape: remove the last dimension
  Shape out_shape;
  if (shape.ndim() == 1) {
    out_shape = {1};
  } else {
    out_shape = Shape(shape.begin(), shape.end() - 1);
  }

  // Edge case: if the last dimension has <= 1 element or tensor is empty
  if (N < 2 || shape.numel() == 0) {
    return hal::reshape(ctx, in, out_shape);
  }

  // Reshape to 2D {M, N} tensor for easier processing
  const auto M = shape.numel() / N;
  spu::Value current = hal::reshape(ctx, in, {M, N});

  int64_t len = N;

  // Find the largest power of 2 that is <= len
  int64_t lower_exp = 1;
  while ((lower_exp << 1) <= len) {
    lower_exp <<= 1;
  }

  // If len > lower_exp, pre-reduce the extra part to make len = lower_exp
  if (len != lower_exp) {
    // c = [lower_exp, len): the extra elements
    // b = [2*lower_exp - len, lower_exp): elements to pair with c
    // a = [0, 2*lower_exp - len): untouched elements
    auto c = hal::slice(ctx, current, {0, lower_exp}, {M, len}, {});
    auto b =
        hal::slice(ctx, current, {0, 2 * lower_exp - len}, {M, lower_exp}, {});
    auto t = fn(ctx, c, b);  // t = c ⊕ b

    auto a = hal::slice(ctx, current, {0, 0}, {M, 2 * lower_exp - len}, {});
    current = hal::concatenate(ctx, {a, t}, 1);
    len = lower_exp;
  }

  // Now len is a power of 2, do standard halving reduction
  while (len > 1) {
    const int64_t half = len / 2;
    auto lhs = hal::slice(ctx, current, {0, 0}, {M, half}, {});
    auto rhs = hal::slice(ctx, current, {0, half}, {M, 2 * half}, {});
    current = fn(ctx, lhs, rhs);
    len = half;
  }

  return hal::reshape(ctx, current, out_shape);
}

}  // namespace spu::kernel::hal