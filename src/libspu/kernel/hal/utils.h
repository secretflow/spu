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

// Reduce the last axis of the input tensor using the given associative binary
// function. Uses a binary tree reduction algorithm with O(N) complexity.
//
// fn: an associative binary Function
// in: a tensor, reduce the last axis
//
// Output shape:
//   - If input shape is (N,), output shape is (1,)
//   - If input shape is (..., N), output shape is (...)
template <typename Fn>
spu::Value associative_reduce(Fn&& fn, SPUContext* ctx, const Value& in) {
  SPU_ENFORCE(in.shape().ndim() >= 1U, "input should not be scalar");

  const Shape& shape = in.shape();
  const auto N = shape.back();

  // Compute output shape: remove the last dimension
  // For 1D input (N,) -> (1,), for higher dims (..., N) -> (...)
  Shape out_shape;
  if (shape.ndim() == 1) {
    out_shape = {1};
  } else {
    out_shape = Shape(shape.begin(), shape.end() - 1);
  }

  // Edge case: if the last dimension has <= 1 element or tensor is empty
  if (N < 2 || shape.numel() == 0) {
    return in;
  }

  // Reshape to 2D {M, N} tensor for easier processing
  const auto M = shape.numel() / N;
  spu::Value current = hal::reshape(ctx, in, {M, N});

  // Use a stack to store tails (odd elements) for later processing
  std::stack<spu::Value> tails;

  // Binary tree reduction using TreeReduce strategy:
  // Split into [0, half) and [half, 2*half), save tail if odd
  int64_t len = N;
  while (len > 1) {
    const int64_t half = len / 2;

    // lhs = [0, half), rhs = [half, 2*half)
    auto lhs = hal::slice(ctx, current, {0, 0}, {M, half}, {});
    auto rhs = hal::slice(ctx, current, {0, half}, {M, 2 * half}, {});

    // Save tail if len is odd
    if (len % 2 == 1) {
      tails.push(hal::slice(ctx, current, {0, 2 * half}, {M, len}, {}));
    }

    current = fn(ctx, lhs, rhs);
    len = half;
  }

  // TODO: this may cause at worst 2*lg(n) time of reducer call, compare the
  // best case log(n) times.
  //
  // consider len = 63, will iterate 5 (31, 15, 7, 3, 1), and generate
  // len(tails) = 5, the total number is 5 + 5 = 10 times.
  //
  // However, the exact log(n) implementation requires more complex logic and
  // has worse cache performance, so we use this simpler method.
  //
  // Process all tails
  while (!tails.empty()) {
    current = fn(ctx, current, tails.top());
    tails.pop();
  }

  return hal::reshape(ctx, current, out_shape);
}

}  // namespace spu::kernel::hal