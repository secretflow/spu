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
  const auto perm_field = in.storage_type().as<Ring2k>()->field();
  auto zero = hal::constant(ctx, 0U, in.dtype(), {1}, perm_field);
  auto pad_even = hal::pad(
      ctx, final_even_elems, zero, {0, 0},
      {0, final_even_elems.numel() == odd_elems.numel() ? 1 : 0}, {0, 1});
  auto pad_odd = hal::pad(
      ctx, odd_elems, zero, {0, 1},
      {0, final_even_elems.numel() == odd_elems.numel() ? 0 : 1}, {0, 1});

  auto ret = hal::_add(ctx, pad_even, pad_odd).setDtype(in.dtype());
  return hal::reshape(ctx, ret, in.shape());
}

}  // namespace spu::kernel::hal