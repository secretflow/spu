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

#include "libspu/kernel/hlo/sort.h"

#include <numeric>

#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hlo {

namespace internal {

using Sort1dFn =
    std::function<std::vector<spu::Value>(absl::Span<const spu::Value>)>;

// Given a & p are vectors, and p is a permutation.
// let b = permute(a, p) where b[i] = a[p[i]]
//
// InversePermute is a function that computes
//     q = InversePermute(p) where
//     c = permute(b, q) = a
//
// To solve the equation, for any i:
//     c[i] = b[q[i]]
//          = a[p[q[i]]]
//          = a[i]
//
// That is p[q[i]] == i
Index InversePermute(const Index &p) {
  Index q(p.size());
  const auto n = static_cast<int64_t>(p.size());
  for (int64_t i = 0; i < n; ++i) {
    q[p[i]] = i;
  }
  return q;
}

std::vector<spu::Value> Sort(SPUContext *ctx,
                             absl::Span<const spu::Value> inputs,
                             int64_t sort_dim, const Sort1dFn &sort_fn) {
  // sanity check.
  SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
  // put the to_sort dimension to last dimension.
  const Shape shape = inputs[0].shape();

  // let
  // - M is the number of inputs.
  // - N is the number of vector to sort
  // - W is the vector length.
  const int64_t M = inputs.size();
  const int64_t W = shape.dim(sort_dim);
  const int64_t N = shape.numel() / W;
  Axes perm(shape.ndim());
  Axes unperm;
  {
    // 2 ==> {0, 1, 4, 3, 2}
    SPU_ENFORCE(sort_dim < shape.ndim());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[sort_dim], perm.back());

    auto q = InversePermute(Index(perm.begin(), perm.end()));
    unperm = Axes(q.begin(), q.end());
  }

  Shape perm_shape(shape.begin(), shape.end());
  std::swap(perm_shape[sort_dim], perm_shape.back());

  // Do sort in 2-dimensions.
  // First, reshape the input to (N, W)
  std::vector<spu::Value> inputs2d;
  for (auto const &input : inputs) {
    auto transposed = hal::transpose(ctx, input, perm);
    auto reshaped = hal::reshape(ctx, transposed, {N, W});
    inputs2d.push_back(reshaped);
  }

  // Call sort1d for each dim to sort.
  // results (N,M,W), each element is a vector with length W.
  std::vector<std::vector<spu::Value>> sorted1d;
  for (int64_t ni = 0; ni < N; ni++) {
    // TODO: all these small sort could be done in parallel.
    std::vector<spu::Value> input_i;
    for (auto const &input : inputs2d) {
      // we need 1-d tensor here
      input_i.push_back(
          hal::reshape(ctx, hal::slice(ctx, input, {ni, 0}, {ni + 1, W}), {W}));
    }

    sorted1d.push_back(sort_fn(input_i));
  }

  // result is (M,shape)
  std::vector<spu::Value> results(M);
  for (int64_t mi = 0; mi < M; mi++) {
    std::vector<spu::Value> output2d;
    for (int64_t ni = 0; ni < N; ni++) {
      output2d.push_back(hal::reshape(ctx, sorted1d[ni][mi], {1, W}));
    }
    auto result = hal::concatenate(ctx, output2d, 0);
    // Permute it back, final result is (M, shape)
    result = hal::reshape(ctx, result, perm_shape);
    results[mi] = hal::transpose(ctx, result, unperm);
  }

  return results;
}

}  // namespace internal

std::vector<spu::Value> Sort(SPUContext *ctx,
                             absl::Span<const spu::Value> inputs,
                             int64_t sort_dim, bool is_stable,
                             const hal::CompFn &comparator_body,
                             Visibility comparator_ret_vis) {
  auto sort_fn = [&](absl::Span<const spu::Value> input) {
    return hal::sort1d(ctx, input, comparator_body, comparator_ret_vis,
                       is_stable);
  };
  return internal::Sort(ctx, inputs, sort_dim, sort_fn);
}

std::vector<spu::Value> SimpleSort(SPUContext *ctx,
                                   absl::Span<const spu::Value> inputs,
                                   int64_t sort_dim,
                                   hal::SortDirection direction) {
  auto sort_fn = [&](absl::Span<const spu::Value> input) {
    return hal::simple_sort1d(ctx, input, direction);
  };
  return internal::Sort(ctx, inputs, sort_dim, sort_fn);
}

}  // namespace spu::kernel::hlo
