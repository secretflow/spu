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

#include "libspu/kernel/hlo/rank.h"

#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hlo {

namespace {
std::vector<spu::Value> TopkApply(SPUContext *ctx, const spu::Value &input,
                                  const Rank1dFn &apply_fn) {
  const Shape &shape = input.shape();

  // Topk always deals last-dimension
  // - N is the number of vector to permute
  // - W is the vector length.
  const int64_t W = shape.back();
  const int64_t N = shape.numel() / W;

  // First, reshape the input to (N, W)
  auto reshaped = hal::reshape(ctx, input, {N, W});

  // Then, do topk in last dimension
  std::vector<std::vector<spu::Value>> topk1d;
  topk1d.reserve(N);
  for (int64_t i = 0; i < N; ++i) {
    // TODO: how to do these parallelly?
    auto input_i =
        hal::reshape(ctx, hal::slice(ctx, reshaped, {i, 0}, {i + 1, W}), {W});
    topk1d.push_back(apply_fn(input_i));
  }

  const bool include_index = topk1d[0].size() == 2;

  // the output shape is (..., k)
  Shape new_shape(shape.begin(), shape.end());
  const auto k = topk1d[0][0].numel();
  new_shape.back() = k;

  // Finally, Reshape back to shape
  std::vector<spu::Value> ret;
  ret.reserve(2);

  std::vector<spu::Value> value2d;
  value2d.reserve(N);
  for (int64_t i = 0; i < N; ++i) {
    value2d.push_back(hal::reshape(ctx, topk1d[i][0], {1, k}));
  }
  auto ret_val = hal::concatenate(ctx, value2d, 0);
  ret.push_back(hal::reshape(ctx, ret_val, new_shape));
  if (include_index) {
    std::vector<spu::Value> index2d;
    index2d.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
      index2d.push_back(hal::reshape(ctx, topk1d[i][1], {1, k}));
    }
    auto ret_inx = hal::concatenate(ctx, index2d, 0);
    ret.push_back(hal::reshape(ctx, ret_inx, new_shape));
  }
  return ret;
}
}  // namespace

std::vector<spu::Value> TopK(SPUContext *ctx, const spu::Value &input,
                             int64_t k_lo, int64_t k_hi, bool largest,
                             bool value_only) {
  const Shape &shape = input.shape();
  SPU_ENFORCE(shape.numel() > 0, "input must non-empty.");
  SPU_ENFORCE(
      k_lo <= shape.back() && k_lo > 0,
      "k_lo should be larger than 0 and smaller than the last dimension.");

  if (k_hi == -1) {
    k_hi = k_lo;
  }

  SPU_ENFORCE(k_lo <= k_hi,
              "k_lo should be smaller than k_hi, got k_lo={}, k_hi={}", k_lo,
              k_hi);

  auto scalar_cmp_fn = [largest](spu::SPUContext *ctx, const spu::Value &lhs,
                                 const spu::Value &rhs) {
    if (largest) {
      return hal::greater(ctx, lhs, rhs);
    } else {
      return hal::less(ctx, lhs, rhs);
    }
  };

  auto topk_fn = [&](const spu::Value &input) {
    return hal::topk_1d(ctx, input, {k_lo, k_hi}, scalar_cmp_fn, value_only);
  };

  return TopkApply(ctx, input, topk_fn);
}

}  // namespace spu::kernel::hlo
