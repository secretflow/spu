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

#pragma once

#include "spu/kernel/hal/permute_util.h"
#include "spu/kernel/hlo/casting.h"
#include "spu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

namespace {

void sliceCopy(spu::Value &dst, const spu::Value &src,
               std::vector<int64_t> dst_indices, size_t dim) {
  auto copy_size = src.shape()[0];
  for (int64_t idx = 0; idx < copy_size; ++idx) {
    dst_indices[dim] = idx;
    dst.copyElementFrom(src, {idx}, dst_indices);
  }
}

}  // namespace

template <typename Comp>
std::vector<spu::Value> Sort(HalContext *ctx,
                             absl::Span<const spu::Value> inputs,
                             int64_t sort_dim, bool is_stable,
                             const Comp &comparator_body) {
  int64_t num_operands = inputs.size();
  auto key_shape = inputs[0].shape();
  auto rank = key_shape.size();
  std::vector<spu::Value> results;
  results.reserve(num_operands);
  for (int64_t i = 0; i < num_operands; ++i) {
    results.emplace_back(
        NdArrayRef(inputs[i].data().eltype(), inputs[i].shape()),
        inputs[i].dtype());
  }
  std::vector<int64_t> zero_base(rank, 0);
  std::vector<int64_t> increment(rank, 1);
  int64_t sort_dim_elements = key_shape[sort_dim];
  YASL_ENFORCE(
      sort_dim >= 0 && sort_dim < static_cast<int64_t>(increment.size()),
      "Unexpected out-of-bound sort dimension {}"
      " accessing increment of size {} ",
      sort_dim, increment.size());
  increment[sort_dim] = sort_dim_elements;
  bool warned = false;

  // Iterate through each dimension except 'sort_dim'.
  forEachIndex(
      key_shape, zero_base, key_shape, increment,
      [&](const std::vector<int64_t> &indices) {
        // Extract a slice from each operand literal that corresponds to
        // exactly the row in dimension 'sort_dim'.
        std::vector<int64_t> limit_indices(indices.begin(), indices.end());
        std::for_each(limit_indices.begin(), limit_indices.end(),
                      [](int64_t &index) { ++index; });
        limit_indices[sort_dim] = sort_dim_elements;
        std::vector<spu::Value> values_to_sort;
        values_to_sort.reserve(num_operands);
        for (int64_t i = 0; i < num_operands; ++i) {
          auto value_to_sort = hal::reshape(
              ctx, hal::slice(ctx, inputs[i], indices, limit_indices, {}),
              {sort_dim_elements});
          values_to_sort.push_back(std::move(value_to_sort));
        }
        std::vector<int64_t> indices_to_sort(sort_dim_elements);
        std::iota(indices_to_sort.begin(), indices_to_sort.end(), 0);
        auto comparator = [&comparator_body, &num_operands, &ctx,
                           &values_to_sort, &warned](int64_t a, int64_t b) {
          std::vector<spu::Value> values;
          values.reserve(2 * num_operands);
          for (int64_t i = 0; i < num_operands; ++i) {
            values.push_back(values_to_sort[i].getElementAt(a));
            values.push_back(values_to_sort[i].getElementAt(b));
          }
          spu::Value ret = comparator_body(values);
          if (ret.isSecret() && ctx->rt_config().reveal_secret_condition()) {
            ret = Reveal(ctx, ret);
            if (!warned) {
              SPDLOG_WARN("Reveal condition region result of SortOp");
              warned = true;
            }
          }
          return getConditionValue(ctx, ret);
        };

        if (is_stable) {
          std::stable_sort(indices_to_sort.begin(), indices_to_sort.end(),
                           comparator);
        } else {
          std::sort(indices_to_sort.begin(), indices_to_sort.end(), comparator);
        }

        std::vector<int64_t> start_indices(rank, 0);
        for (int64_t i = 0; i < num_operands; ++i) {
          auto sorted_value = hal::permute(ctx, values_to_sort[i], 0,
                                           xt::adapt(indices_to_sort));
          sliceCopy(results[i], sorted_value, indices, sort_dim);
        }
      });

  return results;
}

}  // namespace spu::kernel::hlo
