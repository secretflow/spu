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

#include <optional>
#include <stack>

#include "spu/core/ndarray_ref.h"
#include "spu/core/shape_util.h"
#include "spu/device/pphlo/kernels/geometrical.h"
#include "spu/device/pphlo/kernels/utils.h"
#include "spu/hal/shape_ops.h"

namespace spu::device::pphlo::kernel {

struct ReduceWindowConfig {
  absl::Span<const int64_t> window_shape;
  absl::Span<const int64_t> window_strides;
  absl::Span<const int64_t> window_dilations;
  absl::Span<const std::pair<int64_t, int64_t>> window_padding;
  absl::Span<const int64_t> base_dilations;
};

template <typename Reducer>
hal::Value
ReduceWindow(HalContext *ctx, const hal::Value &input,
             const hal::Value &init_val, absl::Span<const int64_t> ret_shape,
             const ReduceWindowConfig &config, const Reducer &reducer) {

  const int64_t rank = input.shape().size();
  std::vector<int64_t> window_index(rank, 0);

  // Init...
  hal::Value ret = hal::expand(ctx, init_val, ret_shape);

  // For each resulting dimension, calculate and assign computed value.
  auto evaluate_impl =
      [&](absl::Span<int64_t> output_index) -> std::optional<hal::Value> {
    std::optional<hal::Value> ret;
    RunOnWindowIndex(config.window_shape, config.window_strides,
                     config.window_dilations, config.window_padding,
                     input.shape(), config.base_dilations, output_index,
                     window_index,
                     [&](absl::Span<const int64_t> operand_index) {
                       ret = input.getElementAt(operand_index);
                     });
    return ret;
  };

  // For each window index
  auto batch = hal::expand(ctx, input.getElementAt({}), ret_shape);

  do {
    // Collect one element from each window
    std::vector<int64_t> output_index(ret_shape.size(), 0);
    do {
      auto r = evaluate_impl(absl::MakeSpan(output_index));
      if (r.has_value()) {
        batch.copyElementFrom(*r, {}, output_index);
      }
    } while (
        bumpIndices(absl::MakeSpan(ret_shape), absl::MakeSpan(output_index)));

    // Now run the batch
    ret = reducer(ret, batch);

  } while (
      bumpIndices<int64_t>(config.window_shape, absl::MakeSpan(window_index)));

  return ret;
}

template <typename Reducer>
std::vector<hal::Value>
TreeReduce(HalContext *ctx, absl::Span<const hal::Value> inputs,
           absl::Span<const hal::Value> init_values,
           absl::Span<const int64_t> dimensions_to_reduce,
           absl::Span<const int64_t> ret_shape, const Reducer &reducer) {

  int64_t num_args = inputs.size();

  std::vector<int64_t> slice_begin(inputs[0].shape().size(), 0);
  std::vector<int64_t> slice_end = inputs[0].shape();
  std::vector<int64_t> slice_strides(inputs[0].shape().size(), 1);

  std::vector<hal::Value> arg_values(inputs.begin(), inputs.end());

  std::vector<hal::Value> lhs(num_args);
  std::vector<hal::Value> rhs(num_args);

  std::stack<std::vector<hal::Value>> tails;
  for (const auto &dimension_to_reduce : dimensions_to_reduce) {
    int64_t reduce_length = arg_values[0].shape()[dimension_to_reduce];
    while (reduce_length > 1) {
      reduce_length = reduce_length / 2;
      // lhs
      slice_begin[dimension_to_reduce] = 0;
      slice_end[dimension_to_reduce] = reduce_length;
      for (size_t idx = 0; idx < arg_values.size(); ++idx) {
        lhs[idx] = kernel::Slice(ctx, arg_values[idx], slice_begin, slice_end,
                                 slice_strides);
      }
      // rhs
      slice_begin[dimension_to_reduce] = reduce_length;
      slice_end[dimension_to_reduce] = 2 * reduce_length;
      for (size_t idx = 0; idx < arg_values.size(); ++idx) {
        rhs[idx] = kernel::Slice(ctx, arg_values[idx], slice_begin, slice_end,
                                 slice_strides);
      }
      // tail
      if (slice_end[dimension_to_reduce] <
          arg_values[0].shape()[dimension_to_reduce]) {
        slice_begin[dimension_to_reduce] = 2 * reduce_length;
        slice_end[dimension_to_reduce] =
            arg_values[0].shape()[dimension_to_reduce];
        std::vector<hal::Value> &tail = tails.emplace(num_args);

        for (size_t idx = 0; idx < arg_values.size(); ++idx) {
          tail[idx] = kernel::Slice(ctx, arg_values[idx], slice_begin,
                                    slice_end, slice_strides);
        }
      }
      arg_values = reducer(lhs, rhs);
    }

    while (!tails.empty()) {
      arg_values = reducer(arg_values, tails.top());
      tails.pop();
    }

    slice_begin[dimension_to_reduce] = 0;
    slice_end[dimension_to_reduce] = 1;
  }

  auto &tail = tails.emplace(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    tail[i] = kernel::Broadcast(ctx, init_values[i], ret_shape, {});
    arg_values[i] = kernel::Reshape(ctx, arg_values[i], ret_shape);
  }

  return reducer(arg_values, tails.top());
}

} // namespace spu::device::pphlo::kernel
