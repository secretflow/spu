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

#include "libspu/kernel/hlo/reduce.h"

#include <algorithm>
#include <cstdint>
#include <future>
#include <iostream>
#include <stack>
#include <vector>

#include "libspu/core/parallel_utils.h"
#include "libspu/core/xt_helper.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {

std::vector<spu::Value> TreeReduce(SPUContext *ctx,
                                   absl::Span<const spu::Value> inputs,
                                   int64_t axis,
                                   const BatchedValueBinaryFn &reducer) {
  const int64_t nargs = inputs.size();

  std::vector<spu::Value> outputs(inputs.begin(), inputs.end());

  std::vector<spu::Value> lhs(nargs);
  std::vector<spu::Value> rhs(nargs);

  std::stack<std::vector<spu::Value>> tails;

  Index slice_begin(inputs.back().shape().size(), 0);
  Index slice_end(inputs.back().shape().begin(), inputs.back().shape().end());
  Strides slice_strides(inputs.back().shape().size(), 1);
  int64_t len = outputs[0].shape()[axis];
  while (len > 1) {
    const int64_t half = len / 2;

    // lhs & rhs
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      slice_begin[axis] = 0;
      slice_end[axis] = half;
      auto len = outputs[idx].shape().size();

      lhs[idx] = hal::slice(
          ctx, outputs[idx],
          Index(slice_begin.begin(), slice_begin.begin() + len),
          Index(slice_end.begin(), slice_end.begin() + len),
          Strides(slice_strides.begin(), slice_strides.begin() + len));

      slice_begin[axis] = half;
      slice_end[axis] = 2 * half;
      rhs[idx] = hal::slice(
          ctx, outputs[idx],
          Index(slice_begin.begin(), slice_begin.begin() + len),
          Index(slice_end.begin(), slice_end.begin() + len),
          Strides(slice_strides.begin(), slice_strides.begin() + len));
    }

    // tail
    if (len % 2 == 1) {
      slice_begin[axis] = 2 * half;
      slice_end[axis] = len;
      std::vector<spu::Value> &tail = tails.emplace(nargs);
      for (size_t idx = 0; idx < outputs.size(); ++idx) {
        auto len = outputs[idx].shape().size();
        tail[idx] = hal::slice(
            ctx, outputs[idx],
            Index(slice_begin.begin(), slice_begin.begin() + len),
            Index(slice_end.begin(), slice_end.begin() + len),
            Strides(slice_strides.begin(), slice_strides.begin() + len));
      }
    }

    outputs = reducer(lhs, rhs);
    len /= 2;

    SPU_ENFORCE(outputs[0].shape()[axis] == len);
  }

  // TODO: this may cause at worst 2*lg(n) time of reducer call, compare the
  // best case lg(n) times.
  //
  // consider len = 63, will iterate 5 (31, 15, 7, 3, 1), and generate
  // len(tails) = 5, the total number is 5 + 5 = 10 times.
  //
  // Optimize ME.
  while (!tails.empty()) {
    outputs = reducer(outputs, tails.top());
    tails.pop();
  }

  return outputs;
}

std::vector<spu::Value> ReduceWindowWithoutDilation(
    SPUContext *ctx, absl::Span<const spu::Value> inputs,
    absl::Span<const spu::Value> init_values, const Shape &window_shape,
    const Strides &window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    bool last_operand_is_window_mask, bool ignore_init_value,
    const Shape &ret_shape, const BatchedValueBinaryFn &reducer) {
  const size_t nargs =
      last_operand_is_window_mask ? inputs.size() - 1 : inputs.size();

  auto window_size = std::accumulate(window_shape.begin(), window_shape.end(),
                                     1, std::multiplies<>());

  // expand the operand, simplify following actions without strides and padding.
  std::vector<spu::Value> expanded;
  for (size_t idx = 0; idx < nargs; ++idx) {
    const auto &input = inputs[idx];
    const auto &init_val = init_values[idx];
    expanded.emplace_back(expandWindow(ctx, input, window_shape, window_strides,
                                       window_padding, init_val));
  }

  if (last_operand_is_window_mask) {
    auto mask = inputs.back();
    Shape shape(expanded.back().shape().size() + 1, 1);
    shape.back() = window_size;

    auto mask_idx = shape.size() - 2;
    for (auto iter = window_shape.rbegin(); iter != window_shape.rend();
         ++iter) {
      shape[mask_idx--] = *iter;
    }

    mask = hal::reshape(ctx, mask, shape);

    for (size_t idx = 0; idx < expanded.back().shape().size(); ++idx) {
      shape[idx] = expanded.back().shape()[idx];
    }

    expanded.emplace_back(hal::broadcast_to(ctx, mask, shape, {}));
  }

  // Flatten the window, to maximize parallel processing.
  Shape tiled_1d_shape(ret_shape.begin(), ret_shape.end());
  tiled_1d_shape.push_back(window_size);
  for (size_t idx = 0; idx < nargs; ++idx) {
    // reshape to tiled_1d
    expanded[idx] = hal::reshape(ctx, expanded[idx], tiled_1d_shape);
  }

  if (last_operand_is_window_mask) {
    Shape tiled_1d_shape_mask(tiled_1d_shape.begin(), tiled_1d_shape.end());
    tiled_1d_shape_mask.emplace_back(window_size);
    expanded.back() = hal::reshape(ctx, expanded.back(), tiled_1d_shape_mask);
  }

  // reduce the last axis
  auto outputs = TreeReduce(ctx, expanded, tiled_1d_shape.size() - 1, reducer);

  // reduce the last axis
  for (size_t idx = 0; idx < nargs; idx++) {
    outputs[idx] = hal::reshape(ctx, outputs[idx], ret_shape);
  }
  if (last_operand_is_window_mask) {
    Shape mask_ret_shape(ret_shape.begin(), ret_shape.end());
    mask_ret_shape.emplace_back(window_size);
    outputs.back() = hal::reshape(ctx, outputs.back(), mask_ret_shape);
  }

  if (!ignore_init_value) {
    // init_values are scalars, broadcast to return shape first.
    std::vector<spu::Value> broadcasted_init_values;
    for (const auto &v : init_values) {
      broadcasted_init_values.push_back(hal::broadcast_to(ctx, v, ret_shape));
    }

    return reducer(outputs, broadcasted_init_values);
  }
  return outputs;
}

std::vector<spu::Value> ReduceWindowImpl(
    SPUContext *ctx, absl::Span<const spu::Value> inputs,
    absl::Span<const spu::Value> init_values, const Shape &ret_shape,
    const ReduceWindowConfig &config, bool last_operand_is_window_mask,
    bool ignore_init_value, const BatchedValueBinaryFn &reducer) {
  if (std::all_of(config.window_dilations.begin(),
                  config.window_dilations.end(),
                  [](const int64_t x) { return x == 1; }) &&
      std::all_of(config.base_dilations.begin(), config.base_dilations.end(),
                  [](const int64_t x) { return x == 1; })) {
    return ReduceWindowWithoutDilation(
        ctx, inputs, init_values, config.window_shape, config.window_strides,
        config.window_padding, last_operand_is_window_mask, ignore_init_value,
        ret_shape, reducer);
  }

  SPU_ENFORCE(!last_operand_is_window_mask);

  const int64_t ndims = inputs[0].shape().size();
  int64_t nargs = inputs.size();

  SPU_ENFORCE_EQ(ndims, static_cast<int64_t>(config.window_padding.size()));
  SPU_ENFORCE_EQ(ndims, static_cast<int64_t>(config.window_dilations.size()));

  Sizes padding_lo(ndims);
  Sizes padding_hi(ndims);
  Sizes padding_in(ndims);

  for (size_t idx = 0; idx < config.window_padding.size(); idx++) {
    padding_lo[idx] = config.window_padding[idx].first;
    padding_hi[idx] = config.window_padding[idx].second;
    padding_in[idx] = config.base_dilations[idx] - 1;
  }

  const Strides &S = config.window_strides;
  const Shape &W = config.window_shape;

  // padding
  std::vector<spu::Value> padded_inputs;
  for (int64_t idx = 0; idx < nargs; ++idx) {
    padded_inputs.emplace_back(hal::pad(ctx, inputs[idx], init_values[idx],
                                        padding_lo, padding_hi, padding_in));
  }

  // iterate windows to reduce
  // reduce dims
  const auto in_shape = padded_inputs[0].shape();
  Axes reduce_dims(in_shape.size(), 0);
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);

  Index window_index(ndims, 0);
  std::vector<std::vector<spu::Value>> reduced_rets;
  do {
    Index start(ndims);
    Index end(ndims);
    for (int64_t dim = 0; dim < ndims; dim++) {
      start[dim] = window_index[dim] * S[dim];
      end[dim] = start[dim] + W[dim] +
                 (W[dim] - 1) * (config.window_dilations[dim] - 1);
    }
    std::vector<Value> windows(nargs);
    for (int64_t idx = 0; idx < nargs; ++idx) {
      windows[idx] = hal::slice(ctx, padded_inputs[idx], start, end,
                                (Strides)config.window_dilations);
    }
    reduced_rets.emplace_back(
        Reduce(ctx, windows, init_values, reduce_dims, reducer));
  } while (bumpIndices(ret_shape, absl::MakeSpan(window_index)));

  SPU_ENFORCE_EQ(static_cast<int64_t>(reduced_rets.size()), ret_shape.numel());

  std::vector<spu::Value> rets;
  for (int64_t input_idx = 0; input_idx < nargs; ++input_idx) {
    std::vector<Value> reduced_values;

    for (size_t widx = 0; widx < reduced_rets.size(); ++widx) {
      Shape new_shape = reduced_rets[widx][input_idx].shape();
      new_shape.insert(new_shape.begin(), 1);

      reduced_values.emplace_back(
          hal::reshape(ctx, reduced_rets[widx][input_idx], new_shape));
    }

    rets.emplace_back(
        hal::reshape(ctx, hal::concatenate(ctx, reduced_values, 0), ret_shape));
  }

  return rets;
}

std::vector<spu::Value> ReduceWindow(SPUContext *ctx,
                                     absl::Span<const spu::Value> inputs,
                                     absl::Span<const spu::Value> init_values,
                                     const Shape &ret_shape,
                                     const ReduceWindowConfig &config,
                                     const BatchedValueBinaryFn &reducer,
                                     bool ignore_init_values) {
  return ReduceWindowImpl(ctx, inputs, init_values, ret_shape, config, false,
                          ignore_init_values, reducer);
}

std::vector<spu::Value> Reduce(SPUContext *ctx,
                               absl::Span<const spu::Value> inputs,
                               absl::Span<const spu::Value> init_values,
                               const Axes &dims_to_reduce,
                               const BatchedValueBinaryFn &reducer,
                               bool ignore_init_values) {
  // Reduce multiple dimension
  //
  // The straight-forward method iterates dimension_to_reduce with each dim a
  // TreeReduce kernel. In SPU, we tries to minimize the reducer call.
  //
  // The algorithm is summarized below:
  //
  // Input:
  //   shape       2 3 4 5 6
  //   dims          X   X
  //
  // Steps:
  //   perm        2 4 6 3 5         0 2 4 1 3
  //   flatten     2 4 6 15
  //   reduce      2 4 6 1
  //   result      2 1 4 1 6
  //
  // Note(jint), theoretically, this method will reduce number of reducer calls,
  // in this example, from
  //   ceil(lg(3)) + ceil(lg(5)) = 2 + 3 = 5
  // to
  //   ceil(lg(3 * 5)) = 4
  //
  // But in current TreeReduce (unoptimized) implementation, this method is
  // slower.
  //
  // Note(jint): this `lowering` progress is easy to be ported to
  // compile-time.

  const auto in_shape = inputs[0].shape();

  Axes perm(in_shape.size(), 0);
  std::iota(perm.begin(), perm.end(), 0);
  // swap axes, move the dims to reduce to inner most.
  std::stable_partition(perm.begin(), perm.end(), [&](int64_t axis) {
    return std::find(dims_to_reduce.begin(), dims_to_reduce.end(), axis) ==
           dims_to_reduce.end();
  });

  Shape flat_shape;
  int64_t numel_to_reduce = 1;
  for (size_t axis = 0; axis < in_shape.size(); axis++) {
    if (std::find(dims_to_reduce.begin(), dims_to_reduce.end(), axis) ==
        dims_to_reduce.end()) {
      flat_shape.push_back(in_shape[axis]);
    } else {
      numel_to_reduce *= in_shape[axis];
    }
  }
  flat_shape.push_back(numel_to_reduce);

  std::vector<spu::Value> flattened;
  for (const auto &input : inputs) {
    flattened.push_back(
        hal::reshape(ctx, hal::transpose(ctx, input, perm), flat_shape));
  }

  // reduce the inner most axis
  auto results =
      TreeReduce(ctx, flattened, flattened[0].shape().size() - 1, reducer);

  // broadcast to origin shape.
  Shape out_shape = inputs[0].shape();
  for (const auto &axis : dims_to_reduce) {
    out_shape[axis] = 1;
  }

  for (auto &result : results) {
    result = hal::reshape(ctx, result, out_shape);
  }

  if (ignore_init_values) {
    return results;
  }

  std::vector<spu::Value> broadcasted_init_values;
  // init_values are scalars, broadcast to return shape first.
  for (const auto &v : init_values) {
    broadcasted_init_values.push_back(hal::broadcast_to(ctx, v, out_shape));
  }

  return reducer(results, broadcasted_init_values);
}

// So idea here..
// When windows size is 2x2, tile and run parallel on window element level has
// way to much overhead (both memory and computation).
// Just do a window level parallel is good enough
// And without dilation and padding, this can be achieved through just slicing
// FIXME: This is a super special case...consider generalize it a little bit
std::pair<spu::Value, spu::Value> ArgMax1x2x2x1NoPaddingWithoutDilation(
    SPUContext *ctx, const spu::Value &input, const Strides &window_strides) {
  auto input_shape = input.shape();

  spu::Value h_max;
  spu::Value h_idx_max;

  Strides strides(window_strides.size(), 1);
  {
    // Get to horizontal slices
    strides[2] = window_strides[2];
    auto lhs = hal::slice(
        ctx, input, {0, 0, 0, 0},
        {input_shape[0], input_shape[1], input_shape[2] - 1, input_shape[3]},
        strides);
    auto rhs = hal::slice(
        ctx, input, {0, 0, 1, 0},
        {input_shape[0], input_shape[1], input_shape[2], input_shape[3]},
        strides);

    strides[2] = 1;
    // Do a less comp
    auto h_comp = hal::less(ctx, rhs, lhs);
    // make comp an ashare
    h_comp = hal::_prefer_a(ctx, h_comp);

    auto h_i_comp = hal::reshape(ctx, h_comp,
                                 {h_comp.shape()[0], h_comp.shape()[1],
                                  h_comp.shape()[2], h_comp.shape()[3], 1});

    // Now do two selections
    auto mask_shape = h_comp.shape();
    mask_shape.emplace_back(2);

    // Now compute horizontal max...
    h_max = hal::select(ctx, h_comp, lhs, rhs);

    // Mask index
    h_idx_max =
        hal::concatenate(ctx, {h_i_comp, hal::logical_not(ctx, h_i_comp)}, 4);
  }

  // Now do vertical compare...
  strides[1] = window_strides[1];
  auto upper_value = hal::slice(ctx, h_max, {0, 0, 0, 0},
                                {h_max.shape()[0], h_max.shape()[1] - 1,
                                 h_max.shape()[2], h_max.shape()[3]},
                                strides);
  auto bottom_value = hal::slice(
      ctx, h_max, {0, 1, 0, 0},
      {h_max.shape()[0], h_max.shape()[1], h_max.shape()[2], h_max.shape()[3]},
      strides);

  auto v_comp = hal::less(ctx, bottom_value, upper_value);
  v_comp = hal::_prefer_a(ctx, v_comp);

  // Compute max value
  auto max_ret = hal::select(ctx, v_comp, upper_value, bottom_value);

  // Compute max indices
  auto v_comp_not = hal::logical_not(ctx, v_comp);

  auto v_i_comp = hal::reshape(ctx, v_comp,
                               {v_comp.shape()[0], v_comp.shape()[1],
                                v_comp.shape()[2], v_comp.shape()[3], 1});
  v_i_comp = hal::broadcast_to(ctx, v_i_comp,
                               {v_i_comp.shape()[0], v_i_comp.shape()[1],
                                v_i_comp.shape()[2], v_i_comp.shape()[3], 2});

  auto v_i_comp_not =
      hal::reshape(ctx, v_comp_not,
                   {v_comp_not.shape()[0], v_comp_not.shape()[1],
                    v_comp_not.shape()[2], v_comp_not.shape()[3], 1});
  v_i_comp_not =
      hal::broadcast_to(ctx, v_i_comp_not,
                        {v_i_comp_not.shape()[0], v_i_comp_not.shape()[1],
                         v_i_comp_not.shape()[2], v_i_comp_not.shape()[3], 2});

  strides.emplace_back(1);
  auto upper_slice = hal::slice(
      ctx, h_idx_max, {0, 0, 0, 0, 0},
      {h_idx_max.shape()[0], h_idx_max.shape()[1] - 1, h_idx_max.shape()[2],
       h_idx_max.shape()[3], h_idx_max.shape()[4]},
      strides);

  auto bottom_slice = hal::slice(
      ctx, h_idx_max, {0, 1, 0, 0, 0},
      {h_idx_max.shape()[0], h_idx_max.shape()[1], h_idx_max.shape()[2],
       h_idx_max.shape()[3], h_idx_max.shape()[4]},
      strides);

  upper_slice = hal::mul(ctx, v_i_comp, upper_slice);
  bottom_slice = hal::mul(ctx, v_i_comp_not, bottom_slice);

  auto max_indices = hal::concatenate(ctx, {upper_slice, bottom_slice}, 4);

  return {max_ret, max_indices};
}

std::pair<spu::Value, spu::Value> ArgMax(SPUContext *ctx,
                                         const spu::Value &input,
                                         const Shape &ret_shape,
                                         const ReduceWindowConfig &config) {
  // Add a fast 1x2x2x1, no padding fast reduce
  auto no_padding =
      std::all_of(config.window_padding.begin(), config.window_padding.end(),
                  [](const std::pair<int64_t, int64_t> &p) {
                    return p.first == 0 && p.second == 0;
                  });
  if (config.window_shape == absl::Span<const int64_t>{1, 2, 2, 1} &&
      no_padding) {
    return ArgMax1x2x2x1NoPaddingWithoutDilation(ctx, input,
                                                 config.window_strides);
  }

  // Create eye
  size_t window_size =
      std::accumulate(config.window_shape.begin(), config.window_shape.end(), 1,
                      std::multiplies<size_t>());
  xt::xarray<bool> e = xt::eye<bool>({window_size, window_size}, 0);

  auto mask = hal::constant(ctx, e, DT_I1);

  auto result = ReduceWindowImpl(
      ctx, {input, mask}, {spu::Value(), spu::Value()}, ret_shape, config, true,
      true,
      [&](absl::Span<spu::Value const> lhs,
          absl::Span<spu::Value const> rhs) -> std::vector<spu::Value> {
        SPU_ENFORCE(lhs.size() == 2);
        auto c = hal::less(ctx, rhs[0], lhs[0]);
        c = hal::_prefer_a(ctx, c);
        // Select value
        auto v = hal::select(ctx, c, lhs[0], rhs[0]);
        // Select index
        auto c_i_shape = c.shape();
        c_i_shape.emplace_back(1);
        auto c_i = hal::reshape(ctx, c, c_i_shape);
        c_i_shape.back() = window_size;
        c_i = hal::broadcast_to(ctx, c_i, c_i_shape);
        auto i = hal::select(ctx, c_i, lhs[1], rhs[1]);

        return {v, i};
      });
  return {result[0], result[1]};
}

}  // namespace spu::kernel::hlo
