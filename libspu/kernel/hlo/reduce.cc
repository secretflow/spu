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
#include "libspu/core/shape_util.h"
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

  std::vector<int64_t> slice_begin(inputs.back().shape().size(), 0);
  std::vector<int64_t> slice_end = inputs.back().shape();
  std::vector<int64_t> slice_strides(inputs.back().shape().size(), 1);
  int64_t len = outputs[0].shape()[axis];
  while (len > 1) {
    const int64_t half = len / 2;

    // lhs & rhs
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      slice_begin[axis] = 0;
      slice_end[axis] = half;

      lhs[idx] = hal::slice(
          ctx, outputs[idx],
          absl::MakeSpan(slice_begin.data(), outputs[idx].shape().size()),
          absl::MakeSpan(slice_end.data(), outputs[idx].shape().size()),
          absl::MakeSpan(slice_strides.data(), outputs[idx].shape().size()));

      slice_begin[axis] = half;
      slice_end[axis] = 2 * half;
      rhs[idx] = hal::slice(
          ctx, outputs[idx],
          absl::MakeSpan(slice_begin.data(), outputs[idx].shape().size()),
          absl::MakeSpan(slice_end.data(), outputs[idx].shape().size()),
          absl::MakeSpan(slice_strides.data(), outputs[idx].shape().size()));
    }

    // tail
    if (len % 2 == 1) {
      slice_begin[axis] = 2 * half;
      slice_end[axis] = len;
      std::vector<spu::Value> &tail = tails.emplace(nargs);
      for (size_t idx = 0; idx < outputs.size(); ++idx) {
        tail[idx] = hal::slice(
            ctx, outputs[idx],
            absl::MakeSpan(slice_begin.data(), outputs[idx].shape().size()),
            absl::MakeSpan(slice_end.data(), outputs[idx].shape().size()),
            absl::MakeSpan(slice_strides.data(), outputs[idx].shape().size()));
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

spu::Value ExpandStridedWindow(
    SPUContext *ctx, const spu::Value &base,
    absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
  const auto &base_shape = base.shape();
  const size_t ndim = base_shape.size();

  SPU_ENFORCE(ndim == window_shape.size() &&    //
              ndim == window_strides.size() &&  //
              ndim == padding.size());

  // calculate output shape
  std::vector<int64_t> expanded_shape(ndim, 0);
  for (size_t dim = 0; dim < ndim; dim++) {
    int64_t padded_size =
        padding[dim].first + padding[dim].second + base_shape[dim];
    expanded_shape[dim] =
        ((padded_size - window_shape[dim]) / window_strides[dim] + 1) *
        window_shape[dim];
  }

  const std::vector<int64_t> window_dilations(window_shape.size(), 1);
  const std::vector<int64_t> base_dilations(base.shape().size(), 1);
  // expand it, assume padding & dialation element is zero.
  spu::Value expanded = hal::zeros(ctx, base.dtype(), expanded_shape);
  if (base.isSecret()) {
    expanded = hal::seal(ctx, expanded);
  }

  auto numel = calcNumel(expanded_shape);

  yacl::parallel_for(
      0, numel, computeTaskSize(numel), [&](int64_t begin, int64_t end) {
        std::vector<int64_t> expanded_index =
            unflattenIndex(begin, expanded_shape);

        std::vector<int64_t> window_count_index(ndim, 0);
        std::vector<int64_t> window_index(ndim, 0);

        for (int64_t idx = begin; idx < end; ++idx) {
          for (size_t dim = 0; dim < ndim; dim++) {
            window_index[dim] = expanded_index[dim] % window_shape[dim];
            window_count_index[dim] = expanded_index[dim] / window_shape[dim];
          }

          std::vector<int64_t> base_index(ndim, 0);
          bool out_of_bound = getBaseIndexFromWindowIndex(
              window_shape, window_strides, window_dilations, padding,
              base_shape, base_dilations, window_count_index, window_index,
              absl::MakeSpan(base_index));

          if (!out_of_bound) {
            // TODO: anti-pattern, do not use .data(), use ops instead.
            expanded.data().update_slice(
                base.data().slice_scalar_at(base_index), expanded_index);
          }

          if (!bumpIndices<int64_t>(expanded_shape,
                                    absl::MakeSpan(expanded_index))) {
            break;
          }
        }
      });

  return expanded;
}

spu::Value ConvertToTiledLayout(SPUContext *ctx, const spu::Value &in,
                                absl::Span<const int64_t> block_shape) {
  // Note(jint): use pad+reshape+transpose to convert from column layout to
  // tiled layout.
  //
  // For example, in shape = [6, 12], window = [2, 3]
  // The result is [3, 4, 2, 3]
  SPU_ENFORCE(in.shape().size() == block_shape.size());
  std::vector<int64_t> tiled_shape;
  for (size_t dim = 0; dim < in.shape().size(); dim++) {
    SPU_ENFORCE(in.shape()[dim] % block_shape[dim] == 0);
    tiled_shape.push_back(in.shape()[dim] / block_shape[dim]);
    tiled_shape.push_back(block_shape[dim]);
  }
  std::vector<int64_t> perm(tiled_shape.size(), 0);
  std::iota(perm.begin(), perm.end(), 0);
  std::stable_partition(perm.begin(), perm.end(),
                        [](int64_t x) { return x % 2 == 0; });

  auto out = hal::reshape(ctx, in, tiled_shape);
  return hal::transpose(ctx, out, perm);
}

std::vector<spu::Value> ReduceWindowWithoutDilation(
    SPUContext *ctx, absl::Span<const spu::Value> inputs,
    absl::Span<const spu::Value> init_values,
    absl::Span<const int64_t> window_shape,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> window_padding,
    bool last_operand_is_window_mask, bool ignore_init_value,
    absl::Span<const int64_t> ret_shape, const BatchedValueBinaryFn &reducer) {
  const size_t nargs =
      last_operand_is_window_mask ? inputs.size() - 1 : inputs.size();

  auto window_size = std::accumulate(window_shape.begin(), window_shape.end(),
                                     1, std::multiplies<>());

  // expand the operand, simplify following actions without strides and padding.
  std::vector<spu::Value> expanded;
  for (size_t idx = 0; idx < nargs; ++idx) {
    const auto &input = inputs[idx];
    auto x = ExpandStridedWindow(ctx, input, window_shape, window_strides,
                                 window_padding);
    expanded.emplace_back(ConvertToTiledLayout(ctx, x, window_shape));
  }

  if (last_operand_is_window_mask) {
    auto mask = inputs.back();
    std::vector<int64_t> shape(expanded.back().shape().size() + 1, 1);
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
  std::vector<int64_t> tiled_1d_shape(ret_shape.begin(), ret_shape.end());
  tiled_1d_shape.push_back(window_size);
  for (size_t idx = 0; idx < nargs; ++idx) {
    // reshape to tiled_1d
    expanded[idx] = hal::reshape(ctx, expanded[idx], tiled_1d_shape);
  }

  if (last_operand_is_window_mask) {
    std::vector<int64_t> tiled_1d_shape_mask(tiled_1d_shape.begin(),
                                             tiled_1d_shape.end());
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
    std::vector<int64_t> mask_ret_shape(ret_shape.begin(), ret_shape.end());
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
    absl::Span<const spu::Value> init_values,
    absl::Span<const int64_t> ret_shape, const ReduceWindowConfig &config,
    bool last_operand_is_window_mask, bool ignore_init_value,
    const BatchedValueBinaryFn &reducer) {
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
  std::vector<int64_t> window_index(ndims, 0);
  int64_t nargs = inputs.size();

  // Init...
  std::vector<spu::Value> rets(nargs);
  for (int64_t idx = 0; idx < nargs; ++idx) {
    rets[idx] = hal::expand(ctx, init_values[idx], ret_shape);
  }

  // For each resulting dimension, calculate and assign computed value.
  auto evaluate_impl =
      [&](absl::Span<int64_t const> output_index) -> std::vector<spu::Value> {
    std::vector<spu::Value> ret;
    RunOnWindowIndex(
        config.window_shape, config.window_strides, config.window_dilations,
        config.window_padding, inputs[0].shape(), config.base_dilations,
        output_index, window_index,
        [&](absl::Span<const int64_t> operand_index) {
          for (int64_t idx = 0; idx < nargs; ++idx) {
            auto element =
                hal::slice_scalar_at(ctx, inputs[idx], operand_index);
            ret.emplace_back(std::move(element));
          }
        });
    return ret;
  };

  // For each window index
  std::vector<spu::Value> batches(nargs);
  for (int64_t idx = 0; idx < nargs; ++idx) {
    batches[idx] =
        hal::expand(ctx, hal::slice_scalar_at(ctx, inputs[idx], {}), ret_shape);
  }

  do {
    // Collect one element from each window
    std::vector<int64_t> output_index(ret_shape.size(), 0);
    do {
      auto r = evaluate_impl(output_index);
      if (!r.empty()) {
        for (int64_t idx = 0; idx < nargs; ++idx) {
          batches[idx].data().update_slice(r[idx].data(), output_index);
        }
      }
    } while (bumpIndices(ret_shape, absl::MakeSpan(output_index)));

    // Now run the batch
    rets = reducer(rets, batches);

  } while (bumpIndices(config.window_shape, absl::MakeSpan(window_index)));

  return rets;
}

std::vector<spu::Value> ReduceWindow(SPUContext *ctx,
                                     absl::Span<const spu::Value> inputs,
                                     absl::Span<const spu::Value> init_values,
                                     absl::Span<const int64_t> ret_shape,
                                     const ReduceWindowConfig &config,
                                     const BatchedValueBinaryFn &reducer,
                                     bool ignore_init_values) {
  return ReduceWindowImpl(ctx, inputs, init_values, ret_shape, config, false,
                          ignore_init_values, reducer);
}

std::vector<spu::Value> Reduce(SPUContext *ctx,
                               absl::Span<const spu::Value> inputs,
                               absl::Span<const spu::Value> init_values,
                               absl::Span<const int64_t> dims_to_reduce,
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

  const std::vector<int64_t> in_shape = inputs[0].shape();

  std::vector<int64_t> perm(in_shape.size(), 0);
  std::iota(perm.begin(), perm.end(), 0);
  // swap axes, move the dims to reduce to inner most.
  std::stable_partition(perm.begin(), perm.end(), [&](int64_t axis) {
    return std::find(dims_to_reduce.begin(), dims_to_reduce.end(), axis) ==
           dims_to_reduce.end();
  });

  std::vector<int64_t> flat_shape;
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
  std::vector<int64_t> out_shape = inputs[0].shape();
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
    SPUContext *ctx, const spu::Value &input,
    absl::Span<const int64_t> window_strides) {
  auto input_shape = input.shape();

  spu::Value h_max;
  spu::Value h_idx_max;

  std::vector<int64_t> strides(window_strides.size(), 1);
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
                                         absl::Span<const int64_t> ret_shape,
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
      ctx, {input, mask}, {}, ret_shape, config, true, true,
      [&](absl::Span<spu::Value const> lhs,
          absl::Span<spu::Value const> rhs) -> std::vector<spu::Value> {
        SPU_ENFORCE(lhs.size() == 2);
        auto c = hal::less(ctx, rhs[0], lhs[0]);
        c = hal::_prefer_a(ctx, c);
        // Select value
        auto v = hal::select(ctx, c, lhs[0], rhs[0]);
        // Select index
        std::vector<int64_t> c_i_shape = c.shape();
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
