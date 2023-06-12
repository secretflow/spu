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

#include "libspu/kernel/hlo/indexing.h"

#include <cstring>

#include "llvm/ADT/STLExtras.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/hal.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/basic_ternary.h"
#include "libspu/kernel/hlo/basic_unary.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/reduce.h"
#include "libspu/kernel/hlo/utils.h"

// forward
namespace spu::kernel::hal::detail {
void hintNumberOfBits(const Value &a, size_t nbits);
}

namespace {
struct IndexIterationSpace {
  std::vector<int64_t> index_base;
  std::vector<int64_t> index_count;
  std::vector<int64_t> index_incr;
};

// Returns an IndexIterationSpace that iterates over the output batch
// dimensions while keeping the rest of the output dimensions clamped to 0.
IndexIterationSpace iterationSpaceForOutputBatchIndices(
    absl::Span<const int64_t> output_shape,
    const spu::kernel::hlo::GatherConfig &config) {
  int64_t output_rank = output_shape.size();
  std::vector<int64_t> index_base(output_rank, 0);
  std::vector<int64_t> index_count;
  index_count.reserve(output_rank);

  for (int64_t i = 0; i < output_rank; i++) {
    bool is_output_batch_dim = !std::binary_search(config.offsetDims.begin(),
                                                   config.offsetDims.end(), i);
    index_count.push_back(is_output_batch_dim ? output_shape[i] : 1);
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(output_rank, 1)};
}

// Return an IndexIterationSpace that iterates over the output slice
// dimensions while keeping the rest of the output dimensions clamped to 0.
IndexIterationSpace iterationSpaceForOutputOffsetIndices(
    int64_t output_rank, const spu::kernel::hlo::GatherConfig &config) {
  std::vector<int64_t> index_base(output_rank, 0);
  std::vector<int64_t> index_count(output_rank, 1);
  int64_t slice_sizes_idx = 0;

  for (int64_t i = 0; i < output_rank; i++) {
    bool is_output_window_dim = std::binary_search(config.offsetDims.begin(),
                                                   config.offsetDims.end(), i);
    if (is_output_window_dim) {
      while (std::binary_search(config.collapsedSliceDims.begin(),
                                config.collapsedSliceDims.end(),
                                slice_sizes_idx)) {
        slice_sizes_idx++;
      }
      index_count[i] = config.sliceSizes[slice_sizes_idx++];
    }
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(output_rank, 1)};
}

// This functor computes the contribution of start_indices to an input index
// corresponding to an output index.  That is, given an output index I, it
// picks out the batch indices in I and uses them to look up a starting index,
// G, from the start indices tensor, and expands G into the input space
// according to start_index_map.
class OutputBatchIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputBatchIndexToInputIndex(
      const spu::kernel::hlo::GatherConfig &config,
      absl::Span<const int64_t> input_shape,
      absl::Span<const int64_t> output_shape,
      const xt::xarray<int64_t> &start_indices)
      : config_(config), start_indices_(start_indices) {
    for (int64_t i = 0; i < static_cast<int64_t>(output_shape.size()); ++i) {
      output_dim_is_batch_dims_.push_back(!std::binary_search(
          config_.offsetDims.begin(), config_.offsetDims.end(), i));
    }

    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i) {
      int64_t index_of_input_dim_in_index_vector =
          std::distance(config_.startIndexMap.begin(),
                        std::find(config_.startIndexMap.begin(),
                                  config_.startIndexMap.end(), i));

      if (static_cast<size_t>(index_of_input_dim_in_index_vector) ==
          config_.startIndexMap.size()) {
        input_dim_value_to_index_vector_.push_back(-1);
      } else {
        input_dim_value_to_index_vector_.push_back(
            index_of_input_dim_in_index_vector);
      }
    }

    index_vector_index_.resize(start_indices_.shape().size());
    input_index_.resize(input_shape.size());
    int64_t index_vector_size = start_indices_.shape()[config.indexVectorDim];
    index_vector_.resize(index_vector_size);

    start_indices_shape_.reserve(start_indices_.shape().size());
    for (const auto &d : start_indices_.shape()) {
      start_indices_shape_.emplace_back(static_cast<int64_t>(d));
    }
  }

  // Returns the contribution of start_indices to the input index
  // corresponding to output_index.  See gather_inner_loop_body.
  //
  // This is conceptually  a stateless transformation from output_index to the
  // gather input index, but:
  //
  //  - Instead of allocating memory to represent the gather input index on
  //    every invocation we reuse the same storage for the result
  //    (input_index_), mutating it in place.
  //  - Instead of allocating buffers for temporary values like
  //    index_vector_index_ and index_vector on every invocation, we reuse the
  //    same storage for all invocations.
  //
  // This returns a Span into memory owned by the class.
  absl::Span<const int64_t> operator()(absl::Span<const int64_t> output_index) {
    propagateOutputIndexGatherDimsToIndexVectorIndex(output_index);
    fetchIndexVector();
    propagateIndexVectorToInputIndex();
    return input_index_;
  }

 private:
  // Propagates the batch dimensions from the output index into
  // index_vector_index_ by mutating index_vector_index_ in place.  Does not
  // update the dim_numbers.index_vector_dim() dimension -- that's the
  // dimension we iterate over in FetchIndexVector.
  void propagateOutputIndexGatherDimsToIndexVectorIndex(
      absl::Span<const int64_t> output_index) {
    int64_t index_vector_index_i = 0;
    for (int64_t i = 0, e = output_index.size(); i < e; i++) {
      if (!output_dim_is_batch_dims_[i]) {
        continue;
      }

      if (index_vector_index_i == config_.indexVectorDim) {
        index_vector_index_i++;
      }

      index_vector_index_[index_vector_index_i++] = output_index[i];
    }
  }

  // Populates index_vector_ by iterating over start_indices_ according to
  // index_vector_index_.
  void fetchIndexVector() {
    int64_t index_vector_dim = config_.indexVectorDim;
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      index_vector_[i] = start_indices_.data()[spu::flattenIndex(
          index_vector_index_, start_indices_shape_)];
    }
  }

  // Populates input_index_.
  void propagateIndexVectorToInputIndex() {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_index_vector_[i] != -1) {
        input_index_[i] = index_vector_[input_dim_value_to_index_vector_[i]];
      }
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
  // of the input index from the index vector.  See
  // PropagateIndexVectorToInputIndex.
  std::vector<int64_t> input_dim_value_to_index_vector_;

  // output_dim_is_batch_dims_[i] is true iff the output index i is a gather
  // dimension.
  std::vector<bool> output_dim_is_batch_dims_;

  // The buffer into which we construct an index into start_indices_ to fetch
  // the index vector.
  std::vector<int64_t> index_vector_index_;

  // The index vector fetched from start_indices_.
  std::vector<int64_t> index_vector_;

  // The result computed by this functor.  operator() returns a Span into
  // this vector.
  std::vector<int64_t> input_index_;

  const spu::kernel::hlo::GatherConfig &config_;
  const xt::xarray<int64_t> &start_indices_;
  std::vector<int64_t> start_indices_shape_;
};

// This functor computes the contribution of the offset indices in an output
// index to an input index.  That is, given an output index I it picks out the
// output offset indices in I and expands it into an index into the input
// shape.
class OutputOffsetIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputOffsetIndexToInputIndex(
      const spu::kernel::hlo::GatherConfig &config,
      absl::Span<const int64_t> input_shape,
      absl::Span<const int64_t> output_shape) {
    std::vector<int64_t> window_index_to_output_index;
    int64_t output_index_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(output_shape.size()); i++) {
      if (std::binary_search(config.offsetDims.begin(), config.offsetDims.end(),
                             i)) {
        window_index_to_output_index.push_back(output_index_count++);
      } else {
        output_index_count++;
      }
    }

    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); i++) {
      if (std::binary_search(config.collapsedSliceDims.begin(),
                             config.collapsedSliceDims.end(), i)) {
        input_dim_value_to_output_index_.push_back(-1);
      } else {
        input_dim_value_to_output_index_.push_back(
            window_index_to_output_index[window_dim_count++]);
      }
    }

    input_index_.resize(input_shape.size());
  }

  // Returns the contribution of the window indices to the input index
  // corresponding to output_index.  See gather_inner_loop_body.
  //
  // This is conceptually a stateless transformation from output_index to the
  // window input index, but instead of allocating memory to represent the
  // gather input index on every invocation we reuse the same storage for the
  // result (input_index_), mutating it in place.
  //
  // This returns a Span into memory owned by the class.
  absl::Span<const int64_t> operator()(absl::Span<const int64_t> output_index) {
    propagateOutputIndexWindowDimsToInputIndex(output_index);
    return input_index_;
  }

  // Returns for a given 'input_dim' the corresponding output dimension index,
  // or -1 if 'input_dim' is an elided window dimension.
  int64_t input_dim_value_to_output_index(int64_t input_dim) {
    return input_dim_value_to_output_index_[input_dim];
  }

 private:
  // Propagates window dimensions from the output index to input_index_ by
  // mutating input_index_ in place.
  void propagateOutputIndexWindowDimsToInputIndex(
      absl::Span<const int64_t> output_index) {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_output_index_[i] != -1) {
        input_index_[i] = output_index[input_dim_value_to_output_index_[i]];
      }
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
  // of the input index from the output index. See
  // PropagateOutputIndexWindowDimsToInputIndex.
  std::vector<int64_t> input_dim_value_to_output_index_;

  // The result computed by this functor.  operator() returns a Span into
  // this vector.
  std::vector<int64_t> input_index_;
};

spu::Value reshapedGatherIndices(spu::SPUContext *ctx, int64_t index_vector_dim,
                                 const spu::Value &start_indices) {
  if (start_indices.shape().size() != static_cast<size_t>(index_vector_dim)) {
    return start_indices;
  }

  auto new_shape = start_indices.shape();
  new_shape.push_back(1);

  return spu::kernel::hal::reshape(ctx, start_indices, new_shape);
}

spu::Value SecretLinearUpdateIndexing(spu::SPUContext *ctx,
                                      const spu::Value &operand,
                                      const spu::Value &update,
                                      const spu::Value &linear_idx) {
  // TODO: Consider utilizing DLP to improve performance
  SPU_ENFORCE(operand.shape().size() == 1, "operand must be a 1D tensor");
  SPU_ENFORCE(linear_idx.numel() == 1, "index must be a 1D indexing");
  SPU_ENFORCE(update.numel() == 1, "update must be a scalar");

  // Basic idea here:
  // eq(iota, idx) * update + !eq(iota, idx) * operand
  auto linear_idx_broadcasted =
      spu::kernel::hlo::Broadcast(ctx, linear_idx, {operand.numel()}, {});
  spu::Value idx_iota =
      spu::kernel::hlo::Iota(ctx, spu::DT_I64, operand.numel());
  auto mask = spu::kernel::hlo::Equal(ctx, linear_idx_broadcasted, idx_iota);

  auto c0 = spu::kernel::hlo::Constant(ctx, static_cast<int64_t>(0), {});
  auto i0 = spu::kernel::hlo::Cast(ctx, c0, c0.vtype(), operand.dtype());

  auto reverse_mask = spu::kernel::hlo::Not(ctx, mask);

  auto broadcast_update =
      spu::kernel::hlo::Broadcast(ctx, update, operand.shape(), {0});

  return spu::kernel::hlo::Add(
      ctx, spu::kernel::hlo::Mul(ctx, operand, reverse_mask),
      spu::kernel::hlo::Mul(ctx, broadcast_update, mask));
}

std::vector<spu::Value> ClampAndFlattenIndex(
    spu::SPUContext *ctx, absl::Span<const spu::Value> start_indices,
    absl::Span<const int64_t> iterate_shape,
    absl::Span<const int64_t> limit_shape) {
  // Transform start_indices
  // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i]
  // - size_indices[i])

  std::vector<spu::Value> clamped_start(start_indices.size());
  {
    std::vector<spu::Value> reshaped_start_indices;
    std::transform(start_indices.cbegin(), start_indices.cend(),
                   std::back_inserter(reshaped_start_indices),
                   [&](const spu::Value &x) {
                     return spu::kernel::hlo::Reshape(ctx, x, {1});
                   });

    auto concat_idx =
        spu::kernel::hlo::Concatenate(ctx, reshaped_start_indices, 0);
    auto lower_bound = spu::kernel::hlo::Constant(ctx, static_cast<int64_t>(0),
                                                  concat_idx.shape());
    lower_bound = spu::kernel::hlo::Cast(ctx, lower_bound, lower_bound.vtype(),
                                         concat_idx.dtype());

    std::vector<int64_t> upper_bound_pt(start_indices.size());
    for (size_t idx = 0; idx < upper_bound_pt.size(); ++idx) {
      upper_bound_pt[idx] = limit_shape[idx] - iterate_shape[idx];
    }
    auto upper_bound =
        spu::kernel::hlo::Constant(ctx, upper_bound_pt, concat_idx.shape());
    upper_bound = spu::kernel::hlo::Cast(ctx, upper_bound, upper_bound.vtype(),
                                         concat_idx.dtype());

    auto c = spu::kernel::hlo::Clamp(ctx, concat_idx, lower_bound, upper_bound);
    for (int64_t idx = 0; idx < static_cast<int64_t>(clamped_start.size());
         ++idx) {
      clamped_start[idx] = spu::kernel::hlo::Reshape(
          ctx, spu::kernel::hlo::Slice(ctx, c, {idx}, {idx + 1}, {1}), {});
    }
  }

  // Now flatten start index
  auto linear_idx =
      spu::kernel::hlo::Constant(ctx, static_cast<int64_t>(0), {});
  int64_t stride = 1;
  for (int64_t idx = iterate_shape.size() - 1; idx >= 0; --idx) {
    linear_idx = spu::kernel::hlo::Add(
        ctx, linear_idx,
        spu::kernel::hlo::Mul(ctx, clamped_start[idx],
                              spu::kernel::hlo::Constant(ctx, stride, {})));
    stride *= limit_shape[idx];
  }

  // Now compute offsets of each index
  std::vector<int64_t> base(iterate_shape.size(), 0);
  std::vector<int64_t> incr(iterate_shape.size(), 1);

  std::vector<int64_t> flatten_idx;
  spu::kernel::forEachIndex(
      limit_shape, base, iterate_shape, incr,
      [&flatten_idx, &limit_shape](absl::Span<const int64_t> idx) {
        flatten_idx.emplace_back(spu::flattenIndex(idx, limit_shape));
      });

  auto num_index = spu::calcNumel(iterate_shape);
  std::vector<spu::Value> linear_indices;
  linear_indices.reserve(num_index);
  auto added = spu::kernel::hlo::Add(
      ctx,
      spu::kernel::hlo::Broadcast(
          ctx, spu::kernel::hlo::Reshape(ctx, linear_idx, {1}), {num_index},
          {0}),
      spu::kernel::hlo::Constant(ctx, flatten_idx, {num_index}));
  for (int64_t idx = 0; idx < num_index; ++idx) {
    linear_indices.emplace_back(spu::kernel::hlo::Reshape(
        ctx, spu::kernel::hlo::Slice(ctx, added, {idx}, {idx + 1}, {1}), {}));
  }
  return linear_indices;
}

}  // namespace

namespace spu::kernel::hlo {

spu::Value Gather(SPUContext *ctx, const spu::Value &operand,
                  const spu::Value &start_indices, const GatherConfig &config,
                  absl::Span<const int64_t> result_shape) {
  // If input is empty, short circuit
  if (operand.numel() == 0) {
    return operand;
  }

  auto start_indices_value =
      reshapedGatherIndices(ctx, config.indexVectorDim, start_indices);

  SPU_ENFORCE(start_indices.isPublic());

  auto start_induces = getIndices(ctx, start_indices_value);

  // We iterate over the gather dimensions in the output shape in an outer
  // loop nest, and iterate over the window dimensions in the output shape in
  // an inner loop nest.
  IndexIterationSpace start_indices_iteration_space =
      iterationSpaceForOutputBatchIndices(result_shape, config);
  IndexIterationSpace offset_indices_iteration_space =
      iterationSpaceForOutputOffsetIndices(result_shape.size(), config);

  // Scratch buffers that hold an index in the output shape and the
  // corresponding index in the input shape.
  // If input is empty, short circuit it
  auto operand_shape = operand.shape();
  std::vector<int64_t> input_index(operand_shape.size());
  std::vector<int64_t> output_index(result_shape.size());
  std::vector<int64_t> input_index_clamped(operand_shape.size());

  OutputBatchIndexToInputIndex output_batch_index_to_input_index(
      config, /*input_shape=*/operand_shape,
      /*output_shape=*/result_shape, start_induces);
  OutputOffsetIndexToInputIndex output_offset_index_to_input_index(
      config, /*input_shape=*/operand_shape,
      /*output_shape=*/result_shape);

  spu::Value result(NdArrayRef(operand.data().eltype(), result_shape),
                    operand.dtype());

  auto gather_inner_loop_body =
      [&](absl::Span<const int64_t> output_window_index,
          absl::Span<const int64_t> input_gather_index,
          absl::Span<const int64_t> output_gather_index) {
        auto input_window_index =
            output_offset_index_to_input_index(output_window_index);
        for (int i = 0, e = output_index.size(); i < e; i++) {
          output_index[i] = output_gather_index[i] + output_window_index[i];
        }
        for (int i = 0, e = input_gather_index.size(); i < e; i++) {
          int64_t output_dim = output_offset_index_to_input_index
                                   .input_dim_value_to_output_index(i);
          // If 'output_dim' is -1, it means 'i' is an elided window dim. This
          // means we set the iteration index to 0, so for the purpose of the
          // following calculations we can consider the output dimension size
          // to be 1.
          int64_t output_dim_size =
              output_dim == -1 ? 1 : result_shape[output_dim];
          // Clamp the gather index so that the gather region fits in the
          // operand. input_index_clamped[i] = clamp(input_gather_index[i], 0,
          //                                       operand_shape.dimensions(i)
          //                                       - output_dim_size);
          input_index_clamped[i] =
              std::min(operand_shape[i] - output_dim_size,
                       std::max(int64_t{0}, input_gather_index[i]));
        }
        for (int i = 0, e = input_index.size(); i < e; i++) {
          input_index[i] = input_index_clamped[i] + input_window_index[i];
        }

        result.data().update_slice(operand.data().slice_scalar_at(input_index),
                                   output_index);
      };

  auto gather_outer_loop_body =
      [&](absl::Span<const int64_t> output_gather_index) {
        auto input_gather_index =
            output_batch_index_to_input_index(output_gather_index);
        forEachIndex(result_shape, offset_indices_iteration_space.index_base,
                     offset_indices_iteration_space.index_count,
                     offset_indices_iteration_space.index_incr,
                     [&](absl::Span<const int64_t> output_window_index) {
                       return gather_inner_loop_body(output_window_index,
                                                     input_gather_index,
                                                     output_gather_index);
                     });
      };

  forEachIndex(result_shape, start_indices_iteration_space.index_base,
               start_indices_iteration_space.index_count,
               start_indices_iteration_space.index_incr,
               gather_outer_loop_body);

  return result;
}

spu::Value DynamicUpdateSlice(SPUContext *ctx, const spu::Value &operand,
                              const spu::Value &update,
                              absl::Span<const spu::Value> start_indices) {
  // Basic idea here, get a ref slice and update the whole slice..
  SPU_ENFORCE_EQ(start_indices.size(), operand.shape().size());
  SPU_ENFORCE_EQ(start_indices.size(), update.shape().size());
  SPU_ENFORCE(!start_indices.empty());

  if (start_indices[0].isSecret()) {
    // flatten first
    spu::Value flattened_operand =
        hal::reshape(ctx, operand, {operand.numel()});

    spu::Value flattened_update = Reshape(ctx, update, {update.numel()});

    auto flattened_indices = ClampAndFlattenIndex(
        ctx, start_indices, update.shape(), operand.shape());

    spu::Value ret = flattened_operand;

    for (int64_t n = 0; n < static_cast<int64_t>(flattened_indices.size());
         ++n) {
      auto update_slice = Slice(ctx, flattened_update, {n}, {n + 1}, {1});
      ret = SecretLinearUpdateIndexing(ctx, ret, update_slice,
                                       flattened_indices[n]);
    }

    return Reshape(ctx, ret, operand.shape());

  } else {
    // Start indices
    std::vector<int64_t> start_indices_i64(start_indices.size());
    for (const auto &idx : llvm::enumerate(start_indices)) {
      auto v_idx = idx.value();
      if (v_idx.isSecret()) {
        v_idx = hal::reveal(ctx, v_idx);
        SPDLOG_WARN("Reveal {}th start index of DynamicUpdateSlice",
                    idx.index());
      }
      start_indices_i64[idx.index()] = getIndices(ctx, v_idx)[0];
      // Transform start_indices
      // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i]
      // - update.dimension_size[i])
      start_indices_i64[idx.index()] = std::min(
          std::max(start_indices_i64[idx.index()], static_cast<int64_t>(0)),
          operand.shape()[idx.index()] - update.shape()[idx.index()]);
    }

    return UpdateSlice(ctx, operand, update, start_indices_i64);
  }
}

spu::Value UpdateSlice(SPUContext *ctx, const spu::Value &in,
                       const spu::Value &update,
                       absl::Span<const int64_t> start_indices) {
  return hal::update_slice(ctx, in, update, start_indices);
}

spu::Value SecretDynamicSliceImpl(SPUContext *ctx, const spu::Value &operand,
                                  absl::Span<const int64_t> slice_size,
                                  absl::Span<const spu::Value> start_indices) {
  if (slice_size[0] == operand.shape()[0]) {
    if (slice_size.size() == 1) {
      return operand;
    }

    // Full dimension
    std::vector<int64_t> start(operand.shape().size(), 0);
    std::vector<int64_t> limit = operand.shape();
    std::vector<int64_t> strides(operand.shape().size(), 1);

    std::vector<spu::Value> results(operand.shape()[0]);
    for (int64_t idx = 0; idx < operand.shape()[0]; ++idx) {
      start[0] = idx;
      limit[0] = idx + 1;
      // Slice one...
      auto sliced = hal::slice(ctx, operand, start, limit, strides);
      // Remove leading one
      auto reshaped =
          hal::reshape(ctx, sliced, absl::MakeSpan(sliced.shape()).subspan(1));
      // Do indexing
      auto indexed = SecretDynamicSliceImpl(
          ctx, reshaped, slice_size.subspan(1), start_indices.subspan(1));
      // Add leading one dimension back
      std::vector<int64_t> result_shape(indexed.shape().size() + 1, 1);
      for (size_t idx = 0; idx < indexed.shape().size(); ++idx) {
        result_shape[idx + 1] = indexed.shape()[idx];
      }
      results[idx] = hal::reshape(ctx, indexed, result_shape);
    }

    if (results.size() == 1) {
      return results[0];
    }
    return hal::concatenate(ctx, results, 0);
  }

  // equal(adjusted, iota)
  spu::Value mask;
  spu::Value idx_iota =
      hal::iota(ctx, start_indices[0].dtype(), operand.shape()[0]);

  mask = hal::equal(ctx,
                    hal::broadcast_to(ctx, start_indices[0], idx_iota.shape()),
                    idx_iota);

  if (slice_size[0] >= 1) {
    auto pad_value = hal::seal(ctx, hal::constant(ctx, false, mask.dtype()));
    pad_value = hal::_cast_type(ctx, pad_value, mask.storage_type());
    mask = hal::pad(ctx, mask, pad_value, {slice_size[0]}, {0}, {0});
    // FIXME(juhou): we should avoid setting the BShr here
    // However mask.storage_type().as<BShare>->nbits() is not 1 after the
    // padding. We implicitly set mask as a 1-bit BShr so that the following
    // hal::matmul can use a much lighter B2A proc for both ABY3 and CHEETAH.
    hal::detail::hintNumberOfBits(mask, 1);
  }

  // foreach
  std::vector<spu::Value> results(slice_size[0]);

  // Do collapse inner dims when necessary
  auto collapsed_operand = operand;
  if (collapsed_operand.shape().size() > 2) {
    // Reshape from XxYxZ to Xx(Y*Z)
    collapsed_operand =
        hal::reshape(ctx, collapsed_operand,
                     {operand.shape()[0],
                      calcNumel(absl::MakeSpan(operand.shape()).subspan(1))});
  }

  std::vector<int64_t> indexed_shape = operand.shape();
  indexed_shape[0] = 1;

  for (int64_t idx = 0; idx < slice_size[0]; ++idx) {
    auto mask_slice =
        hal::slice(ctx, mask, {mask.numel() - idx - operand.shape()[0]},
                   {mask.numel() - idx}, {1});
    mask_slice = hal::reshape(ctx, mask_slice, {1, mask_slice.numel()});

    results[idx] = hal::matmul(ctx, mask_slice, collapsed_operand);

    results[idx] = hal::reshape(ctx, results[idx], indexed_shape);
  }

  if (slice_size.size() > 1) {
    std::vector<int64_t> result_shape(slice_size.begin(), slice_size.end());
    result_shape[0] = 1;
    // Keep indexing deeper
    for (int64_t idx = 0; idx < slice_size[0]; ++idx) {
      results[idx] = hal::reshape(
          ctx, results[idx], absl::MakeSpan(results[idx].shape()).subspan(1));
      results[idx] = SecretDynamicSliceImpl(
          ctx, results[idx], slice_size.subspan(1), start_indices.subspan(1));
      results[idx] = hal::reshape(ctx, results[idx], result_shape);
    }
  }

  if (results.size() == 1) {
    return results[0];
  }

  return hal::concatenate(ctx, results, 0);
}

spu::Value SecretDynamicSlice(SPUContext *ctx, const spu::Value &operand,
                              absl::Span<const int64_t> slice_size,
                              absl::Span<const spu::Value> start_indices) {
  // Clamp all indices
  auto lower_bound =
      hlo::Constant(ctx, std::vector<int64_t>(slice_size.size(), 0),
                    {static_cast<int64_t>(slice_size.size())});

  std::vector<int64_t> limit = operand.shape();
  for (size_t idx = 0; idx < limit.size(); ++idx) {
    limit[idx] -= slice_size[idx];
  }
  auto upper_bound =
      hlo::Constant(ctx, limit, {static_cast<int64_t>(slice_size.size())});

  // Cast to proper type
  lower_bound = hlo::Cast(ctx, lower_bound, lower_bound.vtype(),
                          start_indices[0].dtype());
  upper_bound = hlo::Cast(ctx, upper_bound, upper_bound.vtype(),
                          start_indices[0].dtype());

  // Reshape from scalar to {1} to make concat happy
  std::vector<spu::Value> adjusted_start_indices;
  std::transform(start_indices.cbegin(), start_indices.cend(),
                 std::back_inserter(adjusted_start_indices),
                 [&](const Value &x) { return hal::reshape(ctx, x, {1}); });

  auto adjusted_all_indices = hal::broadcast_to(
      ctx,
      hal::clamp(ctx, hal::concatenate(ctx, adjusted_start_indices, 0),
                 lower_bound, upper_bound),
      {operand.shape()[0]});

  for (int64_t idx = 0;
       idx < static_cast<int64_t>(adjusted_start_indices.size()); ++idx) {
    adjusted_start_indices[idx] =
        hal::slice(ctx, adjusted_all_indices, {idx}, {idx + 1}, {1});
  }

  return SecretDynamicSliceImpl(ctx, operand, slice_size,
                                adjusted_start_indices);
}

spu::Value DynamicSlice(SPUContext *ctx, const spu::Value &operand,
                        absl::Span<const int64_t> slice_size,
                        absl::Span<const spu::Value> start_indices) {
  SPU_ENFORCE_EQ(slice_size.size(), start_indices.size());
  SPU_ENFORCE_EQ(slice_size.size(), operand.shape().size());
  SPU_ENFORCE(!start_indices.empty());

  if (start_indices[0].isSecret()) {
    return SecretDynamicSlice(ctx, operand, slice_size, start_indices);
  } else {
    // Start indices
    std::vector<int64_t> start_indices_i64(start_indices.size());
    for (const auto &idx : llvm::enumerate(start_indices)) {
      auto v_idx = idx.value();
      start_indices_i64[idx.index()] = getIndices(ctx, v_idx)[0];
      // Transform start_indices
      // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i]
      // - size_indices[i])
      start_indices_i64[idx.index()] = std::min(
          std::max(start_indices_i64[idx.index()], static_cast<int64_t>(0)),
          operand.shape()[idx.index()] - slice_size[idx.index()]);
    }

    // Limit
    std::vector<int64_t> limit(start_indices_i64);
    for (size_t idx = 0; idx < limit.size(); ++idx) {
      limit[idx] += slice_size[idx];
    }

    // Strides is always 1
    std::vector<int64_t> strides(limit.size(), 1);

    return hal::slice(ctx, operand, start_indices_i64, limit, strides);
  }
}

spu::Value FilterByMask(SPUContext *ctx, const spu::Value &operand,
                        absl::Span<const uint8_t> mask) {
  // Sanity
  SPU_ENFORCE(operand.shape().size() == 1, "Operand must be a vector");
  SPU_ENFORCE(mask.size() == (size_t)operand.shape()[0],
              "filter must be same length as operand");

  // Count result size
  int64_t num_true = 0;
  for (auto m : mask) {
    if (m != 0) {
      ++num_true;
    }
  }

  std::vector<int64_t> indices(num_true);
  int64_t indices_counter = 0;
  for (int64_t mask_idx = 0; mask_idx != static_cast<int64_t>(mask.size());
       ++mask_idx) {
    if (mask[mask_idx] != 0) {
      indices[indices_counter++] = mask_idx;
    }
  }

  return Value(operand.data().linear_gather(indices), operand.dtype());
}

spu::Value LinearGather(SPUContext *ctx, const spu::Value &in,
                        absl::Span<const int64_t> indices) {
  return Value(in.data().linear_gather(indices), in.dtype());
}

void LinearScatterInPlace(SPUContext *ctx, spu::Value &in,
                          const spu::Value &update,
                          absl::Span<const int64_t> indices) {
  if (in.data().eltype() != update.data().eltype()) {
    auto common_type =
        hal::_common_type(ctx, in.data().eltype(), update.data().eltype());
    in = hal::_cast_type(ctx, in, common_type).setDtype(in.dtype());
    LinearScatterInPlace(
        ctx, in,
        hal::_cast_type(ctx, update, common_type).setDtype(update.dtype()),
        indices);
    return;
  }
  in.data().linear_scatter(update.data(), indices);
}

}  // namespace spu::kernel::hlo
