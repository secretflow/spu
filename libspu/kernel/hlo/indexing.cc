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

#include "libspu/core/ndarray_ref.h"
#include "libspu/kernel/hal/hal.h"
#include "libspu/kernel/hlo/utils.h"
#include "libspu/kernel/value.h"

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

spu::Value reshapedGatherIndices(spu::HalContext *ctx, int64_t index_vector_dim,
                                 const spu::Value &start_indices) {
  if (start_indices.shape().size() != static_cast<size_t>(index_vector_dim)) {
    return start_indices;
  }

  auto new_shape = start_indices.shape();
  new_shape.push_back(1);

  return spu::kernel::hal::reshape(ctx, start_indices, new_shape);
}

}  // namespace

namespace spu::kernel::hlo {

spu::Value Gather(HalContext *ctx, const spu::Value &operand,
                  const spu::Value &start_indicies, const GatherConfig &config,
                  absl::Span<const int64_t> result_shape) {
  // If input is empty, short circuit
  if (operand.numel() == 0) {
    return operand;
  }

  auto start_indices_value =
      reshapedGatherIndices(ctx, config.indexVectorDim, start_indicies);

  if (start_indices_value.isSecret() &&
      ctx->rt_config().reveal_secret_indicies()) {
    start_indices_value = hal::reveal(ctx, start_indices_value);
    SPDLOG_WARN("Reveal start indicies value of GatherOp");
  }

  auto start_induces = getIndicies(ctx, start_indices_value);

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

        result.copyElementFrom(operand, input_index, output_index);
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

spu::Value FilterByMask(HalContext *ctx, const spu::Value &operand,
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

  spu::Value result({operand.data().eltype(), {num_true}}, operand.dtype());

  const auto *in_ptr = &operand.data().at({0});
  auto *out_ptr = &result.data().at({0});
  auto elsize = operand.elsize();

  // Copy...
  for (int64_t idx = 0; idx < operand.shape()[0]; ++idx) {
    if (mask[idx] != 0) {
      std::memcpy(out_ptr, in_ptr, elsize);
      out_ptr += elsize;
    }
    in_ptr += operand.strides()[0] * elsize;
  }

  return result;
}

}  // namespace spu::kernel::hlo
