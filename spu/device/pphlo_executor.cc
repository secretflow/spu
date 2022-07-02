// Copyright 2021 Ant Group Co., Ltd.
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

#include "spu/device/pphlo_executor.h"

#include <algorithm>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "spdlog/spdlog.h"
#include "yasl/base/exception.h"
#include "yasl/utils/thread_pool.h"

#include "spu/core/shape_util.h"
#include "spu/core/xt_helper.h"
#include "spu/device/frame.h"
#include "spu/dialect/pphlo_dialect.h"
#include "spu/dialect/pphlo_ops.h"
#include "spu/dialect/pphlo_types.h"
#include "spu/hal/constants.h"
#include "spu/hal/context.h"
#include "spu/hal/hal.h"
#include "spu/hal/permute_util.h"
#include "spu/hal/polymorphic.h"
#include "spu/hal/test_util.h"
#include "spu/hal/type_cast.h"
#include "spu/hal/value.h"

namespace spu::device {

namespace {

std::string printLocation(const mlir::Location &loc) {
  std::string pstr;
  llvm::raw_string_ostream ss(pstr);
  loc->print(ss);
  ss.flush();
  return pstr;
}

std::vector<int64_t> build_shape(llvm::ArrayRef<int64_t> shape) {
  std::vector<int64_t> ret(shape.size());

  for (size_t idx = 0; idx < ret.size(); ++idx) {
    ret[idx] = shape[idx];
  }

  return ret;
}

hal::Value makeZeros(HalContext *ctx, Visibility vis, DataType dtype,
                     llvm::ArrayRef<int64_t> shape) {
  hal::Value scalar;
  switch (dtype) {
  case DT_I1: {
    scalar = hal::make_value(ctx, vis, false);
    break;
  }
  case DT_I8: {
    scalar = hal::make_value(ctx, vis, static_cast<std::int8_t>(0));
    break;
  }
  case DT_U8: {
    scalar = hal::make_value(ctx, vis, static_cast<std::uint8_t>(0));
    break;
  }
  case DT_I16: {
    scalar = hal::make_value(ctx, vis, static_cast<std::int16_t>(0));
    break;
  }
  case DT_U16: {
    scalar = hal::make_value(ctx, vis, static_cast<std::uint16_t>(0));
    break;
  }
  case DT_I32: {
    scalar = hal::make_value(ctx, vis, static_cast<std::int32_t>(0));
    break;
  }
  case DT_U32: {
    scalar = hal::make_value(ctx, vis, static_cast<std::uint32_t>(0));
    break;
  }
  case DT_I64: {
    scalar = hal::make_value(ctx, vis, static_cast<std::int64_t>(0));
    break;
  }
  case DT_U64: {
    scalar = hal::make_value(ctx, vis, static_cast<std::uint64_t>(0));
    break;
  }
  case DT_FXP: {
    scalar = hal::make_value(ctx, vis, 0.0F);
    break;
  }
  default: {
    YASL_THROW("Should not hit, dtype = {}", dtype);
  }
  }

  if (shape.empty()) {
    return scalar;
  } else {
    return hal::broadcast_to(ctx, scalar, build_shape(shape));
  }
}

void RunOnWindowIndex(
    absl::Span<int64_t> window_shape, absl::Span<int64_t> window_strides,
    absl::Span<int64_t> window_dilation,
    absl::Span<std::pair<int64_t, int64_t>> window_padding,
    absl::Span<const int64_t> base_shape, absl::Span<int64_t> base_dilation,
    const absl::Span<int64_t> window_count_index,
    absl::Span<const int64_t> window_index,
    const std::function<void(const std::vector<int64_t> &)> &f) {
  const int64_t rank = base_shape.size();
  std::vector<int64_t> base_index(rank);
  bool out_of_bound = false;
  for (int64_t i = 0; i < rank; ++i) {
    // Padding is applied to the dilated base. Say that padding is 3 and
    // dilation is 2 for some dimension. After applying base dilation and
    // padding, the dimension looks like:
    // P P P E D D E D D ... E D D E P P P
    // where E are the elements and D are the holes. So, the elements are
    // located in indices: padding + k*base_dilation for k = {0, 1, 2, ...}.
    // We are accessing elements in the transformed base at indices:
    // window_count_index * stride + window_index * window_dilation.
    // Solving for k gives us
    // (win_count_i * stride + win_i * win_dilation - pad) / base_dilation
    // When this is a natural number, we index an original element.
    // Otherwise, we index a 0 (pad or hole), and we don't need to apply
    // the callback f.
    base_index[i] = window_count_index[i] * window_strides[i] +
                    window_index[i] * window_dilation[i] -
                    window_padding[i].first;
    if (base_index[i] % base_dilation[i] != 0) {
      out_of_bound = true;
      break;
    }
    base_index[i] /= base_dilation[i];
    if (base_index[i] < 0 || base_index[i] >= base_shape[i]) {
      out_of_bound = true;
      break;
    }
  }
  if (!out_of_bound) {
    f(base_index);
  }
}

} // namespace

class RegionExecutor {
public:
  explicit RegionExecutor(HalContext *ctx, Frame *aFrame,
                          PPHloExecutor *func_executor)
      : hctx_(ctx), frame_(aFrame), func_executor_(func_executor) {
    aFrame->enterRegion();
  }

  ~RegionExecutor() { frame_->leaveRegion(); }

  std::vector<hal::Value> executeRegion(mlir::Region &region,
                                        llvm::ArrayRef<hal::Value> inputs);

  HalContext *getContext() const { return hctx_; }

private:
  std::vector<hal::Value> executeBlock(mlir::Block &block);
  std::vector<hal::Value> executeTerminator(mlir::Operation &op);

  void debug_print(mlir::Operation &op, bool before_execution);

  template <typename OpT, typename... MoreOpT>
  void dispatchOp(mlir::Operation &op) {
    if (auto casted = llvm::dyn_cast<OpT>(op)) {
      // Pre-execution meta
      if (!suppress_pphlo_trace_ && hctx_->rt_config().enable_pphlo_trace()) {
        debug_print(op, true);
      }

      auto tp = func_executor_->profileStart();

      // Execute op
      execute(casted);

      func_executor_->profileEnd(op.getName().getStringRef(), tp);

      if (!suppress_pphlo_trace_ && hctx_->rt_config().enable_pphlo_trace()) {
        debug_print(op, false);
      }
    } else {
      if constexpr (!sizeof...(MoreOpT)) {
        // If there is no more op types to dispatch, and the previous cast
        // fails..print error message
        errorUnknownOp(op);
      } else {
        dispatchOp<MoreOpT...>(op);
      }
    }
  }

  /// Unary ops
  void execute(mlir::pphlo::ReciprocalOp &op);
  void execute(mlir::pphlo::NegOp &op);
  void execute(mlir::pphlo::ExpOp &op);
  void execute(mlir::pphlo::LogOp &op);
  void execute(mlir::pphlo::Log1pOp &op);
  void execute(mlir::pphlo::CeilOp &op);
  void execute(mlir::pphlo::FloorOp &op);
  void execute(mlir::pphlo::AbsOp &op);
  void execute(mlir::pphlo::TransposeOp &op);
  void execute(mlir::pphlo::LogisticOp &op);
  void execute(mlir::pphlo::NotOp &op);
  void execute(mlir::pphlo::TanhOp &op);

  /// Binary ops
  void execute(mlir::pphlo::EqualOp &op);
  void execute(mlir::pphlo::LessOp &op);
  void execute(mlir::pphlo::GreaterOp &op);

  void execute(mlir::pphlo::AddOp &op);
  void execute(mlir::pphlo::SubOp &op);
  void execute(mlir::pphlo::MulOp &op);
  void execute(mlir::pphlo::PowOp &op);
  void execute(mlir::pphlo::RemOp &op);
  void execute(mlir::pphlo::MaxOp &op);
  void execute(mlir::pphlo::MinOp &op);
  void execute(mlir::pphlo::DotOp &op);
  void execute(mlir::pphlo::ShiftLeftOp &op);
  void execute(mlir::pphlo::ShiftRightArithmeticOp &op);
  void execute(mlir::pphlo::ShiftRightLogicalOp &op);

  /// Ternary ops
  void execute(mlir::pphlo::ClampOp &op);

  /// Logical ops
  void execute(mlir::pphlo::AndOp &op);
  void execute(mlir::pphlo::OrOp &op);
  void execute(mlir::pphlo::XorOp &op);

  /// Shape ops
  void execute(mlir::pphlo::BroadcastOp &op);
  void execute(mlir::pphlo::ReshapeOp &op);
  void execute(mlir::pphlo::ConcatenateOp &op);
  void execute(mlir::pphlo::SliceOp &op);
  void execute(mlir::pphlo::GatherOp &op);
  void execute(mlir::pphlo::PadOp &op);
  void execute(mlir::pphlo::ReverseOp &op);

  /// Data generator ops
  void execute(mlir::pphlo::ConstOp &op);
  void execute(mlir::pphlo::IotaOp &op);

  /// Other ops
  void execute(mlir::pphlo::RngUniformOp &op);
  void execute(mlir::pphlo::ConvertOp &op);
  void execute(mlir::pphlo::BitcastConvertOp &op);
  void execute(mlir::pphlo::ConvOp &op);
  void execute(mlir::pphlo::SortOp &op);
  void execute(mlir::pphlo::DynamicUpdateSliceOp &op);
  void execute(mlir::pphlo::DynamicSliceOp &op);

  /// Reduce ops
  void execute(mlir::pphlo::ReduceOp &op);
  void execute(mlir::pphlo::ReduceWindowOp &op);

  /// Control flow ops
  void execute(mlir::pphlo::WhileOp &op);
  void execute(mlir::pphlo::IfOp &op);

  /// Debug ops
  void execute(mlir::pphlo::DbgPrintOp &op);

  /// Lowered ops (All these ops will throw at run time)
  void execute(mlir::pphlo::SqrtOp &op);
  void execute(mlir::pphlo::SelectOp &op);
  void execute(mlir::pphlo::SelectAndScatterOp &op);
  void execute(mlir::pphlo::ReturnOp &op);
  void execute(mlir::pphlo::NotEqualOp &op);
  void execute(mlir::pphlo::LessEqualOp &op);
  void execute(mlir::pphlo::GreaterEqualOp &op);
  void execute(mlir::pphlo::DivOp &op);
  void errorUnknownOp(mlir::Operation &op);

  Frame *getFrame() { return frame_; }

  template <class T, typename Fn>
  void shift_imp(T &op, const Fn &f) {
    const auto &rhs = lookupValue(op.rhs());
    const auto &lhs = lookupValue(op.lhs());

    YASL_ENFORCE(rhs.isPublic(), "shift bit value needs to be a public");
    YASL_ENFORCE(rhs.shape() == lhs.shape());

    std::vector<int64_t> indicies(lhs.shape().size(), 0);

    // Depend on protocl, shift result might be different, AShr vs BShr.
    // So delay the preallocation
    // FIXME: maybe we can do something better?
    std::optional<hal::Value> result;

    do {
      auto bits = extractShiftBits(rhs.getElementAt(indicies));
      const auto lhs_el = lhs.getElementAt(indicies);
      auto ret_el = f(hctx_, lhs_el, bits);
      if (!result.has_value()) {
        result = hal::Value({ret_el.storage_type(), lhs.shape()}, lhs.dtype());
      }
      result->copyElementFrom(ret_el, {}, indicies);
    } while (bumpIndices<int64_t>(lhs.shape(), absl::MakeSpan(indicies)));

    getFrame()->addValue(op.getResult(), result.value());
  }

  void conv2D(mlir::pphlo::ConvOp &op);

  const hal::Value &lookupValue(::mlir::Value v) const;
  size_t extractShiftBits(const hal::Value &v) const;
  bool getConditionValue(const hal::Value &v) const;

  template <typename T>
  hal::Value iotaHelper(size_t numel, Visibility vis) {
    std::vector<T> tmp(numel);
    std::iota(tmp.begin(), tmp.end(), 0);
    auto c = hal::constant(hctx_, tmp);
    if (vis == VIS_PUBLIC) {
      return c;
    } else {
      return hal::p2s(hctx_, c);
    }
  }

  HalContext *hctx_{nullptr};
  Frame *frame_{nullptr};
  PPHloExecutor *func_executor_{nullptr};
  mlir::pphlo::TypeTools type_tools_;

  //
  bool suppress_type_check_ = false;
  bool suppress_pphlo_trace_ = false;
};

namespace {

std::mutex ErrorHandlerMutex;

void SPUErrorHandler(void * /*use_data*/, const char *reason,
                     bool /*gen_crash_diag*/) {
  YASL_THROW(reason);
}

template <typename T>
std::vector<T> build_vec_idx(const mlir::DenseIntElementsAttr &attr) {
  std::vector<T> ret;

  for (const auto &v : attr) {
    ret.emplace_back(static_cast<T>(v.getLimitedValue()));
  }

  return ret;
}

PtType getPtType(const mlir::Type &type) {
  if (auto ft = type.dyn_cast<mlir::FloatType>()) {
    switch (ft.getWidth()) {
    case 32:
      return PT_F32;
    case 64:
      return PT_F64;
    }
  }
  if (auto it = type.dyn_cast<mlir::IntegerType>()) {
    if (it.getWidth() == 1) {
      return PT_BOOL;
    }
    // In mlir, isSigned is for si[1-9][0-9]* type, isUnsigned is for
    // ui[1-9][0-9]*, i[1-9][0-9]* is signless IntegerType... So here, we only
    // check for isUnsigned, signless we treat it as signed.
    // See https://reviews.llvm.org/D72533
    switch (it.getWidth()) {
    case 8:
      return it.isUnsigned() ? PT_U8 : PT_I8;
    case 16:
      return it.isUnsigned() ? PT_U16 : PT_I16;
    case 32:
      return it.isUnsigned() ? PT_U32 : PT_I32;
    case 64:
      return it.isUnsigned() ? PT_U64 : PT_I64;
    }
  }
  YASL_THROW("Hit unknown pt_type");
}

// All sorts of gather helper functions
hal::Value reshapedGatherIndices(HalContext *ctx, int64_t index_vector_dim,
                                 const hal::Value &start_indices) {

  if (start_indices.shape().size() != static_cast<size_t>(index_vector_dim)) {
    return start_indices;
  }

  auto new_shape = start_indices.shape();
  new_shape.push_back(1);

  return hal::reshape(ctx, start_indices, new_shape);
}

struct IndexIterationSpace {
  std::vector<int64_t> index_base;
  std::vector<int64_t> index_count;
  std::vector<int64_t> index_incr;
};

// Returns an IndexIterationSpace that iterates over the output batch
// dimensions while keeping the rest of the output dimensions clamped to 0.
IndexIterationSpace iterationSpaceForOutputBatchIndices(
    llvm::ArrayRef<int64_t> output_shape,
    const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers) {
  int64_t output_rank = output_shape.size();
  std::vector<int64_t> index_base(output_rank, 0);
  std::vector<int64_t> index_count;
  index_count.reserve(output_rank);

  for (int64_t i = 0; i < output_rank; i++) {
    bool is_output_batch_dim =
        !std::binary_search(dim_numbers.getOffsetDims().begin(),
                            dim_numbers.getOffsetDims().end(), i);
    index_count.push_back(is_output_batch_dim ? output_shape[i] : 1);
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(output_rank, 1)};
}

// Return an IndexIterationSpace that iterates over the output slice
// dimensions while keeping the rest of the output dimensions clamped to 0.
IndexIterationSpace iterationSpaceForOutputOffsetIndices(
    int64_t output_rank, const mlir::DenseIntElementsAttr &slice_sizes,
    const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers) {

  std::vector<int64_t> index_base(output_rank, 0);
  std::vector<int64_t> index_count(output_rank, 1);
  int64_t slice_sizes_idx = 0;

  for (int64_t i = 0; i < output_rank; i++) {
    bool is_output_window_dim =
        std::binary_search(dim_numbers.getOffsetDims().begin(),
                           dim_numbers.getOffsetDims().end(), i);
    if (is_output_window_dim) {
      while (std::binary_search(dim_numbers.getCollapsedSliceDims().begin(),
                                dim_numbers.getCollapsedSliceDims().end(),
                                slice_sizes_idx)) {
        slice_sizes_idx++;
      }
      index_count[i] =
          *(slice_sizes.getValues<int64_t>().begin() + slice_sizes_idx++);
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
      const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers,
      llvm::ArrayRef<int64_t> input_shape, llvm::ArrayRef<int64_t> output_shape,
      const xt::xarray<int64_t> &start_indices)
      : dim_numbers_(dim_numbers), start_indices_(start_indices) {

    for (int64_t i = 0; i < static_cast<int64_t>(output_shape.size()); ++i) {
      output_dim_is_batch_dims_.push_back(
          !std::binary_search(dim_numbers_.getOffsetDims().begin(),
                              dim_numbers_.getOffsetDims().end(), i));
    }

    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i) {
      int64_t index_of_input_dim_in_index_vector =
          std::distance(dim_numbers_.getStartIndexMap().begin(),
                        std::find(dim_numbers_.getStartIndexMap().begin(),
                                  dim_numbers_.getStartIndexMap().end(), i));

      if (static_cast<size_t>(index_of_input_dim_in_index_vector) ==
          dim_numbers_.getStartIndexMap().size()) {
        input_dim_value_to_index_vector_.push_back(-1);
      } else {
        input_dim_value_to_index_vector_.push_back(
            index_of_input_dim_in_index_vector);
      }
    }

    index_vector_index_.resize(start_indices_.shape().size());
    input_index_.resize(input_shape.size());
    int64_t index_vector_size =
        start_indices_.shape()[dim_numbers_.getIndexVectorDim()];
    index_vector_.resize(index_vector_size);

    start_indices_shape.reserve(start_indices_.shape().size());
    for (const auto &d : start_indices_.shape()) {
      start_indices_shape.emplace_back(static_cast<int64_t>(d));
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
  llvm::ArrayRef<int64_t> operator()(llvm::ArrayRef<int64_t> output_index) {
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
      llvm::ArrayRef<int64_t> output_index) {
    int64_t index_vector_index_i = 0;
    for (int64_t i = 0, e = output_index.size(); i < e; i++) {
      if (!output_dim_is_batch_dims_[i]) {
        continue;
      }

      if (index_vector_index_i == dim_numbers_.getIndexVectorDim()) {
        index_vector_index_i++;
      }

      index_vector_index_[index_vector_index_i++] = output_index[i];
    }
  }

  // Populates index_vector_ by iterating over start_indices_ according to
  // index_vector_index_.
  void fetchIndexVector() {
    int64_t index_vector_dim = dim_numbers_.getIndexVectorDim();
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      index_vector_[i] =
          start_indices_
              .data()[flattenIndex(index_vector_index_, start_indices_shape)];
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

  const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers_;
  const xt::xarray<int64_t> &start_indices_;
  std::vector<int64_t> start_indices_shape;
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
      const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers,
      llvm::ArrayRef<int64_t> input_shape,
      llvm::ArrayRef<int64_t> output_shape) {

    std::vector<int64_t> window_index_to_output_index;
    int64_t output_index_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(output_shape.size()); i++) {
      if (std::binary_search(dim_numbers.getOffsetDims().begin(),
                             dim_numbers.getOffsetDims().end(), i)) {
        window_index_to_output_index.push_back(output_index_count++);
      } else {
        output_index_count++;
      }
    }

    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); i++) {
      if (std::binary_search(dim_numbers.getCollapsedSliceDims().begin(),
                             dim_numbers.getCollapsedSliceDims().end(), i)) {
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
  llvm::ArrayRef<int64_t> operator()(llvm::ArrayRef<int64_t> output_index) {
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
      llvm::ArrayRef<int64_t> output_index) {
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

template <typename FnTy>
void forEachIndex(llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> base,
                  llvm::ArrayRef<int64_t> count, llvm::ArrayRef<int64_t> incr,
                  FnTy &&visitor_function) {
  YASL_ENFORCE_EQ(shape.size(), base.size());
  YASL_ENFORCE_EQ(incr.size(), base.size());
  YASL_ENFORCE_EQ(count.size(), base.size());

  const auto rank = static_cast<int64_t>(shape.size());
  // Allows handling R0 arrays, such that the visitor function will be called
  // once with the proper empty indexes.
  int64_t n = -1;
  std::vector<int64_t> indexes(base.begin(), base.end());

  while (n < rank) {
    visitor_function(indexes);
    // Increments dimensions in minor to major order.
    for (n = 0; n < rank; ++n) {
      indexes[n] += incr[n];
      if (indexes[n] < base[n] + count[n]) {
        break;
      }
      indexes[n] = base[n];
    }
  }
}

template <typename FnType>
void forEachIndex(llvm::ArrayRef<int64_t> shape,
                  const FnType &visitor_function) {
  std::vector<int64_t> base(shape.size());
  std::vector<int64_t> incr(shape.size(), 1);
  return forEachIndex(shape, base,
                      /*count=*/shape, incr, visitor_function);
}

xt::xarray<int64_t> getIndicies(HalContext *ctx, const hal::Value &value,
                                llvm::StringRef opName) {
  YASL_ENFORCE(value.isInt(), "{} indicies value must be integers.",
               opName.str());
  YASL_ENFORCE(value.isPublic(), "{} indicies value must be public.",
               opName.str());
  return hal::test::dump_public_as<int64_t>(ctx, value);
}

std::vector<int64_t> MakeDimMultipliers(llvm::ArrayRef<int64_t> shape) {
  std::vector<int64_t> v(shape.size());
  int64_t scale = 1;
  for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
    v[dim] = scale;
    scale *= shape[dim];
  }
  return v;
}

void sliceCopy(hal::Value &dst, const hal::Value &src,
               std::vector<int64_t> dst_indices, size_t dim) {
  auto copy_size = src.shape()[0];
  for (int64_t idx = 0; idx < copy_size; ++idx) {
    dst_indices[dim] = idx;
    dst.copyElementFrom(src, {idx}, dst_indices);
  }
}

} // namespace

const hal::Value &RegionExecutor::lookupValue(::mlir::Value v) const {
  const auto *val = frame_->getValue(v);
  if (val == nullptr) {
    // Somehow cannot find this value on stack, print a reasonable error
    // message.
    std::string str;
    llvm::raw_string_ostream debug_s(str);
    v.getDefiningOp()->print(debug_s);
    YASL_ENFORCE(false, "Try to get a non-exist value, defined at {}",
                 debug_s.str());
  }
  return *val;
}

void RegionExecutor::execute(mlir::pphlo::ReduceOp &op) {
  int64_t num_args = op->getNumOperands() / 2;
  std::vector<int64_t> dimensions_to_reduce =
      build_vec_idx<int64_t>(op.dimensions());

  llvm::SmallVector<hal::Value, 2> input_args(num_args);
  llvm::SmallVector<hal::Value, 2> init_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = lookupValue(op.inputs()[i]);
    init_values[i] = lookupValue(op.init_values()[i]);
  }

  // All args and results have the same dimensions, so pick an arbitrary one.
  const auto &output_shape =
      op->getResultTypes()[0].dyn_cast<mlir::RankedTensorType>().getShape();

  std::vector<int64_t> arg_dimensions = input_args[0].shape();

  // All increments are set to 0.
  std::vector<int64_t> arg_dim_steps(arg_dimensions.size());

  // All counts are set to 0.
  std::vector<int64_t> arg_dim_counts(arg_dimensions.size());

  // Set steps and counts for reduced dimensions.
  // This avoids iterating over non-reduced dimensions, as their step
  // and count is set to zero.
  for (const int64_t dim : dimensions_to_reduce) {
    arg_dim_steps[dim] = 1;
    arg_dim_counts[dim] = arg_dimensions[dim];
  }

  // Map each dimension in the result to a dimension in arg that isn't
  // being reduced.
  std::vector<int64_t> result_to_arg_index;
  for (int64_t i = 0; i < static_cast<int64_t>(arg_dimensions.size()); ++i) {
    if (arg_dim_steps[i] == 0) {
      result_to_arg_index.push_back(i);
    }
  }

  std::vector<hal::Value> results(num_args);
  std::vector<hal::Value> arg_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    results[i] = hal::broadcast_to(hctx_, init_values[i], output_shape);
    arg_values[i] =
        hal::broadcast_to(hctx_, input_args[i].getElementAt({}), output_shape);
  }

  // Iterates only over reduced shape, as counts and steps are set to zero
  // for all non-reduced dimensions.
  // FIXME(azheng): Use vectorized reduce reduce #op from sizeof(reduce dims) to
  // log(sizeof(reduce dims))
  suppress_type_check_ = true;
  suppress_pphlo_trace_ = true;
  std::vector<int64_t> slice_begin(arg_dimensions.size(), 0);
  std::vector<int64_t> slice_end = input_args[0].shape();
  std::vector<int64_t> slice_strides(arg_dimensions.size(), 1);
  std::vector<int64_t> base(arg_dimensions.size());
  forEachIndex(arg_dimensions, base, arg_dim_counts, arg_dim_steps,
               [&](llvm::ArrayRef<int64_t> partial_input_index) {
                 for (const int64_t dim : dimensions_to_reduce) {
                   slice_begin[dim] = partial_input_index[dim];
                   slice_end[dim] = slice_begin[dim] + 1;
                 }

                 for (int64_t i = 0; i < num_args; ++i) {
                   arg_values[i] = hal::reshape(
                       hctx_,
                       hal::slice(hctx_, input_args[i], slice_begin, slice_end,
                                  slice_strides),
                       results[i].shape());
                 }

                 // Now do reduce
                 // Evaluate computation with specified literal operands.
                 std::vector<hal::Value> embedded_operands;
                 embedded_operands.reserve(op.body().getNumArguments());
                 for (const auto &accumulator : results) {
                   embedded_operands.push_back(accumulator);
                 }
                 for (const auto &local_input : arg_values) {
                   embedded_operands.push_back(local_input);
                 }

                 results = executeRegion(op.body(), embedded_operands);
               });
  suppress_type_check_ = false;
  suppress_pphlo_trace_ = false;

  for (int64_t i = 0; i < num_args; ++i) {
    getFrame()->addValue(op->getResult(i), results[i]);
  }
}

bool RegionExecutor::getConditionValue(const hal::Value &value) const {
  YASL_ENFORCE(value.numel() == 1, "Condition value must be a scalar tensor.");
  YASL_ENFORCE(value.dtype() == DT_I1, "Expect bool, got {}", value.dtype());

  const auto public_val = hal::dump_public(hctx_, value);
  return (public_val.at<bool>({}) != 0);
}

void RegionExecutor::execute(mlir::pphlo::IfOp &op) {
  auto conditional = lookupValue(op.condition());
  if (conditional.isSecret() && hctx_->rt_config().reveal_secret_condition()) {
    SPDLOG_WARN("Reveal condition variable of {} from: {}",
                op->getName().getStringRef().str(),
                printLocation(op->getLoc()));
    conditional = hal::reveal(hctx_, conditional);
  }
  bool v = getConditionValue(conditional);

  auto results = executeRegion(v ? op.true_branch() : op.false_branch(), {});

  // Copy output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    getFrame()->addValue(ret.value(), results[ret.index()]);
  }
}

/// While evalation order:
/// 1. Forward all args into cond block
/// 2. Evaluate condition
/// 3. If true -> run body with all args forward into body block
/// 4. If false -> done, set output
void RegionExecutor::execute(mlir::pphlo::WhileOp &op) {
  // First inputs vectors
  std::vector<hal::Value> inputs;
  inputs.reserve(op->getNumOperands());

  // Prepare inputs
  for (const auto operand : op->getOperands()) {
    inputs.emplace_back(lookupValue(operand));
  }

  bool warned = false;

  // Push frame
  auto eval_cond = [&](llvm::ArrayRef<hal::Value> inputs) -> bool {
    // Sanity inputs
    YASL_ENFORCE(inputs.size() == op.cond().getNumArguments());

    // Now evaluate cond
    auto ret = executeRegion(op.cond(), inputs);

    // Get cond result
    YASL_ENFORCE(
        ret.size() == 1,
        "WhileOp condition body should not return more than 1 result.");

    if (ret[0].isSecret() && hctx_->rt_config().reveal_secret_condition()) {
      ret[0] = hal::reveal(hctx_, ret[0]);
      if (!warned) {
        SPDLOG_WARN("Reveal condition region result of {} from",
                    op->getName().getStringRef().str(),
                    printLocation(op->getLoc()));
        warned = true;
      }
    }

    return getConditionValue(ret[0]);
  };

  while (eval_cond(inputs)) {
    // Sanity inputs
    YASL_ENFORCE(inputs.size() == op.body().getNumArguments());

    // dispatch body
    inputs = executeRegion(op.body(), inputs);
  }

  // Assign output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    getFrame()->addValue(ret.value(), inputs[ret.index()]);
  }
}

#define STANDARD_UNARY_OP_EXEC_IMPL(OpName, Fn)                                \
  void RegionExecutor::execute(mlir::pphlo::OpName &op) {                      \
    getFrame()->addValue(op.getResult(),                                       \
                         Fn(hctx_, lookupValue(op.getOperand())));             \
  } // namespace spu::device

STANDARD_UNARY_OP_EXEC_IMPL(ReciprocalOp, hal::reciprocal)
STANDARD_UNARY_OP_EXEC_IMPL(NegOp, hal::negate)
STANDARD_UNARY_OP_EXEC_IMPL(ExpOp, hal::exp)
STANDARD_UNARY_OP_EXEC_IMPL(LogOp, hal::log)
STANDARD_UNARY_OP_EXEC_IMPL(Log1pOp, hal::log1p)
STANDARD_UNARY_OP_EXEC_IMPL(FloorOp, hal::floor)
STANDARD_UNARY_OP_EXEC_IMPL(CeilOp, hal::ceil)
STANDARD_UNARY_OP_EXEC_IMPL(AbsOp, hal::abs)
STANDARD_UNARY_OP_EXEC_IMPL(LogisticOp, hal::logistic)
STANDARD_UNARY_OP_EXEC_IMPL(TanhOp, hal::tanh)

#undef STANDARD_UNARY_OP_EXEC_IMPL

#define STANDARD_BINARY_OP_EXEC_IMPL(OpName, Fn)                               \
  void RegionExecutor::execute(mlir::pphlo::OpName &op) {                      \
    getFrame()->addValue(op.getResult(), Fn(hctx_, lookupValue(op.lhs()),      \
                                            lookupValue(op.rhs())));           \
  } // namespace spu::device

STANDARD_BINARY_OP_EXEC_IMPL(AddOp, hal::add)
STANDARD_BINARY_OP_EXEC_IMPL(EqualOp, hal::equal);
STANDARD_BINARY_OP_EXEC_IMPL(SubOp, hal::sub)
STANDARD_BINARY_OP_EXEC_IMPL(LessOp, hal::less)
STANDARD_BINARY_OP_EXEC_IMPL(GreaterOp, hal::greater)
STANDARD_BINARY_OP_EXEC_IMPL(MulOp, hal::mul)
STANDARD_BINARY_OP_EXEC_IMPL(PowOp, hal::power)
STANDARD_BINARY_OP_EXEC_IMPL(MaxOp, hal::max)
STANDARD_BINARY_OP_EXEC_IMPL(MinOp, hal::min)
STANDARD_BINARY_OP_EXEC_IMPL(AndOp, hal::bitwise_and)
STANDARD_BINARY_OP_EXEC_IMPL(OrOp, hal::bitwise_or)
STANDARD_BINARY_OP_EXEC_IMPL(XorOp, hal::bitwise_xor)
STANDARD_BINARY_OP_EXEC_IMPL(DivOp, hal::div)

#undef STANDARD_BINARY_OP_EXEC_IMPL

#define LOWERED_OP_IMPL(OpName)                                                \
  void RegionExecutor::execute(mlir::pphlo::OpName &) {                        \
    YASL_THROW("Lowered op should not occur at backend");                      \
  }

LOWERED_OP_IMPL(SqrtOp)
LOWERED_OP_IMPL(ReturnOp)
LOWERED_OP_IMPL(NotEqualOp)
LOWERED_OP_IMPL(LessEqualOp)
LOWERED_OP_IMPL(GreaterEqualOp)

#undef LOWERED_OP_IMPL

#define UNIMPL_OP(OpName)                                                      \
  void RegionExecutor::execute(mlir::pphlo::OpName &op) {                      \
    YASL_THROW("Missing Runtime Impl Op {}", op->getName().getStringRef());    \
  }

#undef UNIMPL_OP

void RegionExecutor::execute(mlir::pphlo::SelectOp &op) {
  auto pred = lookupValue(op.pred());
  auto on_true = lookupValue(op.on_true());
  auto on_false = lookupValue(op.on_false());

  getFrame()->addValue(op.getResult(),
                       hal::select(hctx_, pred, on_true, on_false));
}

void RegionExecutor::execute(mlir::pphlo::RemOp &op) {
  // FIXME: When hal has a remainder, use that
  auto lhs = lookupValue(op.lhs());
  auto rhs = lookupValue(op.rhs());

  YASL_ENFORCE(lhs.dtype() == rhs.dtype(), "dtype mismatch {} != {}",
               lhs.dtype(), rhs.dtype());

  auto lhs_f = lhs;
  auto rhs_f = rhs;

  // 1st: find quotient by x/y
  if (lhs_f.isInt()) {
    lhs_f = hal::dtype_cast(hctx_, lhs_f, DT_FXP);
    rhs_f = hal::dtype_cast(hctx_, rhs_f, DT_FXP);
  }

  auto quotient = hal::div(hctx_, lhs_f, rhs_f);
  // 2nd: round to nearst number through (x >= 0.0) ? floor(x) : ceil(x)...
  auto zero = hal::constant(hctx_, 0.0F, quotient.shape());
  auto rquot =
      hal::select(hctx_, hal::greater_equal(hctx_, quotient, zero),
                  hal::floor(hctx_, quotient), hal::ceil(hctx_, quotient));
  // 3rd: rem = numer - rquot * denom
  auto ret = hal::sub(hctx_, lhs_f, hal::mul(hctx_, rquot, rhs_f));

  if (lhs.isInt()) {
    ret = hal::dtype_cast(hctx_, ret, lhs.dtype());
  }
  getFrame()->addValue(op.getResult(), std::move(ret));
}

void RegionExecutor::execute(mlir::pphlo::TransposeOp &op) {
  getFrame()->addValue(
      op.getResult(), hal::transpose(hctx_, lookupValue(op.getOperand()),
                                     build_vec_idx<int64_t>(op.permutation())));
}

void RegionExecutor::execute(mlir::pphlo::NotOp &op) {
  const auto &in = lookupValue(op.getOperand());
  getFrame()->addValue(op.getResult(), hal::bitwise_not(hctx_, in));
}

void RegionExecutor::execute(mlir::pphlo::DotOp &op) {
  const auto &lhs = lookupValue(op.lhs());
  const auto &rhs = lookupValue(op.rhs());
  YASL_ENFORCE(!lhs.shape().empty() && lhs.shape().size() <= 2);
  YASL_ENFORCE(!rhs.shape().empty() && rhs.shape().size() <= 2);

  getFrame()->addValue(op.getResult(), hal::matmul(hctx_, lhs, rhs));
}

void RegionExecutor::execute(mlir::pphlo::BroadcastOp &op) {
  auto to_shape =
      build_shape(op.getType().dyn_cast<mlir::RankedTensorType>().getShape());
  getFrame()->addValue(
      op.getResult(),
      hal::broadcast_to(hctx_, lookupValue(op.getOperand()), to_shape,
                        build_vec_idx<size_t>(op.broadcast_dimensions())));
}

void RegionExecutor::execute(mlir::pphlo::ReshapeOp &op) {
  auto to_shape =
      build_shape(op.getType().dyn_cast<mlir::RankedTensorType>().getShape());
  getFrame()->addValue(
      op.getResult(),
      hal::reshape(hctx_, lookupValue(op.getOperand()), to_shape));
}

void RegionExecutor::execute(mlir::pphlo::RngUniformOp &op) {
  auto to_shape =
      build_shape(op.getType().dyn_cast<mlir::RankedTensorType>().getShape());
  getFrame()->addValue(op.getResult(),
                       hal::rng_uniform(hctx_, lookupValue(op.a()),
                                        lookupValue(op.b()), to_shape));
}

static DataType getDtypeFromMlirType(::mlir::Type mlir_ty) {
  mlir::pphlo::TypeTools tool;
  if (auto int_ty =
          tool.getExpressedType(mlir_ty).dyn_cast<::mlir::IntegerType>()) {
    switch (int_ty.getWidth()) {
    case 1:
      return DT_I1;
    case 8:
      return int_ty.isUnsigned() ? DT_U8 : DT_I8;
    case 16:
      return int_ty.isUnsigned() ? DT_U16 : DT_I16;
    case 32:
      return int_ty.isUnsigned() ? DT_U32 : DT_I32;
    case 64:
      return int_ty.isUnsigned() ? DT_U64 : DT_I64;
    default:
      YASL_THROW("unsupported int type {}");
    }
  }
  auto flp_ty = tool.getExpressedType(mlir_ty).dyn_cast<::mlir::FloatType>();
  YASL_ENFORCE(flp_ty, "invalid type");
  return DT_FXP;
}

void RegionExecutor::execute(mlir::pphlo::ConvertOp &op) {
  mlir::pphlo::TypeTools tool;
  auto dst_dtype = getDtypeFromMlirType(op.getType());
  auto dst_vtype = tool.isMPCType<mlir::pphlo::PublicType>(op.getType())
                       ? VIS_PUBLIC
                       : VIS_SECRET;
  auto ret = lookupValue(op.getOperand());
  if (ret.vtype() != dst_vtype) {
    if (dst_vtype == VIS_PUBLIC) {
      ret = hal::reveal(hctx_, ret);
    } else {
      ret = hal::p2s(hctx_, ret);
    }
  }
  if (ret.dtype() != dst_dtype) {
    ret = hal::dtype_cast(hctx_, ret, dst_dtype);
  }
  getFrame()->addValue(op.getResult(), ret);
}

void RegionExecutor::execute(mlir::pphlo::ConstOp &op) {
  const auto &val = op.value();
  const auto &dea = val.dyn_cast<mlir::DenseElementsAttr>();
  const auto &type = val.getType().dyn_cast<mlir::RankedTensorType>();
  const auto &dst_shape = build_shape(type.getShape());
  const auto &pt_type = getPtType(type.getElementType());

  if (dea.isSplat()) {
    PtBufferView view(dea.getRawData().data(), pt_type, {}, {});
    const auto scalar = hal::constant(hctx_, view);
    getFrame()->addValue(op.getResult(),
                         hal::broadcast_to(hctx_, scalar, dst_shape));
  } else {
    PtBufferView view(dea.getRawData().data(), pt_type, dst_shape,
                      makeCompactStrides(dst_shape));
    getFrame()->addValue(op.getResult(), hal::constant(hctx_, view));
  }
}

#define DISPATCH_ALL_NONE_BOOL_PT_TYPES(PT_TYPE, NAME, ...)                    \
  [&] {                                                                        \
    switch (PT_TYPE) {                                                         \
      __CASE_PT_TYPE(spu::PT_I8, NAME, __VA_ARGS__)                            \
      __CASE_PT_TYPE(spu::PT_U8, NAME, __VA_ARGS__)                            \
      __CASE_PT_TYPE(spu::PT_I16, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_U16, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_I32, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_U32, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_I64, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_U64, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_F32, NAME, __VA_ARGS__)                           \
      __CASE_PT_TYPE(spu::PT_F64, NAME, __VA_ARGS__)                           \
    default:                                                                   \
      YASL_THROW("{} not implemented for pt_type={}", #NAME, PT_TYPE);         \
    }                                                                          \
  }()

void RegionExecutor::execute(mlir::pphlo::IotaOp &op) {
  const auto &ret_type =
      op.output().getType().dyn_cast<mlir::RankedTensorType>();
  const size_t numel = ret_type.getShape()[op.iota_dimension()];

  auto ret_el_type = type_tools_.getExpressedType(ret_type);
  auto pt_type = getPtType(ret_el_type);

  hal::Value iota_ret;
  DISPATCH_ALL_NONE_BOOL_PT_TYPES(pt_type, "_", [&] {
    iota_ret = iotaHelper<_PtTypeT>(numel, VIS_PUBLIC);
  });

  if (ret_type.getShape().size() > 1) {
    // Need a broadcast
    iota_ret =
        hal::broadcast_to(hctx_, iota_ret, build_shape(ret_type.getShape()));
  }

  getFrame()->addValue(op.output(), std::move(iota_ret));
}

void RegionExecutor::execute(mlir::pphlo::ConcatenateOp &op) {
  std::vector<hal::Value> values;

  // FIXME: Need a better way to normalize inputs storage types
  // Normalize all inputs to ashr
  for (auto operand : op->getOperands()) {
    auto v = lookupValue(operand);
    auto zero = hal::constant(hctx_, v.isInt() ? 0 : 0.0F, v.shape());
    zero = hal::dtype_cast(hctx_, zero, v.dtype());
    values.emplace_back(hal::add(hctx_, v, zero));
  }

  // set result
  getFrame()->addValue(op.getResult(),
                       hal::concatenate(hctx_, values, op.dimension()));
}

void RegionExecutor::execute(mlir::pphlo::SliceOp &op) {
  getFrame()->addValue(op.getResult(),
                       hal::slice(hctx_, lookupValue(op.getOperand()),
                                  build_vec_idx<int64_t>(op.start_indices()),
                                  build_vec_idx<int64_t>(op.limit_indices()),
                                  build_vec_idx<int64_t>(op.strides())));
}

void RegionExecutor::execute(mlir::pphlo::DbgPrintOp &op) {
  hal::dbg_print(hctx_, lookupValue(op.operand()));
}

void RegionExecutor::execute(mlir::pphlo::ClampOp &op) {
  getFrame()->addValue(op.getResult(), hal::clamp(hctx_, lookupValue(op.min()),
                                                  lookupValue(op.operand()),
                                                  lookupValue(op.max())));
}

void RegionExecutor::execute(mlir::pphlo::BitcastConvertOp &op) {
  const auto &in_type =
      op.getOperand().getType().dyn_cast<mlir::RankedTensorType>();
  const auto &out_type =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>();

  // bitcast should not change total #bytes, so if sizeof(in_t) !=
  // sizeof(out_t) will result to a shape change, thus it's enough to just
  // ensure in_shape == out_shape
  YASL_ENFORCE(in_type.getShape() == out_type.getShape(),
               "bitcast with different size is not supported yet");

  getFrame()->addValue(op.getResult(),
                       hal::bitcast(hctx_, lookupValue(op.getOperand()),
                                    getDtypeFromMlirType(out_type),
                                    op.elsize()));
}

size_t RegionExecutor::extractShiftBits(const hal::Value &v) const {
  YASL_ENFORCE(v.isInt());
  const auto arr = hal::dump_public(hctx_, v);

  return DISPATCH_ALL_PT_TYPES(arr.eltype().as<PtTy>()->pt_type(), "", [&] {
    return static_cast<size_t>(arr.at<_PtTypeT>({}));
  });
}

void RegionExecutor::execute(mlir::pphlo::ShiftLeftOp &op) {
  return shift_imp(op, hal::left_shift);
}

void RegionExecutor::execute(mlir::pphlo::ShiftRightArithmeticOp &op) {
  return shift_imp(op, hal::right_shift_arithmetic);
}

void RegionExecutor::execute(mlir::pphlo::ShiftRightLogicalOp &op) {
  return shift_imp(op, hal::right_shift_logical);
}

void RegionExecutor::errorUnknownOp(mlir::Operation &op) {
  // These lines of code in theory should not hit.
  // If hit, make a proper error message.
  std::string err_str;
  llvm::raw_string_ostream err(err_str);
  op.print(err);
  YASL_THROW("Unhandled mlir op {}", err.str());
}

void RegionExecutor::execute(mlir::pphlo::ReverseOp &op) {
  getFrame()->addValue(op.getResult(),
                       hal::reverse(hctx_, lookupValue(op.getOperand()),
                                    build_vec_idx<size_t>(op.dimensions())));
}

void RegionExecutor::execute(mlir::pphlo::PadOp &op) {
  const auto &operand = lookupValue(op.operand());
  const size_t operand_rank = operand.shape().size();
  const auto &padding_value = lookupValue(op.padding_value());
  YASL_ENFORCE(padding_value.shape().empty());

  auto edge_padding_low = build_vec_idx<int64_t>(op.edge_padding_low());
  YASL_ENFORCE(edge_padding_low.size() == operand_rank);
  auto edge_padding_high = build_vec_idx<int64_t>(op.edge_padding_high());
  YASL_ENFORCE(edge_padding_high.size() == operand_rank);
  auto interior_padding = build_vec_idx<int64_t>(op.interior_padding());
  YASL_ENFORCE(interior_padding.size() == operand_rank);
  YASL_ENFORCE(std::all_of(interior_padding.begin(), interior_padding.end(),
                           [](int64_t i) { return i >= 0; }));

  getFrame()->addValue(op.getResult(),
                       hal::pad(hctx_, operand, padding_value, edge_padding_low,
                                edge_padding_high, interior_padding));
}

void RegionExecutor::execute(mlir::pphlo::ReduceWindowOp &op) {

  YASL_ENFORCE(op->getNumResults() == 1,
               "Variadic reduce window is not supported yet");

  const auto &input = lookupValue(op.inputs());
  const auto &init_val = lookupValue(op.init_values());

  auto window_shape = build_vec_idx<int64_t>(op.window_dimensions());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().hasValue()) {
    window_strides = build_vec_idx<int64_t>(*op.window_strides());
  }

  // window dilation
  std::vector<int64_t> window_dilations(window_shape.size(), 1);
  if (op.window_dilations().hasValue()) {
    window_dilations = build_vec_idx<int64_t>(*op.window_dilations());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().hasValue()) {
    const auto v = *op.padding();

    YASL_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  // base dilation
  std::vector<int64_t> base_dilation(window_shape.size(), 1);
  if (op.base_dilations().hasValue()) {
    base_dilation = build_vec_idx<int64_t>(*op.base_dilations());
  }

  const int64_t rank = input.shape().size();
  std::vector<int64_t> window_index(rank, 0);

  // Init...
  auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();
  hal::Value ret = hal::broadcast_to(hctx_, init_val, build_shape(ret_shape));

  // For each resulting dimension, calculate and assign computed value.
  auto evaluate_impl =
      [&](absl::Span<int64_t> output_index) -> std::optional<hal::Value> {
    std::optional<hal::Value> ret;
    RunOnWindowIndex(
        absl::MakeSpan(window_shape), absl::MakeSpan(window_strides),
        absl::MakeSpan(window_dilations), absl::MakeSpan(window_padding),
        absl::MakeSpan(input.shape()), absl::MakeSpan(base_dilation),
        output_index, absl::MakeSpan(window_index),
        [&](absl::Span<const int64_t> operand_index) {
          ret = input.getElementAt(operand_index);
        });
    return ret;
  };

  // For each window index
  suppress_pphlo_trace_ = true;
  suppress_type_check_ = true;

  auto batch = hal::broadcast_to(hctx_, input.getElementAt({}), ret_shape);

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
    ret = executeRegion(op.body(), {ret, batch})[0];

  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  suppress_pphlo_trace_ = false;
  suppress_type_check_ = false;

  getFrame()->addValue(op.getResult(), std::move(ret));
}

// This is ported from
// https://github.com/tensorflow/tensorflow/blob/bf4c6ad46dac1f7f69911e2bfc48e141a39b40af/tensorflow/compiler/xla/service/hlo_evaluator.cc#L1774
void RegionExecutor::execute(mlir::pphlo::GatherOp &op) {
  // If input is empty, short circuit
  const auto &operand = lookupValue(op.operand());
  if (operand.numel() == 0) {
    getFrame()->addValue(op.getResult(), operand);
  }

  const auto &output_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();
  const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers =
      op.dimension_numbers();

  auto start_indices_value = reshapedGatherIndices(
      hctx_, dim_numbers.getIndexVectorDim(), lookupValue(op.start_indices()));

  if (start_indices_value.isSecret() &&
      hctx_->rt_config().reveal_secret_indicies()) {
    start_indices_value = hal::reveal(hctx_, start_indices_value);
    SPDLOG_WARN("Reveal start indicies value of {}",
                op->getName().getStringRef().str(),
                printLocation(op->getLoc()));
  }

  auto start_induces =
      getIndicies(hctx_, start_indices_value, op->getName().getStringRef());

  // We iterate over the gather dimensions in the output shape in an outer
  // loop nest, and iterate over the window dimensions in the output shape in
  // an inner loop nest.
  IndexIterationSpace start_indices_iteration_space =
      iterationSpaceForOutputBatchIndices(output_shape, dim_numbers);
  IndexIterationSpace offset_indices_iteration_space =
      iterationSpaceForOutputOffsetIndices(output_shape.size(),
                                           op.slice_sizes(), dim_numbers);

  // Scratch buffers that hold an index in the output shape and the
  // corresponding index in the input shape.
  // If input is empty, short circuit it
  auto operand_shape =
      op.operand().getType().dyn_cast<mlir::RankedTensorType>().getShape();
  std::vector<int64_t> input_index(operand_shape.size());
  std::vector<int64_t> output_index(output_shape.size());
  std::vector<int64_t> input_index_clamped(operand_shape.size());

  OutputBatchIndexToInputIndex output_batch_index_to_input_index(
      dim_numbers, /*input_shape=*/operand_shape,
      /*output_shape=*/output_shape, start_induces);
  OutputOffsetIndexToInputIndex output_offset_index_to_input_index(
      dim_numbers, /*input_shape=*/operand_shape,
      /*output_shape=*/output_shape);

  hal::Value result = hal::broadcast_to(
      hctx_,
      hal::make_value(hctx_, operand.vtype(),
                      operand.isInt() ? PtBufferView(0) : PtBufferView(0.0F)),
      build_shape(output_shape));

  auto gather_inner_loop_body = [&](llvm::ArrayRef<int64_t> output_window_index,
                                    llvm::ArrayRef<int64_t> input_gather_index,
                                    llvm::ArrayRef<int64_t>
                                        output_gather_index) {
    llvm::ArrayRef<int64_t> input_window_index =
        output_offset_index_to_input_index(output_window_index);
    for (int i = 0, e = output_index.size(); i < e; i++) {
      output_index[i] = output_gather_index[i] + output_window_index[i];
    }
    for (int i = 0, e = input_gather_index.size(); i < e; i++) {
      int64_t output_dim =
          output_offset_index_to_input_index.input_dim_value_to_output_index(i);
      // If 'output_dim' is -1, it means 'i' is an elided window dim. This
      // means we set the iteration index to 0, so for the purpose of the
      // following calculations we can consider the output dimension size
      // to be 1.
      int64_t output_dim_size = output_dim == -1 ? 1 : output_shape[output_dim];
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
      [&](llvm::ArrayRef<int64_t> output_gather_index) {
        llvm::ArrayRef<int64_t> input_gather_index =
            output_batch_index_to_input_index(output_gather_index);
        forEachIndex(output_shape, offset_indices_iteration_space.index_base,
                     offset_indices_iteration_space.index_count,
                     offset_indices_iteration_space.index_incr,
                     [&](llvm::ArrayRef<int64_t> output_window_index) {
                       return gather_inner_loop_body(output_window_index,
                                                     input_gather_index,
                                                     output_gather_index);
                     });
      };

  forEachIndex(output_shape, start_indices_iteration_space.index_base,
               start_indices_iteration_space.index_count,
               start_indices_iteration_space.index_incr,
               gather_outer_loop_body);

  getFrame()->addValue(op.getResult(), std::move(result));
}

// This is a optimized conv with im2col
void RegionExecutor::conv2D(mlir::pphlo::ConvOp &op) {
  // 01io -> o01i
  hal::Value input = lookupValue(op.lhs());
  auto kernel = lookupValue(op.rhs());
  const auto &dnums = op.dimension_numbers();

  const auto batch = input.shape()[dnums.getInputBatchDimension()];
  const auto feature = input.shape()[dnums.getInputFeatureDimension()];
  auto input_h = input.shape()[dnums.getInputSpatialDimensions()[0]];
  auto input_w = input.shape()[dnums.getInputSpatialDimensions()[1]];
  const auto out = kernel.shape()[dnums.getKernelOutputFeatureDimension()];
  auto kernel_h = kernel.shape()[dnums.getKernelSpatialDimensions()[0]];
  auto kernel_w = kernel.shape()[dnums.getKernelSpatialDimensions()[1]];

  // transpose to b01f_01io
  input = hal::transpose(
      hctx_, input,
      {dnums.getInputBatchDimension(), dnums.getInputSpatialDimensions()[0],
       dnums.getInputSpatialDimensions()[1], dnums.getInputFeatureDimension()});
  kernel = hal::transpose(hctx_, kernel,
                          {dnums.getKernelSpatialDimensions()[0],
                           dnums.getKernelSpatialDimensions()[1],
                           dnums.getKernelOutputFeatureDimension(),
                           dnums.getKernelInputFeatureDimension()});

  std::vector<int64_t> padding(4, 0);
  std::vector<int64_t> lhs_dilation(2, 1);
  bool lhs_need_padding = false;
  if (op.padding().hasValue()) {
    for (size_t idx = 0; idx < 4; ++idx) {
      padding[idx] = op.padding()->getValues<int64_t>()[idx];
    }
    lhs_need_padding |= std::any_of(padding.begin(), padding.end(),
                                    [](int64_t i) { return i != 0; });
  }

  if (op.lhs_dilation().hasValue()) {
    lhs_dilation[0] = op.lhs_dilation()->getValues<int64_t>()[0];
    lhs_dilation[1] = op.lhs_dilation()->getValues<int64_t>()[1];

    lhs_need_padding |= std::any_of(lhs_dilation.begin(), lhs_dilation.end(),
                                    [](int64_t i) { return i != 1; });
  }

  if (lhs_need_padding) {
    // add padding
    auto padding_value = makeZeros(hctx_, input.vtype(), input.dtype(), {});
    input = pad(hctx_, input, padding_value, {0, padding[0], padding[2], 0},
                {0, padding[1], padding[3], 0},
                {0, lhs_dilation[0] - 1, lhs_dilation[1] - 1, 0});
    input_h = input_h + (input_h - 1) * (lhs_dilation[0] - 1);
    input_w = input_w + (input_w - 1) * (lhs_dilation[1] - 1);
  }

  if (op.rhs_dilation().hasValue()) {
    auto rhs_dilation = op.rhs_dilation()->getValues<int64_t>();
    bool need_dilate = std::any_of(rhs_dilation.begin(), rhs_dilation.end(),
                                   [](int64_t i) { return i != 1; });
    if (need_dilate) {
      auto padding_value = makeZeros(hctx_, kernel.vtype(), kernel.dtype(), {});
      kernel = pad(hctx_, kernel, padding_value, {0, 0, 0, 0}, {0, 0, 0, 0},
                   {rhs_dilation[0] - 1, rhs_dilation[1] - 1, 0, 0});
      kernel_h = kernel_h + (kernel_h - 1) * (rhs_dilation[0] - 1);
      kernel_w = kernel_w + (kernel_w - 1) * (rhs_dilation[1] - 1);
    }
  }

  YASL_ENFORCE((input_h + padding[0] + padding[1]) >= kernel_h);
  YASL_ENFORCE((input_w + padding[2] + padding[3]) >= kernel_w);

  std::vector<int64_t> window_strides(2, 1);
  if (op.window_strides().hasValue()) {
    window_strides[0] = op.window_strides()->getValues<int64_t>()[0];
    window_strides[1] = op.window_strides()->getValues<int64_t>()[1];
  }

  const auto out_h =
      (input_h - kernel_h + padding[0] + padding[1]) / window_strides[0] + 1;
  const auto out_w =
      (input_w - kernel_w + padding[2] + padding[3]) / window_strides[1] + 1;

  std::vector<hal::Value> im2col_elements;

  kernel = reshape(hctx_, kernel, {out, feature * kernel_h * kernel_w});
  for (int64_t i = 0; i <= input_h - kernel_h + padding[0] + padding[1];
       i += window_strides[0]) {
    for (int64_t j = 0; j <= input_w - kernel_w + padding[2] + padding[3];
         j += window_strides[1]) {
      const auto sliced_input =
          reshape(hctx_,
                  slice(hctx_, input, {0, i, j, 0},
                        {batch, i + kernel_h, j + kernel_w, feature}, {}),
                  {batch, feature * kernel_h * kernel_w});

      im2col_elements.emplace_back(sliced_input);
    }
  }

  auto im2col = concatenate(hctx_, im2col_elements, 1);

  im2col = reshape(hctx_, im2col,
                   {batch * out_h * out_w, feature * kernel_h * kernel_w});

  auto ret = matmul(hctx_, im2col, transpose(hctx_, kernel));

  ret = reshape(hctx_, ret, {batch, out_h, out_w, out});

  // Transpose output from b01f
  std::vector<int64_t> permutation(4);
  permutation[dnums.getOutputBatchDimension()] = 0;
  permutation[dnums.getOutputSpatialDimensions()[0]] = 1;
  permutation[dnums.getOutputSpatialDimensions()[1]] = 2;
  permutation[dnums.getOutputFeatureDimension()] = 3;
  getFrame()->addValue(op.getResult(), hal::transpose(hctx_, ret, permutation));
}

// This is a port of hlo evoluator's HandleConvolutionWithLiterals
// See
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_evaluator_typed_visitor.h
void RegionExecutor::execute(mlir::pphlo::ConvOp &op) {
  const auto &dnums = op.dimension_numbers();
  const size_t num_spatial_dims = dnums.getOutputSpatialDimensions().size();
  YASL_ENFORCE(num_spatial_dims == dnums.getInputSpatialDimensions().size());
  YASL_ENFORCE(num_spatial_dims == dnums.getKernelSpatialDimensions().size());

  const auto &lhs_shape =
      op.lhs().getType().dyn_cast<mlir::TensorType>().getShape();

  const auto &rhs_shape =
      op.rhs().getType().dyn_cast<mlir::TensorType>().getShape();

  bool fast_path = true;
  // fast path restrictions
  // Restriction 1.
  fast_path &= (op.feature_group_count() == 1);
  fast_path &= (op.batch_group_count() == 1);

  if (lhs_shape.size() == 4 && rhs_shape.size() == 4 && fast_path) {
    conv2D(op);
    return;
  }

  std::vector<int64_t> window_shape;
  for (auto i : dnums.getKernelSpatialDimensions()) {
    window_shape.push_back(rhs_shape[i]);
  }

  auto lhs_dim_multipliers = MakeDimMultipliers(lhs_shape);
  auto rhs_dim_multipliers = MakeDimMultipliers(rhs_shape);

  auto lhs_value = lookupValue(op.lhs());
  if (!lhs_value.data().isCompact()) {
    lhs_value = lhs_value.clone();
  }
  auto rhs_value = lookupValue(op.rhs());
  if (!rhs_value.data().isCompact()) {
    rhs_value = rhs_value.clone();
  }

  const int64_t feature_group_count = op.feature_group_count();
  const int64_t batch_group_count = op.batch_group_count();

  // Dimension number applicable for input (lhs).
  const int64_t input_batch_dim = dnums.getInputBatchDimension();
  const int64_t input_z_dim = dnums.getInputFeatureDimension();
  // Dimension number applicable for kernel (rhs).
  const int64_t kernel_input_z_dim = dnums.getKernelInputFeatureDimension();
  const int64_t kernel_output_z_dim = dnums.getKernelOutputFeatureDimension();
  // Dimension number applicable for output.
  const int64_t output_batch_dim = dnums.getOutputBatchDimension();
  const int64_t output_z_dim = dnums.getOutputFeatureDimension();

  const int64_t input_z_size = lhs_shape[input_z_dim];

  const int64_t input_batch_size = lhs_shape[input_batch_dim];

  const int64_t batch_group_size = input_batch_size / batch_group_count;

  // The size of an input feature group.
  const int64_t input_feature_group_size = input_z_size / feature_group_count;

  const int64_t output_z_size = rhs_shape[kernel_output_z_dim];
  // The output feature dimension is a concatenation of convolution results
  // from the different groups.
  const int64_t output_feature_group_size = output_z_size / feature_group_count;

  // Start computing
  auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();
  hal::Value ret = makeZeros(hctx_, VIS_PUBLIC, lhs_value.dtype(), ret_shape);

  // Iterate on window
  std::vector<int64_t> window_index(dnums.getKernelSpatialDimensions().size(),
                                    0);

  do {
    hal::Value lhs_slice =
        makeZeros(hctx_, lhs_value.vtype(), lhs_value.dtype(), ret_shape);
    hal::Value rhs_slice =
        makeZeros(hctx_, rhs_value.vtype(), rhs_value.dtype(), ret_shape);

    forEachIndex(ret_shape, [&](llvm::ArrayRef<int64_t> output_index) {
      // Calculate the group index to which the current output index
      // belongs.
      const int64_t feature_group_index =
          output_index[output_z_dim] / output_feature_group_size;

      const int64_t depthwise_multiplier =
          batch_group_count > 1 ? output_z_size / input_batch_size : 1;
      const int64_t batch_group_index =
          output_index[output_z_dim] / depthwise_multiplier;

      // Find corresponding spatial dimension index for input (lhs).
      int64_t lhs_linear_spatial_index = 0;
      int64_t rhs_linear_spatial_index = 0;
      for (int64_t ki = 0; ki < static_cast<int64_t>(window_index.size());
           ++ki) {
        // Spatial dimension number for input (lhs) and output.
        const int64_t input_spatial_dim = dnums.getInputSpatialDimensions()[ki];
        const int64_t output_spatial_dim =
            dnums.getOutputSpatialDimensions()[ki];

        // Calculate lhs (input) index without taking base dilation into
        // account.
        const int64_t undilated_index =
            output_index[output_spatial_dim] *
                (op.window_strides().hasValue()
                     ? op.window_strides()->getValues<int64_t>()[ki]
                     : 1) -
            (op.padding().hasValue()
                 ? op.padding()->getValues<int64_t>()[2 * ki]
                 : 0) +
            window_index[ki] *
                (op.rhs_dilation().hasValue()
                     ? op.rhs_dilation()->getValues<int64_t>()[ki]
                     : 1);
        // Skip if the lhs (input) index is to be dilated.  As an
        // optimization, skip this mod if there's no dilation.
        if (op.lhs_dilation().hasValue() &&
            op.lhs_dilation()->getValues<int64_t>()[ki] > 1 &&
            undilated_index % op.lhs_dilation()->getValues<int64_t>()[ki] !=
                0) {
          return;
        }

        // Calculate the actual lhs (input) index after dilation.  As an
        // optimization, skip this integer divide if there's no dilation.
        int64_t lhs_spatial_index;
        if (op.lhs_dilation().hasValue() &&
            op.lhs_dilation()->getValues<int64_t>()[ki] > 1) {
          lhs_spatial_index =
              undilated_index / op.lhs_dilation()->getValues<int64_t>()[ki];
        } else {
          lhs_spatial_index = undilated_index;
        }

        // Skip if input index is not in bounds.
        if (!(lhs_spatial_index >= 0 &&
              lhs_spatial_index < lhs_shape[input_spatial_dim])) {
          return;
        }

        lhs_linear_spatial_index +=
            lhs_spatial_index * lhs_dim_multipliers[input_spatial_dim];
        rhs_linear_spatial_index +=
            window_index[ki] *
            rhs_dim_multipliers[dnums.getKernelSpatialDimensions()[ki]];
      }

      for (int64_t rhs_iz = 0; rhs_iz < input_feature_group_size; ++rhs_iz) {
        const int64_t iz =
            feature_group_index * input_feature_group_size + rhs_iz;

        int64_t lhs_linear_index = lhs_linear_spatial_index;
        lhs_linear_index += output_index[output_batch_dim] *
                            lhs_dim_multipliers[input_batch_dim];

        // We are scraping only the diagonal elements in the resultant
        // convolution output when batch_group_count is greater than 1,
        // where 1 is the default. No scraping is done in that case.
        // This approach works out automatically for 'groups' in batches
        // with group_size > 1, because we already descend down the batch
        // dimension for the 'output_batch_dim' above.
        lhs_linear_index +=
            ((batch_group_index * batch_group_size) % input_batch_size) *
            lhs_dim_multipliers[input_batch_dim];

        lhs_linear_index += iz * lhs_dim_multipliers[input_z_dim];
        int64_t rhs_linear_index = rhs_linear_spatial_index;

        rhs_linear_index += output_index[output_z_dim] *
                            rhs_dim_multipliers[kernel_output_z_dim];
        rhs_linear_index += rhs_iz * rhs_dim_multipliers[kernel_input_z_dim];

        lhs_slice.copyElementFrom(lhs_value.getElementAt(lhs_linear_index), {},
                                  output_index);
        rhs_slice.copyElementFrom(rhs_value.getElementAt(rhs_linear_index), {},
                                  output_index);
      }
    });

    // Work on current slice
    auto mul_ret = hal::mul(hctx_, lhs_slice, rhs_slice);
    ret = hal::add(hctx_, mul_ret, ret);
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  getFrame()->addValue(op.getResult(), std::move(ret));
}

void RegionExecutor::execute(mlir::pphlo::SortOp &op) {

  auto key_shape =
      op->getOperand(0).getType().dyn_cast<mlir::RankedTensorType>().getShape();
  auto rank = key_shape.size();
  std::vector<hal::Value> inputs;
  for (int64_t i = 0; i < op->getNumOperands(); ++i) {
    inputs.emplace_back(lookupValue(op->getOperand(i)));
  }
  std::vector<hal::Value> results;
  results.reserve(op->getNumOperands());
  for (int64_t i = 0; i < op->getNumOperands(); ++i) {
    results.emplace_back(
        NdArrayRef(inputs[i].data().eltype(), inputs[i].shape()),
        inputs[i].dtype());
  }
  std::vector<int64_t> zero_base(rank, 0);
  std::vector<int64_t> increment(rank, 1);
  int64_t sort_dim = op.dimension();
  int64_t sort_dim_elements = key_shape[sort_dim];
  YASL_ENFORCE(sort_dim >= 0 &&
                   sort_dim < static_cast<int64_t>(increment.size()),
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
        std::vector<hal::Value> values_to_sort;
        values_to_sort.reserve(op->getNumOperands());
        for (int64_t i = 0; i < op->getNumOperands(); ++i) {
          auto value_to_sort = hal::reshape(
              hctx_, hal::slice(hctx_, inputs[i], indices, limit_indices, {}),
              {sort_dim_elements});
          values_to_sort.push_back(std::move(value_to_sort));
        }
        std::vector<int64_t> indices_to_sort(sort_dim_elements);
        std::iota(indices_to_sort.begin(), indices_to_sort.end(), 0);
        auto comparator = [&op, this, &values_to_sort, &warned](int64_t a,
                                                                int64_t b) {
          std::vector<hal::Value> values;
          values.reserve(2 * op->getNumOperands());
          for (int64_t i = 0; i < op->getNumOperands(); ++i) {
            values.push_back(values_to_sort[i].getElementAt(a));
            values.push_back(values_to_sort[i].getElementAt(b));
          }
          auto ret = this->executeRegion(op.comparator(), values);
          if (ret[0].isSecret() &&
              hctx_->rt_config().reveal_secret_condition()) {
            ret[0] = hal::reveal(hctx_, ret[0]);
            if (!warned) {
              SPDLOG_WARN("Reveal condition region result of {} from {}",
                          op->getName().getStringRef().str(),
                          printLocation(op->getLoc()));
              warned = true;
            }
          }
          return getConditionValue(ret[0]);
        };

        if (op.is_stable()) {
          std::stable_sort(indices_to_sort.begin(), indices_to_sort.end(),
                           comparator);
        } else {
          std::sort(indices_to_sort.begin(), indices_to_sort.end(), comparator);
        }

        std::vector<int64_t> start_indices(rank, 0);
        for (int64_t i = 0; i < op->getNumOperands(); ++i) {
          auto sorted_value = hal::permute(hctx_, values_to_sort[i], 0,
                                           xt::adapt(indices_to_sort));
          sliceCopy(results[i], sorted_value, indices, sort_dim);
        }
      });

  for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
    getFrame()->addValue(op->getResult(idx), std::move(results[idx]));
  }
}

void RegionExecutor::execute(mlir::pphlo::DynamicUpdateSliceOp &op) {
  // Basic idea here, get a ref slice and update the whole slice..
  // Start indicies
  std::vector<int64_t> start_indicies(op.start_indices().size());
  const auto &operand = lookupValue(op.operand());
  const auto &update = lookupValue(op.update());

  for (const auto &idx : llvm::enumerate(op.start_indices())) {
    auto op_index = lookupValue(idx.value());
    if (op_index.isSecret() && hctx_->rt_config().reveal_secret_indicies()) {
      op_index = hal::reveal(hctx_, op_index);
      SPDLOG_WARN("Reveal {}th start index of {} from {}", idx.index(),
                  op->getName().getStringRef(), printLocation(op->getLoc()));
    }
    start_indicies[idx.index()] =
        getIndicies(hctx_, op_index, op->getName().getStringRef())[0];
    // Transform start_indicies
    // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] -
    // update.dimension_size[i])
    start_indicies[idx.index()] =
        std::min(std::max(start_indicies[idx.index()], int64_t(0)),
                 operand.shape()[idx.index()] - update.shape()[idx.index()]);
  }

  // Limit
  std::vector<int64_t> limit(start_indicies);
  for (size_t idx = 0; idx < limit.size(); ++idx) {
    limit[idx] = start_indicies[idx] + update.shape()[idx];
  }

  // Strides is always 1
  std::vector<int64_t> strides(limit.size(), 1);

  // First get a slice
  auto result = operand.clone();
  auto slice = hal::slice(hctx_, result, start_indicies, limit, strides);

  // (xiaochen): I know it's hacky here, but make life easier
  YASL_ENFORCE(slice.data().buf()->data() == result.data().buf()->data(),
               "slice needs to return a ref to input");
  YASL_ENFORCE(slice.shape() == update.shape(),
               "slice shape should equal to update shape");

  std::vector<int64_t> indicies(slice.shape().size(), 0);
  do {
    slice.copyElementFrom(update, indicies, indicies);
  } while (bumpIndices<int64_t>(slice.shape(), absl::MakeSpan(indicies)));

  getFrame()->addValue(op.getResult(), result);
}

void RegionExecutor::execute(mlir::pphlo::DynamicSliceOp &op) {
  // Start indicies
  std::vector<int64_t> start_indicies(op.start_indices().size());
  std::vector<int64_t> slice_size = build_vec_idx<int64_t>(op.slice_sizes());
  const auto &operand = lookupValue(op.operand());

  for (const auto &idx : llvm::enumerate(op.start_indices())) {
    auto op_index = lookupValue(idx.value());
    if (op_index.isSecret() && hctx_->rt_config().reveal_secret_indicies()) {
      op_index = hal::reveal(hctx_, op_index);
      SPDLOG_WARN("Reveal {}th start index of {} from {}", idx.index(),
                  op->getName().getStringRef(), printLocation(op->getLoc()));
    }
    start_indicies[idx.index()] =
        getIndicies(hctx_, op_index, op->getName().getStringRef())[0];
    // Transform start_indicies
    // start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] -
    // size_indices[i])
    start_indicies[idx.index()] =
        std::min(std::max(start_indicies[idx.index()], int64_t(0)),
                 operand.shape()[idx.index()] - slice_size[idx.index()]);
  }

  // Limit
  std::vector<int64_t> limit_indicies(op.slice_sizes().size());
  for (const auto &idx : llvm::enumerate(op.slice_sizes())) {
    limit_indicies[idx.index()] =
        start_indicies[idx.index()] + idx.value().getLimitedValue();
  }

  // Strides is always 1
  std::vector<int64_t> strides(limit_indicies.size(), 1);

  getFrame()->addValue(
      op.getResult(),
      hal::slice(hctx_, operand, start_indicies, limit_indicies, strides));
}

std::vector<hal::Value> RegionExecutor::executeTerminator(mlir::Operation &op) {
  if (llvm::isa<mlir::func::ReturnOp>(op) ||
      llvm::isa<mlir::pphlo::ReturnOp>(op)) {
    std::vector<hal::Value> results;
    results.reserve(op.getNumOperands());
    for (const auto operand : op.getOperands()) {
      results.emplace_back(lookupValue(operand));
    }
    return results;
  }
  llvm_unreachable("Unknown block terminator");
}

void RegionExecutor::execute(mlir::pphlo::SelectAndScatterOp &op) {
  auto operand = lookupValue(op.operand());
  auto source = lookupValue(op.source());
  auto init_val = lookupValue(op.init_value());

  auto result = hal::broadcast_to(hctx_, init_val, operand.shape());

  auto window_shape = build_vec_idx<int64_t>(op.window_dimensions().getValue());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().hasValue()) {
    window_strides = build_vec_idx<int64_t>(*op.window_strides());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().hasValue()) {
    const auto v = *op.padding();

    YASL_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  std::vector<int64_t> window_dilations(window_shape.size(), 1);
  std::vector<int64_t> base_dilations(operand.shape().size(), 1);

  auto idx_matrix =
      hal::reshape(hctx_, iotaHelper<int64_t>(operand.numel(), operand.vtype()),
                   operand.shape());

  suppress_pphlo_trace_ = true;
  suppress_type_check_ = true;

  const auto rank = operand.shape().size();
  std::vector<int64_t> window_index(rank, 0);

  auto current_val =
      hal::broadcast_to(hctx_, operand.getElementAt({}), source.shape());
  auto current_idx =
      hal::broadcast_to(hctx_, idx_matrix.getElementAt({}), source.shape());
  hal::Value selected_val;
  hal::Value selected_idx;
  bool first_iter = true;

  do {
    std::vector<int64_t> output_index(source.shape().size(), 0);
    do {
      RunOnWindowIndex(
          absl::MakeSpan(window_shape), absl::MakeSpan(window_strides),
          absl::MakeSpan(window_dilations), absl::MakeSpan(window_padding),
          absl::MakeSpan(operand.shape()), absl::MakeSpan(base_dilations),
          absl::MakeSpan(output_index), absl::MakeSpan(window_index),
          [&](absl::Span<const int64_t> operand_index) {
            current_val.copyElementFrom(operand, operand_index, output_index);
            current_idx.copyElementFrom(idx_matrix, operand_index,
                                        output_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));
    if (first_iter) {
      // First iter, don't do the real compute, just copy to selected
      selected_val = current_val.clone();
      selected_idx = current_idx.clone();
      first_iter = false;
    } else {
      auto ret = executeRegion(op.select(), {selected_val, current_val});
      selected_val = hal::select(hctx_, ret[0], selected_val, current_val);
      selected_idx = hal::select(hctx_, ret[0], selected_idx, current_idx);
    }
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  // Scatter
  std::fill(window_index.begin(), window_index.end(), 0);
  auto idx_slice =
      hal::broadcast_to(hctx_, idx_matrix.getElementAt({}), source.shape());
  auto result_slice =
      hal::broadcast_to(hctx_, result.getElementAt({}), source.shape());

  do {
    std::vector<int64_t> output_index(source.shape().size(), 0);
    do {
      RunOnWindowIndex(
          absl::MakeSpan(window_shape), absl::MakeSpan(window_strides),
          absl::MakeSpan(window_dilations), absl::MakeSpan(window_padding),
          absl::MakeSpan(operand.shape()), absl::MakeSpan(base_dilations),
          absl::MakeSpan(output_index), absl::MakeSpan(window_index),
          [&](absl::Span<const int64_t> operand_index) {
            idx_slice.copyElementFrom(idx_matrix, operand_index, output_index);
            result_slice.copyElementFrom(result, operand_index, output_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));

    auto mask = hal::equal(hctx_, selected_idx, idx_slice);

    auto added = hal::add(hctx_, result_slice, source);
    result_slice = hal::select(hctx_, mask, added, result_slice);

    // Reset, copy window again...
    std::fill(output_index.begin(), output_index.end(), 0);
    do {
      RunOnWindowIndex(
          absl::MakeSpan(window_shape), absl::MakeSpan(window_strides),
          absl::MakeSpan(window_dilations), absl::MakeSpan(window_padding),
          absl::MakeSpan(operand.shape()), absl::MakeSpan(base_dilations),
          absl::MakeSpan(output_index), absl::MakeSpan(window_index),
          [&](absl::Span<const int64_t> operand_index) {
            result.copyElementFrom(result_slice, output_index, operand_index);
          });
    } while (
        bumpIndices<int64_t>(source.shape(), absl::MakeSpan(output_index)));
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));

  suppress_pphlo_trace_ = false;
  suppress_type_check_ = false;

  getFrame()->addValue(op.getResult(), std::move(result));
}

std::vector<hal::Value> RegionExecutor::executeBlock(mlir::Block &block) {
  for (auto &op : block.without_terminator()) {
    dispatchOp<
#define GET_OP_LIST
#include "spu/dialect/pphlo_ops.cc.inc"
        >(op);
  }

  if (auto *termOp = block.getTerminator()) {
    if (!suppress_pphlo_trace_ && hctx_->rt_config().enable_pphlo_trace()) {
      debug_print(*termOp, true);
    }
    return executeTerminator(*termOp);
  }

  // No terminator
  return {};
}

std::vector<hal::Value>
RegionExecutor::executeRegion(mlir::Region &region,
                              llvm::ArrayRef<hal::Value> inputs) {
  getFrame()->enterRegion();
  getFrame()->setTypeCheker(!suppress_type_check_ &&
                            getContext()->rt_config().enable_type_checker());
  for (const auto &blkarg : region.getArguments()) {
    getFrame()->addValue(blkarg, inputs[blkarg.getArgNumber()]);
  }

  auto ret = executeBlock(region.front());
  getFrame()->leaveRegion();
  getFrame()->setTypeCheker(getContext()->rt_config().enable_type_checker());
  return ret;
}

std::vector<hal::Value>
PPHloExecutor::executeFunc(mlir::func::FuncOp &fcn,
                           llvm::ArrayRef<hal::Value> inputs) {
  Frame callFrame;
  RegionExecutor executor(getContext(), &callFrame, this);
  return executor.executeRegion(fcn.getBody(), inputs);
}

PPHloExecutor::PPHloExecutor(HalContext *ctx) : Executor(ctx) {
  // Set an error handler
  {
    std::lock_guard<std::mutex> guard(ErrorHandlerMutex);
    llvm::remove_fatal_error_handler();
    llvm::install_fatal_error_handler(SPUErrorHandler);
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::pphlo::PPHloDialect, mlir::func::FuncDialect>();
  mlir_context_ = std::make_unique<mlir::MLIRContext>(registry);

  hctx_->clearProfilingRecords();
  hctx_->prot()->clearProfilingRecords();
}

PPHloExecutor::~PPHloExecutor() {
  std::lock_guard<std::mutex> guard(ErrorHandlerMutex);
  llvm::remove_fatal_error_handler();
}

std::vector<hal::Value>
PPHloExecutor::run(const std::string &code,
                   const std::vector<hal::Value> &inputs) {
  auto moduleOpRef =
      mlir::parseSourceString<mlir::ModuleOp>(code, mlir_context_.get());

  auto entry_function = moduleOpRef->lookupSymbol<mlir::FuncOp>("main");
  YASL_ENFORCE(entry_function);

  return executeFunc(entry_function, inputs);
}

mlir::OwningOpRef<mlir::ModuleOp>
PPHloExecutor::parseSourceString(const std::string &code) {
  auto moduleOp =
      mlir::parseSourceString<mlir::ModuleOp>(code, mlir_context_.get());
  return moduleOp;
}

void RegionExecutor::debug_print(mlir::Operation &op, bool before) {
  if (before) {
    if (hctx_->lctx() && hctx_->lctx()->Rank() == 0) {
      std::string buf;
      llvm::raw_string_ostream debug_stream(buf);
      op.print(debug_stream);
      SPDLOG_INFO("PPHLO {}", debug_stream.str());
    }
  } else {
    for (const auto &ret : llvm::enumerate(op.getResults())) {
      const auto &v = lookupValue(ret.value());
      if (hctx_->lctx() && hctx_->lctx()->Rank() == 0) {
        SPDLOG_INFO("PPHLO ret {}, storage_type {}", ret.index(),
                    v.storage_type());
      }
      hal::dbg_print(hctx_, v);
    }
  }
}

} // namespace spu::device
