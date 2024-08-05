// Copyright 2023 Ant Group Co., Ltd.
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

#include <numeric>
#include <unordered_set>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

bool GatherIsBroadcast(GatherOp &op) {
  auto gather_slice_size = op.getSliceSizes().getValues<int64_t>();
  auto op_shape = op.getOperand().getType().getShape();
  return (gather_slice_size.size() == op_shape.size()) &&
         (std::equal(gather_slice_size.begin(), gather_slice_size.end(),
                     op_shape.begin()));
}

std::vector<int64_t> DeleteDimensions(llvm::ArrayRef<int64_t> dims_to_delete,
                                      llvm::ArrayRef<int64_t> shape) {
  std::unordered_set<int64_t> ordered_dims_to_delete(dims_to_delete.begin(),
                                                     dims_to_delete.end());

  std::vector<int64_t> result;
  result.reserve(shape.size() - ordered_dims_to_delete.size());

  for (size_t idx = 0; idx < shape.size(); ++idx) {
    if (ordered_dims_to_delete.count(idx) != 0) {
      continue;
    }
    result.emplace_back(idx);
  }
  return result;
}

mlir::DenseIntElementsAttr
ConvertDimensions(OpBuilder *builder, llvm::ArrayRef<int64_t> op_dimensions) {
  llvm::SmallVector<APInt, 8> dimensions;
  dimensions.reserve(op_dimensions.size());
  for (auto value : op_dimensions) {
    dimensions.emplace_back(APInt(64, value));
  }

  return DenseIntElementsAttr::get(
      RankedTensorType::get(dimensions.size(), builder->getIntegerType(64)),
      dimensions);
}

// Computes how many trips a loop implementing this gather op would take.
int64_t GatherLoopTripCount(GatherOp op) {
  auto start_indices = op.getStartIndices();
  const auto start_indices_shape = start_indices.getType().getShape();
  const auto &dim_numbers = op.getDimensionNumbers();

  int64_t trip_count = 1;
  for (int64_t i = 0, e = start_indices_shape.size(); i < e; i++) {
    if (i != dim_numbers.getIndexVectorDim()) {
      trip_count *= start_indices_shape[i];
    }
  }
  return trip_count;
}

llvm::SmallVector<int64_t>
ComputePermutedShape(llvm::ArrayRef<int64_t> shape,
                     llvm::ArrayRef<int64_t> permutation) {
  llvm::SmallVector<int64_t> result_shape;
  for (auto dim : permutation) {
    result_shape.emplace_back(shape[dim]);
  }
  return result_shape;
}

TypedValue<RankedTensorType>
TransposeIndexVectorDimToLast(TypedValue<RankedTensorType> &start_indices,
                              int64_t index_vector_dim) {
  const auto start_indices_shape = start_indices.getType().getShape();

  if (static_cast<int64_t>(start_indices_shape.size()) == index_vector_dim) {
    return start_indices;
  }

  if (index_vector_dim ==
      static_cast<int64_t>(start_indices_shape.size() - 1)) {
    return start_indices;
  }

  std::vector<int64_t> permutation;
  permutation.reserve(start_indices_shape.size());
  for (int64_t i = 0, e = start_indices_shape.size(); i < e; i++) {
    if (i != index_vector_dim) {
      permutation.emplace_back(i);
    }
  }
  permutation.emplace_back(index_vector_dim);

  auto result_shape = ComputePermutedShape(start_indices_shape, permutation);

  OpBuilder builder(start_indices.getContext());
  if (auto *ip = start_indices.getDefiningOp()) {
    builder.setInsertionPointAfter(ip);
  } else {
    builder.setInsertionPointToStart(start_indices.getParentBlock());
  }

  auto transpose = builder.create<TransposeOp>(
      start_indices.getLoc(),
      RankedTensorType::get(result_shape,
                            start_indices.getType().getElementType()),
      start_indices, ConvertDimensions(&builder, permutation));

  return transpose.getResult();
}

TypedValue<RankedTensorType>
PrependDegenerateDims(TypedValue<RankedTensorType> operand, int64_t n) {
  SPU_ENFORCE(n > 0);
  std::vector<int64_t> new_shape_dims;
  const auto operand_shape = operand.getType().getShape();
  new_shape_dims.reserve(n + operand_shape.size());
  new_shape_dims.insert(new_shape_dims.begin(), n, 1);
  std::copy(operand_shape.begin(), operand_shape.end(),
            std::back_inserter(new_shape_dims));

  OpBuilder builder(operand.getContext());
  if (auto *ip = operand.getDefiningOp()) {
    builder.setInsertionPointAfter(ip);
  } else {
    builder.setInsertionPointToStart(operand.getParentBlock());
  }

  auto reshape = builder.create<ReshapeOp>(
      operand.getLoc(),
      RankedTensorType::get(new_shape_dims, operand.getType().getElementType()),
      operand);

  return reshape.getResult();
}

TypedValue<RankedTensorType>
CollapseFirstNDims(TypedValue<RankedTensorType> operand, int64_t n) {
  SPU_ENFORCE(n > 0);

  const auto operand_shape = operand.getType().getShape();
  SPU_ENFORCE((int64_t)operand_shape.size() >= n);

  int64_t new_shape_leading_bound = 1;
  for (int64_t i = 0; i < n; i++) {
    new_shape_leading_bound *= operand_shape[i];
  }

  std::vector<int64_t> new_shape_dims;
  new_shape_dims.reserve(operand_shape.size() - n + 1);
  new_shape_dims.push_back(new_shape_leading_bound);

  std::copy(operand_shape.begin() + n, operand_shape.end(),
            std::back_inserter(new_shape_dims));

  auto output_type =
      RankedTensorType::get(new_shape_dims, operand.getType().getElementType());

  OpBuilder builder(operand.getContext());
  if (auto *ip = operand.getDefiningOp()) {
    builder.setInsertionPointAfter(ip);
  } else {
    builder.setInsertionPointToStart(operand.getParentBlock());
  }

  auto reshape =
      builder.create<ReshapeOp>(operand.getLoc(), output_type, operand);

  return reshape.getResult();
}

// Canonicalizes the start_indices tensors so that we only have deal with some
// specific cases in the while loop that does the heavy lifting.
//
// See the "High Level Algorithm" section for a broader picture.
TypedValue<RankedTensorType>
CanonicalizeGatherIndices(TypedValue<RankedTensorType> &start_indices,
                          int64_t index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  auto transposed_start_indices =
      TransposeIndexVectorDimToLast(start_indices, index_vector_dim);
  bool indices_are_scalar =
      index_vector_dim ==
      static_cast<int64_t>(start_indices.getType().getShape().size());

  // The number of dimensions in start_indices that are index dimensions.
  const int64_t index_dims_in_start_indices = indices_are_scalar ? 0 : 1;

  // If there is only one index (i.e. start_indices has rank 1 and this gather
  // is really just a dynamic slice) add a leading degenerate dimension for
  // uniformity.  Otherwise create a "collapsed" leading dimension that subsumes
  // all of the non-index-vector dimensions.
  const auto shape = transposed_start_indices.getType().getShape();
  if (static_cast<int64_t>(shape.size()) == index_dims_in_start_indices) {
    return PrependDegenerateDims(transposed_start_indices, 1);
  } else {
    // Collapse all but the dimensions (0 or 1) in start_indices containing the
    // index vectors.
    return CollapseFirstNDims(transposed_start_indices,
                              shape.size() - index_dims_in_start_indices);
  }
}

TypedValue<RankedTensorType> CreateGatherLoopAccumulatorInitValue(
    GatherOp op, Type element_type, DenseIntElementsAttr slice_sizes,
    int64_t gather_loop_trip_count,
    const GatherDimensionNumbersAttr &dim_numbers) {
  std::vector<int64_t> accumulator_state_shape_dims;
  auto array_slice_size = slice_sizes.getValues<int64_t>();
  accumulator_state_shape_dims.reserve(1 + array_slice_size.size());
  accumulator_state_shape_dims.push_back(gather_loop_trip_count);
  for (int64_t i = 0; i < static_cast<int64_t>(array_slice_size.size()); i++) {
    if (!std::binary_search(dim_numbers.getCollapsedSliceDims().begin(),
                            dim_numbers.getCollapsedSliceDims().end(), i)) {
      accumulator_state_shape_dims.emplace_back(array_slice_size[i]);
    }
  }

  OpBuilder builder(op);
  TypeTools type_tools;

  auto c = builder.create<ConstantOp>(
      op->getLoc(), builder.getZeroAttr(RankedTensorType::get(
                        accumulator_state_shape_dims,
                        type_tools.getExpressedType(element_type))));

  if (type_tools.getTypeVisibility(element_type) != Visibility::VIS_PUBLIC) {
    auto convert = builder.create<ConvertOp>(
        op.getLoc(),
        RankedTensorType::get(accumulator_state_shape_dims, element_type),
        c.getResult());
    return convert.getResult();
  } else {
    return c.getResult();
  }
}

TypedValue<RankedTensorType>
ExpandFirstDimIntoNDims(TypedValue<RankedTensorType> operand,
                        llvm::ArrayRef<int64_t> expanded_dims) {
  SPU_ENFORCE_GT(operand.getType().getShape().size(), size_t(0));
  SPU_ENFORCE_EQ(operand.getType().getShape()[0],
                 std::accumulate(expanded_dims.begin(), expanded_dims.end(), 1,
                                 std::multiplies()));

  std::vector<int64_t> expanded_shape_dim_bounds;
  expanded_shape_dim_bounds.reserve(expanded_dims.size() +
                                    operand.getType().getShape().size() - 1);
  std::copy(expanded_dims.begin(), expanded_dims.end(),
            std::back_inserter(expanded_shape_dim_bounds));
  std::copy(operand.getType().getShape().begin() + 1,
            operand.getType().getShape().end(),
            std::back_inserter(expanded_shape_dim_bounds));

  auto result_type = RankedTensorType::get(expanded_shape_dim_bounds,
                                           operand.getType().getElementType());

  OpBuilder builder(operand.getContext());
  if (auto *ip = operand.getDefiningOp()) {
    builder.setInsertionPointAfter(ip);
  } else {
    builder.setInsertionPointToStart(operand.getParentBlock());
  }
  auto reshaped =
      builder.create<ReshapeOp>(operand.getLoc(), result_type, operand);
  return reshaped.getResult();
}

TypedValue<RankedTensorType>
ElideDegenerateDims(OpBuilder *builder, TypedValue<RankedTensorType> operand,
                    absl::Span<const int64_t> dims_to_elide) {
  std::unordered_set<int64_t> dims_to_elide_set(dims_to_elide.begin(),
                                                dims_to_elide.end());
  std::vector<int64_t> new_shape;
  for (size_t idx = 0; idx < operand.getType().getShape().size(); ++idx) {
    if (dims_to_elide_set.count(idx) > 0) {
      continue;
    }
    new_shape.emplace_back(operand.getType().getShape()[idx]);
  }

  auto reshape = builder->create<ReshapeOp>(
      operand.getLoc(),
      RankedTensorType::get(new_shape, operand.getType().getElementType()),
      operand);
  return reshape.getResult();
}

// Expands out or contracts away the gather dimensions in the accumulator
// produced by the while loop.
TypedValue<RankedTensorType> AdjustBatchDimsInAccumulator(
    OpBuilder *builder, llvm::ArrayRef<int64_t> start_indices_shape,
    TypedValue<RankedTensorType> accumulator, int64_t index_vector_dim) {
  std::vector<int64_t> batch_dim_bounds;
  batch_dim_bounds.reserve(start_indices_shape.size());
  for (int64_t i = 0, e = start_indices_shape.size(); i < e; i++) {
    if (i != index_vector_dim) {
      batch_dim_bounds.push_back(start_indices_shape[i]);
    }
  }

  if (batch_dim_bounds.empty()) {
    // If batch_dim_bounds is empty we must be lowering a (effectively)
    // dynamic-slice.  In that case, there is a leading degenerate gather
    // dimension that we added to make this special case play well with the
    // general while loop which we need to remove now.
    return ElideDegenerateDims(builder, accumulator, {0});
  }

  return ExpandFirstDimIntoNDims(accumulator, batch_dim_bounds);
}

void BuildWhileCondition(Region &cond, Value counter,
                         Value canonical_start_indices, Value accumulator_init,
                         Value loop_upper_bound) {
  OpBuilder builder(cond);
  TypeTools type_tool;

  auto lt = builder.create<LessOp>(
      cond.getLoc(),
      RankedTensorType::get(
          {}, type_tool.getTypeWithVisibility(builder.getI1Type(),
                                              Visibility::VIS_PUBLIC)),
      cond.getArgument(0), loop_upper_bound);

  builder.create<ReturnOp>(cond.getLoc(), ValueRange{lt.getResult()});
}

int64_t FindIndex(llvm::ArrayRef<int64_t> c, int64_t value) {
  const auto *it = std::find(c.begin(), c.end(), value);
  return std::distance(c.begin(), it);
}

// Expand an index vector from the start_indices tensor into a vector that can
// be used to dynamic-slice out of the gather operand.
llvm::SmallVector<Value> ExpandIndexVectorIntoOperandSpace(
    OpBuilder *builder, TypedValue<RankedTensorType> index_vector,
    const GatherDimensionNumbersAttr &dim_numbers, int64_t operand_rank) {

  TypeTools typetool;
  auto index_type =
      typetool.getExpressedType(index_vector.getType().getElementType());

  if (operand_rank == 0) {
    // This is Gather from a scalar. So, the index vector in operand space must
    // be a zero-sized vector.
    // return computation->AddInstruction(HloInstruction::CreateConstant(
    //     LiteralUtil::CreateFromDimensions(index_shape.element_type(), {0})));
    auto zero_const = builder->create<ConstantOp>(
        index_vector.getLoc(),
        builder->getZeroAttr(RankedTensorType::get({}, index_type)));
    return {zero_const.getResult()};
  }

  auto p_zero_const = builder->create<ConstantOp>(
      index_vector.getLoc(),
      builder->getZeroAttr(RankedTensorType::get({}, index_type)));

  auto zero_const = builder->create<ConvertOp>(
      index_vector.getLoc(),
      RankedTensorType::get({}, typetool.toMPCType<SecretType>(index_type)),
      p_zero_const);

  // We extract out individual components from the smaller index and concatenate
  // them (interspersing zeros as needed) into the larger index.
  llvm::SmallVector<Value> expanded_index_components;

  for (int64_t i = 0; i < operand_rank; i++) {
    int64_t index_vector_dim_index =
        FindIndex(dim_numbers.getStartIndexMap(), i);
    if (index_vector_dim_index !=
        static_cast<int64_t>(dim_numbers.getStartIndexMap().size())) {

      auto component_to_concat = builder->create<SliceOp>(
          index_vector.getLoc(),
          RankedTensorType::get({1}, index_vector.getType().getElementType()),
          index_vector, ConvertDimensions(builder, {index_vector_dim_index}),
          ConvertDimensions(builder, {index_vector_dim_index + 1}),
          ConvertDimensions(builder, {1}));
      auto reshaped = builder->create<ReshapeOp>(
          index_vector.getLoc(),
          RankedTensorType::get({}, index_vector.getType().getElementType()),
          component_to_concat);
      expanded_index_components.push_back(reshaped);
    } else {
      expanded_index_components.push_back(zero_const);
    }
  }

  return expanded_index_components;
}

// This generates the body of the while that implements the main data movement
// behavior of gather using dynamic-slice and dynamic-update-slice.
void GatherLoopBody(GatherOp gather, Region &body,
                    TypedValue<RankedTensorType> operand,
                    TypedValue<RankedTensorType> start_indices) {
  OpBuilder builder(body);

  auto induction_var = body.getArgument(0);
  auto output_accumulator = body.getArgument(1);

  TypeTools typetools;
  auto index_type = typetools.getExpressedType(
      induction_var.getType().dyn_cast<RankedTensorType>().getElementType());

  // Increment counter first
  auto const_one = builder.create<ConstantOp>(
      gather->getLoc(),
      DenseElementsAttr::get(RankedTensorType::get({}, index_type),
                             builder.getIntegerAttr(index_type, 1)));

  // counter + 1
  auto incremented_counter =
      builder.create<AddOp>(induction_var.getLoc(), induction_var.getType(),
                            induction_var, const_one);

  const auto &dim_numbers = gather.getDimensionNumbers();

  bool has_scalar_indices = start_indices.getType().getShape().size() == 1;
  SPU_ENFORCE_EQ(
      has_scalar_indices,
      dim_numbers.getIndexVectorDim() ==
          (int64_t)gather.getStartIndices().getType().getShape().size());

  auto index_zero = builder.create<ConstantOp>(
      gather->getLoc(),
      builder.getZeroAttr(RankedTensorType::get({}, index_type)));

  TypedValue<RankedTensorType> index_vector;

  if (has_scalar_indices) {
    // In this case start_indices has rank 1 and induction_var_as_vector (of
    // shape {1}) is an index into this rank 1 tensor.
    auto ds = builder.create<DynamicSliceOp>(gather->getLoc(), start_indices,
                                             ValueRange{induction_var},
                                             ConvertDimensions(&builder, {1}));
    index_vector = ds.getResult();
  } else {
    // In this case start_indices has rank 2 and induction_var_as_vector (of
    // shape {1}) is an index into just the first dimension of this rank 2
    // tensor.

    int64_t index_vector_size = start_indices.getType().getShape()[1];

    auto index_vector_2d = builder.create<DynamicSliceOp>(
        gather->getLoc(), start_indices, ValueRange{induction_var, index_zero},
        ConvertDimensions(&builder, {1, index_vector_size}));

    index_vector = ElideDegenerateDims(&builder, index_vector_2d, {0});
  }

  auto gathered_slice_start = ExpandIndexVectorIntoOperandSpace(
      &builder, index_vector, dim_numbers, operand.getType().getShape().size());

  auto gathered_slice = builder.create<DynamicSliceOp>(
      gather->getLoc(), operand, gathered_slice_start, gather.getSliceSizes());

  auto gathered_slice_with_dims_collapsed = ElideDegenerateDims(
      &builder, gathered_slice, dim_numbers.getCollapsedSliceDims());

  auto gathered_slice_for_update =
      PrependDegenerateDims(gathered_slice_with_dims_collapsed, 1);

  SmallVector<Value> index_vector_into_accumulator;
  index_vector_into_accumulator.push_back(induction_var);
  for (size_t idx = 0;
       idx < gathered_slice_with_dims_collapsed.getType().getShape().size();
       ++idx) {
    index_vector_into_accumulator.push_back(index_zero);
  }

  auto updated_accumulator = builder.create<DynamicUpdateSliceOp>(
      gather->getLoc(), output_accumulator, gathered_slice_for_update,
      index_vector_into_accumulator);

  builder.create<ReturnOp>(
      gather->getLoc(), ValueRange{incremented_counter, updated_accumulator});
}

struct GatherConverter : public OpRewritePattern<GatherOp> {
  explicit GatherConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {

    TypeTools type_tool;
    if (type_tool.getTypeVisibility(op.getStartIndices().getType()) !=
        Visibility::VIS_SECRET) {
      // Do not expand public gather
      return failure();
    }

    OpBuilder builder(op);

    // Secret gather
    if (GatherIsBroadcast(op)) {
      // Replace gather with broadcast
      auto broadcast_operand_shape =
          DeleteDimensions(op.getDimensionNumbers().getCollapsedSliceDims(),
                           op.getOperand().getType().getShape());
      auto reshaped_type = RankedTensorType::get(
          broadcast_operand_shape, op.getOperand().getType().getElementType());
      auto broadcast_operand = builder.create<ReshapeOp>(
          op->getLoc(), reshaped_type, op.getOperand());
      rewriter.replaceOpWithNewOp<BroadcastOp>(
          op, op->getResults().getType(), broadcast_operand,
          ConvertDimensions(&builder,
                            op.getDimensionNumbers().getOffsetDims()));
      return success();
    }

    auto index_type = type_tool.getExpressedType(
        op.getStartIndices().getType().getElementType());
    auto operand = op.getOperand();
    auto start_indices = op.getStartIndices();
    auto output_type = op->getResultTypes()[0].dyn_cast<ShapedType>();
    auto output_shape = output_type.getShape();
    int64_t output_rank = output_shape.size();

    const auto &dim_numbers = op.getDimensionNumbers();

    int64_t gather_loop_trip_count = GatherLoopTripCount(op);

    auto canonical_start_indices = CanonicalizeGatherIndices(
        start_indices, dim_numbers.getIndexVectorDim());

    SPU_ENFORCE(gather_loop_trip_count ==
                canonical_start_indices.getType().getShape()[0]);

    auto accumulator_init = CreateGatherLoopAccumulatorInitValue(
        op, output_type.getElementType(), op.getSliceSizes(),
        gather_loop_trip_count, op.getDimensionNumbers());

    auto loopUpperBound = builder.create<ConstantOp>(
        op->getLoc(),
        DenseElementsAttr::get(
            RankedTensorType::get({}, index_type),
            builder.getIntegerAttr(index_type, gather_loop_trip_count)));

    auto counter = builder.create<ConstantOp>(
        op->getLoc(),
        builder.getZeroAttr(RankedTensorType::get({}, index_type)));

    auto loop = builder.create<WhileOp>(
        op->getLoc(),
        TypeRange{counter.getResult().getType(), accumulator_init.getType()},
        ValueRange{counter, accumulator_init});
    {
      loop.getCond().push_back(new Block());
      loop.getCond().front().addArguments(
          TypeRange{counter.getType(), accumulator_init.getType()},
          {counter.getLoc(), accumulator_init.getLoc()});
    }
    {
      loop.getBody().push_back(new Block());

      loop.getBody().front().addArguments(
          TypeRange{counter.getType(), accumulator_init.getType()},
          {counter.getLoc(), accumulator_init.getLoc()});
    }
    // Generate loop condition
    BuildWhileCondition(loop.getCond(), counter.getResult(),
                        canonical_start_indices, accumulator_init,
                        loopUpperBound.getResult());

    GatherLoopBody(op, loop.getBody(), operand, canonical_start_indices);

    OpResult accumulator_result = loop->getResults().back();

    auto accumulator_with_batch_dims_decanonicalized =
        AdjustBatchDimsInAccumulator(
            &builder, start_indices.getType().getShape(),
            cast<mlir::TypedValue<RankedTensorType>>(accumulator_result),
            dim_numbers.getIndexVectorDim());

    std::vector<int64_t> permutation;
    permutation.reserve(output_rank);

    int64_t batch_idx_counter = 0;
    int64_t offset_idx_counter =
        output_rank - dim_numbers.getOffsetDims().size();
    for (int64_t i = 0; i < output_rank; i++) {
      bool is_offset_dim =
          std::binary_search(dim_numbers.getOffsetDims().begin(),
                             dim_numbers.getOffsetDims().end(), i);
      if (is_offset_dim) {
        permutation.push_back(offset_idx_counter++);
      } else {
        permutation.push_back(batch_idx_counter++);
      }
    }

    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getResult().getType(),
        accumulator_with_batch_dims_decanonicalized,
        ConvertDimensions(&builder, permutation));

    return success();
  }
};

struct ExpandSecretGather : public ExpandSecretGatherBase<ExpandSecretGather> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<GatherConverter>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createExpandSecretGatherPass() {
  return std::make_unique<ExpandSecretGather>();
}

} // namespace mlir::pphlo
