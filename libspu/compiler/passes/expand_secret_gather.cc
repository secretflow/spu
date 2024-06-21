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

#include <numeric>
#include <unordered_set>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/core/prelude.h"
#include "libspu/dialect/pphlo/ops.h"

namespace mlir::spu::pphlo {

namespace {

bool GatherIsBroadcast(CustomCallOp &op) {
  auto operand = op->getOperands()[0];
  auto attr =
      mlir::dyn_cast<mlir::DictionaryAttr>(op->getAttr("pphlo.attributes"));
  auto gather_slice_size =
      mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("slice_sizes"))
          .asArrayRef();
  auto op_shape = mlir::dyn_cast<ShapedType>(operand.getType()).getShape();
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

// Computes how many trips a loop implementing this gather op would take.
int64_t GatherLoopTripCount(CustomCallOp op) {
  auto start_indices = op->getOperands()[1];
  auto start_indices_shape =
      mlir::dyn_cast<ShapedType>(start_indices.getType()).getShape();
  auto attr =
      mlir::dyn_cast<mlir::DictionaryAttr>(op->getAttr("pphlo.attributes"));

  int64_t trip_count = 1;
  for (int64_t i = 0, e = start_indices_shape.size(); i < e; i++) {
    if (i != mlir::dyn_cast<mlir::IntegerAttr>(attr.get("index_vector_dim"))
                 .getInt()) {
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

Value TransposeIndexVectorDimToLast(Value &start_indices,
                                    int64_t index_vector_dim) {
  const auto start_indices_shape =
      mlir::dyn_cast<ShapedType>(start_indices.getType()).getShape();

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
      RankedTensorType::get(result_shape, mlir::dyn_cast<RankedTensorType>(
                                              start_indices.getType())
                                              .getElementType()),
      start_indices, permutation);

  return transpose.getResult();
}

Value PrependDegenerateDims(Value operand, int64_t n) {
  SPU_ENFORCE(n > 0);
  std::vector<int64_t> new_shape_dims;
  const auto operand_shape =
      mlir::dyn_cast<ShapedType>(operand.getType()).getShape();
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
      RankedTensorType::get(
          new_shape_dims,
          mlir::dyn_cast<RankedTensorType>(operand.getType()).getElementType()),
      operand);

  return reshape.getResult();
}

Value CollapseFirstNDims(Value operand, int64_t n) {
  SPU_ENFORCE(n > 0);

  const auto operand_shape =
      mlir::dyn_cast<ShapedType>(operand.getType()).getShape();
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

  auto output_type = RankedTensorType::get(
      new_shape_dims,
      mlir::dyn_cast<RankedTensorType>(operand.getType()).getElementType());

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
Value CanonicalizeGatherIndices(Value start_indices, int64_t index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  auto transposed_start_indices =
      TransposeIndexVectorDimToLast(start_indices, index_vector_dim);
  bool indices_are_scalar =
      index_vector_dim ==
      static_cast<int64_t>(mlir::dyn_cast<ShapedType>(start_indices.getType())
                               .getShape()
                               .size());

  // The number of dimensions in start_indices that are index dimensions.
  const int64_t index_dims_in_start_indices = indices_are_scalar ? 0 : 1;

  // If there is only one index (i.e. start_indices has rank 1 and this gather
  // is really just a dynamic slice) add a leading degenerate dimension for
  // uniformity.  Otherwise create a "collapsed" leading dimension that subsumes
  // all of the non-index-vector dimensions.
  const auto shape =
      mlir::dyn_cast<ShapedType>(transposed_start_indices.getType()).getShape();
  if (static_cast<int64_t>(shape.size()) == index_dims_in_start_indices) {
    return PrependDegenerateDims(transposed_start_indices, 1);
  } else {
    // Collapse all but the dimensions (0 or 1) in start_indices containing the
    // index vectors.
    return CollapseFirstNDims(transposed_start_indices,
                              shape.size() - index_dims_in_start_indices);
  }
}

Value CreateGatherLoopAccumulatorInitValue(CustomCallOp op, Type element_type,
                                           int64_t gather_loop_trip_count,
                                           DictionaryAttr attr) {
  std::vector<int64_t> accumulator_state_shape_dims;
  const auto slice_size =
      mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("slice_sizes"))
          .asArrayRef();
  const auto collapsed_slice_dims =
      mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("collapsed_slice_dims"))
          .asArrayRef();
  accumulator_state_shape_dims.reserve(1 + slice_size.size());
  accumulator_state_shape_dims.push_back(gather_loop_trip_count);
  for (int64_t i = 0; i < static_cast<int64_t>(slice_size.size()); i++) {
    if (!std::binary_search(collapsed_slice_dims.begin(),
                            collapsed_slice_dims.end(), i)) {
      accumulator_state_shape_dims.emplace_back(slice_size[i]);
    }
  }

  OpBuilder builder(op);
  TypeTools type_tools(op->getContext());

  auto express_type = type_tools.getExpressedType(element_type);
  auto shaped_type =
      RankedTensorType::get(accumulator_state_shape_dims, express_type);
  auto zero_attr = builder.getZeroAttr(shaped_type);

  if (zero_attr == nullptr && mlir::isa<mlir::ComplexType>(express_type)) {
    std::complex<APFloat> zero = {APFloat(0.0F), APFloat(0.0F)};
    zero_attr = DenseElementsAttr::get(shaped_type,
                                       std::vector<std::complex<llvm::APFloat>>(
                                           shaped_type.getNumElements(), zero));
  }

  auto c = builder.create<ConstantOp>(op->getLoc(), zero_attr);

  if (type_tools.getTypeVisibility(element_type) != Visibility::PUBLIC) {
    auto convert = builder.create<ConvertOp>(
        op.getLoc(),
        RankedTensorType::get(accumulator_state_shape_dims, element_type),
        c.getResult());
    return convert.getResult();
  } else {
    return c.getResult();
  }
}

Value ExpandFirstDimIntoNDims(Value operand,
                              llvm::ArrayRef<int64_t> expanded_dims) {
  const auto &operand_shape =
      mlir::dyn_cast<ShapedType>(operand.getType()).getShape();
  SPU_ENFORCE_GT(operand_shape.size(), size_t(0));
  SPU_ENFORCE_EQ(operand_shape[0],
                 std::accumulate(expanded_dims.begin(), expanded_dims.end(), 1,
                                 std::multiplies()));

  std::vector<int64_t> expanded_shape_dim_bounds;
  expanded_shape_dim_bounds.reserve(expanded_dims.size() +
                                    operand_shape.size() - 1);
  std::copy(expanded_dims.begin(), expanded_dims.end(),
            std::back_inserter(expanded_shape_dim_bounds));
  std::copy(operand_shape.begin() + 1, operand_shape.end(),
            std::back_inserter(expanded_shape_dim_bounds));

  auto result_type = RankedTensorType::get(
      expanded_shape_dim_bounds,
      mlir::dyn_cast<RankedTensorType>(operand.getType()).getElementType());

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

Value ElideDegenerateDims(OpBuilder *builder, Value operand,
                          absl::Span<const int64_t> dims_to_elide) {
  std::unordered_set<int64_t> dims_to_elide_set(dims_to_elide.begin(),
                                                dims_to_elide.end());
  std::vector<int64_t> new_shape;
  const auto &operand_shape =
      mlir::dyn_cast<ShapedType>(operand.getType()).getShape();
  for (size_t idx = 0; idx < operand_shape.size(); ++idx) {
    if (dims_to_elide_set.count(idx) > 0) {
      continue;
    }
    new_shape.emplace_back(operand_shape[idx]);
  }

  auto reshape = builder->create<ReshapeOp>(
      operand.getLoc(),
      RankedTensorType::get(
          new_shape,
          mlir::dyn_cast<RankedTensorType>(operand.getType()).getElementType()),
      operand);
  return reshape.getResult();
}

// Expands out or contracts away the gather dimensions in the accumulator
// produced by the while loop.
Value AdjustBatchDimsInAccumulator(OpBuilder *builder,
                                   llvm::ArrayRef<int64_t> start_indices_shape,
                                   Value accumulator,
                                   int64_t index_vector_dim) {
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

void BuildWhileCondition(MLIRContext *ctx, Region &cond, Value /*counter*/,
                         Value /*canonical_start_indices*/,
                         Value /*accumulator_init*/, Value loop_upper_bound) {
  OpBuilder builder(cond);
  TypeTools type_tool(ctx);

  auto lt = builder.create<LessOp>(
      cond.getLoc(),
      RankedTensorType::get(
          {}, type_tool.getType(builder.getI1Type(), Visibility::PUBLIC)),
      cond.getArgument(0), loop_upper_bound);

  builder.create<ReturnOp>(cond.getLoc(), ValueRange{lt.getResult()});
}

int64_t FindIndex(llvm::ArrayRef<int64_t> c, int64_t value) {
  const auto *it = std::find(c.begin(), c.end(), value);
  return std::distance(c.begin(), it);
}

// Expand an index vector from the start_indices tensor into a vector that can
// be used to dynamic-slice out of the gather operand.
llvm::SmallVector<Value>
ExpandIndexVectorIntoOperandSpace(MLIRContext *ctx, OpBuilder *builder,
                                  Value index_vector, DictionaryAttr attr,
                                  int64_t operand_rank) {

  TypeTools typetool(ctx);
  auto index_type = typetool.getExpressedType(
      mlir::dyn_cast<RankedTensorType>(index_vector.getType())
          .getElementType());

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
      RankedTensorType::get({},
                            typetool.getType(index_type, Visibility::SECRET)),
      p_zero_const);

  // We extract out individual components from the smaller index and concatenate
  // them (interspersing zeros as needed) into the larger index.
  llvm::SmallVector<Value> expanded_index_components;

  const auto start_index_map =
      mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("start_index_map"))
          .asArrayRef();
  for (int64_t i = 0; i < operand_rank; i++) {
    int64_t index_vector_dim_index = FindIndex(start_index_map, i);
    if (index_vector_dim_index !=
        static_cast<int64_t>(start_index_map.size())) {

      auto component_to_concat = builder->create<SliceOp>(
          index_vector.getLoc(),
          RankedTensorType::get(
              {1}, mlir::dyn_cast<RankedTensorType>(index_vector.getType())
                       .getElementType()),
          index_vector,
          DenseI64ArrayAttr::get(builder->getContext(),
                                 {index_vector_dim_index}),
          DenseI64ArrayAttr::get(builder->getContext(),
                                 {index_vector_dim_index + 1}),
          DenseI64ArrayAttr::get(builder->getContext(), {1}));
      auto reshaped = builder->create<ReshapeOp>(
          index_vector.getLoc(),
          RankedTensorType::get(
              {}, mlir::dyn_cast<RankedTensorType>(index_vector.getType())
                      .getElementType()),
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
void GatherLoopBody(CustomCallOp gather, Region &body, Value operand,
                    Value start_indices) {
  OpBuilder builder(body);

  auto induction_var = body.getArgument(0);
  auto output_accumulator = body.getArgument(1);

  TypeTools typetools(gather->getContext());
  auto index_type = typetools.getExpressedType(
      mlir::dyn_cast<RankedTensorType>(induction_var.getType())
          .getElementType());

  // Increment counter first
  auto const_one = builder.create<ConstantOp>(
      gather->getLoc(),
      DenseElementsAttr::get(RankedTensorType::get({}, index_type),
                             builder.getIntegerAttr(index_type, 1)));

  // counter + 1
  auto incremented_counter =
      builder.create<AddOp>(induction_var.getLoc(), induction_var.getType(),
                            induction_var, const_one);
  auto attr =
      mlir::dyn_cast<mlir::DictionaryAttr>(gather->getAttr("pphlo.attributes"));
  auto index_vector_dim =
      mlir::dyn_cast<mlir::IntegerAttr>(attr.get("index_vector_dim")).getInt();

  const auto &start_indices_shape =
      mlir::dyn_cast<ShapedType>(start_indices.getType()).getShape();
  bool has_scalar_indices = start_indices_shape.size() == 1;
  SPU_ENFORCE_EQ(has_scalar_indices,
                 index_vector_dim == (int64_t)start_indices_shape.size());

  auto index_zero = builder.create<ConstantOp>(
      gather->getLoc(),
      builder.getZeroAttr(RankedTensorType::get({}, index_type)));

  Value index_vector;

  if (has_scalar_indices) {
    // In this case start_indices has rank 1 and induction_var_as_vector (of
    // shape {1}) is an index into this rank 1 tensor.
    auto ds = builder.create<DynamicSliceOp>(
        gather->getLoc(), start_indices, ValueRange{induction_var},
        DenseI64ArrayAttr::get(builder.getContext(), {1}));
    index_vector = ds.getResult();
  } else {
    // In this case start_indices has rank 2 and induction_var_as_vector (of
    // shape {1}) is an index into just the first dimension of this rank 2
    // tensor.

    int64_t index_vector_size = start_indices_shape[1];

    auto index_vector_2d = builder.create<DynamicSliceOp>(
        gather->getLoc(), start_indices, ValueRange{induction_var, index_zero},
        DenseI64ArrayAttr::get(builder.getContext(), {1, index_vector_size}));

    index_vector = ElideDegenerateDims(&builder, index_vector_2d, {0});
  }

  auto gathered_slice_start = ExpandIndexVectorIntoOperandSpace(
      gather->getContext(), &builder, index_vector, attr,
      mlir::dyn_cast<ShapedType>(operand.getType()).getShape().size());

  auto gathered_slice = builder.create<DynamicSliceOp>(
      gather->getLoc(), operand, gathered_slice_start,
      mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("slice_sizes"))
          .asArrayRef());

  auto gathered_slice_with_dims_collapsed = ElideDegenerateDims(
      &builder, gathered_slice,
      mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("collapsed_slice_dims"))
          .asArrayRef());

  auto gathered_slice_for_update =
      PrependDegenerateDims(gathered_slice_with_dims_collapsed, 1);

  SmallVector<Value> index_vector_into_accumulator;
  index_vector_into_accumulator.push_back(induction_var);
  for (size_t idx = 0; idx < mlir::dyn_cast<ShapedType>(
                                 gathered_slice_with_dims_collapsed.getType())
                                 .getShape()
                                 .size();
       ++idx) {
    index_vector_into_accumulator.push_back(index_zero);
  }

  auto updated_accumulator = builder.create<DynamicUpdateSliceOp>(
      gather->getLoc(), output_accumulator, gathered_slice_for_update,
      index_vector_into_accumulator);

  builder.create<ReturnOp>(
      gather->getLoc(), ValueRange{incremented_counter, updated_accumulator});
}

// pphlo.gather is custom call now
struct GatherConverter : public OpRewritePattern<CustomCallOp> {
  explicit GatherConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getCallTargetName() != "pphlo.gather") {
      return failure();
    }

    auto operand = op.getOperands()[0];
    auto start_indices = op.getOperands()[1];

    TypeTools type_tool(op->getContext());
    if (type_tool.getTypeVisibility(start_indices.getType()) !=
        Visibility::SECRET) {
      // Do not expand public gather
      return failure();
    }

    OpBuilder builder(op);
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(op->getAttr("pphlo.attributes"));

    // Secret gather
    if (GatherIsBroadcast(op)) {
      // Replace gather with broadcast
      auto broadcast_operand_shape = DeleteDimensions(
          mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
              attr.get("collapsed_slice_dims"))
              .asArrayRef(),
          mlir::dyn_cast<ShapedType>(operand.getType()).getShape());
      auto reshaped_type = RankedTensorType::get(
          broadcast_operand_shape,
          mlir::dyn_cast<RankedTensorType>(operand.getType()).getElementType());
      auto broadcast_operand =
          builder.create<ReshapeOp>(op->getLoc(), reshaped_type, operand);
      rewriter.replaceOpWithNewOp<BroadcastOp>(
          op, op->getResults().getType(), broadcast_operand,
          DenseI64ArrayAttr::get(
              builder.getContext(),
              mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("offset_dims"))
                  .asArrayRef()));
      return success();
    }

    auto index_type = type_tool.getExpressedType(
        mlir::dyn_cast<RankedTensorType>(start_indices.getType())
            .getElementType());
    auto output_type = mlir::dyn_cast<ShapedType>(op->getResultTypes()[0]);
    auto output_shape = output_type.getShape();
    int64_t output_rank = output_shape.size();

    int64_t gather_loop_trip_count = GatherLoopTripCount(op);

    auto canonical_start_indices = CanonicalizeGatherIndices(
        start_indices,
        mlir::dyn_cast<mlir::IntegerAttr>(attr.get("index_vector_dim"))
            .getInt());

    SPU_ENFORCE(gather_loop_trip_count ==
                mlir::dyn_cast<ShapedType>(canonical_start_indices.getType())
                    .getShape()[0]);

    auto accumulator_init = CreateGatherLoopAccumulatorInitValue(
        op, output_type.getElementType(), gather_loop_trip_count, attr);

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
    BuildWhileCondition(op->getContext(), loop.getCond(), counter.getResult(),
                        canonical_start_indices, accumulator_init,
                        loopUpperBound.getResult());

    GatherLoopBody(op, loop.getBody(), operand, canonical_start_indices);

    OpResult accumulator_result = loop->getResults().back();

    auto accumulator_with_batch_dims_decanonicalized =
        AdjustBatchDimsInAccumulator(
            &builder,
            mlir::dyn_cast<ShapedType>(start_indices.getType()).getShape(),
            cast<mlir::Value>(accumulator_result),
            mlir::dyn_cast<mlir::IntegerAttr>(attr.get("index_vector_dim"))
                .getInt());

    std::vector<int64_t> permutation;
    permutation.reserve(output_rank);

    auto offset_dims =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("offset_dims"))
            .asArrayRef();
    int64_t batch_idx_counter = 0;
    int64_t offset_idx_counter = output_rank - offset_dims.size();
    for (int64_t i = 0; i < output_rank; i++) {
      bool is_offset_dim =
          std::binary_search(offset_dims.begin(), offset_dims.end(), i);
      if (is_offset_dim) {
        permutation.push_back(offset_idx_counter++);
      } else {
        permutation.push_back(batch_idx_counter++);
      }
    }

    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getResults()[0].getType(),
        accumulator_with_batch_dims_decanonicalized,
        DenseI64ArrayAttr::get(builder.getContext(), permutation));

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

} // namespace mlir::spu::pphlo