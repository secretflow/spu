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

#include "libspu/dialect/pphlo/ops.h"

#include <numeric>

#include "fmt/format.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "libspu/dialect/pphlo/attrs.h"
#include "libspu/dialect/pphlo/base_enums.h"
#include "libspu/dialect/pphlo/ops.h.inc"

namespace mlir::spu::pphlo {

#include "libspu/dialect/pphlo/patterns.cc.inc"

namespace {

// Checks if the vector `nums` has duplicates.
bool hasDuplicates(const ArrayRef<int64_t> nums) {
  llvm::SmallDenseSet<int64_t> set(nums.begin(), nums.end());
  return set.size() != nums.size();
}

}  // namespace

template <typename T>
static LogicalResult Verify(T /*op*/) {
  return success();
}

// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder&, OperationState& result, Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<DenseElementsAttr>()) {
    auto valueType = elemAttr.getType().dyn_cast<RankedTensorType>();
    type =
        RankedTensorType::get(valueType.getShape(), valueType.getElementType());
  }

  assert(type && "unsupported attribute type for building pphlo.constant");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

OpFoldResult ConstantOp::fold([[maybe_unused]] FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

OpFoldResult ConvertOp::fold(FoldAdaptor) {
  auto operand_ty = getOperand().getType().cast<TensorType>();
  auto result_ty = getResult().getType().cast<TensorType>();
  if (operand_ty == result_ty) {
    return getOperand();
  }

  return {};
}

namespace {

bool isConsecutive(ArrayRef<int64_t> array) {
  for (size_t i = 1; i < array.size(); ++i) {
    if (array[i] - array[i - 1] != 1) {
      return false;
    }
  }
  return true;
}

// This is a pass ported from iree, which simplified dot_general's order of
// dimensions on lhs and rhs
// Ref:
// https://github.com/iree-org/iree/blob/a3481172de519d27b6ec215ee3355ce4f91531fa/compiler/src/iree/compiler/InputConversion/MHLO/MHLOToMHLOPreprocessing.cpp#L277
class TransposeReshapeGenericDotGeneral
    : public OpRewritePattern<DotGeneralOp> {
 public:
  using OpRewritePattern<DotGeneralOp>::OpRewritePattern;

  static Value TransposeIfNonConsecutive(OpBuilder& b, Location loc, Value src,
                                         ArrayRef<int64_t> target_order) {
    if (isConsecutive(target_order)) {
      return src;
    }
    auto type = src.getType().cast<RankedTensorType>();
    SmallVector<int64_t, 4> transposeShape;
    for (auto i : target_order) {
      transposeShape.push_back(type.getDimSize(i));
    }
    return b.create<pphlo::TransposeOp>(
        loc, RankedTensorType::get(transposeShape, type.getElementType()), src,
        target_order);
  }

  Value ReshapeIfNonStandard(OpBuilder& b, Location loc, Value src,
                             size_t dims_border0, size_t dims_border1) const {
    auto type = cast<RankedTensorType>(src.getType());
    ArrayRef<int64_t> shape = type.getShape();
    if (dims_border0 <= 1 && dims_border1 - dims_border0 <= 1 &&
        shape.size() - dims_border1 <= 1) {
      return src;
    }

    SmallVector<int64_t> result_shape = {
        std::accumulate(shape.begin(), shape.begin() + dims_border0, 1,
                        std::multiplies<int64_t>()),
        std::accumulate(shape.begin() + dims_border0,
                        shape.begin() + dims_border1, 1,
                        std::multiplies<int64_t>()),
        std::accumulate(shape.begin() + dims_border1, shape.end(), 1,
                        std::multiplies<int64_t>())};
    return b.create<pphlo::ReshapeOp>(
        loc, RankedTensorType::get(result_shape, type.getElementType()), src);
  }

  LogicalResult matchAndRewrite(pphlo::DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    auto lhsShapeType = op.getLhs().getType().dyn_cast<RankedTensorType>();
    auto rhsShapeType = op.getRhs().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!lhsShapeType || !rhsShapeType || !resultType) {
      return failure();
    }

    SmallVector<int64_t> lhsTargetOrder;
    SmallVector<int64_t> rhsTargetOrder;
    pphlo::DotDimensionNumbersAttr dimNumbers = op.getDotDimensionNumbers();
    auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();

    // No contraction dims means this can be represented as a mul.
    if (lhsContractingDims.empty()) {
      return failure();
    }
    if (rhsContractingDims.empty()) {
      return failure();
    }

    // No batching dimensions means this can be represented a dot.
    if (lhsBatchingDims.empty()) {
      return failure();
    }
    if (rhsBatchingDims.empty()) {
      return failure();
    }

    SmallVector<bool> isLhsParallel(lhsShapeType.getRank(), true);
    for (auto i : lhsBatchingDims) {
      lhsTargetOrder.push_back(i);
      isLhsParallel[i] = false;
    }
    for (auto i : lhsContractingDims) {
      isLhsParallel[i] = false;
    }
    for (int64_t i = 0, e = lhsShapeType.getRank(); i < e; ++i) {
      if (isLhsParallel[i]) {
        lhsTargetOrder.push_back(i);
      }
    }
    for (auto i : lhsContractingDims) {
      lhsTargetOrder.push_back(i);
    }

    SmallVector<bool> isRhsParallel(rhsShapeType.getRank(), true);

    for (auto i : rhsBatchingDims) {
      rhsTargetOrder.push_back(i);
      isRhsParallel[i] = false;
    }
    for (auto i : rhsContractingDims) {
      rhsTargetOrder.push_back(i);
      isRhsParallel[i] = false;
    }
    for (int64_t i = 0, e = rhsShapeType.getRank(); i < e; ++i) {
      if (isRhsParallel[i]) {
        rhsTargetOrder.push_back(i);
      }
    }

    Value lhs = TransposeIfNonConsecutive(rewriter, op.getLoc(), op.getLhs(),
                                          lhsTargetOrder);
    Value rhs = TransposeIfNonConsecutive(rewriter, op.getLoc(), op.getRhs(),
                                          rhsTargetOrder);

    // The dimensions of this will always be transposed into {batch_dims,
    // parallel_dims, contraction_dims}, and the
    // following logic is based on this assumption.
    // TODO(#7443): If we consider transpose performance, the above assumptions
    // may not be true.
    int64_t numLhsContractionDims = lhsContractingDims.size();
    int64_t lhsContractionBase = lhsShapeType.getRank() - numLhsContractionDims;
    int64_t rhsContractionBase = rhsBatchingDims.size();
    int64_t numRhsContractionDims =
        rhsContractionBase + rhsContractingDims.size();

    lhs = ReshapeIfNonStandard(rewriter, op.getLoc(), lhs,
                               rhsBatchingDims.size(), lhsContractionBase);
    rhs = ReshapeIfNonStandard(rewriter, op.getLoc(), rhs,
                               rhsBatchingDims.size(), numRhsContractionDims);

    if (lhs == op.getLhs() && rhs == op.getRhs()) {
      return failure();
    }

    auto dimensionNumbers = pphlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/0,
        /*rhsBatchingDimensions=*/0,
        /*lhsContractingDimensions=*/
        lhs.getType().dyn_cast<ShapedType>().getRank() - 1,
        /*rhsContractingDimensions=*/1);
    auto lhsNewType = lhs.getType().cast<RankedTensorType>();
    auto rhsNewType = rhs.getType().cast<RankedTensorType>();

    // if lhs's shape or rhs's shape has collapsed, we need reshape the result
    bool needReshapeResult = lhsNewType.getRank() < lhsShapeType.getRank() ||
                             rhsNewType.getRank() < rhsShapeType.getRank();
    // batching、lhs parallel、rhs parallel this order is a convention
    SmallVector<int64_t, 4> newShape = {lhsNewType.getShape()[0],
                                        lhsNewType.getShape()[1],
                                        rhsNewType.getShape()[2]};
    auto newResultType =
        needReshapeResult
            ? RankedTensorType::get(newShape, resultType.getElementType())
            : op.getType();

    auto newOp = rewriter.create<pphlo::DotGeneralOp>(
        op.getLoc(), newResultType, lhs, rhs, dimensionNumbers);

    Value result = newOp.getResult();
    if (needReshapeResult) {
      result =
          rewriter.create<pphlo::ReshapeOp>(op.getLoc(), resultType, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

namespace {

llvm::SmallVector<int64_t> InversePermutation(
    llvm::ArrayRef<int64_t> input_permutation) {
  llvm::SmallVector<int64_t> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation[input_permutation[i]] = i;
  }
  return output_permutation;
}

bool IsSameShape(llvm::ArrayRef<int64_t> lhs, llvm::ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t idx = 0; idx < lhs.size(); ++idx) {
    if (lhs[idx] != rhs[idx]) {
      return false;
    }
  }
  return true;
}

}  // namespace

// This is piece of code is ported from tensorflow/xla
// Ref
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/cpu/conv_canonicalization.cc
class NormalizeDimensionOrder : public OpRewritePattern<ConvolutionOp> {
 private:
  static bool needTranspose(llvm::ArrayRef<int64_t> old_shape,
                            llvm::ArrayRef<int64_t> new_shape,
                            llvm::ArrayRef<int64_t> permutation) {
    if (!IsSameShape(old_shape, new_shape)) {
      return true;
    }
    // Same shape, check permutation
    for (size_t idx = 0; idx < permutation.size(); ++idx) {
      if (permutation[idx] != static_cast<int64_t>(idx)) {
        return true;
      }
    }
    return false;
  }

 public:
  using OpRewritePattern<ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    const auto& dnums = op.getDimensionNumbers();
    auto input_batch_dim = dnums.getInputBatchDimension();
    auto input_feature_dim = dnums.getInputFeatureDimension();
    auto kernel_input_feature_dim = dnums.getKernelInputFeatureDimension();
    auto kernel_output_feature_dim = dnums.getKernelOutputFeatureDimension();

    const int64_t num_spatial_dims = dnums.getOutputSpatialDimensions().size();
    const int64_t num_dims = num_spatial_dims + 2;

    // A canonical convolution's dimension numbers need to satisfy the
    // following conditions (see cs/PotentiallyImplementedAsEigenConvolution).
    //
    // - the input is in NHWC order.
    // - the kernel is in HWIO order.
    //
    // For simplicity, as a first step, we reshape the input and filter to
    // NHWC and HWIO order, respectively. This may lose precision but won't
    // break the soundness.
    auto input = op.getLhs();
    auto input_type = input.getType().dyn_cast<RankedTensorType>();
    auto input_shape = input_type.getShape();

    llvm::SmallVector<int64_t> new_input_dim_order(num_dims);
    llvm::SmallVector<int64_t> new_input_dims(num_dims);
    new_input_dim_order[0] = input_batch_dim;
    new_input_dims[0] = input_shape[input_batch_dim];
    for (int64_t i = 0; i < num_spatial_dims; ++i) {
      new_input_dim_order[i + 1] = dnums.getInputSpatialDimensions()[i];
      new_input_dims[i + 1] = input_shape[dnums.getInputSpatialDimensions()[i]];
    }
    new_input_dim_order[num_dims - 1] = input_feature_dim;
    new_input_dims[num_dims - 1] = input_shape[input_feature_dim];

    mlir::Value new_input = input;
    if (needTranspose(input_shape, new_input_dims, new_input_dim_order)) {
      auto new_input_type =
          RankedTensorType::get(new_input_dims, input_type.getElementType());
      new_input = rewriter.create<pphlo::TransposeOp>(
          op->getLoc(), new_input_type, input, new_input_dim_order);
    }

    auto kernel = op.getRhs();

    auto kernel_type = kernel.getType().dyn_cast<RankedTensorType>();
    auto kernel_shape = kernel_type.getShape();

    llvm::SmallVector<int64_t> new_kernel_dim_order(num_dims);
    llvm::SmallVector<int64_t> new_kernel_dims(num_dims);

    for (int64_t i = 0; i < num_spatial_dims; ++i) {
      new_kernel_dim_order[i] = dnums.getKernelSpatialDimensions()[i];
      new_kernel_dims[i] = kernel_shape[dnums.getKernelSpatialDimensions()[i]];
    }
    new_kernel_dim_order[num_dims - 2] = kernel_input_feature_dim;
    new_kernel_dims[num_dims - 2] = kernel_shape[kernel_input_feature_dim];
    new_kernel_dim_order[num_dims - 1] = kernel_output_feature_dim;
    new_kernel_dims[num_dims - 1] = kernel_shape[kernel_output_feature_dim];

    mlir::Value new_kernel = kernel;
    if (needTranspose(kernel_shape, new_kernel_dims, new_kernel_dim_order)) {
      auto new_kernel_type =
          RankedTensorType::get(new_kernel_dims, kernel_type.getElementType());
      new_kernel = rewriter.create<TransposeOp>(op->getLoc(), new_kernel_type,
                                                kernel, new_kernel_dim_order);
    }

    if (input == new_input && kernel == new_kernel) {
      return failure();
    }

    llvm::SmallVector<int64_t> new_output_dim_order(num_dims);
    llvm::SmallVector<int64_t> new_conv_dims(num_dims);
    auto output_batch_dim = dnums.getOutputBatchDimension();
    auto output_feature_dim = dnums.getOutputFeatureDimension();
    new_output_dim_order[0] = output_batch_dim;

    auto result_type = op->getResultTypes()[0].dyn_cast<RankedTensorType>();
    auto result_shape = result_type.getShape();

    new_conv_dims[0] = result_shape[output_batch_dim];
    for (int64_t i = 0; i < num_spatial_dims; ++i) {
      new_output_dim_order[i + 1] = dnums.getOutputSpatialDimensions()[i];
      new_conv_dims[i + 1] =
          result_shape[dnums.getOutputSpatialDimensions()[i]];
    }
    new_output_dim_order[num_dims - 1] = output_feature_dim;
    new_conv_dims[num_dims - 1] = result_shape[output_feature_dim];

    auto new_conv_type =
        RankedTensorType::get(new_conv_dims, result_type.getElementType());

    llvm::SmallVector<int64_t> input_sd(num_spatial_dims);
    llvm::SmallVector<int64_t> kernel_sd(num_spatial_dims);
    llvm::SmallVector<int64_t> output_sd(num_spatial_dims);

    for (int64_t i = 0; i < num_spatial_dims; ++i) {
      input_sd[i] = i + 1;
      kernel_sd[i] = i;
      output_sd[i] = i + 1;
    }

    auto new_dnums = ConvDimensionNumbersAttr::get(
        op->getContext(), 0, num_dims - 1, input_sd, num_dims - 2, num_dims - 1,
        kernel_sd, 0, num_dims - 1, output_sd);

    // The window of the old convolution is reused, because reshapes only
    // change the dimension mapping but not the dimension sizes. For
    // example, input height and width are the same as before the reshapes.
    auto new_conv = rewriter.create<ConvolutionOp>(
        op->getLoc(), new_conv_type, new_input, new_kernel,
        DenseI64ArrayAttr::get(op->getContext(),
                               op.getWindowStrides().value_or(std::nullopt)),
        new_dnums, op.getFeatureGroupCount(), op.getBatchGroupCount());

    // Reshape the output back to the shape of the original convolution.
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op->getResultTypes()[0], new_conv,
        InversePermutation(new_output_dim_order));

    return success();
  }
};

}  // namespace

OpFoldResult ReverseOp::fold(FoldAdaptor) {
  auto input = getOperand();

  // No dimensions to reverse.
  auto dims = getDimensions();
  if (dims.empty()) {
    return input;
  }

  // If the dimensions to reverse are all statically 1, then the reverse is a
  // no-op.
  auto shapedType = input.getType().cast<ShapedType>();
  if (llvm::all_of(
          dims, [&](int64_t dim) { return shapedType.getDimSize(dim) == 1; })) {
    return input;
  }
  return {};
}

LogicalResult ReverseOp::verify() {
  //(C1) operand and result have the same type.
  auto inputType = getOperand().getType().cast<RankedTensorType>();
  auto retType = getResult().getType().cast<RankedTensorType>();

  TypeTools tools(getContext());
  if (tools.getExpressedType(inputType) != tools.getExpressedType(retType)) {
    return emitOpError("operand and result type mismatch");
  }

  //(C2) All dimensions in dimensions are unique.
  auto dims = getDimensions();
  llvm::SmallDenseSet<int64_t> unique_dims(dims.begin(), dims.end());

  if (unique_dims.size() != dims.size()) {
    return emitOpError("dimensions are not unique");
  }

  //(C3) For all dimensions k in dimensions, 0 <= dimensions[k] < rank(result).
  for (int64_t dim : unique_dims) {
    if (dim < 0) {
      return emitOpError(llvm::formatv(
          "all dimensions should be non-negative. Got dimension: {0}.", dim));
    }
    if (retType && dim >= retType.getRank()) {
      return emitOpError(llvm::formatv(
          "all dimensions should be between [0, {0}). Got dimension: {1}.",
          retType.getRank(), dim));
    }
  }
  return success();
}

void DotGeneralOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                               MLIRContext* context) {
  results.add<TransposeReshapeGenericDotGeneral>(context);
}

void ConvolutionOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<NormalizeDimensionOrder>(context);
}

void SelectOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                           ::mlir::MLIRContext* context) {
  results.add<FusePredNegIntoSelect>(context);
}

OpFoldResult ReciprocalOp::fold(FoldAdaptor operands) {
  auto val = operands.getOperands()[0].dyn_cast_or_null<DenseFPElementsAttr>();

  if (!val) {
    return {};
  }

  if (val.isSplat()) {
    auto splat_val = val.getSplatValue<APFloat>();
    APFloat one(splat_val.getSemantics(), 1);

    return SplatElementsAttr::get(val.getType().dyn_cast<ShapedType>(),
                                  one / splat_val);
  }

  llvm::SmallVector<APFloat, 4> values;
  values.reserve(val.getNumElements());

  auto first_val = *val.getValues<APFloat>().begin();
  APFloat one(first_val.getSemantics(), 1);

  for (auto it : val.getValues<APFloat>()) {
    values.push_back(one / it);
  }

  return DenseFPElementsAttr::get(val.getType().dyn_cast<ShapedType>(), values);
}

LogicalResult verifyReduceOpInputsAndInferShape(
    std::optional<Location> location, SmallVector<TensorType> inputArgTypes,
    SmallVector<TensorType> /*initValueTypes*/,
    llvm::ArrayRef<int64_t> dimensions, SmallVector<int64_t>& /*newDimensions*/,
    Attribute& /*encoding*/) {
  uint64_t numInputs = inputArgTypes.size();

  for (uint64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    if (failed(mlir::verifyCompatibleShape(inputArgTypes[0],
                                           inputArgTypes[inputIdx]))) {
      return emitOptionalError(
          location, "expects all inputs to have compatible shapes. Shape at",
          " input-index ", inputIdx,
          " is not compatible with shape at input-index 0");
    }
  }

  DenseSet<int64_t> dimensionsToReduceSet;
  for (int64_t dimension : dimensions) {
    if ((dimension >= inputArgTypes[0].getRank()) || dimension < 0) {
      return emitOptionalError(
          location, "Out-of-bounds dimension ", dimension,
          " for input-tensor rank: ", inputArgTypes[0].getRank());
    }

    if (!dimensionsToReduceSet.insert(dimension).second) {
      return emitOptionalError(location,
                               "Duplicate reduction dimension: ", dimension);
    }
  }

  return success();
}

LogicalResult verifyReducerShape(std::optional<Location> loc, Block& block,
                                 ArrayRef<TensorType> /*inputArgTypes*/,
                                 ArrayRef<TensorType> initValueTypes,
                                 int64_t numInputs,
                                 ArrayRef<int64_t> allowedDimensions) {
  // Check that the number of reduction-region arguments matches with that of
  // reduce-op's arguments.
  if (static_cast<int64_t>(block.getArguments().size()) != numInputs * 2) {
    return emitOptionalError(loc, "Reduction-region must take ", numInputs * 2,
                             " parameters, but takes ",
                             block.getArguments().size(), " parameter(s)");
  }

  // Check if the reduction-region produces non-zero outputs.
  if (block.getTerminator()->getOperands().empty()) {
    return emitOptionalError(
        loc, "The reduction-region expected to return some value(s)");
  }

  // Check that the reduction-region returns list- of tensors.
  // The number of result-tensors must match the `numInputs`.
  if (static_cast<int64_t>(block.getTerminator()->getOperands().size()) !=
      numInputs) {
    return emitOptionalError(loc, "Reduction-region here must produce ",
                             numInputs, " tensors, but produces ",
                             block.getTerminator()->getOperands().size(),
                             " instead");
  }

  SmallVector<TensorType> accumulatorSubShapes;
  for (Value retOperand : block.getTerminator()->getOperands()) {
    auto tensorTy = retOperand.getType().dyn_cast<TensorType>();
    if (!tensorTy) {
      return emitOptionalError(loc,
                               "Reduction-region here must produce "
                               "tensor-typed result(s), but "
                               "produces ",
                               retOperand.getType(), " instead");
    }

    accumulatorSubShapes.push_back(tensorTy);
  }

  // Consider typical reduce-* op syntax:
  //
  //      op(I(i), V(j)):
  //       block(BI(i), BV(j)):
  //         ... some computation ...
  //         return(R(i))
  //
  // where
  //  I(i)  : i-th input of op
  //  V(j)  : j-th init-value of op
  //  BI(i) : i-th input of reducer-function
  //  BV(j) : j-th init-value of reducer-function
  //  R(i)  : i-th return-type
  //
  //  Note that: |I(i)| == |V(j)| == |BI(i)| == |BV(j)| == |R(i)|
  //
  //  Here are the type-constraints among V(j), BI(i), BV(j), and R(i).
  //    C1 : Check that BI(i) and R(i) have same shape and element-type.
  //    C2 : Check that BV(j) and R(i) have same shape and element-type.
  //    C3 : Check that V(j) and R(i) have same shape and element-type.
  //
  //  From C1, C2, and C3, we can infer that V(j), BI(i), BV(j), and R(i) all
  //  have compatible shapes and element-types.
  //  The next check, C4, adds constraints on how the type if I(i) is related
  //  to any_of(V(j), BI(i), BV(j), and R(i)), say BV(j);
  //
  //  C4.1 : Check that I(i) and BV(j) have same element-type.
  //  C4.2 : Check that shape of BV(j) is a 'sub-sequence' of
  //         'allowedDimensions'. 'allowedDimensions' is a list of dimensions
  //         which any of BI(i), BV(j), and R(i) is allowed to have.
  for (int64_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    // Check C1.
    if (failed(verifyCompatibleShape(accumulatorSubShapes[inputIdx],
                                     block.getArgument(inputIdx).getType()))) {
      return emitOptionalError(
          loc, "The type of reduction-region's parameter at index ", inputIdx,
          " is different than the corresponding result type: ",
          block.getArgument(inputIdx).getType(), " vs ",
          accumulatorSubShapes[inputIdx]);
    }

    // Check C2.
    if (failed(verifyCompatibleShape(
            accumulatorSubShapes[inputIdx],
            block.getArgument(numInputs + inputIdx).getType()))) {
      return emitOptionalError(
          loc, "The type of reduction-region's parameter at index ",
          numInputs + inputIdx,
          " is different than the corresponding result type: ",
          block.getArgument(numInputs + inputIdx).getType(), " vs ",
          accumulatorSubShapes[inputIdx]);
    }

    // Check C3.
    if (failed(verifyCompatibleShape(accumulatorSubShapes[inputIdx],
                                     initValueTypes[inputIdx]))) {
      return emitOptionalError(
          loc, "The type of reduction-region's result type at index ", inputIdx,
          " differs from the op's corresponding init-value type: ",
          accumulatorSubShapes[inputIdx], " vs ", initValueTypes[inputIdx]);
    }

    // Check C4.2.
    Type blockArgType = block.getArgument(numInputs + inputIdx).getType();
    auto blockArgTensorTy = blockArgType.cast<TensorType>();

    auto argShape = blockArgTensorTy.getShape();
    if (argShape.size() > allowedDimensions.size()) {
      return emitOptionalError(
          loc, "The rank of reduction-region's argument at index ",
          numInputs + inputIdx,
          " is expected to be <= ", allowedDimensions.size(), ", got ",
          argShape.size());
    }

    int64_t argShapeIdx = 0;
    for (int64_t outputShapeIdx = 0;
         outputShapeIdx < static_cast<int64_t>(allowedDimensions.size()) &&
         argShapeIdx < static_cast<int64_t>(argShape.size());
         outputShapeIdx++) {
      if (allowedDimensions[outputShapeIdx] == argShape[argShapeIdx]) {
        argShapeIdx++;
      }
    }

    if (argShapeIdx != static_cast<int64_t>(argShape.size())) {
      return emitOptionalError(
          loc, "The shape of reduction-region's argument at index ",
          numInputs + inputIdx,
          " is not compatible with that of reduce-op's input-parameter "
          "at index ",
          inputIdx);
    }
  }

  return success();
}

// Ported from:
// https://github.com/openxla/stablehlo/blob/311f14ce78c7fe35d304ee91007b58c335cf821e/stablehlo/dialect/TypeInference.cpp#L3603
LogicalResult ReduceOp::verify() {
  SmallVector<TensorType> inputArgTypes{llvm::map_range(
      getInputs().getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};
  SmallVector<TensorType> initValueTypes{llvm::map_range(
      getInitValues().getTypes(),
      [](Type t) -> TensorType { return t.cast<TensorType>(); })};

  // P1. & P2.
  SmallVector<int64_t> newDimensions;
  Attribute encoding;
  if (failed(verifyReduceOpInputsAndInferShape(getLoc(), inputArgTypes,
                                               initValueTypes, getDimensions(),
                                               newDimensions, encoding))) {
    return failure();
  }

  // P3.
  uint64_t numInputs = getInputs().size();

  Block& block = getBody().front();
  if (failed(verifyReducerShape(getLoc(), block, inputArgTypes, initValueTypes,
                                numInputs, newDimensions))) {
    return failure();
  }
  return success();
}

OpFoldResult ReshapeOp::fold(FoldAdaptor) {
  auto operand_shape = getOperand().getType().cast<TensorType>().getShape();
  auto result_shape = getResult().getType().cast<TensorType>().getShape();
  if (operand_shape == result_shape) {
    return getOperand();
  }
  return {};
}

OpFoldResult TransposeOp::fold(FoldAdaptor) {
  for (const auto& it : llvm::enumerate(getPermutation())) {
    if (static_cast<int64_t>(it.index()) != it.value()) {
      return {};
    }
  }
  return getOperand();
}

LogicalResult TransposeOp::verify() {
  // Constraints
  // (C1) operand and result have the same element type.
  auto inputType = getOperand().getType().cast<RankedTensorType>();
  auto retType = getResult().getType().cast<RankedTensorType>();

  TypeTools tools(getContext());
  if (tools.getExpressedType(inputType.getElementType()) !=
      tools.getExpressedType(retType.getElementType())) {
    return emitOpError("operand and result type mismatch");
  }

  // (C2) permutation is a permutation of [0, 1, ..., R-1] where R is the rank
  // of operand.
  auto max_rank = inputType.getRank();
  auto permutation = getPermutation();
  for (auto p : permutation) {
    if (p < 0 || p > max_rank - 1) {
      return emitOpError(llvm::formatv("permutation {0} out of range [0, {1}]",
                                       p, max_rank - 1));
    }
  }

  // There seems a spec inconsistency here between xla and stablehlo...we
  // actually follows xla semantic here
  // (C3) For all dimensions i in operand,
  // dim(operand, i) = dim(result,j) where j = permutation[i].
  auto input_shape = inputType.getShape();
  auto ret_shape = retType.getShape();
  for (int64_t d = 0; d < max_rank; ++d) {
    if (input_shape[permutation[d]] != ret_shape[d]) {
      return emitOpError(
          fmt::format("shape mismatch input shape = {}, result shape = {}, "
                      "permutation = {}",
                      fmt::join(input_shape, "x"), fmt::join(ret_shape, "x"),
                      fmt::join(permutation, "x")));
    }
  }

  return success();
}

LogicalResult ConcatenateOp::verify() {
  RankedTensorType firstRankedType;
  int firstRankedIndex;
  int numOperands = getNumOperands();
  auto concatDimension = static_cast<int64_t>(getDimension());
  if (concatDimension < 0) {
    return emitOpError(
        llvm::formatv("dimension {0} is negative", concatDimension));
  }
  for (int i = 0; i < numOperands; i++) {
    auto secondType = getOperand(i).getType().dyn_cast<ShapedType>();
    if (!secondType.hasRank()) {
      continue;
    }

    if (!firstRankedType) {
      firstRankedType = secondType.cast<RankedTensorType>();
      firstRankedIndex = i;
      if (firstRankedType.getRank() == 0) {
        return emitOpError(
            llvm::formatv("rank-0 values cannot be concatenated"));
      }
      if (concatDimension >= firstRankedType.getRank()) {
        return emitOpError(
            llvm::formatv("dimension {0} is out-of-bounds for input rank {1}",
                          concatDimension, firstRankedType.getRank()));
      }
      continue;
    }

    if (firstRankedType.getRank() != secondType.getRank()) {
      return emitOpError(llvm::formatv(
          "operands ({0}) and ({1}) do not match rank", firstRankedIndex, i));
    }

    auto firstShape = firstRankedType.getShape();
    auto secondShape = secondType.getShape();
    for (int d = 0; d < firstRankedType.getRank(); ++d) {
      if (!ShapedType::isDynamic(firstShape[d]) &&
          !ShapedType::isDynamic(secondShape[d]) &&
          firstShape[d] != secondShape[d] && d != concatDimension) {
        return emitOpError(llvm::formatv(
            "shapes of operand ({0}) and ({1}) do not match at non-concat "
            "index: ({2}) != ({3}) at non-concat index {4}",
            firstRankedIndex, i,
            llvm::make_range(firstShape.begin(), firstShape.end()),
            llvm::make_range(secondShape.begin(), secondShape.end()), d));
      }
    }
  }
  return success();
}

LogicalResult BroadcastOp::verify() {
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();

  auto operandRank = operandType.getRank();

  if (getBroadcastDimensions().empty()) {
    if (operandRank == 0) {
      return success();
    }
    return emitOpError(
        llvm::formatv("broadcast_dimensions is absent, but required because "
                      "operand has non-zero rank ({0})",
                      operandRank));
  }

  auto dimensionsSize = getBroadcastDimensions().size();
  if (static_cast<int64_t>(dimensionsSize) != operandRank) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        dimensionsSize, operandRank));
  }

  auto dimensions = getBroadcastDimensions();
  if (hasDuplicates(dimensions)) {
    return emitOpError("broadcast_dimensions should not have duplicates");
  }

  auto resultType = getResult().getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();

  for (size_t i = 0; i != dimensionsSize; ++i) {
    auto dimIndex = dimensions[i];
    if ((dimIndex >= resultRank) || (dimIndex < 0)) {
      return emitOpError(
          llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    if (!operandType.isDynamicDim(i)) {
      auto dimSize = operandType.getDimSize(i);
      auto resultDimSize = resultType.getDimSize(dimIndex);
      if (dimSize != 1 && dimSize != resultDimSize) {
        return emitOpError(
            llvm::formatv("size of operand dimension {0} ({1}) is not equal to "
                          "1 or size of result dimension {2} ({3})",
                          i, dimSize, dimIndex, resultDimSize));
      }
    }
  }

  return success();
}

LogicalResult IotaOp::verify() {
  auto shape = getType().cast<ShapedType>();
  if (!shape.hasRank()) {
    return success();
  }

  if (shape.getRank() == 0) {
    return emitOpError() << "does not support scalars.";
  }

  auto iotaDimension = static_cast<int64_t>(this->getIotaDimension());
  if (iotaDimension >= shape.getRank() || iotaDimension < 0) {
    return emitOpError()
           << "iota dimension cannot go beyond the output rank or be negative.";
  }
  return success();
}

LogicalResult SliceOp::verify() {
  auto rankedTy = getOperand().getType();

  // slice_c2
  int64_t rank = rankedTy.getRank();
  if (static_cast<int64_t>(getStartIndices().size()) != rank) {
    return emitOpError(
        llvm::formatv("the number of elements in start_indices ({0}) does not "
                      "match the rank of the operand ({1})",
                      getStartIndices().size(), rank));
  }

  auto start = getStartIndices();
  auto limit = getLimitIndices();
  auto strideVals = getStrides();

  for (int64_t i = 0, e = rank; i != e; i++) {
    // slice_c3
    if (start[i] < 0) {
      return emitOpError(llvm::formatv(
          "negative start index {0} in dimension {1}", start[i], i));
    }

    int64_t operandSize = rankedTy.getDimSize(i);
    // slice_c3
    if (limit[i] > operandSize) {
      return emitOpError(llvm::formatv(
          "limit index {0} is larger than dimension size {1} in dimension {2}",
          limit[i], operandSize, i));
    }

    // slice_c3
    if (start[i] > limit[i]) {
      return emitOpError(llvm::formatv(
          "start index {0}  is larger than limit index {1} in dimension {2}",
          start[i], limit[i], i));
    }

    // slice_c4
    if (strideVals[i] <= 0) {
      return emitOpError(
          llvm::formatv("stride must be positive but got {0} in dimension {1}",
                        strideVals[i], i));
    }
  }

  return success();
}

void CustomCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // CustomCall has "all possible effects" unless the has_side_effect is present
  // and set to false.
  auto hasSideEffect = (*this)->getAttrOfType<BoolAttr>("has_side_effect");
  if (hasSideEffect && !hasSideEffect.getValue()) {
    return;
  }
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Free::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get());
}

}  // namespace mlir::spu::pphlo

#define GET_OP_CLASSES
#include "libspu/dialect/pphlo/ops.cc.inc"
