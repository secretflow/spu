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

#include "libspu/dialect/pphlo_ops.h"

#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "stablehlo/dialect/TypeInference.h"

#include "libspu/dialect/pphlo_attrs.h"
#include "libspu/dialect/pphlo_ops.h.inc"

namespace mlir::pphlo {

#include "libspu/dialect/pphlo_patterns.cc.inc"

namespace {

Type convertPtTypeToPPhloType(Type pt_type) {
  return pphlo::PublicType::get(pt_type.getContext(), pt_type);
}

// Checks if the vector `nums` has duplicates.
bool hasDuplicates(const ArrayRef<int64_t> nums) {
  llvm::SmallDenseSet<int64_t> set(nums.begin(), nums.end());
  return set.size() != nums.size();
}

}  // namespace

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder& builder, OperationState& result,
                       Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<DenseElementsAttr>()) {
    auto valueType = elemAttr.getType().dyn_cast<RankedTensorType>();
    type = RankedTensorType::get(
        valueType.getShape(),
        convertPtTypeToPPhloType(valueType.getElementType()));
  }

  assert(type && "unsupported attribute type for building pphlo.constant");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
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
        b.getI64TensorAttr(target_order));
  }

  static Value ReshapeIfMorethan3D(OpBuilder& b, Location loc, Value src,
                                   size_t dims_border0, size_t dims_border1) {
    auto type = src.getType().cast<RankedTensorType>();
    if (type.getRank() <= 3) {
      return src;
    }
    auto shape = type.getShape();
    SmallVector<int64_t, 4> result_shape = {
        std::accumulate(shape.begin(), shape.begin() + dims_border0, 1,
                        std::multiplies<>()),
        std::accumulate(shape.begin() + dims_border0,
                        shape.begin() + dims_border1, 1, std::multiplies<>()),
        std::accumulate(shape.begin() + dims_border1, shape.end(), 1,
                        std::multiplies<>())};
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

    lhs = ReshapeIfMorethan3D(rewriter, op.getLoc(), lhs,
                              rhsBatchingDims.size(), lhsContractionBase);
    rhs = ReshapeIfMorethan3D(rewriter, op.getLoc(), rhs,
                              rhsBatchingDims.size(), numRhsContractionDims);

    if (lhs == op.getLhs() && rhs == op.getRhs()) {
      return failure();
    }

    auto dimensionNumbers = pphlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/0,
        /*rhsBatchingDimensions=*/0,
        /*lhsContractingDimensions=*/2, /*rhsContractingDimensions=*/1);
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

std::vector<int64_t> InversePermutation(
    absl::Span<const int64_t> input_permutation) {
  std::vector<int64_t> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation.at(input_permutation.at(i)) = i;
  }
  return output_permutation;
}

mlir::DenseIntElementsAttr ConvertToDenseIntElementAttr(
    OpBuilder* builder, llvm::ArrayRef<int64_t> value) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(value.size(), builder->getIntegerType(64)), value);
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

    std::vector<int64_t> new_input_dim_order(num_dims);
    std::vector<int64_t> new_input_dims(num_dims);
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
          op->getLoc(), new_input_type, input,
          ConvertToDenseIntElementAttr(&rewriter, new_input_dim_order));
    }

    auto kernel = op.getRhs();

    auto kernel_type = kernel.getType().dyn_cast<RankedTensorType>();
    auto kernel_shape = kernel_type.getShape();

    std::vector<int64_t> new_kernel_dim_order(num_dims);
    std::vector<int64_t> new_kernel_dims(num_dims);

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
      new_kernel = rewriter.create<TransposeOp>(
          op->getLoc(), new_kernel_type, kernel,
          ConvertToDenseIntElementAttr(&rewriter, new_kernel_dim_order));
    }

    if (input == new_input && kernel == new_kernel) {
      return failure();
    }

    std::vector<int64_t> new_output_dim_order(num_dims);
    std::vector<int64_t> new_conv_dims(num_dims);
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

    std::vector<int64_t> input_sd(num_spatial_dims);
    std::vector<int64_t> kernel_sd(num_spatial_dims);
    std::vector<int64_t> output_sd(num_spatial_dims);

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
        op.getWindowStrides().value_or(nullptr), new_dnums,
        op.getFeatureGroupCount(), op.getBatchGroupCount());

    // Reshape the output back to the shape of the original convolution.
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op->getResultTypes()[0], new_conv,
        ConvertToDenseIntElementAttr(&rewriter,
                                     InversePermutation(new_output_dim_order)));

    return success();
  }
};

}  // namespace

OpFoldResult ReverseOp::fold(FoldAdaptor) {
  auto input = getOperand();

  // No dimensions to reverse.
  auto dims = getDimensions();
  if (dims.getNumElements() == 0) {
    return input;
  }

  // If the dimensions to reverse are all statically 1, then the reverse is a
  // no-op.
  auto shapedType = input.getType().cast<ShapedType>();
  if (llvm::all_of(dims.getValues<int64_t>(), [&](int64_t dim) {
        return shapedType.getDimSize(dim) == 1;
      })) {
    return input;
  }
  return {};
}

LogicalResult ReverseOp::verify() {
  //(C1) operand and result have the same type.
  auto inputType = getOperand().getType().cast<RankedTensorType>();
  auto retType = getResult().getType().cast<RankedTensorType>();

  TypeTools tools;
  if (tools.getExpressedType(inputType) != tools.getExpressedType(retType)) {
    return emitOpError("operand and result type mismatch");
  }

  // dimensions: 1-dimensional tensor constant of type si64
  if (getDimensions().getType().getRank() != 1) {
    return emitOpError("dimensions must be a 1-dimensional tensor");
  }

  //(C2) All dimensions in dimensions are unique.
  auto dims = getDimensions().getValues<int64_t>();
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
    SmallVector<TensorType> initValueTypes, DenseIntElementsAttr dimensions,
    SmallVector<int64_t>& newDimensions, Attribute& encoding) {
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
  for (int64_t dimension : dimensions.getValues<int64_t>()) {
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
                                 ArrayRef<TensorType> inputArgTypes,
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
  for (const auto& it : llvm::enumerate(getPermutation().getValues<APInt>())) {
    if (it.index() != it.value()) {
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

  TypeTools tools;
  if (tools.getExpressedType(inputType) != tools.getExpressedType(retType)) {
    return emitOpError("operand and result type mismatch");
  }

  // (C2) permutation is a permutation of [0, 1, ..., R-1] where R is the rank
  // of operand.
  auto max_rank = inputType.getRank();
  auto permutation = getPermutation().getValues<int64_t>();
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

LogicalResult PadOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  PadOp::Adaptor adaptor(operands, attributes, {}, regions);
  SmallVector<Type> types;
  auto status = hlo::inferPadOp(
      location, adaptor.getOperand().getType(),
      adaptor.getPaddingValue().getType(), adaptor.getEdgePaddingLow(),
      adaptor.getEdgePaddingHigh(), adaptor.getInteriorPadding(), types);

  // Convert type to STC
  for (auto& t : types) {
    auto rt = t.dyn_cast<RankedTensorType>();
    inferredReturnShapes.emplace_back(rt.getShape(), rt.getElementType());
  }

  return status;
}

LogicalResult PadOp::inferReturnTypes(
    MLIRContext* context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, {}, regions);
  return hlo::inferPadOp(location, adaptor.getOperand().getType(),
                         adaptor.getPaddingValue().getType(),
                         adaptor.getEdgePaddingLow(),
                         adaptor.getEdgePaddingHigh(),
                         adaptor.getInteriorPadding(), inferredReturnTypes);
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

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, {}, regions);
  return hlo::inferConcatenateOp(location, adaptor.getVal().getTypes(),
                                 adaptor.getDimension(), inferredReturnTypes);
}

LogicalResult BroadcastOp::verify() {
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();

  auto operandRank = operandType.getRank();

  if (!getBroadcastDimensions()) {
    if (operandRank == 0) {
      return success();
    }
    return emitOpError(
        llvm::formatv("broadcast_dimensions is absent, but required because "
                      "operand has non-zero rank ({0})",
                      operandRank));
  }

  auto dimensionsType = getBroadcastDimensions().getType();
  auto dimensionsRank = dimensionsType.getRank();
  if (dimensionsRank != 1) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions has rank {0} instead of rank 1", dimensionsRank));
  }

  auto dimensionsSize = dimensionsType.getNumElements();
  if (dimensionsSize != operandRank) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        dimensionsSize, operandRank));
  }

  auto dimensions =
      llvm::to_vector(getBroadcastDimensions().getValues<int64_t>());
  if (hasDuplicates(dimensions)) {
    return emitOpError("broadcast_dimensions should not have duplicates");
  }

  auto resultType = getResult().getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();

  for (int i = 0; i != dimensionsSize; ++i) {
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
  // slice_i2
  ShapedType attrTy = getStartIndices().getType();
  if (attrTy.getRank() != 1) {
    return emitOpError(
        llvm::formatv("start_indices has rank {0} instead of required rank 1",
                      attrTy.getRank()));
  }

  // slice_c2
  int64_t rank = rankedTy.getRank();
  if (attrTy.getNumElements() != rank) {
    return emitOpError(
        llvm::formatv("the number of elements in start_indices ({0}) does not "
                      "match the rank of the operand ({1})",
                      attrTy.getNumElements(), rank));
  }

  auto start = getStartIndices().getValues<int64_t>();
  auto limit = getLimitIndices().getValues<int64_t>();
  auto strideVals = getStrides().getValues<int64_t>();

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

LogicalResult inferDynamicSliceOp(std::optional<Location> location,
                                  Type operandType, TypeRange startIndicesTypes,
                                  DenseIntElementsAttr sliceSizes,
                                  SmallVectorImpl<Type>& inferredReturnTypes) {
  // dynamic_slice_i3
  if (sliceSizes.getType().getRank() != 1) {
    return emitOptionalError(location,
                             "slice_sizes should be rank 1, but got rank ",
                             sliceSizes.getType().getRank(), ".");
  }
  // dynamic_slice_c2
  int numSliceSizes = sliceSizes.getNumElements();
  int numStartIndices = startIndicesTypes.size();
  if (numStartIndices != numSliceSizes) {
    return emitOptionalError(location, "has mismatched number of slice sizes (",
                             numSliceSizes, ") and number of start indices (",
                             numStartIndices, ")");
  }
  auto rankedOperandType = operandType.dyn_cast<RankedTensorType>();
  // dynamic_slice_c2
  if (rankedOperandType.getRank() != numStartIndices) {
    return emitOptionalError(
        location, "has mismatched number of start indices (", numStartIndices,
        ") and the rank of operand (", rankedOperandType.getRank(), ")");
  }

  // dynamic_slice_c4
  for (int i = 0; i < numSliceSizes; ++i) {
    int64_t sliceSize = sliceSizes.getValues<int64_t>()[i];
    if (sliceSize < 0) {
      return emitOptionalError(
          location, "has negative size index to dynamic slice: ", sliceSize);
    }
    if (!rankedOperandType.isDynamicDim(i)) {
      int64_t dimSize = rankedOperandType.getDimSize(i);
      if (sliceSize > dimSize) {
        return emitOptionalError(location, "has slice size ", sliceSize,
                                 " greater than dimension size ", dimSize,
                                 " in dimension ", i, " of operand");
      }
    }
  }

  std::vector<int64_t> slice_size(sliceSizes.getValues<int64_t>().begin(),
                                  sliceSizes.getValues<int64_t>().end());
  // dynamic_slice_c5
  inferredReturnTypes.emplace_back(
      RankedTensorType::get(slice_size, rankedOperandType.getElementType()));
  return success();
}

LogicalResult DynamicSliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, {}, regions);
  return inferDynamicSliceOp(location, adaptor.getOperand().getType(),
                             adaptor.getStartIndices().getTypes(),
                             adaptor.getSliceSizes(), inferredReturnTypes);
}

LogicalResult inferDynamicUpdateSliceOp(
    std::optional<Location> location, Value operand, Value update,
    ValueRange startIndices, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operandType = operand.getType().cast<ShapedType>();
  auto updateType = update.getType().cast<ShapedType>();

  // dynamic_update_slice_c3
  if (updateType.hasRank() && operandType.hasRank() &&
      updateType.getRank() != operandType.getRank()) {
    return emitOptionalError(
        location,
        "update rank does not match operand rank: ", updateType.getRank(),
        " vs ", operandType.getRank(), ".");
  }

  // dynamic_update_slice_c4
  if (operandType.hasRank() &&
      static_cast<int64_t>(startIndices.size()) != operandType.getRank()) {
    return emitOptionalError(
        location, "expects number of start_indices to match operand rank: ",
        startIndices.size(), " vs ", operandType.getRank(), ".");
  }

  // dynamic_update_slice_c6
  if (operandType.hasRank() && updateType.hasRank()) {
    for (auto [index, dims] : llvm::enumerate(
             llvm::zip(operandType.getShape(), updateType.getShape()))) {
      auto [operandDim, updateDim] = dims;
      if (updateDim < 0 || updateDim > operandDim) {
        return emitOptionalError(location, "expects size at dimension ", index,
                                 " of update to be in range [0, ", operandDim,
                                 "]. Got: ", updateDim, ".");
      }
    }
  }

  // dynamic_update_slice_c1
  inferredReturnTypes.emplace_back(RankedTensorType::get(
      operandType.getShape(), operandType.getElementType()));
  return success();
}

LogicalResult DynamicUpdateSliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  DynamicUpdateSliceOp::Adaptor adaptor(operands, attributes, {}, regions);

  return inferDynamicUpdateSliceOp(
      location, adaptor.getOperand(), adaptor.getUpdate(),
      adaptor.getStartIndices(), inferredReturnTypes);
}

template <typename T>
static void printField(AsmPrinter& printer, StringRef name, T field,
                       StringRef& separator) {
  if (field != 0) {
    printer << separator << name << " = " << field;
    separator = ", ";
  }
}
template <typename T>
static void printField(AsmPrinter& printer, StringRef name, ArrayRef<T> field,
                       StringRef& separator) {
  if (!field.empty()) {
    printer << separator << name << " = [";
    llvm::interleaveComma(field, printer);
    printer << "]";
    separator = ", ";
  }
}

template <typename... Ts>
static void printStruct(AsmPrinter& printer, StringRef name,
                        Ts... print_fields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(mhlo-team): this can be simplified when TF moves to C++17.
  using Unused = int[];
  (void)Unused{0, (printField(printer, std::get<0>(print_fields),
                              std::get<1>(print_fields), separator),
                   0)...};
  printer << ">";
}

/// Parse a custom attribute that resembles a struct of the form
/// <
///   foo = something_parsed_by_custom_parser,
///   bar = something_parsed_by_different_custom_parser,
///   baz something_parsed_by_another_custom_parser
/// >
/// The optional argument `parse_equal` array can be used to denote if
/// '=' follows the keyword (see baz in the example above) for a field. If
/// not provided, all fields must be followed by a '='.
static ParseResult parseStruct(
    AsmParser& parser, ArrayRef<StringRef> keywords,
    ArrayRef<llvm::function_ref<ParseResult()>> parse_funcs,
    ArrayRef<bool> parse_equal = {}) {
  assert(keywords.size() == parse_funcs.size());
  assert(parse_equal.empty() || parse_equal.size() == keywords.size());
  SmallVector<bool> seen(keywords.size(), false);
  while (failed(parser.parseOptionalGreater())) {
    bool foundOne = false;
    for (const auto& it : llvm::enumerate(keywords)) {
      size_t index = it.index();
      StringRef keyword = it.value();
      if (succeeded(parser.parseOptionalKeyword(keyword))) {
        if (seen[index]) {
          return parser.emitError(parser.getCurrentLocation())
                 << "duplicated `" << keyword << "` entry";
        }
        if (parse_equal.empty() || parse_equal[index]) {
          if (failed(parser.parseEqual())) {
            return failure();
          }
        }
        if (failed(parse_funcs[index]())) {
          return failure();
        }
        if (failed(parser.parseOptionalComma())) {
          return parser.parseGreater();
        }
        seen[index] = true;
        foundOne = true;
      }
    }
    if (!foundOne) {
      auto parseError = parser.emitError(parser.getCurrentLocation())
                        << "expected one of: ";
      llvm::interleaveComma(keywords, parseError, [&](StringRef kw) {
        parseError << '`' << kw << '`';
      });
      return parseError;
    }
  }
  return success();
}

static ParseResult parseDims(AsmParser& parser, SmallVector<int64_t>& dims) {
  dims.clear();
  if (parser.parseLSquare()) {
    return failure();
  }
  while (failed(parser.parseOptionalRSquare())) {
    dims.emplace_back();
    if (parser.parseInteger(dims.back())) {
      return failure();
    }
    (void)parser.parseOptionalComma();
  }
  return success();
}

void GatherDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "gather", std::make_pair("offset_dims", getOffsetDims()),
              std::make_pair("collapsed_slice_dims", getCollapsedSliceDims()),
              std::make_pair("start_index_map", getStartIndexMap()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute GatherDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }

  SmallVector<int64_t> offset_dims;
  SmallVector<int64_t> collapsed_slice_dims;
  SmallVector<int64_t> start_index_map;
  int64_t index_vector_dim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offset_dims); },
           [&]() { return parseDims(parser, collapsed_slice_dims); },
           [&]() { return parseDims(parser, start_index_map); },
           [&]() { return parser.parseInteger(index_vector_dim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(parser.getContext(), offset_dims,
                                         collapsed_slice_dims, start_index_map,
                                         index_vector_dim);
}

// Custom printer and parser for DotDimensionNumbersAttr.
void DotDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(
      printer, "dot",
      std::make_pair("lhs_batching_dimensions", getLhsBatchingDimensions()),
      std::make_pair("rhs_batching_dimensions", getRhsBatchingDimensions()),
      std::make_pair("lhs_contracting_dimensions",
                     getLhsContractingDimensions()),
      std::make_pair("rhs_contracting_dimensions",
                     getRhsContractingDimensions()));
}

Attribute DotDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }

  SmallVector<int64_t> lhsBatchingDimensions;
  SmallVector<int64_t> rhsBatchingDimensions;
  SmallVector<int64_t> lhsContractingDimensions;
  SmallVector<int64_t> rhsContractingDimensions;

  if (failed(parseStruct(
          parser,
          {"lhs_batching_dimensions", "rhs_batching_dimensions",
           "lhs_contracting_dimensions", "rhs_contracting_dimensions"},
          {[&]() { return parseDims(parser, lhsBatchingDimensions); },
           [&]() { return parseDims(parser, rhsBatchingDimensions); },
           [&]() { return parseDims(parser, lhsContractingDimensions); },
           [&]() { return parseDims(parser, rhsContractingDimensions); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return {};
  }
  return DotDimensionNumbersAttr::get(
      parser.getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

namespace {
enum NonSpatialDim : int64_t {
  IOBatch = -1,    // Input or output batch dimension
  IOFeature = -2,  // Input or output feature dimension
  KIFeature = -3,  // Kernel input feature dimension
  KOFeature = -4,  // Kernel output feature dimensions.
};

struct DenseMapInfoNonSpatialDim {
  static inline NonSpatialDim getEmptyKey() {
    return static_cast<NonSpatialDim>(DenseMapInfo<int64_t>::getEmptyKey());
  }

  static inline NonSpatialDim getTombstoneKey() {
    return static_cast<NonSpatialDim>(DenseMapInfo<int64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const NonSpatialDim& key) {
    return DenseMapInfo<int64_t>::getHashValue(key);
  }

  static bool isEqual(const NonSpatialDim& lhs, const NonSpatialDim& rhs) {
    return lhs == rhs;
  }
};

char NonSpatialDimToString(NonSpatialDim dim) {
  switch (dim) {
    case IOBatch:
      return 'b';
    case IOFeature:
      return 'f';
    case KIFeature:
      return 'i';
    case KOFeature:
      return 'o';
  }
  llvm_unreachable("Unexpected spatial dim");
}
}  // namespace

// Custom printer and parser for struct attributes.
void printConvolutionDimensions(AsmPrinter& p, ConvDimensionNumbersAttr dnums) {
  constexpr int64_t kUnknownDim = std::numeric_limits<int64_t>::min();
  auto print_dim =
      [&](ArrayRef<int64_t> spatial_dims,
          ArrayRef<std::pair<int64_t, NonSpatialDim>> non_spatial_dims) {
        int64_t num_dims = 0;
        if (!spatial_dims.empty()) {
          num_dims =
              *std::max_element(spatial_dims.begin(), spatial_dims.end()) + 1;
        }
        for (const auto& dim : non_spatial_dims) {
          num_dims = std::max(num_dims, dim.first + 1);
        }

        llvm::SmallVector<int64_t> dims(num_dims, kUnknownDim);
        // Fill each element of dims with a (< 0) NonSpatialDim enum or a (>=0)
        // spatial dimension index.
        for (const std::pair<int64_t, NonSpatialDim>& non_spatial_dim :
             non_spatial_dims) {
          dims[non_spatial_dim.first] = non_spatial_dim.second;
        }
        for (const auto& spatial_dim : llvm::enumerate(spatial_dims)) {
          dims[spatial_dim.value()] = static_cast<int64_t>(spatial_dim.index());
        }

        // Each dimension numbers will be printed as a comma separated list
        // surrounded by square brackets, e.g., [b, 0, 1, 2, f]
        p << '[';
        llvm::interleaveComma(dims, p, [&](int64_t dim) {
          if (dim == kUnknownDim) {
            p << "?";
          } else if (dim >= 0) {
            p << dim;
          } else {
            p << NonSpatialDimToString(static_cast<NonSpatialDim>(dim));
          }
        });
        p << ']';
      };

  print_dim(dnums.getInputSpatialDimensions(),
            {{dnums.getInputBatchDimension(), IOBatch},
             {dnums.getInputFeatureDimension(), IOFeature}});
  p << "x";
  print_dim(dnums.getKernelSpatialDimensions(),
            {{dnums.getKernelInputFeatureDimension(), KIFeature},
             {dnums.getKernelOutputFeatureDimension(), KOFeature}});
  p << "->";
  print_dim(dnums.getOutputSpatialDimensions(),
            {{dnums.getOutputBatchDimension(), IOBatch},
             {dnums.getOutputFeatureDimension(), IOFeature}});
}

void printConvolutionDimensions(AsmPrinter& p, Operation* /*unused*/,
                                ConvDimensionNumbersAttr dnums) {
  printConvolutionDimensions(p, dnums);
}

void ConvDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printer << "<";
  printConvolutionDimensions(printer, *this);
  printer << ">";
}

ParseResult parseConvolutionDimensions(AsmParser& parser,
                                       ConvDimensionNumbersAttr& dnums) {
  // Parsing a single set of dim numbers gives the spatial dimensions as a
  // single ArrayRef<int64_t> and a list of non-spatial dimensions as
  // IntegerAttrs (indexed by the NonSpatialDim enum).
  using ParseDimResultT =
      std::pair<llvm::SmallVector<int64_t>,
                llvm::SmallDenseMap<NonSpatialDim, int64_t, 4,
                                    DenseMapInfoNonSpatialDim>>;

  // Note that the allowed_non_spatial_dims is a set (as opposed to unordered
  // set) because its used to print a list of allowed non spatial dims in the
  // error messages, so making it a set keeps the error messages deterministic.
  auto parse_dims =
      [&](std::set<NonSpatialDim, std::greater<>> allowed_non_spatial_dims,
          ParseDimResultT& parsed_dims) -> ParseResult {
    auto& spatial_dims = std::get<0>(parsed_dims);
    auto& non_spatial_dims = std::get<1>(parsed_dims);
    spatial_dims.clear();
    non_spatial_dims.clear();

    // Parse the starting [
    if (parser.parseLSquare()) {
      return failure();
    }

    llvm::SmallDenseMap<int64_t, int64_t> spatial_dims_map;
    constexpr int64_t kInvalidDimension = -1;
    // Keep track of the maximum spatial dimension parsed as we expect to see
    // all the dimensions from 0 to maximum dimension parsed.
    int64_t max_parsed_spatial_dim = kInvalidDimension;

    int64_t index = 0;
    do {
      int64_t spatial_dim{};
      auto dim_location = parser.getCurrentLocation();
      OptionalParseResult parseResult =
          parser.parseOptionalInteger(spatial_dim);
      if (parseResult.has_value()) {
        if (parseResult.value().failed()) {
          return failure();
        }
        // We were successful in parsing an integer. Check if it is a valid
        // dimension (non-negative and no duplicate) and add its index to the
        // spatial dims map.
        if (spatial_dim < 0) {
          return parser.emitError(dim_location)
                 << "Unexpected dimension " << spatial_dim;
        }
        if (!spatial_dims_map
                 .insert(std::pair<int64_t, int64_t>(spatial_dim, index))
                 .second) {
          return parser.emitError(dim_location)
                 << "Duplicate entries for spatial dimension " << spatial_dim;
        }
        max_parsed_spatial_dim = std::max(spatial_dim, max_parsed_spatial_dim);
      } else if (!parser.parseOptionalQuestion()) {
        // Do nothing other than increment `index` at the bottom of the loop;
        // '?' means "unknown dimension", and it's not represented in the
        // return value of this function.
      } else {
        // We did not parse an integer or question mark. We expect a keyword
        // token.
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) {
          return failure();
        }
        if (keyword.size() != 1 || allowed_non_spatial_dims.empty()) {
          return parser.emitError(dim_location, "Unexpected keyword ")
                 << keyword;
        }
        // Check if the keyword matches one of the allowed non-spatial dims.
        // If so, add it to the non_spatial dims and remove it from the
        // allowed set so that it won't be allowed again.
        bool is_allowed = false;
        for (NonSpatialDim allowed : allowed_non_spatial_dims) {
          if (keyword[0] == NonSpatialDimToString(allowed)) {
            non_spatial_dims.insert({allowed, index});
            allowed_non_spatial_dims.erase(allowed);
            is_allowed = true;
            break;
          }
        }

        if (!is_allowed) {
          mlir::InFlightDiagnostic diag =
              parser.emitError(dim_location, "Unexpected dimension ");
          diag << keyword << ", expecting ";
          llvm::interleaveComma(
              allowed_non_spatial_dims, diag,
              [&](NonSpatialDim dim) { diag << NonSpatialDimToString(dim); });
          return diag;
        }
      }
      index++;
    } while (parser.parseOptionalComma().succeeded());

    // Make sure all expected non-spatial dimensions are parsed.
    if (!allowed_non_spatial_dims.empty()) {
      mlir::InFlightDiagnostic diag =
          parser.emitError(parser.getCurrentLocation(), "Expected dimensions ");
      llvm::interleaveComma(
          allowed_non_spatial_dims, diag,
          [&](NonSpatialDim dim) { diag << NonSpatialDimToString(dim); });
      diag << " not specified";
      return diag;
    }

    // parse ending ]
    if (parser.parseRSquare()) {
      return failure();
    }

    // Number of expected spatial dimensions is one more than the maximum parsed
    // spatial dimension. For example, if we parse [0, 3, 2, b, i, 1], then the
    // maximum parsed spatial dimension is 3 and the number of expected spatial
    // dimensions is 4.
    int64_t num_spatial_dimensions = max_parsed_spatial_dim + 1;
    spatial_dims.resize(num_spatial_dimensions);
    // Store spatial dimensions in a vector which maps spatial dim (vector
    // index) -> index in the tensor dimensions. For example, for parsed
    // dimension numbers [0, 3, 2, b, i, 1] the spatial dimension vector would
    // be [0, 5, 2, 1].
    //
    // Get all the unspecified spatial dimensions to throw a more descriptive
    // error later.
    llvm::SmallVector<int64_t> unspecified_spatial_dims;
    constexpr int kPrintUnspecifiedDimsMax = 10;
    for (int dim = 0; dim < num_spatial_dimensions; ++dim) {
      auto it = spatial_dims_map.find(dim);
      if (it == spatial_dims_map.end()) {
        // Have an upper bound on the number of unspecified dimensions to print
        // in the error message.
        if (unspecified_spatial_dims.size() < kPrintUnspecifiedDimsMax) {
          unspecified_spatial_dims.push_back(dim);
        }
        continue;
      }
      spatial_dims[dim] = it->second;
    }

    // Verify that we got all spatial dimensions between 0 and maximum parsed
    // spatial dimension.
    if (!unspecified_spatial_dims.empty()) {
      mlir::InFlightDiagnostic diag = parser.emitError(
          parser.getCurrentLocation(), "Expected spatial dimensions ");
      llvm::interleaveComma(unspecified_spatial_dims, diag);
      diag << " not specified";
      return diag;
    }

    return success();
  };

  ParseDimResultT parsed_dims;
  if (parse_dims({IOBatch, IOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> input_spatial_dimensions = parsed_dims.first;
  int64_t input_batch_dimension = parsed_dims.second[IOBatch];
  int64_t input_feature_dimension = parsed_dims.second[IOFeature];
  if (parser.parseKeyword("x")) {
    return failure();
  }
  if (parse_dims({KIFeature, KOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> kernel_spatial_dimensions = parsed_dims.first;
  int64_t kernel_input_feature_dimension = parsed_dims.second[KIFeature];
  int64_t kernel_output_feature_dimension = parsed_dims.second[KOFeature];
  if (parser.parseArrow()) {
    return failure();
  }
  if (parse_dims({IOBatch, IOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> output_spatial_dimensions = parsed_dims.first;
  int64_t output_batch_dimension = parsed_dims.second[IOBatch];
  int64_t output_feature_dimension = parsed_dims.second[IOFeature];
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), input_batch_dimension,
      input_feature_dimension, input_spatial_dimensions,
      kernel_input_feature_dimension, kernel_output_feature_dimension,
      kernel_spatial_dimensions, output_batch_dimension,
      output_feature_dimension, output_spatial_dimensions);

  return success();
}

Attribute ConvDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  ConvDimensionNumbersAttr dnums;

  if (failed(parseConvolutionDimensions(parser, dnums))) {
    return {};
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return dnums;
}

namespace {

// Custom formatting for convolution window attributes.
void printWindowAttribute(OpAsmPrinter& p, DenseElementsAttr attribute) {
  if (attribute.getElementType().isInteger(/*width=*/1)) {
    // boolean attribute.
    llvm::interleaveComma(attribute.getValues<bool>(), p,
                          [&](bool b) { p << (b ? 1 : 0); });
    return;
  }
  if (attribute.getType().getRank() == 2) {
    // Padding is Nx2 attribute.
    auto it = attribute.value_begin<int64_t>();
    std::vector<std::pair<int64_t, int64_t>> values(attribute.getNumElements() /
                                                    2);
    for (auto& item : values) {
      int64_t first = *it;
      ++it;
      int64_t second = *it;
      ++it;
      item = {first, second};
    }
    llvm::interleaveComma(
        values, p, [&](const std::pair<int64_t, int64_t> pair) {
          p << '[' << pair.first << ", " << pair.second << ']';
        });
  } else {
    llvm::interleaveComma(attribute.getValues<int64_t>(), p);
  }
}

}  // namespace

void printWindowAttributes(OpAsmPrinter& p, Operation* op,
                           std::optional<DenseIntElementsAttr> window_strides) {
  using PairT = std::pair<DenseElementsAttr, StringRef>;
  std::array<PairT, 1> printed_attributes = {{
      {window_strides ? *window_strides : nullptr, "stride"},
  }};

  // Do not print attributes that do no exist.
  auto non_null_attributes = llvm::make_filter_range(
      printed_attributes,
      [](const PairT& a) { return static_cast<bool>(a.first); });

  llvm::interleaveComma(non_null_attributes, p, [&](const PairT& a) {
    p << a.second << " = [";
    printWindowAttribute(p, a.first);
    p << "]";
  });
}

ParseResult parseWindowAttributes(OpAsmParser& parser,
                                  DenseIntElementsAttr& window_strides) {
  StringRef attribute_name;

  // Helper to parse an array of the form [ e0, e1, .. ]
  auto parse_array = [&](const std::function<ParseResult(void)>& parse_element,
                         std::optional<size_t> expected_size =
                             std::nullopt) -> ParseResult {
    if (parser.parseLSquare()) {
      return failure();
    }
    size_t size = 0;
    do {
      if (parse_element()) {
        return failure();
      }
      size++;
    } while (parser.parseOptionalComma().succeeded());
    if (parser.parseRSquare()) {
      return failure();
    }
    if (expected_size && size != *expected_size) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected array with")
             << *expected_size << " elements, got " << size
             << " elements instead";
    }
    return success();
  };

  llvm::StringSet<> allowed_attribute_names{{"stride"}};

  while (parser.parseOptionalKeyword(&attribute_name).succeeded()) {
    // Verify that the attribute name is valid and erase it.
    if (!allowed_attribute_names.erase(attribute_name)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Unexpected keyword ")
             << attribute_name;
    }

    if (parser.parseEqual()) {
      return failure();
    }

    // parse the attribute value. We need to support either 1D and Nx2 array
    // of integers to parse.
    llvm::SmallVector<int64_t> values;
    auto int64_parser = [&]() {
      return parser.parseInteger(values.emplace_back(0));
    };

    // Parse 1D array of integers.
    if (parse_array(int64_parser)) {
      return failure();
    }
    auto attr = parser.getBuilder().getI64TensorAttr(values);

    if (attribute_name == "stride") {
      window_strides = attr;
    } else {
      llvm_unreachable("Unexpected attribute name");
    }

    // continue parsing if there is a comma at the end.
    if (parser.parseOptionalComma().failed()) {
      break;
    }
  }
  return success();
}

}  // namespace mlir::pphlo

#define GET_OP_CLASSES
#include "libspu/dialect/pphlo_ops.cc.inc"
