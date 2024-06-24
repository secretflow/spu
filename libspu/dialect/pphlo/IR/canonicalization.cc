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

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "libspu/dialect/pphlo/IR/ops.h"

namespace mlir::spu::pphlo {

llvm::SmallVector<int64_t> inversePermutation(
    llvm::ArrayRef<int64_t> input_permutation) {
  llvm::SmallVector<int64_t> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation[input_permutation[i]] = i;
  }
  return output_permutation;
}

bool isSameShape(llvm::ArrayRef<int64_t> lhs, llvm::ArrayRef<int64_t> rhs) {
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
    auto type = mlir::dyn_cast<RankedTensorType>(src.getType());
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
    auto lhsShapeType = mlir::dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsShapeType = mlir::dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resultType =
        mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
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
        mlir::dyn_cast<ShapedType>(lhs.getType()).getRank() - 1,
        /*rhsContractingDimensions=*/1);
    auto lhsNewType = mlir::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsNewType = mlir::dyn_cast<RankedTensorType>(rhs.getType());

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

// This is piece of code is ported from tensorflow/xla
// Ref
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/cpu/conv_canonicalization.cc
class NormalizeDimensionOrder : public OpRewritePattern<ConvolutionOp> {
 private:
  static bool needTranspose(llvm::ArrayRef<int64_t> old_shape,
                            llvm::ArrayRef<int64_t> new_shape,
                            llvm::ArrayRef<int64_t> permutation) {
    if (!isSameShape(old_shape, new_shape)) {
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
    auto input_type = mlir::dyn_cast<RankedTensorType>(input.getType());
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

    auto kernel_type = mlir::dyn_cast<RankedTensorType>(kernel.getType());
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

    auto result_type =
        mlir::dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
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
        new_dnums);

    // Reshape the output back to the shape of the original convolution.
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op->getResultTypes()[0], new_conv,
        inversePermutation(new_output_dim_order));

    return success();
  }
};

class NormalizeConv1D : public OpRewritePattern<ConvolutionOp> {
 public:
  using OpRewritePattern<ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    // Check 1D conv
    auto dnums = op.getDimensionNumbers();
    if (dnums.getInputSpatialDimensions().size() != 1) {
      return failure();
    }

    // Check in [b, 0, f]x[0, i, o] -> [b, 0, f]
    if (dnums.getInputBatchDimension() != 0 &&
        dnums.getInputFeatureDimension() != 2) {
      return failure();
    }
    if (dnums.getKernelInputFeatureDimension() != 1 &&
        dnums.getKernelOutputFeatureDimension() != 2) {
      return failure();
    }
    if (dnums.getOutputBatchDimension() != 0 &&
        dnums.getOutputFeatureDimension() != 2) {
      return failure();
    }

    auto lhs_type = mlir::dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhs_type = mlir::dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto ret_type = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());

    // reshape lhs to [b, 1, s0, f]
    auto reshaped_lhs = rewriter.create<ReshapeOp>(
        op->getLoc(),
        RankedTensorType::get({lhs_type.getShape()[0], 1,
                               lhs_type.getShape()[1], lhs_type.getShape()[2]},
                              lhs_type.getElementType()),
        op.getLhs());

    // reshape rhs to [1, s0, i, o]
    auto reshaped_rhs = rewriter.create<ReshapeOp>(
        op->getLoc(),
        RankedTensorType::get({1, rhs_type.getShape()[0],
                               rhs_type.getShape()[1], rhs_type.getShape()[2]},
                              rhs_type.getElementType()),
        op.getRhs());

    auto new_dnums = ConvDimensionNumbersAttr::get(
        op->getContext(), dnums.getInputBatchDimension(),
        dnums.getInputFeatureDimension() + 1, {1, 2},
        dnums.getKernelInputFeatureDimension() + 1,
        dnums.getKernelOutputFeatureDimension() + 1, {0, 1},
        dnums.getOutputBatchDimension(), dnums.getOutputFeatureDimension() + 1,
        {1, 2});

    llvm::SmallVector<int64_t> window_strides(2, 1);
    if (op.getWindowStrides().has_value()) {
      window_strides[1] = (*op.getWindowStrides())[0];
    }

    // create a new 2d conv
    auto new_conv = rewriter.create<ConvolutionOp>(
        op->getLoc(),
        RankedTensorType::get({ret_type.getShape()[0], 1,
                               ret_type.getShape()[1], ret_type.getShape()[2]},
                              ret_type.getElementType()),
        reshaped_lhs, reshaped_rhs,
        DenseI64ArrayAttr::get(op->getContext(), window_strides), new_dnums);

    // Reshape back
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, ret_type, new_conv);

    return success();
  }
};

class DivToReciprocal : public OpRewritePattern<DivOp> {
 public:
  using OpRewritePattern<DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::DivOp op,
                                PatternRewriter& rewriter) const override {
    TypeTools tools_(op->getContext());
    if (!tools_.isFloatType(op.getType())) {
      return failure();
    }

    auto lhs_def = op.getLhs().getDefiningOp<ConstantOp>();

    if (lhs_def == nullptr) {
      return failure();
    }

    auto const_value = lhs_def.getValue();
    if (!const_value.isSplat()) {
      return failure();
    }

    if (const_value.getSplatValue<llvm::APFloat>().convertToDouble() == 1.0F) {
      rewriter.replaceOpWithNewOp<ReciprocalOp>(op, op.getRhs());
    }

    return failure();
  }
};

class NormalizeDotShape : public OpRewritePattern<DotOp> {
 public:
  using OpRewritePattern<DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp op,
                                PatternRewriter& rewriter) const override {
    auto lhs_type = mlir::dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhs_type = mlir::dyn_cast<RankedTensorType>(op.getRhs().getType());

    // Semantics listed at https://openxla.org/xla/operation_semantics#dot
    // scalar dot scalar
    Value new_dot;
    if (lhs_type.getRank() == 1 && rhs_type.getRank() == 1) {
      // n dot n -> 1xn dot nx1 -> 1x1 -> scalar
      auto new_lhs_type = RankedTensorType::get({1, lhs_type.getNumElements()},
                                                lhs_type.getElementType());
      auto new_rhs_type = RankedTensorType::get({rhs_type.getNumElements(), 1},
                                                rhs_type.getElementType());
      auto lhs =
          rewriter.create<ReshapeOp>(op->getLoc(), new_lhs_type, op.getLhs());
      auto rhs =
          rewriter.create<ReshapeOp>(op->getLoc(), new_rhs_type, op.getRhs());

      new_dot = rewriter.create<DotOp>(op->getLoc(), lhs, rhs);
    } else if (lhs_type.getRank() == 2 && rhs_type.getRank() == 1) {
      // matrix dot vector
      // mxk dot k -> mxk dot kx1 -> mx1 -> m
      auto new_rhs_type = RankedTensorType::get({rhs_type.getNumElements(), 1},
                                                rhs_type.getElementType());
      auto rhs =
          rewriter.create<ReshapeOp>(op->getLoc(), new_rhs_type, op.getRhs());

      new_dot = rewriter.create<DotOp>(op->getLoc(), op.getLhs(), rhs);
    } else if (lhs_type.getRank() == 1 && rhs_type.getRank() == 2) {
      // vector dot matrix
      // k dot k*n -> 1xk * k*n -> 1xn -> n
      auto new_lhs_type = RankedTensorType::get({1, lhs_type.getNumElements()},
                                                lhs_type.getElementType());
      auto lhs =
          rewriter.create<ReshapeOp>(op->getLoc(), new_lhs_type, op.getLhs());

      new_dot = rewriter.create<DotOp>(op->getLoc(), lhs, op.getRhs());
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getResult().getType(),
                                           new_dot);
    return success();
  }
};

class MarkValueOnlyTopK : public OpRewritePattern<CustomCallOp> {
 public:
  using OpRewritePattern<CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getCallTargetName() != "mhlo.topk" || op->getNumResults() != 2) {
      return failure();
    }

    auto indices = op.getResult(1);
    if (!indices.use_empty()) {
      return failure();
    }

    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(op->getAttr("mhlo.attributes"));

    auto new_op = rewriter.create<CustomCallOp>(
        op->getLoc(), TypeRange{op->getResultTypes()[0]}, op->getOperands(),
        op.getCallTargetName());

    auto new_attr = DictionaryAttr::get(
        op->getContext(),
        {NamedAttribute(rewriter.getStringAttr("k"), attr.get("k")),
         NamedAttribute(rewriter.getStringAttr("largest"), attr.get("largest")),
         NamedAttribute(rewriter.getStringAttr("value_only"),
                        rewriter.getBoolAttr(true))});
    new_op->setAttr("mhlo.attributes", new_attr);

    rewriter.replaceAllUsesWith(op->getResult(0), new_op->getResult(0));

    return success();
  }
};

class MergeMulConstant : public OpRewritePattern<MulOp> {
 private:
  ConstantOp getActualDefiningConstant(Value v) const {
    if (auto op = v.getDefiningOp<ConstantOp>()) {
      return op;
    }

    if (auto op = v.getDefiningOp<ConvertOp>()) {
      return getActualDefiningConstant(op->getOperand(0));
    }

    return nullptr;
  }

 public:
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::MulOp op,
                                PatternRewriter& rewriter) const override {
    auto lhs = getActualDefiningConstant(op.getLhs());
    auto rhs = getActualDefiningConstant(op.getRhs());
    if (!lhs && !rhs) {
      return failure();
    }

    // x * 1 -> x
    if (rhs && (matchPattern(rhs.getValue(), m_One()) ||
                matchPattern(rhs.getValue(), m_OneFloat()))) {
      rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getResult().getType(),
                                             op.getLhs());
      return success();
    }

    // x * 0 -> 0
    if (rhs && (matchPattern(rhs.getValue(), m_Zero()) ||
                matchPattern(rhs.getValue(), m_AnyZeroFloat()))) {
      rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getResult().getType(),
                                             op.getRhs());
      return success();
    }

    return failure();
  }
};

#include "libspu/dialect/pphlo/IR/canonicalization_patterns.cc.inc"

void DotGeneralOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                               MLIRContext* context) {
  results.add<TransposeReshapeGenericDotGeneral>(context);
}

void DotOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<NormalizeDotShape>(context);
}

void ConvolutionOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<NormalizeDimensionOrder, NormalizeConv1D>(context);
}

void SelectOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                           ::mlir::MLIRContext* context) {
  results.add<FusePredNegIntoSelect>(context);
}

void DivOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context) {
  results.add<DivToReciprocal>(context);
}

void CustomCallOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                               MLIRContext* context) {
  results.add<MarkValueOnlyTopK>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MergeMulConstant>(context);
}

}  // namespace mlir::spu::pphlo
