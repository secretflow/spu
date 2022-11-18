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

#include "spu/dialect/pphlo_ops.h"

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

#include "spu/dialect/pphlo_attrs.h"
#include "spu/dialect/pphlo_ops.h.inc"

namespace mlir::pphlo {

#include "spu/dialect/pphlo_patterns.cc.inc"

namespace {
Type convertPtTypeToPPhloType(Type ptType) {
  return pphlo::PublicType::get(ptType.getContext(), ptType);
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

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
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

  Value TransposeIfNonConsecutive(OpBuilder& b, Location loc, Value src,
                                  ArrayRef<int64_t> targetOrder) const {
    if (isConsecutive(targetOrder)) {
      return src;
    }
    auto type = src.getType().cast<RankedTensorType>();
    SmallVector<int64_t, 4> transposeShape;
    for (auto i : targetOrder) {
      transposeShape.push_back(type.getDimSize(i));
    }
    return b.create<pphlo::TransposeOp>(
        loc, RankedTensorType::get(transposeShape, type.getElementType()), src,
        b.getI64TensorAttr(targetOrder));
  }

  Value ReshapeIfMorethan3D(OpBuilder& b, Location loc, Value src,
                            size_t dimsBorder0, size_t dimsBorder1) const {
    auto type = src.getType().cast<RankedTensorType>();
    if (type.getRank() <= 3) {
      return src;
    }
    auto shape = type.getShape();
    SmallVector<int64_t, 4> result_shape = {
        std::accumulate(shape.begin(), shape.begin() + dimsBorder0, 1,
                        std::multiplies<int64_t>()),
        std::accumulate(shape.begin() + dimsBorder0,
                        shape.begin() + dimsBorder1, 1,
                        std::multiplies<int64_t>()),
        std::accumulate(shape.begin() + dimsBorder1, shape.end(), 1,
                        std::multiplies<int64_t>())};
    return b.create<pphlo::ReshapeOp>(
        loc, RankedTensorType::get(result_shape, type.getElementType()), src);
  }

  LogicalResult matchAndRewrite(pphlo::DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    auto lhsShapeType = op.lhs().getType().dyn_cast<RankedTensorType>();
    auto rhsShapeType = op.rhs().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!lhsShapeType || !rhsShapeType || !resultType) {
      return failure();
    }

    SmallVector<int64_t> lhsTargetOrder;
    SmallVector<int64_t> rhsTargetOrder;
    pphlo::DotDimensionNumbersAttr dimNumbers = op.dot_dimension_numbers();
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

    Value lhs = TransposeIfNonConsecutive(rewriter, op.getLoc(), op.lhs(),
                                          lhsTargetOrder);
    Value rhs = TransposeIfNonConsecutive(rewriter, op.getLoc(), op.rhs(),
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

    if (lhs == op.lhs() && rhs == op.rhs()) {
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
    // batching、lhs parallel、rhs parallel this order is a convension
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
    const auto& dnums = op.dimension_numbers();
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
    auto input = op.lhs();
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

    auto kernel = op.rhs();

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
        op.window_strides().value_or(nullptr), new_dnums,
        op.feature_group_count(), op.batch_group_count());

    // Reshape the output back to the shape of the original convolution.
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op->getResultTypes()[0], new_conv,
        ConvertToDenseIntElementAttr(&rewriter,
                                     InversePermutation(new_output_dim_order)));

    return success();
  }
};

}  // namespace

OpFoldResult ReverseOp::fold(ArrayRef<Attribute> operands) {
  auto input = operand();

  // No dimensions to reverse.
  auto dims = dimensions();
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

OpFoldResult ReciprocalOp::fold(ArrayRef<Attribute> operands) {
  auto val = operands[0].dyn_cast_or_null<DenseFPElementsAttr>();

  if (!val) {
    return {};
  }

  if (val.isSplat()) {
    auto splat_val = val.getSplatValue<APFloat>();
    APFloat one(splat_val.getSemantics(), 1);

    return SplatElementsAttr::get(operands[0].getType().dyn_cast<ShapedType>(),
                                  one / splat_val);
  }

  llvm::SmallVector<APFloat, 4> values;
  values.reserve(val.getNumElements());

  auto first_val = *val.getValues<APFloat>().begin();
  APFloat one(first_val.getSemantics(), 1);

  for (auto it : val.getValues<APFloat>()) {
    values.push_back(one / it);
  }

  return DenseFPElementsAttr::get(operands[0].getType().dyn_cast<ShapedType>(),
                                  values);
}

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  auto operand_shape = getOperand().getType().cast<TensorType>().getShape();
  auto result_shape = getResult().getType().cast<TensorType>().getShape();
  if (operand_shape == result_shape) {
    return getOperand();
  }
  return {};
}

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  for (auto it : llvm::enumerate(permutation().getValues<APInt>())) {
    if (it.index() != it.value()) {
      return {};
    }
  }
  return getOperand();
}

LogicalResult PadOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  PadOp::Adaptor adaptor(operands, attributes, regions);
  auto inputType = adaptor.operand().getType().cast<RankedTensorType>();
  auto padType = adaptor.padding_value().getType().cast<RankedTensorType>();

  if (padType.getRank() != 0) {
    return emitOptionalError(
        location, llvm::formatv("padding value type should be a rank-0 "
                                "tensor, is rank {0}",
                                padType.getRank()));
  }

  const auto& paddingLow = adaptor.edge_padding_low();
  if (paddingLow.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "edge_padding_low length ({0}) must match operand rank ({1})",
            paddingLow.getType().getNumElements(), inputType.getRank()));
  }

  const auto& paddingHigh = adaptor.edge_padding_high();
  if (paddingHigh.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "edge_padding_high length ({0}) must match operand rank ({1})",
            paddingHigh.getType().getNumElements(), inputType.getRank()));
  }

  const auto& paddingInterior = adaptor.interior_padding();
  if (paddingInterior.getType().getNumElements() != inputType.getRank()) {
    return emitOptionalError(
        location,
        llvm::formatv(
            "interior_padding length ({0}) must match operand rank ({1})",
            paddingInterior.getType().getNumElements(), inputType.getRank()));
  }

  auto inputShape = inputType.getShape();
  SmallVector<int64_t> resultShape;
  for (int i = 0, e = inputShape.size(); i < e; i++) {
    int64_t paddingLowVal = paddingLow.getValues<APInt>()[i].getSExtValue();
    int64_t paddingHighVal = paddingHigh.getValues<APInt>()[i].getSExtValue();
    int64_t paddingInteriorVal =
        paddingInterior.getValues<APInt>()[i].getSExtValue();
    if (paddingInteriorVal < 0) {
      return emitOptionalError(
          location, llvm::formatv("Interior padding cannot be negative: {0}",
                                  paddingInteriorVal));
    }
    int64_t expectedOutput =
        inputShape[i] + paddingLowVal + paddingHighVal +
        std::max<int64_t>(inputShape[i] - 1, 0LL) * paddingInteriorVal;
    if (expectedOutput < 0) {
      return emitOptionalError(
          location,
          llvm::formatv("Padding result in negative size for dimension {0}",
                        i));
    }
    resultShape.push_back(expectedOutput);
  }
  inferredReturnShapes.emplace_back(resultShape, inputType.getElementType());

  return success();
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
                        Ts... printFields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(mhlo-team): this can be simplified when TF moves to C++17.
  using unused = int[];
  (void)unused{0, (printField(printer, std::get<0>(printFields),
                              std::get<1>(printFields), separator),
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
    ArrayRef<llvm::function_ref<ParseResult()>> parseFuncs,
    ArrayRef<bool> parse_equal = {}) {
  assert(keywords.size() == parseFuncs.size());
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
        if (failed(parseFuncs[index]())) {
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
    return NonSpatialDim(DenseMapInfo<int64_t>::getEmptyKey());
  }

  static inline NonSpatialDim getTombstoneKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const NonSpatialDim& Key) {
    return DenseMapInfo<int64_t>::getHashValue(Key);
  }

  static bool isEqual(const NonSpatialDim& LHS, const NonSpatialDim& RHS) {
    return LHS == RHS;
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
  using parse_dim_result_t =
      std::pair<llvm::SmallVector<int64_t>,
                llvm::SmallDenseMap<NonSpatialDim, int64_t, 4,
                                    DenseMapInfoNonSpatialDim>>;

  // Note that the allowed_non_spatial_dims is a set (as opposed to unordered
  // set) because its used to print a list of allowed non spatial dims in the
  // error messages, so making it a set keeps the error messages deterministic.
  auto parse_dims =
      [&](std::set<NonSpatialDim, std::greater<>> allowed_non_spatial_dims,
          parse_dim_result_t& parsed_dims) -> ParseResult {
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
      if (parseResult.hasValue()) {
        if (parseResult.getValue().failed()) {
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
        if (unspecified_spatial_dims.size() < kPrintUnspecifiedDimsMax)
          unspecified_spatial_dims.push_back(dim);
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

  parse_dim_result_t parsed_dims;
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

void printWindowAttributes(
    OpAsmPrinter& p, Operation* op,
    llvm::Optional<DenseIntElementsAttr> window_strides) {
  using pair_t = std::pair<DenseElementsAttr, StringRef>;
  std::array<pair_t, 1> printed_attributes = {{
      {window_strides ? *window_strides : nullptr, "stride"},
  }};

  // Do not print attributes that do no exist.
  auto non_null_attributes = llvm::make_filter_range(
      printed_attributes,
      [](const pair_t& a) { return static_cast<bool>(a.first); });

  llvm::interleaveComma(non_null_attributes, p, [&](const pair_t& a) {
    p << a.second << " = [";
    printWindowAttribute(p, a.first);
    p << "]";
  });
}

ParseResult parseWindowAttributes(OpAsmParser& parser,
                                  DenseIntElementsAttr& window_strides) {
  StringRef attribute_name;

  // Helper to parse an array of the form [ e0, e1, .. ]
  auto parse_array = [&](std::function<ParseResult(void)> parse_element,
                         llvm::Optional<size_t> expected_size =
                             llvm::None) -> ParseResult {
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

    // parse the attribute value. We need to support either 1D and Nx2 array of
    // integers to parse.
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
#include "spu/dialect/pphlo_ops.cc.inc"