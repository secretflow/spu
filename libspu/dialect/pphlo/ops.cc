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

#include "fmt/format.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

#include "libspu/dialect/pphlo/attrs.h"
#include "libspu/dialect/pphlo/base_enums.h"
#include "libspu/dialect/pphlo/ops.h.inc"

namespace mlir::spu::pphlo {

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
