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

#include "stablehlo/dialect/TypeInference.h"

#include "libspu/dialect/pphlo/ops.h"

namespace mlir::spu::pphlo {

namespace {

llvm::SmallVector<int64_t> inferShape(
    llvm::SmallVector<llvm::ArrayRef<int64_t>> shapes) {
  for (size_t idx = 1; idx < shapes.size(); ++idx) {
    if (shapes[idx] != shapes[0]) {
      (void)emitOptionalError(std::nullopt, "Shape mismatch");
    }
  }
  return {shapes[0].begin(), shapes[0].end()};
}

Type inferElementType(llvm::SmallVector<Type> types) {
  IntegerType it;
  FloatType ft;

  for (auto type : types) {
    if (auto i = type.dyn_cast<IntegerType>()) {
      if (it && it != i) {
        (void)emitOptionalError(std::nullopt, "IntegerType mismatch");
      }
      it = i;
    } else if (auto f = type.dyn_cast<FloatType>()) {
      if (ft) {
        ft = f.getWidth() > ft.getWidth() ? f : ft;
      } else {
        ft = f;
      }
    } else {
      llvm_unreachable("Should not hit");
    }
  }

  if (ft) {
    return ft;
  }
  return it;
}

Type inferPlainType(MLIRContext* context, llvm::ArrayRef<Type> types) {
  llvm::SmallVector<Type> element_types{};
  llvm::SmallVector<llvm::ArrayRef<int64_t>> shapes{};

  for (auto type : types) {
    element_types.emplace_back(
        type.dyn_cast<RankedTensorType>().getElementType());
    shapes.emplace_back(type.dyn_cast<RankedTensorType>().getShape());
  }

  auto inferred_shape = inferShape(shapes);
  auto inferred_element_type = inferElementType(element_types);

  return RankedTensorType::get(inferred_shape, inferred_element_type);
}

Type inferTypes(MLIRContext* context, ValueRange::type_range types) {
  TypeTools tools(context);

  llvm::SmallVector<Visibility, 4> vis;
  llvm::SmallVector<Type, 4> plain_types;
  for (auto type : types) {
    vis.emplace_back(tools.getTypeVisibility(type));
    plain_types.emplace_back(tools.getExpressedType(type));
  }

  auto common_vis = tools.computeCommonVisibility(vis);

  auto inferred = inferPlainType(context, plain_types);

  return tools.getType(inferred, common_vis);
}

LogicalResult inferReturnTypesFromOperands(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  if (operands.empty()) {
    return emitOptionalError(location, "Missing operand");
  }

  inferredReturnTypes.emplace_back(inferTypes(context, operands.getTypes()));
  return success();
}

}  // namespace

#define INFER_RETURN_TYPES_FROM_OPERANDS(Op)                                   \
  LogicalResult Op::inferReturnTypes(                                          \
      ::mlir::MLIRContext* context,                                            \
      ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, \
      ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties,  \
      ::mlir::RegionRange regions,                                             \
      ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {            \
    return inferReturnTypesFromOperands(context, location, operands,           \
                                        attributes, properties, regions,       \
                                        inferredReturnTypes);                  \
  }

INFER_RETURN_TYPES_FROM_OPERANDS(AddOp)
INFER_RETURN_TYPES_FROM_OPERANDS(AndOp)
INFER_RETURN_TYPES_FROM_OPERANDS(Atan2Op)
INFER_RETURN_TYPES_FROM_OPERANDS(DivOp)
INFER_RETURN_TYPES_FROM_OPERANDS(MaxOp)
INFER_RETURN_TYPES_FROM_OPERANDS(MinOp)
INFER_RETURN_TYPES_FROM_OPERANDS(MulOp)
INFER_RETURN_TYPES_FROM_OPERANDS(OrOp)
INFER_RETURN_TYPES_FROM_OPERANDS(PowOp)
INFER_RETURN_TYPES_FROM_OPERANDS(RemOp)
INFER_RETURN_TYPES_FROM_OPERANDS(ShiftLeftOp)
INFER_RETURN_TYPES_FROM_OPERANDS(ShiftRightArithmeticOp)
INFER_RETURN_TYPES_FROM_OPERANDS(ShiftRightLogicalOp)
INFER_RETURN_TYPES_FROM_OPERANDS(SubtractOp)
INFER_RETURN_TYPES_FROM_OPERANDS(XorOp)

#undef INFER_RETURN_TYPES_FROM_OPERANDS

#define INFER_RETURN_TYPES_COMP(Op)                                            \
  LogicalResult Op::inferReturnTypes(                                          \
      ::mlir::MLIRContext* context,                                            \
      ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, \
      ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties,  \
      ::mlir::RegionRange regions,                                             \
      ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {            \
    TypeTools tools(context);                                                  \
    auto types = operands.getTypes();                                          \
    auto shape = types.front().dyn_cast<RankedTensorType>().getShape();        \
    llvm::SmallVector<Visibility, 2> vis;                                      \
    for (auto type : types) {                                                  \
      vis.emplace_back(tools.getTypeVisibility(type));                         \
    }                                                                          \
    auto common_vis = tools.computeCommonVisibility(vis);                      \
    auto ret_type = tools.getType(                                             \
        RankedTensorType::get(                                                 \
            shape,                                                             \
            IntegerType::get(context, 1,                                       \
                             IntegerType::SignednessSemantics::Signless)),     \
        common_vis);                                                           \
    inferredReturnTypes.emplace_back(ret_type);                                \
    return success();                                                          \
  }

INFER_RETURN_TYPES_COMP(EqualOp)
INFER_RETURN_TYPES_COMP(GreaterOp)
INFER_RETURN_TYPES_COMP(LessOp)
INFER_RETURN_TYPES_COMP(GreaterEqualOp)
INFER_RETURN_TYPES_COMP(LessEqualOp)
INFER_RETURN_TYPES_COMP(NotEqualOp)

#undef INFER_RETURN_TYPES_COMP

LogicalResult PadOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, {}, regions);
  return hlo::inferPadOp(location, adaptor.getOperand().getType(),
                         adaptor.getPaddingValue().getType(),
                         adaptor.getEdgePaddingLow(),
                         adaptor.getEdgePaddingHigh(),
                         adaptor.getInteriorPadding(), inferredReturnTypes);
}

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties, RegionRange regions,
    SmallVectorImpl<Type>& inferred_return_types) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, {}, regions);
  return hlo::inferConcatenateOp(location, adaptor.getInputs().getTypes(),
                                 adaptor.getDimension(), inferred_return_types);
}

LogicalResult inferDynamicSliceOp(std::optional<Location> location,
                                  Type operandType, TypeRange startIndicesTypes,
                                  llvm::ArrayRef<int64_t> sliceSizes,
                                  SmallVectorImpl<Type>& inferredReturnTypes) {
  // dynamic_slice_c2
  int numSliceSizes = sliceSizes.size();
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
    int64_t sliceSize = sliceSizes[i];
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

  TypeTools tools(operandType.getContext());
  // dynamic_slice_c5
  llvm::SmallVector<Visibility> vis(startIndicesTypes.size() + 1);
  vis[0] = tools.getTypeVisibility(operandType);
  for (const auto& index_type : llvm::enumerate(startIndicesTypes)) {
    vis[index_type.index() + 1] = tools.getTypeVisibility(index_type.value());
  }

  inferredReturnTypes.emplace_back(RankedTensorType::get(
      sliceSizes, tools.getType(rankedOperandType.getElementType(),
                                tools.computeCommonVisibility(vis))));

  return success();
}

LogicalResult DynamicSliceOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties, RegionRange regions,
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
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  DynamicUpdateSliceOp::Adaptor adaptor(operands, attributes, {}, regions);

  return inferDynamicUpdateSliceOp(
      location, adaptor.getOperand(), adaptor.getUpdate(),
      adaptor.getStartIndices(), inferredReturnTypes);
}

}  // namespace mlir::spu::pphlo
