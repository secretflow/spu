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

#include "mlir/IR/TypeUtilities.h"
#include "stablehlo/dialect/TypeInference.h"

#include "libspu/core/prelude.h"
#include "libspu/dialect/pphlo/IR/ops.h"

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
  FixedPointType fxpt;

  for (auto type : types) {
    if (auto i = mlir::dyn_cast<IntegerType>(type)) {
      if (it && it != i) {
        (void)emitOptionalError(std::nullopt, "IntegerType mismatch");
      }
      it = i;
    } else if (auto f = mlir::dyn_cast<FloatType>(type)) {
      if (ft) {
        ft = f.getWidth() > ft.getWidth() ? f : ft;
      } else {
        ft = f;
      }
    } else if (auto fxp = mlir::dyn_cast<FixedPointType>(type)) {
      if (fxpt) {
        fxpt = fxp.getWidth() > fxpt.getWidth() ? fxp : fxpt;
      } else {
        fxpt = fxp;
      }
    } else {
      llvm_unreachable("Should not hit");
    }
  }

  if (fxpt) {
    return fxpt;
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
        mlir::dyn_cast<RankedTensorType>(type).getElementType());
    shapes.emplace_back(mlir::dyn_cast<RankedTensorType>(type).getShape());
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
INFER_RETURN_TYPES_FROM_OPERANDS(ClampOp)
INFER_RETURN_TYPES_FROM_OPERANDS(Atan2Op)
INFER_RETURN_TYPES_FROM_OPERANDS(DivOp)
INFER_RETURN_TYPES_FROM_OPERANDS(MaxOp)
INFER_RETURN_TYPES_FROM_OPERANDS(MinOp)
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
    auto shape = mlir::dyn_cast<RankedTensorType>(types.front()).getShape();   \
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

Type inferMulLikeReturnElementType(Type lhs_type, Type rhs_type) {
  auto lhs_fxpt = mlir::dyn_cast<FixedPointType>(lhs_type);
  auto rhs_fxpt = mlir::dyn_cast<FixedPointType>(rhs_type);

  // Both fxp, result width = max(lhs, rhs), fraction = lhs+rhs
  if (lhs_fxpt && rhs_fxpt) {
    return FixedPointType::get(
        lhs_type.getContext(),
        std::max(lhs_fxpt.getWidth(), rhs_fxpt.getWidth()),
        lhs_fxpt.getFraction() + rhs_fxpt.getFraction());
  }

  if (lhs_fxpt) {
    if (mlir::isa<FloatType>(rhs_type)) {
      return FixedPointType::get(lhs_fxpt.getContext(), lhs_fxpt.getWidth(),
                                 2 * lhs_fxpt.getFraction());
    }
    return lhs_type;
  }

  if (rhs_fxpt) {
    if (mlir::isa<FloatType>(lhs_type)) {
      return FixedPointType::get(rhs_fxpt.getContext(), rhs_fxpt.getWidth(),
                                 2 * rhs_fxpt.getFraction());
    }
    return rhs_type;
  }

  if (lhs_type == rhs_type) {
    return lhs_type;
  }

  auto lhs_ft = mlir::dyn_cast<FloatType>(lhs_type);
  auto rhs_ft = mlir::dyn_cast<FloatType>(rhs_type);
  // Both float, but different, returns a wider one
  if (lhs_ft && rhs_ft) {
    return lhs_ft.getWidth() > rhs_ft.getWidth() ? lhs_type : rhs_type;
  }
  // Only one float, returns that float
  if (lhs_ft || rhs_ft) {
    return lhs_ft ? lhs_type : rhs_type;
  }

  auto lhs_it = mlir::dyn_cast<IntegerType>(lhs_type);
  auto rhs_it = mlir::dyn_cast<IntegerType>(rhs_type);

  SPU_ENFORCE(lhs_it && rhs_it);

  IntegerType::SignednessSemantics sign = lhs_it.getSignedness();

  if (lhs_it.getWidth() == 1) {
    sign = rhs_it.getSignedness();
  } else if (rhs_it.getWidth() == 1) {
    sign = lhs_it.getSignedness();
  } else if (lhs_it.getSignedness() == rhs_it.getSignedness()) {
    sign = lhs_it.getSignedness();
  } else if (lhs_it.getSignedness() !=
             IntegerType::SignednessSemantics::Unsigned) {
    sign = lhs_it.getSignedness();
  } else {
    sign = rhs_it.getSignedness();
  }

  return IntegerType::get(lhs_type.getContext(),
                          std::max(lhs_it.getWidth(), rhs_it.getWidth()), sign);
}

LogicalResult MulOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  auto types = operands.getTypes();
  TypeTools tools(context);

  // Result shape, guaranteed by op
  auto shape = mlir::dyn_cast<RankedTensorType>(types.front()).getShape();

  // common vis
  auto common_vis = tools.computeCommonVisibility(
      {tools.getTypeVisibility(types[0]), tools.getTypeVisibility(types[1])});

  // element type
  auto element_type = inferMulLikeReturnElementType(
      getElementTypeOrSelf(tools.getExpressedType(types[0])),
      getElementTypeOrSelf(tools.getExpressedType(types[1])));

  inferredReturnTypes.emplace_back(
      tools.getType(RankedTensorType::get(shape, element_type), common_vis));
  return success();
}

LogicalResult DotOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  auto types = operands.getTypes();
  TypeTools tools(context);

  auto lhsType = mlir::dyn_cast<RankedTensorType>(types[0]);
  auto rhsType = mlir::dyn_cast<RankedTensorType>(types[1]);

  llvm::SmallVector<int64_t> dimensions;

  // Result shape, guaranteed by op
  if (1 == lhsType.getRank() && 1 == rhsType.getRank() &&
      // vector dot vector
      (lhsType.getDimSize(0) == rhsType.getDimSize(0))) {
  } else if (2 == lhsType.getRank() && 1 == rhsType.getRank() &&
             (lhsType.getDimSize(1) == rhsType.getDimSize(0))) {
    // matrix dot vector
    dimensions.push_back(lhsType.getDimSize(0));
  } else if (1 == lhsType.getRank() && 2 == rhsType.getRank() &&
             (lhsType.getDimSize(0) == rhsType.getDimSize(0))) {
    // vector dot matrix
    dimensions.push_back(rhsType.getDimSize(1));
  } else if (2 == lhsType.getRank() && 2 == rhsType.getRank() &&
             (lhsType.getDimSize(1) == rhsType.getDimSize(0))) {
    // matrix dot matrix
    dimensions.push_back(lhsType.getDimSize(0));
    dimensions.push_back(rhsType.getDimSize(1));
  } else {
    return emitOptionalError(location,
                             "expected both lhs/rhs ranks to be "
                             "either 1 or 2");
  }

  // common vis
  auto common_vis = tools.computeCommonVisibility(
      {tools.getTypeVisibility(types[0]), tools.getTypeVisibility(types[1])});

  // element type
  auto element_type = inferMulLikeReturnElementType(
      getElementTypeOrSelf(tools.getExpressedType(types[0])),
      getElementTypeOrSelf(tools.getExpressedType(types[1])));

  inferredReturnTypes.emplace_back(tools.getType(
      RankedTensorType::get(dimensions, element_type), common_vis));
  return success();
}

LogicalResult PadOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferPadOp(location, adaptor.getOperand().getType(),
                         adaptor.getPaddingValue().getType(),
                         adaptor.getEdgePaddingLow(),
                         adaptor.getEdgePaddingHigh(),
                         adaptor.getInteriorPadding(), inferredReturnTypes);
}

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferred_return_types) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferConcatenateOp(location, adaptor.getInputs().getTypes(),
                                 adaptor.getDimension(), inferred_return_types);
}

LogicalResult TransposeOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferred_return_types) {
  TransposeOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTransposeOp(location, adaptor.getOperand(),
                               adaptor.getPermutation(), inferred_return_types);
}

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferred_return_types) {
  SliceOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferSliceOp(location, adaptor.getOperand().getType(),
                           adaptor.getStartIndices(), adaptor.getLimitIndices(),
                           adaptor.getStrides(), inferred_return_types);
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
  auto rankedOperandType = mlir::dyn_cast<RankedTensorType>(operandType);
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
  auto vis = llvm::map_to_vector(
      startIndicesTypes, [&](Type t) { return tools.getTypeVisibility(t); });
  vis.emplace_back(tools.getTypeVisibility(operandType));

  inferredReturnTypes.emplace_back(RankedTensorType::get(
      sliceSizes, tools.getType(rankedOperandType.getElementType(),
                                tools.computeCommonVisibility(vis))));

  return success();
}

LogicalResult DynamicSliceOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, properties, regions);
  return inferDynamicSliceOp(location, adaptor.getOperand().getType(),
                             adaptor.getStartIndices().getTypes(),
                             adaptor.getSliceSizes(), inferredReturnTypes);
}

LogicalResult inferDynamicUpdateSliceOp(
    std::optional<Location> location, Value operand, Value update,
    ValueRange startIndices, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operandType = mlir::dyn_cast<ShapedType>(operand.getType());
  auto updateType = mlir::dyn_cast<ShapedType>(update.getType());

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
  TypeTools tools(operand.getContext());

  auto vis = llvm::map_to_vector(startIndices, [&](mlir::Value v) {
    return tools.getTypeVisibility(v.getType());
  });
  vis.emplace_back(tools.getTypeVisibility(operand.getType()));
  vis.emplace_back(tools.getTypeVisibility(update.getType()));

  inferredReturnTypes.emplace_back(
      RankedTensorType::get(operandType.getShape(),
                            tools.getType(operandType.getElementType(),
                                          tools.computeCommonVisibility(vis))));
  return success();
}

LogicalResult DynamicUpdateSliceOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  DynamicUpdateSliceOp::Adaptor adaptor(operands, attributes, properties,
                                        regions);

  return inferDynamicUpdateSliceOp(
      location, adaptor.getOperand(), adaptor.getUpdate(),
      adaptor.getStartIndices(), inferredReturnTypes);
}

}  // namespace mlir::spu::pphlo
