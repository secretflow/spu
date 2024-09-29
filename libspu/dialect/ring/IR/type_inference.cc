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

#include "libspu/dialect/ring/IR/ops.h"

namespace mlir::spu::ring {

namespace {

bool isSecretType(Type in) {
  auto el = getElementTypeOrSelf(in);
  return mlir::isa<SecretType>(el);
}

Type replaceRingType(Type in, Type new_ring_ty, bool is_secret) {
  auto rt = mlir::dyn_cast<RankedTensorType>(in);
  if (is_secret) {
    new_ring_ty = SecretType::get(new_ring_ty);
  }
  return RankedTensorType::get(rt.getShape(), new_ring_ty);
}

Type inferRingType(MLIRContext* context, TypeRange types) {
  bool isSigned = false;
  unsigned int max_width = 0;
  for (const auto& t : types) {
    auto it = mlir::dyn_cast<IntegerType>(t);
    isSigned = isSigned || !it.isUnsignedInteger();
    max_width = std::max(max_width, it.getWidth());
  }

  return IntegerType::get(context, max_width,
                          isSigned
                              ? IntegerType::SignednessSemantics::Signless
                              : IntegerType::SignednessSemantics::Unsigned);
}

Type inferTypes(MLIRContext* context, ValueRange::type_range types) {
  llvm::SmallVector<Type, 4> ring_types;
  bool isSecret = false;
  for (auto type : types) {
    isSecret = isSecret || isSecretType(type);
    ring_types.emplace_back(getBaseType(type));
  }

  return replaceRingType(types.front(), inferRingType(context, ring_types),
                         isSecret);
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
INFER_RETURN_TYPES_FROM_OPERANDS(XorOp)

#undef INFER_RETURN_TYPES_FROM_OPERANDS

LogicalResult MsbOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  auto in_type = mlir::dyn_cast<RankedTensorType>(operands.front().getType());
  auto is_secret = mlir::dyn_cast<SecretType>(getElementTypeOrSelf(in_type));

  Type new_eltype = IntegerType::get(context, 1);
  if (is_secret) {
    new_eltype = SecretType::get(new_eltype);
  }
  inferredReturnTypes.emplace_back(
      RankedTensorType::get(in_type.getShape(), new_eltype));

  return success();
}

LogicalResult EqualOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  auto lhs_type = mlir::dyn_cast<RankedTensorType>(operands[0].getType());
  auto rhs_type = mlir::dyn_cast<RankedTensorType>(operands[1].getType());

  auto is_secret = mlir::isa<SecretType>(getElementTypeOrSelf(lhs_type)) ||
                   mlir::isa<SecretType>(getElementTypeOrSelf(rhs_type));

  Type new_eltype = IntegerType::get(context, 1);
  if (is_secret) {
    new_eltype = SecretType::get(new_eltype);
  }
  inferredReturnTypes.emplace_back(
      RankedTensorType::get(lhs_type.getShape(), new_eltype));

  return success();
}

LogicalResult LessOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  auto lhs_type = mlir::dyn_cast<RankedTensorType>(operands[0].getType());
  auto rhs_type = mlir::dyn_cast<RankedTensorType>(operands[1].getType());

  auto is_secret = mlir::isa<SecretType>(getElementTypeOrSelf(lhs_type)) ||
                   mlir::isa<SecretType>(getElementTypeOrSelf(rhs_type));

  Type new_eltype = IntegerType::get(context, 1);
  if (is_secret) {
    new_eltype = SecretType::get(new_eltype);
  }
  inferredReturnTypes.emplace_back(
      RankedTensorType::get(lhs_type.getShape(), new_eltype));

  return success();
}

}  // namespace mlir::spu::ring
