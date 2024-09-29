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

#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/type_converter.h"

#include "mlir/IR/TypeUtilities.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/utils/lowering_intrinsic.h"

namespace mlir::spu::pphlo {

namespace {
static std::optional<Value> materialize(OpBuilder& builder, Type type,
                                        ValueRange inputs, Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);

  if (mlir::isa<FloatType>(fromType) && mlir::isa<FixedPointType>(toType)) {
    // Use EncodeToFxp to do float->fixedpoint conversions.
    return builder
        .create<pphlo::CustomCallOp>(loc, TypeRange{type},
                                     ValueRange{inputs[0]}, ENCODE_TO_FXP,
                                     false, true)
        ->getResult(0);
  }

  if (mlir::isa<FixedPointType>(fromType) && mlir::isa<FloatType>(toType)) {
    // Use DecodeFromFxp to do fixedpoint->float conversions.
    return builder
        .create<pphlo::CustomCallOp>(loc, TypeRange{type},
                                     ValueRange{inputs[0]}, DECODE_FROM_FXP,
                                     false, true)
        ->getResult(0);
  }

  return {};
}

Type ConvertFloatImpl(FloatType type, const FxpWidthConfig& config) {
  switch (type.getWidth()) {
    case 16:
      return FixedPointType::get(type.getContext(), config.f16_width,
                                 config.f16_frac_bits);
    case 32:
      return FixedPointType::get(type.getContext(), config.f32_width,
                                 config.f32_frac_bits);
    case 64:
      return FixedPointType::get(type.getContext(), config.f64_width,
                                 config.f64_frac_bits);
    default:
      llvm_unreachable("Unhandled float point");
  }
}

}  // namespace

Value convertFloatToFixed(OpBuilder& builder, Location loc, Value in,
                          Type fxp_type) {
  auto ret = materialize(builder, fxp_type, ValueRange{in}, loc);
  if (ret.has_value()) {
    return *ret;
  }
  return in;
}

ShapedType SecretFloatConverter::toFixedPointIfPossible(ShapedType in) const {
  auto el_t = getElementTypeOrSelf(in);
  if (auto f = mlir::dyn_cast<FloatType>(el_t)) {
    return in.clone(convertFloatType(f));
  }
  return in;
}

Type FloatConverter::convertFloatType(FloatType type) const {
  return ConvertFloatImpl(type, config_);
}

FloatConverter::FloatConverter(FxpWidthConfig config) : config_(config) {
  addConversion([&](FloatType type) -> Type { return convertFloatType(type); });
  addConversion([&](RankedTensorType type) -> Type {
    auto eltype = type.getElementType();
    if (auto ft = mlir::dyn_cast<FloatType>(eltype)) {
      return RankedTensorType::get(type.getShape(), convertFloatType(ft));
    }
    return type;
  });

  // illegal to legal
  addTargetMaterialization(materialize);
}

Type SecretFloatConverter::convertFloatType(FloatType type) const {
  return ConvertFloatImpl(type, config_);
}

Type SecretFloatConverter::convertSecretType(SecretType type) const {
  if (auto ft = mlir::dyn_cast<FloatType>(type.getBaseType())) {
    return SecretType::get(convertFloatType(ft));
  }
  return type;
}

SecretFloatConverter::SecretFloatConverter(FxpWidthConfig config)
    : config_(config) {
  addConversion([&](RankedTensorType type) -> Type {
    auto eltype = type.getElementType();
    if (auto st = mlir::dyn_cast<SecretType>(eltype)) {
      return RankedTensorType::get(type.getShape(), convertSecretType(st));
    }
    return type;
  });
}

}  // namespace mlir::spu::pphlo
