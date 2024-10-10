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

#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/builder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "libspu/core/half.h"
#include "libspu/core/prelude.h"
#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/utils/utils.h"

namespace mlir::spu::pphlo::fixedpoint::builder {

namespace {

Type fxpToFpTypeConversion(Type fxp, bool strip_vis = true) {
  if (auto rt = mlir::dyn_cast<RankedTensorType>(fxp)) {
    return RankedTensorType::get(rt.getShape(),
                                 fxpToFpTypeConversion(rt.getElementType()));
  }
  if (auto st = mlir::dyn_cast<SecretType>(fxp)) {
    if (strip_vis) {
      return fxpToFpTypeConversion(st.getBaseType());
    }
    return SecretType::get(fxpToFpTypeConversion(st.getBaseType()));
  }

  switch (mlir::dyn_cast<FixedPointType>(fxp).getWidth()) {
    case 32:
      return Float16Type::get(fxp.getContext());
    case 64:
      return Float32Type::get(fxp.getContext());
    case 128:
      return Float64Type::get(fxp.getContext());
  }
  llvm_unreachable("Should not hit");
}

Type inferTruncType(Type fxp, int64_t bits_to_trunc) {
  if (auto rt = mlir::dyn_cast<RankedTensorType>(fxp)) {
    return RankedTensorType::get(
        rt.getShape(), inferTruncType(rt.getElementType(), bits_to_trunc));
  }
  if (auto st = mlir::dyn_cast<SecretType>(fxp)) {
    return SecretType::get(inferTruncType(st.getBaseType(), bits_to_trunc));
  }

  return FixedPointType::get(
      fxp.getContext(), mlir::dyn_cast<FixedPointType>(fxp).getWidth(),
      mlir::dyn_cast<FixedPointType>(fxp).getFraction() - bits_to_trunc);
}

}  // namespace

int64_t FxpBuilder::getCurrentFxpBits() {
  return tools_.getFxpBits(base_fxp_value_.getType());
}

int64_t FxpBuilder::getCurrentFxpWidth() {
  return tools_.getFxpWidth(base_fxp_value_.getType());
}

Type FxpBuilder::getIntTypeWithSameWidth(Type t, bool unsign) {
  Visibility vis = getTypeTools().getTypeVisibility(t);
  Type base_type = getTypeTools().getExpressedType(t);

  auto st = mlir::dyn_cast<ShapedType>(base_type);
  auto shape = st.getShape();

  Type result_type;

  if (auto fxp = mlir::dyn_cast<FixedPointType>(st.getElementType())) {
    result_type =
        IntegerType::get(base_type.getContext(), fxp.getWidth(),
                         unsign && fxp.getWidth() != 128
                             ? IntegerType::SignednessSemantics::Unsigned
                             : IntegerType::SignednessSemantics::Signless);

  } else {
    result_type =
        IntegerType::get(base_type.getContext(), st.getElementTypeBitWidth(),
                         unsign && st.getElementTypeBitWidth() != 128
                             ? IntegerType::SignednessSemantics::Unsigned
                             : IntegerType::SignednessSemantics::Signless);
  }

  result_type = RankedTensorType::get(shape, result_type);

  return getTypeTools().getType(result_type, vis);
}

Value FxpBuilder::fxp_constant_with_type(Type fxp_type,
                                         llvm::ArrayRef<double> value) {
  SPU_ENFORCE(value.size() > 1);
  // Get corresponding floating point type
  auto fp_type = mlir::cast<ShapedType>(fxpToFpTypeConversion(fxp_type));
  auto fp_base_type = fp_type.getElementType();

  if (fp_base_type.isF64()) {
    return builder_.create<arith::ConstantOp>(
        loc_, DenseFPElementsAttr::get(fp_type, value));
  }

  if (fp_base_type.isF32()) {
    llvm::SmallVector<float> casted(value);
    return builder_.create<arith::ConstantOp>(
        loc_, DenseFPElementsAttr::get(fp_type, casted));
  }

  if (fp_base_type.isF16()) {
    llvm::SmallVector<half_float::half> casted(value.size());
    for (size_t idx = 0; idx < value.size(); ++idx) {
      casted[idx] = value[idx];
    }
    return builder_.create<arith::ConstantOp>(
        loc_, DenseFPElementsAttr::get(fp_type, casted));
  }

  llvm_unreachable("Should not hit");
}

IntegerAttr FxpBuilder::getIntegerAttr(Type t, int128_t value) {
  auto base = mlir::cast<IntegerType>(
      getElementTypeOrSelf(getTypeTools().getBaseType(t)));
  return builder_.getIntegerAttr(
      IntegerType::get(t.getContext(), base.getWidth()),
      convertFromInt128(base.getWidth(), value));
}

FloatAttr FxpBuilder::getFloatAttr(Type t, double value) {
  auto base = mlir::cast<FloatType>(getElementTypeOrSelf(
      getTypeTools().getBaseType(fxpToFpTypeConversion(t))));
  if (base.isF64()) {
    return builder_.getFloatAttr(base, value);
  }

  if (base.isF32()) {
    return builder_.getFloatAttr(base, static_cast<float>(value));
  }

  if (base.isF16()) {
    return builder_.getFloatAttr(base, static_cast<half_float::half>(value));
  }

  llvm_unreachable("Should not hit");
}

Value FxpBuilder::int_constant_with_type(Type int_type,
                                         llvm::ArrayRef<int128_t> value) {
  SPU_ENFORCE(value.size() > 1);

  auto type = mlir::cast<ShapedType>(tools_.getExpressedType(int_type));
  auto it = mlir::dyn_cast<IntegerType>(type.getElementType());
  bool isSigneless = it.isSignless();
  llvm::SmallVector<APInt> casted(value.size());
  for (size_t idx = 0; idx < value.size(); ++idx) {
    if (it.getWidth() == 128) {
      isSigneless = true;
      int_type = tools_.replaceBaseType(
          int_type,
          IntegerType::get(int_type.getContext(), 128,
                           IntegerType::SignednessSemantics::Signless));
      casted[idx] = convertFromInt128(it.getWidth(), value[idx]);
    } else {
      casted[idx] = APInt(it.getWidth(), value[idx], isSigneless);
    }
  }

  return builder_.create<arith::ConstantOp>(
      loc_, DenseElementsAttr::get(
                RankedTensorType::get({static_cast<int64_t>(value.size())}, it),
                casted));
}

Value FxpBuilder::int_constant(int128_t value) {
  auto int_attr =
      getIntegerAttr(getIntTypeWithSameWidth(base_fxp_value_.getType()), value);
  return splatifyConstant(builder_, int_attr, base_fxp_value_);
}

Value FxpBuilder::uint_constant(uint128_t value) {
  auto int_attr =
      getIntegerAttr(getIntTypeWithSameWidth(base_fxp_value_.getType()), value);
  auto c = splatifyConstant(builder_, int_attr, base_fxp_value_);

  return bitcast(c, getIntTypeWithSameWidth(c.getType(), true));
}

Value FxpBuilder::fxp_constant(double value) {
  auto fp_attr = getFloatAttr(base_fxp_value_.getType(), value);
  return splatifyConstant(builder_, fp_attr, base_fxp_value_);
}

Value FxpBuilder::mul(Value lhs, Value rhs, SignType sign) {
  auto mul = builder_.create<MulOp>(loc_, lhs, rhs);

  auto result_frac_bits = tools_.getFxpBits(mul.getResult().getType());

  return truncation(mul, result_frac_bits / 2, sign);
}

Value FxpBuilder::mul_no_trunc(Value lhs, Value rhs) {
  return builder_.create<MulOp>(loc_, lhs, rhs);
}

Value FxpBuilder::square(Value in) { return mul(in, in, SignType::Positive); }

Value FxpBuilder::dot(Value lhs, Value rhs) {
  auto dot = builder_.create<DotOp>(loc_, lhs, rhs);

  auto result_frac_bits = tools_.getFxpBits(dot.getResult().getType());

  return truncation(dot, result_frac_bits / 2);
}

Value FxpBuilder::truncation(Value in, int64_t bits_to_trunc, SignType sign) {
  return builder_.create<TruncOp>(
      loc_, inferTruncType(in.getType(), bits_to_trunc), in, sign);
}

Value FxpBuilder::add(Value lhs, Value rhs) {
  return builder_.create<AddOp>(loc_, lhs, rhs);
}

Value FxpBuilder::substract(Value lhs, Value rhs) {
  return builder_.create<SubtractOp>(loc_, lhs, rhs);
}

Value FxpBuilder::select(Value pred, Value on_true, Value on_false,
                         Type result_type) {
  return builder_.create<SelectOp>(loc_, result_type, pred, on_true, on_false);
}

Value FxpBuilder::greater(Value lhs, Value rhs) {
  return builder_.create<GreaterOp>(loc_, lhs, rhs);
}

Value FxpBuilder::less(Value lhs, Value rhs) {
  return builder_.create<LessOp>(loc_, lhs, rhs);
}

Value FxpBuilder::equal(Value lhs, Value rhs) {
  return builder_.create<EqualOp>(loc_, lhs, rhs);
}

Value FxpBuilder::negate(Value in) {
  return builder_.create<NegOp>(loc_, in.getType(), in);
}

Value FxpBuilder::bitcast(Value in, Type result_type) {
  return builder_.create<BitcastConvertOp>(loc_, result_type, in);
}

Value FxpBuilder::convert(Value in, Type result_type) {
  auto in_vis = tools_.getTypeVisibility(in.getType());
  auto out_vis = tools_.getTypeVisibility(result_type);
  result_type = tools_.getType(
      result_type, tools_.computeCommonVisibility({in_vis, out_vis}));
  return builder_.create<ConvertOp>(loc_, result_type, in);
}

Value FxpBuilder::concate(llvm::ArrayRef<Value> ops, int64_t axis) {
  return builder_.create<ConcatenateOp>(loc_, ops, axis);
}

Value FxpBuilder::reshape(Value in, llvm::ArrayRef<int64_t> shape) {
  auto in_t = mlir::dyn_cast<ShapedType>(in.getType()).getElementType();
  return builder_.create<ReshapeOp>(loc_, RankedTensorType::get(shape, in_t),
                                    in);
}

Value FxpBuilder::clamp(Value min, Value in, Value max) {
  return builder_.create<ClampOp>(loc_, min, in, max);
}

Value FxpBuilder::floor(Value in) { return builder_.create<FloorOp>(loc_, in); }

Value FxpBuilder::prefix_or(Value in) {
  return builder_.create<PrefixOrOp>(loc_, in);
}

Value FxpBuilder::rshift(Value in, int64_t bits) {
  auto c = getIntegerAttr(in.getType(), bits);
  return builder_.create<ShiftRightLogicalOp>(loc_, in, c);
}

Value FxpBuilder::arshift(Value in, int64_t bits) {
  auto c = getIntegerAttr(in.getType(), bits);
  return builder_.create<ShiftRightArithmeticOp>(loc_, in, c);
}

Value FxpBuilder::lshift(Value in, int64_t bits) {
  auto c = getIntegerAttr(in.getType(), bits);
  return builder_.create<ShiftLeftOp>(loc_, in, c);
}

Value FxpBuilder::xor_(Value lhs, Value rhs) {
  return builder_.create<XorOp>(loc_, lhs, rhs);
}

Value FxpBuilder::and_(Value lhs, Value rhs) {
  return builder_.create<AndOp>(loc_, lhs, rhs);
}

Value FxpBuilder::bitrev(Value in, int64_t start, int64_t end) {
  return builder_.create<BitRevOp>(loc_, in, start, end);
}

Value FxpBuilder::bitdeintel(Value in) {
  return builder_.create<BitDeintlOp>(loc_, in);
}

Value FxpBuilder::bitparity(Value in, int64_t bits) {
  return builder_.create<BitParityOp>(
      loc_, tools_.replaceBaseType(in.getType(), builder_.getI1Type()), in,
      bits);
}

Value FxpBuilder::popcnt(Value in, int64_t bits) {
  if (bits > 0) {
    return builder_.create<PopcntOp>(loc_, in,
                                     builder_.getI64IntegerAttr(bits));
  }
  return builder_.create<PopcntOp>(loc_, in, IntegerAttr());
}

Value FxpBuilder::sign(Value in, bool ignore_sign) {
  return builder_.create<SignOp>(loc_, in, ignore_sign);
}

void FxpBuilder::debug_print(Value in) {
  builder_.create<CustomCallOp>(loc_, TypeRange{}, ValueRange{in}, DBG_PRINT,
                                true);
}

}  // namespace mlir::spu::pphlo::fixedpoint::builder
