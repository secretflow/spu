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

// Base mlir headers
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// depending dialects
#include "mlir/Dialect/Math/IR/Math.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {
namespace {

bool isUnsignedInteger(Type in) {
  auto el_t = getElementTypeOrSelf(in);
  // Consider bool as unsigned int.
  return el_t.isUnsignedInteger() || el_t.isSignlessInteger(1);
}

Type convertInteger(Type in) {
  if (auto rt = mlir::dyn_cast<RankedTensorType>(in)) {
    return rt.clone(convertInteger(rt.getElementType()));
  }

  if (in.isInteger()) {
    return IntegerType::get(in.getContext(), in.getIntOrFloatBitWidth());
  }

  if (auto fxp = mlir::dyn_cast<FixedPointType>(in)) {
    return IntegerType::get(in.getContext(), fxp.getWidth());
  }

  return in;
}

inline Value getConstantOrSplat(OpBuilder &b, Location loc, Type t,
                                Attribute v) {
  if (auto vecType = dyn_cast<ShapedType>(t)) {
    v = SplatElementsAttr::get(vecType, v);
  }
  return b.create<arith::ConstantOp>(loc, t, cast<TypedAttr>(v));
}

inline Value ToSignlessInteger(OpBuilder &b, Location loc, Value in) {
  return b.createOrFold<pphlo::BitcastConvertOp>(
      loc, convertInteger(in.getType()), in);
}

inline Value FromSinglessToType(OpBuilder &b, Location loc, Type t, Value in) {
  return b.createOrFold<pphlo::BitcastConvertOp>(loc, t, in);
}

#define STANDARD_INT_AND_FP_LOWER_DISPATCH(OpType)                    \
  LogicalResult matchAndRewrite(OpType op, PatternRewriter &rewriter) \
      const override {                                                \
    Type elementType = getElementTypeOrSelf(op.getType());            \
    if (isa<FloatType>(elementType)) {                                \
      return lowerFloat(op, rewriter);                                \
    }                                                                 \
    if (isa<IntegerType>(elementType)) {                              \
      return lowerInteger(op, rewriter);                              \
    }                                                                 \
    return failure();                                                 \
  }

#define STANDARD_UNARY_FP_REWRITE(OldOpType, NewOpType)                     \
  LogicalResult lowerFloat(OldOpType op, PatternRewriter &rewriter) const { \
    auto loc = op->getLoc();                                                \
    Value in = op.getOperand();                                             \
    Value new_op = rewriter.create<NewOpType>(loc, in);                     \
    rewriter.replaceOp(op, new_op);                                         \
    return success();                                                       \
  }

#define STANDARD_UNARY_INT_REWRITE(OldOpType, NewOpSIType, NewOpUIType)       \
  LogicalResult lowerInteger(OldOpType op, PatternRewriter &rewriter) const { \
    auto loc = op->getLoc();                                                  \
    auto in = ToSignlessInteger(rewriter, loc, op.getOperand());              \
    Value new_op;                                                             \
    if (isUnsignedInteger(op.getType())) {                                    \
      new_op = rewriter.create<NewOpUIType>(op->getLoc(), in);                \
    } else {                                                                  \
      new_op = rewriter.create<NewOpSIType>(op->getLoc(), in);                \
    }                                                                         \
    new_op = FromSinglessToType(rewriter, loc, op.getType(), new_op);         \
    rewriter.replaceOp(op, new_op);                                           \
    return success();                                                         \
  }

#define STANDARD_BINARY_FP_REWRITE(OldOpType, NewOpType)                    \
  LogicalResult lowerFloat(OldOpType op, PatternRewriter &rewriter) const { \
    auto loc = op->getLoc();                                                \
    auto lhs = op.getLhs();                                                 \
    auto rhs = op.getRhs();                                                 \
    Value new_op = rewriter.create<NewOpType>(loc, lhs, rhs);               \
    rewriter.replaceOp(op, new_op);                                         \
    return success();                                                       \
  }

#define STANDARD_BINARY_INT_REWRITE(OldOpType, NewOpSIType, NewOpUIType)      \
  LogicalResult lowerInteger(OldOpType op, PatternRewriter &rewriter) const { \
    auto loc = op->getLoc();                                                  \
    auto lhs = ToSignlessInteger(rewriter, loc, op.getLhs());                 \
    auto rhs = ToSignlessInteger(rewriter, loc, op.getRhs());                 \
    Value new_op;                                                             \
    if (isUnsignedInteger(op.getLhs().getType())) {                           \
      new_op = rewriter.create<NewOpUIType>(op->getLoc(), lhs, rhs);          \
    } else {                                                                  \
      new_op = rewriter.create<NewOpSIType>(op->getLoc(), lhs, rhs);          \
    }                                                                         \
    new_op = FromSinglessToType(rewriter, loc, op.getType(), new_op);         \
    rewriter.replaceOp(op, new_op);                                           \
    return success();                                                         \
  }

template <typename OldOpType, typename NewOpFType, typename NewOpSIType,
          typename NewOpUIType>

struct StandardArithBinaryOpConverter : public OpRewritePattern<OldOpType> {
 private:
  STANDARD_BINARY_FP_REWRITE(OldOpType, NewOpFType)
  STANDARD_BINARY_INT_REWRITE(OldOpType, NewOpSIType, NewOpUIType)

 public:
  using OpRewritePattern<OldOpType>::OpRewritePattern;
  STANDARD_INT_AND_FP_LOWER_DISPATCH(OldOpType)
};

template <typename OldOpType, typename NewOpFType, typename NewOpSIType,
          typename NewOpUIType>

struct StandardArithUnaryOpConverter : public OpRewritePattern<OldOpType> {
 private:
  STANDARD_UNARY_FP_REWRITE(OldOpType, NewOpFType)
  STANDARD_UNARY_INT_REWRITE(OldOpType, NewOpSIType, NewOpUIType)

 public:
  using OpRewritePattern<OldOpType>::OpRewritePattern;
  STANDARD_INT_AND_FP_LOWER_DISPATCH(OldOpType)
};

template <typename OldOpType, typename NewOpType>
struct StandardArithIntegerOnlyOpConverter
    : public OpRewritePattern<OldOpType> {
  using OpRewritePattern<OldOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OldOpType op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());

    if (!tools.isPublicType(op.getType())) {
      return failure();  // Only do public
    }

    auto loc = op.getLoc();

    Value new_op;

    if (op->getNumOperands() == 1) {
      // bitcast to signless int
      auto in = ToSignlessInteger(rewriter, loc, op->getOperand(0));
      new_op = rewriter.create<NewOpType>(op->getLoc(), in);

    } else if (op->getNumOperands() == 2) {
      // bitcast to signless int
      auto lhs = ToSignlessInteger(rewriter, loc, op->getOperand(0));
      auto rhs = ToSignlessInteger(rewriter, loc, op->getOperand(1));

      new_op = rewriter.create<NewOpType>(op->getLoc(), ValueRange{lhs, rhs});
    }

    new_op = FromSinglessToType(rewriter, loc, op.getType(), new_op);

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <typename OldOpType, typename NewOpType>
struct StandardArithFloatOnlyOpConverter : public OpRewritePattern<OldOpType> {
 private:
  LogicalResult lowerBinary(OldOpType op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value new_op = rewriter.create<NewOpType>(loc, ValueRange{lhs, rhs});
    rewriter.replaceOp(op, new_op);
    return success();
  }

  LogicalResult lowerUnary(OldOpType op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    Value in = op->getOperand(0);
    Value new_op = rewriter.create<NewOpType>(loc, ValueRange{in});
    rewriter.replaceOp(op, new_op);
    return success();
  }

 public:
  using OpRewritePattern<OldOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OldOpType op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());

    if (!tools.isPublicType(op.getType())) {
      return failure();  // Only do public
    }

    if (op->getNumOperands() == 1) {
      return lowerUnary(op, rewriter);
    }

    if (op->getNumOperands() == 2) {
      return lowerBinary(op, rewriter);
    }

    return failure();
  }
};

template <typename T, arith::CmpIPredicate SI_Pred,
          arith::CmpIPredicate UI_Pred, arith::CmpFPredicate F_Pred>
struct ComparatorConverter : public OpRewritePattern<T> {
 private:
  LogicalResult lowerFloat(T op, PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, F_Pred, lhs, rhs);
    return success();
  }

  LogicalResult lowerInteger(T op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto lhs = ToSignlessInteger(rewriter, loc, op.getLhs());
    auto rhs = ToSignlessInteger(rewriter, loc, op.getRhs());

    auto in_el_type = op.getLhs().getType().getElementType();
    bool isUnsigned;

    if (in_el_type.getIntOrFloatBitWidth() == 1) {
      isUnsigned = true;
    } else {
      isUnsigned = in_el_type.isUnsignedInteger();
    }

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, isUnsigned ? UI_Pred : SI_Pred, lhs, rhs);
    return success();
  }

 public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());
    if (!tools.isPublicType(op.getType())) {
      return failure();
    }
    Type elementType = getElementTypeOrSelf(op.getLhs().getType());
    if (isa<FloatType>(elementType)) {
      return lowerFloat(op, rewriter);
    }
    if (isa<IntegerType, FixedPointType>(elementType)) {
      return lowerInteger(op, rewriter);
    }
    return failure();
  }
};

template <typename OldOpType, typename NewOpType>
struct StandardArithShiftOpConverter : public OpRewritePattern<OldOpType> {
  using OpRewritePattern<OldOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OldOpType op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());

    if (!tools.isPublicType(op.getType())) {
      return failure();  // Only do public
    }

    auto loc = op.getLoc();

    // bitcast to signless int
    auto lhs = ToSignlessInteger(rewriter, loc, op.getLhs());
    auto rhs = ToSignlessInteger(rewriter, loc, op.getRhs());

    // Shift
    auto shifted = rewriter.create<NewOpType>(op->getLoc(), lhs, rhs);

    // "Saturate" if the shift is greater than the bitwidth of the type
    auto bitWidthInt = getElementTypeOrSelf(lhs).getIntOrFloatBitWidth();
    auto bitWidth = getConstantOrSplat(
        rewriter, loc, rhs.getType(),
        rewriter.getIntegerAttr(getElementTypeOrSelf(rhs), bitWidthInt));

    // rhs < limit
    auto cmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, rhs, bitWidth);

    Value saturated;

    if constexpr (std::is_same_v<OldOpType, pphlo::ShiftRightArithmeticOp>) {
      Value maxShift = getConstantOrSplat(
          rewriter, loc, lhs.getType(),
          rewriter.getIntegerAttr(getElementTypeOrSelf(lhs.getType()),
                                  bitWidthInt - 1));
      saturated = rewriter.create<NewOpType>(loc, lhs, maxShift);
    } else {
      saturated = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(lhs.getType()));
    }

    // select(cmp, shifted, saturated)
    Value new_op =
        rewriter.create<arith::SelectOp>(loc, cmp, shifted, saturated);

    new_op = FromSinglessToType(rewriter, loc, op.getType(), new_op);

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

struct AbsConverter : public OpRewritePattern<pphlo::AbsOp> {
 private:
  STANDARD_UNARY_FP_REWRITE(pphlo::AbsOp, ::mlir::math::AbsFOp)

  LogicalResult lowerInteger(pphlo::AbsOp op, PatternRewriter &rewriter) const {
    // pphlo.abs(x, result) ->  result = select((x > 0), x, sub(0, x))
    auto loc = op->getLoc();
    Value in = op.getOperand();

    IntegerType int_type =
        mlir::cast<IntegerType>(getElementTypeOrSelf(op.getType()));
    RankedTensorType int_ret_type = mlir::cast<RankedTensorType>(op.getType());

    Value zeroIntval = getConstantOrSplat(rewriter, loc, int_ret_type,
                                          rewriter.getZeroAttr(int_type));

    auto inGtZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, in, zeroIntval);

    auto negVal = rewriter.create<arith::SubIOp>(loc, zeroIntval, in);

    rewriter.replaceOpWithNewOp<::mlir::arith::SelectOp>(op, inGtZero, in,
                                                         negVal);

    return success();
  }

 public:
  using OpRewritePattern::OpRewritePattern;

  STANDARD_INT_AND_FP_LOWER_DISPATCH(pphlo::AbsOp)
};

struct SelectConverter : public OpRewritePattern<pphlo::SelectOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::SelectOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(getContext());

    if (!tools.isPublicType(op.getPred().getType())) {
      return failure();  // Only handle public select
    }

    auto loc = op->getLoc();

    // Make sure all ture/false is result type
    auto on_true =
        rewriter.create<pphlo::ConvertOp>(loc, op.getType(), op.getOnTrue());
    auto on_false =
        rewriter.create<pphlo::ConvertOp>(loc, op.getType(), op.getOnFalse());

    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, op.getPred(), on_true,
                                                 on_false);

    return success();
  }
};

struct ConvertOpConverter : public OpRewritePattern<pphlo::ConvertOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(getContext());

    auto loc = op->getLoc();
    auto arg_type = op.getOperand().getType();
    auto result_type = op.getType();

    if (!tools.isPublicType(result_type) || !tools.isPublicType(arg_type) ||
        tools.isFixedPointType(result_type) ||
        tools.isComplexFixedPointType(arg_type)) {
      return failure();  // Only handle public <-> public convert
    }

    Value converted_in = ToSignlessInteger(rewriter, loc, op.getOperand());
    Type sourceType = getElementTypeOrSelf(arg_type);
    Type targetType = getElementTypeOrSelf(result_type);
    Type convertedSourceType = getElementTypeOrSelf(convertInteger(arg_type));

    // A boolean value is considered to be unsigned when converting to
    // floating-point. Otherwise, it will become `-1`.
    if (isUnsignedInteger(sourceType) &&
        mlir::arith::UIToFPOp::areCastCompatible(convertedSourceType,
                                                 targetType)) {
      rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(
          op, result_type, converted_in, std::nullopt);
      return success();
    }
    if (mlir::arith::SIToFPOp::areCastCompatible(sourceType, targetType)) {
      rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(
          op, result_type, converted_in, std::nullopt);
      return success();
    }
    if (isa<FloatType>(sourceType) && isa<FloatType>(targetType)) {
      auto src = cast<FloatType>(sourceType);
      auto res = cast<FloatType>(targetType);
      if (src.getWidth() > res.getWidth()) {
        rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(
            op, result_type, converted_in, std::nullopt);
        return success();
      }
      if (src.getWidth() < res.getWidth()) {
        rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(
            op, result_type, converted_in, std::nullopt);
        return success();
      }
      return failure();
    }
    if (targetType.isInteger(/*width=*/1)) {
      // When casting to bool, we need to compare whether the value is equal to
      // zero.
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(converted_in.getType()));
      if (sourceType.isInteger()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, arith::CmpIPredicate::ne, converted_in, zero);
        return success();
      }
      if (isa<FloatType>(sourceType)) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, arith::CmpFPredicate::UNE, converted_in, zero);
        return success();
      }
    }
    if (isa<IntegerType>(sourceType) && isa<IntegerType>(targetType)) {
      auto src = cast<IntegerType>(sourceType);
      auto res = cast<IntegerType>(targetType);
      if (src.getWidth() > res.getWidth()) {
        rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(
            op, result_type, converted_in, std::nullopt);
        return success();
      }
      if (src.getWidth() < res.getWidth()) {
        // Special case boolean values, so they get casted to `1` instead of
        // `-1`.
        if (isUnsignedInteger(src)) {
          rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(
              op, result_type, converted_in, std::nullopt);
          return success();
        }
        rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(
            op, result_type, converted_in, std::nullopt);
        return success();
      }
      // Only ui<->i conversion, use bitcast
      rewriter.replaceOpWithNewOp<pphlo::BitcastConvertOp>(op, result_type,
                                                           converted_in);
      return success();
    }
    if (targetType.isUnsignedInteger() &&
        mlir::arith::FPToUIOp::areCastCompatible(convertedSourceType,
                                                 targetType)) {
      rewriter.replaceOpWithNewOp<mlir::arith::FPToUIOp>(
          op, result_type, converted_in, std::nullopt);
      return success();
    }
    if (mlir::arith::FPToSIOp::areCastCompatible(convertedSourceType,
                                                 targetType)) {
      rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(
          op, result_type, converted_in, std::nullopt);
      return success();
    }

    return failure();
  }
};

struct DivConverter : public OpRewritePattern<pphlo::DivOp> {
 private:
  STANDARD_BINARY_FP_REWRITE(pphlo::DivOp, arith::DivFOp)

  LogicalResult lowerInteger(pphlo::DivOp op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto lhs = ToSignlessInteger(rewriter, loc, op.getLhs());
    auto rhs = ToSignlessInteger(rewriter, loc, op.getRhs());

    Value new_op;
    bool isUnsigned = getElementTypeOrSelf(op.getType()).isUnsignedInteger();

    if (isUnsigned) {
      new_op = rewriter.create<arith::DivUIOp>(loc, lhs, rhs);
    } else {
      new_op = rewriter.create<arith::DivSIOp>(loc, lhs, rhs);
    }

    new_op = FromSinglessToType(rewriter, loc, op.getType(), new_op);
    rewriter.replaceOp(op, new_op);
    return success();
  }

 public:
  using OpRewritePattern::OpRewritePattern;

  STANDARD_INT_AND_FP_LOWER_DISPATCH(pphlo::DivOp)
};

struct NotConverter : public OpRewritePattern<pphlo::NotOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::NotOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(getContext());

    if (!tools.isPublicType(op.getType())) {
      return failure();  // Only handle public select
    }

    auto loc = op->getLoc();

    // Make signless
    auto in = ToSignlessInteger(rewriter, loc, op.getOperand());

    auto el_type = getElementTypeOrSelf(in.getType());

    // pphlo.not(x) -> x ^ -1
    Value allOnes = getConstantOrSplat(
        rewriter, loc, in.getType(),
        rewriter.getIntegerAttr(
            el_type, APInt::getAllOnes(el_type.getIntOrFloatBitWidth())));

    Value ret = rewriter.create<::mlir::arith::XOrIOp>(loc, allOnes, in);

    ret = FromSinglessToType(rewriter, loc, op.getType(), ret);

    rewriter.replaceOp(op, ret);

    return success();
  }
};

struct LogisticConverter : public OpRewritePattern<pphlo::LogisticOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::LogisticOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());

    if (!tools.isPublicType(op.getType())) {
      return failure();  // Only do public
    }

    auto loc = op->getLoc();

    // logistic(x) = 1 / (1 + exp(-x))

    // -x
    Value ret = rewriter.create<arith::NegFOp>(loc, op.getOperand());

    // exp(-x)
    ret = rewriter.create<math::ExpOp>(loc, ret);

    // 1 + exp(-x)
    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getOneAttr(op.getType()));
    ret = rewriter.create<arith::AddFOp>(loc, one, ret);

    // 1 / (1 + exp(-x))
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, one, ret);

    return success();
  }
};

struct NegateConverter : public OpRewritePattern<pphlo::NegOp> {
 private:
  STANDARD_UNARY_FP_REWRITE(pphlo::NegOp, arith::NegFOp)

  LogicalResult lowerInteger(pphlo::NegOp op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto in = ToSignlessInteger(rewriter, loc, op.getOperand());

    // pphlo.neg(x, result) -> result = sub(0, x)
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(in.getType()));
    Value new_op = rewriter.create<arith::SubIOp>(loc, zero, in);
    new_op = FromSinglessToType(rewriter, loc, op.getType(), new_op);
    rewriter.replaceOp(op, new_op);
    return success();
  }

 public:
  using OpRewritePattern::OpRewritePattern;
  STANDARD_INT_AND_FP_LOWER_DISPATCH(pphlo::NegOp)
};

struct ReciprocalConverter : public OpRewritePattern<pphlo::ReciprocalOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::ReciprocalOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(getContext());

    if (!tools.isPublicType(op.getType())) {
      return failure();  // Only handle public select
    }

    auto loc = op->getLoc();

    // 1/x
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getOneAttr(op.getType()));

    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, one, op.getOperand());

    return success();
  }
};

struct PowerConverter : public OpRewritePattern<pphlo::PowOp> {
 private:
  STANDARD_BINARY_FP_REWRITE(pphlo::PowOp, math::PowFOp)

  LogicalResult lowerInteger(pphlo::PowOp op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto lhs = ToSignlessInteger(rewriter, loc, op.getLhs());
    auto rhs = ToSignlessInteger(rewriter, loc, op.getRhs());

    auto result_type = convertInteger(op.getType());

    // Exponentiation by squaring:
    // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getOneAttr(result_type));

    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Everything else would overflow for any exponent > 1, as 2^64
    // is the larget possible exponent for a 64-bit integer, and
    // that's 1 << 6.
    Value upperBound = rewriter.create<arith::ConstantIndexOp>(loc, 6);

    auto originalBase = lhs;
    auto originalExponent = rhs;

    Value accum =
        rewriter
            .create<scf::ForOp>(
                loc, lowerBound, upperBound, step,
                SmallVector<Value>({one, originalBase, originalExponent}),
                [&](OpBuilder &b, Location, Value /*v*/, ValueRange iters) {
                  Value accum = iters[0];
                  Value base = iters[1];
                  Value exponent = iters[2];

                  Value condition = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq,
                      b.create<::mlir::arith::AndIOp>(loc, exponent, one), one);
                  Value multiplied =
                      b.create<::mlir::arith::MulIOp>(loc, accum, base);
                  accum = b.create<::mlir::arith::SelectOp>(loc, condition,
                                                            multiplied, accum);
                  base = b.create<::mlir::arith::MulIOp>(loc, base, base);
                  exponent =
                      b.create<::mlir::arith::ShRUIOp>(loc, exponent, one);
                  b.create<scf::YieldOp>(
                      loc, SmallVector<Value>({accum, base, exponent}));
                })
            .getResult(0);

    Value negOne = getConstantOrSplat(
        rewriter, loc, result_type,
        rewriter.getIntegerAttr(getElementTypeOrSelf(result_type), -1));
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(result_type));
    Value two = getConstantOrSplat(
        rewriter, loc, result_type,
        rewriter.getIntegerAttr(getElementTypeOrSelf(result_type), 2));

    Value rhsIsEven = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<arith::RemSIOp>(loc, rhs, two), zero);
    Value rhsIsNegative = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, rhs, zero);
    Value lhsIsOne =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, one);
    Value lhsIsNegOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhs, negOne);

    // The accum is correct when the rhs is non-negative. When rhs is
    // negative, we return 0 for integer, with the exception of lhs values of
    // 1
    // and -1 which have integer results for negative exponents.
    // Specifically,
    // the calulation is the following:
    //
    // - Return accum if the rhs is not negative.
    // - Return 1 or -1 depending on the parity of rhs when the lhs is -1.
    // - Return 1 if lhs is 1.
    // - Else return 0.
    Value ifLhsIsOne =
        rewriter.create<::mlir::arith::SelectOp>(loc, lhsIsOne, one, zero);
    Value ifLhsIsNegOne = rewriter.create<::mlir::arith::SelectOp>(
        loc, lhsIsNegOne,
        rewriter.create<::mlir::arith::SelectOp>(loc, rhsIsEven, one, negOne),
        ifLhsIsOne);
    Value new_op = rewriter.create<::mlir::arith::SelectOp>(
        loc, rhsIsNegative, ifLhsIsNegOne, accum);

    new_op = FromSinglessToType(rewriter, loc, op.getType(), new_op);

    rewriter.replaceOp(op, new_op);
    return success();
  }

 public:
  using OpRewritePattern::OpRewritePattern;
  STANDARD_INT_AND_FP_LOWER_DISPATCH(pphlo::PowOp)
};

struct SignConverter : public OpRewritePattern<pphlo::SignOp> {
 private:
  LogicalResult lowerFloat(pphlo::SignOp op, PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(op.getType()));
    Value ne0I1 = rewriter.create<::mlir::arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, op.getOperand(), zero);
    Value ne0Float =
        rewriter.create<::mlir::arith::UIToFPOp>(loc, zero.getType(), ne0I1);
    Value copySign = rewriter.create<::mlir::math::CopySignOp>(
        loc, op.getType(), ne0Float, op.getOperand());
    auto isNan = rewriter.create<::mlir::arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, op.getOperand(), op.getOperand());
    rewriter.replaceOpWithNewOp<::mlir::arith::SelectOp>(
        op, isNan, op.getOperand(), copySign);

    return success();
  }

  LogicalResult lowerInteger(pphlo::SignOp op,
                             PatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto in = ToSignlessInteger(rewriter, loc, op.getOperand());

    auto integer_type = cast<IntegerType>(getElementTypeOrSelf(in.getType()));

    // sign(x) = x == 0 ? 0 : ((x s>> 31) | 1)

    // x >> bits
    Value bitwidthMinusOne = getConstantOrSplat(
        rewriter, loc, in.getType(),
        rewriter.getIntegerAttr(integer_type, integer_type.getWidth() - 1));
    Value ashr =
        rewriter.create<::mlir::arith::ShRSIOp>(loc, in, bitwidthMinusOne);

    // ((x s>> 31) | 1)
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getOneAttr(in.getType()));

    Value new_op = rewriter.create<::mlir::arith::OrIOp>(loc, ashr, one);

    if (!op.getIgnoreZero()) {
      // x == 0 ? 0 : ((x s>> 31) | 1)
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(in.getType()));

      Value cmp = rewriter.create<::mlir::arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, in, zero);

      new_op = rewriter.create<::mlir::arith::SelectOp>(loc, cmp, zero, new_op);
    }

    new_op = FromSinglessToType(rewriter, loc, op.getType(), new_op);
    rewriter.replaceOp(op, new_op);
    return success();
  }

 public:
  using OpRewritePattern::OpRewritePattern;
  STANDARD_INT_AND_FP_LOWER_DISPATCH(pphlo::SignOp)
};

struct ErfConverter : public OpRewritePattern<pphlo::CustomCallOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());

    if (op.getCallTargetName() != "mhlo.erf" ||
        !tools.isPublicType(op.getResult(0).getType())) {
      return failure();  // Only do public
    }

    rewriter.replaceOpWithNewOp<math::ErfOp>(op, op->getOperand(0));

    return success();
  }
};

struct LegalizeToArith : public LegalizeToArithBase<LegalizeToArith> {
 private:
  void populateRewritePatterns(RewritePatternSet &patterns) {
    auto *context = patterns.getContext();

    // Controlflow
    patterns.insert<
        AbsConverter,  //
        StandardArithBinaryOpConverter<pphlo::AddOp, arith::AddFOp,
                                       arith::AddIOp,
                                       arith::AddIOp>,  //
        StandardArithBinaryOpConverter<pphlo::SubtractOp, arith::SubFOp,
                                       arith::SubIOp,
                                       arith::SubIOp>,  //
        StandardArithBinaryOpConverter<pphlo::MaxOp, arith::MaximumFOp,
                                       arith::MaxSIOp,
                                       arith::MaxUIOp>,  //
        StandardArithBinaryOpConverter<pphlo::MinOp, arith::MinimumFOp,
                                       arith::MinSIOp,
                                       arith::MinUIOp>,  //
        StandardArithBinaryOpConverter<pphlo::MulOp, arith::MulFOp,
                                       arith::MulIOp,
                                       arith::MulIOp>,  //
        StandardArithBinaryOpConverter<pphlo::SubtractOp, arith::SubFOp,
                                       arith::SubIOp,
                                       arith::SubIOp>,  //
        StandardArithBinaryOpConverter<pphlo::RemOp, arith::RemFOp,
                                       arith::RemSIOp,
                                       arith::RemUIOp>,  //
        StandardArithIntegerOnlyOpConverter<pphlo::AndOp,
                                            arith::AndIOp>,  //
        StandardArithIntegerOnlyOpConverter<pphlo::OrOp,
                                            arith::OrIOp>,  //
        StandardArithIntegerOnlyOpConverter<pphlo::XorOp,
                                            arith::XOrIOp>,  //
        StandardArithIntegerOnlyOpConverter<pphlo::PopcntOp,
                                            math::CtPopOp>,  //
        ComparatorConverter<pphlo::LessOp, arith::CmpIPredicate::slt,
                            arith::CmpIPredicate::ult,
                            arith::CmpFPredicate::OLT>,  //
        ComparatorConverter<pphlo::LessEqualOp, arith::CmpIPredicate::sle,
                            arith::CmpIPredicate::ule,
                            arith::CmpFPredicate::OLE>,  //
        ComparatorConverter<pphlo::EqualOp, arith::CmpIPredicate::eq,
                            arith::CmpIPredicate::eq,
                            arith::CmpFPredicate::OEQ>,  //
        ComparatorConverter<pphlo::NotEqualOp, arith::CmpIPredicate::ne,
                            arith::CmpIPredicate::ne,
                            arith::CmpFPredicate::ONE>,  //
        ComparatorConverter<pphlo::GreaterOp, arith::CmpIPredicate::sgt,
                            arith::CmpIPredicate::ugt,
                            arith::CmpFPredicate::OGT>,
        ComparatorConverter<pphlo::GreaterEqualOp, arith::CmpIPredicate::sge,
                            arith::CmpIPredicate::uge,
                            arith::CmpFPredicate::OGE>,                    //
        StandardArithFloatOnlyOpConverter<pphlo::CosineOp, math::CosOp>,   //
        StandardArithFloatOnlyOpConverter<pphlo::Expm1Op, math::ExpM1Op>,  //
        StandardArithFloatOnlyOpConverter<pphlo::ExpOp, math::ExpOp>,      //
        StandardArithFloatOnlyOpConverter<pphlo::Atan2Op, math::Atan2Op>,  //
        StandardArithFloatOnlyOpConverter<pphlo::CeilOp, math::CeilOp>,    //
        StandardArithFloatOnlyOpConverter<pphlo::FloorOp, math::FloorOp>,  //
        StandardArithFloatOnlyOpConverter<pphlo::Log1pOp, math::Log1pOp>,  //
        StandardArithFloatOnlyOpConverter<pphlo::LogOp, math::LogOp>,      //
        StandardArithFloatOnlyOpConverter<pphlo::RoundOp, math::RoundOp>,  //
        StandardArithFloatOnlyOpConverter<pphlo::RoundNearestEvenOp,
                                          math::RoundEvenOp>,  //
        StandardArithFloatOnlyOpConverter<pphlo::RsqrtOp,
                                          math::RsqrtOp>,  //
        StandardArithFloatOnlyOpConverter<pphlo::SqrtOp,
                                          math::SqrtOp>,                 //
        StandardArithFloatOnlyOpConverter<pphlo::SineOp, math::SinOp>,   //
        StandardArithFloatOnlyOpConverter<pphlo::TanhOp, math::TanhOp>,  //
        StandardArithShiftOpConverter<pphlo::ShiftLeftOp, arith::ShLIOp>,
        StandardArithShiftOpConverter<pphlo::ShiftRightLogicalOp,
                                      arith::ShRUIOp>,
        StandardArithShiftOpConverter<pphlo::ShiftRightArithmeticOp,
                                      arith::ShRSIOp>,
        SelectConverter,
        ConvertOpConverter,   //
        DivConverter,         //
        NotConverter,         //
        LogisticConverter,    //
        NegateConverter,      //
        PowerConverter,       //
        ReciprocalConverter,  //
        SignConverter,        //
        ErfConverter>(context);
  }

 public:
  LegalizeToArith(const LegalizeToArith &) = default;
  LegalizeToArith() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);

    populateRewritePatterns(patterns);

    mlir::GreedyRewriteConfig config;
    // There's no point simplifying more than once.
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToArith() {
  return std::make_unique<LegalizeToArith>();
}

}  // namespace mlir::spu::pphlo
