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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// depending dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "libspu/dialect/pphlo/IR/dialect.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/ring/IR/ops.h"
#include "libspu/dialect/ring/transforms/pass_details.h"
#include "libspu/dialect/utils/lowering_intrinsic.h"

namespace mlir::spu {
namespace {

template <typename T>
struct StandardTypeOnlyConverter : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      T op, typename StandardTypeOnlyConverter::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Converter result types
    llvm::SmallVector<Type> result_types;
    if (failed(this->typeConverter->convertTypes(op->getResultTypes(),
                                                 result_types))) {
      return failure();
    }

    // create new op
    Operation *new_op = nullptr;

    if constexpr (std::is_same_v<T, scf::IfOp>) {
      new_op = rewriter.create<T>(op.getLoc(), result_types,
                                  adaptor.getCondition(), true, true);
    } else {
      new_op = rewriter.create<T>(op.getLoc(), result_types,
                                  adaptor.getOperands(), op->getAttrs());
    }

    // Convert regions
    for (size_t idx = 0; idx < op->getNumRegions(); ++idx) {
      auto &source_region = op->getRegions()[idx];
      auto &dest_region = new_op->getRegions()[idx];

      TypeConverter::SignatureConversion sig_conversion(
          source_region.getNumArguments());

      for (const auto &arg : source_region.getArguments()) {
        auto arg_t = this->typeConverter->convertType(arg.getType());
        sig_conversion.addInputs(arg.getArgNumber(), arg_t);
      }

      // inline region
      rewriter.inlineRegionBefore(source_region, dest_region,
                                  dest_region.end());

      // Convert region
      if (failed(rewriter.convertRegionTypes(
              &dest_region, *this->getTypeConverter(), &sig_conversion))) {
        return op->emitOpError("Failed to convert region");
      }
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<pphlo::CustomCallOp>
    : public OpConversionPattern<pphlo::CustomCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::CustomCallOp op, pphlo::CustomCallOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> result_types;

    if (typeConverter->convertTypes(op->getResultTypes(), result_types)
            .failed()) {
      return failure();
    }

    pphlo::TypeTools tools(op->getContext());

    if (op.getCallTargetName() == ENCODE_TO_FXP) {
      rewriter.replaceOpWithNewOp<ring::EncodeToFxpOp>(
          op, result_types.front(), adaptor.getOperands()[0],
          tools.getFxpBits(op->getResultTypes()[0]));
      return success();
    } else if (op.getCallTargetName() == DECODE_FROM_FXP) {
      rewriter.replaceOpWithNewOp<ring::DecodeFromFxpOp>(
          op, result_types.front(), adaptor.getOperands()[0],
          tools.getFxpBits(op->getOperandTypes()[0]));
    } else {
      rewriter.replaceOpWithNewOp<pphlo::CustomCallOp>(
          op, result_types, adaptor.getOperands(), op->getAttrs());
    }

    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<pphlo::CaseOp>
    : public OpConversionPattern<pphlo::CaseOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::CaseOp op, pphlo::CaseOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return emitOptionalError(op->getLoc(),
                             "CaseOp should already been lowered");
  }
};

template <>
struct StandardTypeOnlyConverter<pphlo::MulOp>
    : public OpConversionPattern<pphlo::MulOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::MulOp op, pphlo::MulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ret_type = typeConverter->convertType(op.getType());

    if (ret_type == op.getType()) {
      // One of the input get lower, but result is still the same
      // This is indeed a partial lowered mul
      auto ret = rewriter.create<pphlo::CustomCallOp>(
          op->getLoc(), TypeRange{ret_type}, adaptor.getOperands(),
          PARTIAL_MUL);

      rewriter.replaceOp(op, ret.getResult(0));
    } else {
      rewriter.replaceOpWithNewOp<pphlo::MulOp>(op, ret_type, adaptor.getLhs(),
                                                adaptor.getRhs());
    }

    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<pphlo::DotOp>
    : public OpConversionPattern<pphlo::DotOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::DotOp op, pphlo::DotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ret_type = typeConverter->convertType(op.getType());

    if (ret_type == op.getType()) {
      // One of the input get lower, but result is still the same
      // This is indeed a partial lowered mul
      auto ret = rewriter.create<pphlo::CustomCallOp>(
          op->getLoc(), TypeRange{ret_type}, adaptor.getOperands(),
          PARTIAL_DOT);

      rewriter.replaceOp(op, ret.getResult(0));
    } else {
      rewriter.replaceOpWithNewOp<pphlo::DotOp>(op, ret_type, adaptor.getLhs(),
                                                adaptor.getRhs());
    }

    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<linalg::BroadcastOp>
    : public OpConversionPattern<linalg::BroadcastOp> {
  using OpConversionPattern<linalg::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::BroadcastOp op, linalg::BroadcastOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(
        op, adaptor.getInput(), adaptor.getInit(), op.getDimensions());
    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<linalg::TransposeOp>
    : public OpConversionPattern<linalg::TransposeOp> {
  using OpConversionPattern<linalg::TransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::TransposeOp op, linalg::TransposeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, adaptor.getInput(), adaptor.getInit(), op.getPermutation());
    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<pphlo::ConvertOp>
    : public OpConversionPattern<pphlo::ConvertOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::ConvertOp op, pphlo::ConvertOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Do not actually lower convert
    auto ret_type = typeConverter->convertType(op.getType());

    auto loc = op->getLoc();

    auto in = rewriter
                  .create<pphlo::CustomCallOp>(
                      loc, TypeRange{op.getOperand().getType()},
                      adaptor.getOperands(), PARTIAL_BITCONVERT)
                  .getResult(0);

    auto old_cast =
        rewriter
            .create<pphlo::CustomCallOp>(loc, TypeRange{op.getType()},
                                         ValueRange{in}, PARTIAL_CONVERT)
            .getResult(0);

    rewriter.replaceOpWithNewOp<pphlo::CustomCallOp>(
        op, TypeRange{ret_type}, ValueRange{old_cast}, PARTIAL_BITCONVERT);

    return success();
  }
};

template <>
struct StandardTypeOnlyConverter<pphlo::TruncOp>
    : public OpConversionPattern<pphlo::TruncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pphlo::TruncOp op, pphlo::TruncOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Rounding
    // AxB = (AxB >> 14) + ((AxB >> 13) & 1);
    // See
    // https://stackoverflow.com/questions/14008330/how-do-you-multiply-two-fixed-point-numbers
    // Under certain pattern, like sum(mul(A, B)), error can accumulate in a
    // fairly significant way
    pphlo::TypeTools tools(getContext());
    auto loc = op->getLoc();

    auto in_fxp_bits = tools.getFxpBits(op.getOperand().getType());
    auto out_fxp_bits = tools.getFxpBits(op.getResult().getType());

    if (out_fxp_bits >= in_fxp_bits) {
      return emitOptionalError(loc, "invalid trunction");
    }

    auto result_type =
        mlir::cast<ShapedType>(typeConverter->convertType(op.getType()));
    auto el_type = getElementTypeOrSelf(result_type);

    auto trunc_bits = in_fxp_bits - out_fxp_bits;

    auto s1_bits = rewriter.create<arith::ConstantOp>(
        loc, SplatElementsAttr::get(
                 result_type, rewriter.getIntegerAttr(el_type, trunc_bits)));

    auto s2_bits = rewriter.create<arith::ConstantOp>(
        loc, SplatElementsAttr::get(result_type, rewriter.getIntegerAttr(
                                                     el_type, trunc_bits - 1)));

    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getOneAttr(result_type));

    // x >> n
    Value s1 =
        rewriter.create<arith::ShRSIOp>(loc, adaptor.getOperand(), s1_bits);
    // x >> (n-1)
    Value s2 =
        rewriter.create<arith::ShRSIOp>(loc, adaptor.getOperand(), s2_bits);

    // x >> (n-1) & 1
    s2 = rewriter.create<arith::AndIOp>(loc, s2, one);

    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, s1, s2);

    return success();
  }
};

class PPHLOToRingTypeConverter : public TypeConverter {
 private:
  static Type convertFxpType(pphlo::FixedPointType type) {
    return IntegerType::get(type.getContext(), type.getWidth());
  }

  static std::optional<Value> materializeCastFromIllegal(OpBuilder &builder,
                                                         Type type,
                                                         ValueRange inputs,
                                                         Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
  }

 public:
  explicit PPHLOToRingTypeConverter() {
    addConversion([](Type t) { return t; });
    addConversion([&](ShapedType type) -> Type {
      auto eltype = type.getElementType();
      if (auto fxpt = mlir::dyn_cast<pphlo::FixedPointType>(eltype)) {
        return type.clone(convertFxpType(fxpt));
      }
      return type;
    });
    addTargetMaterialization(materializeCastFromIllegal);
  }
};

struct DecayPublicFxp : public ring::DecayPublicFixedPointBase<DecayPublicFxp> {
 private:
  template <typename OpT>
  void addTypeOnlyConversion(RewritePatternSet &patterns,
                             PPHLOToRingTypeConverter &converter,
                             MLIRContext *context) {
    patterns.insert<StandardTypeOnlyConverter<OpT>>(converter, context);
  }

  template <typename OpT, typename OpT2, typename... OpTs>
  void addTypeOnlyConversion(RewritePatternSet &patterns,
                             PPHLOToRingTypeConverter &converter,
                             MLIRContext *context) {
    addTypeOnlyConversion<OpT>(patterns, converter, context);
    addTypeOnlyConversion<OpT2, OpTs...>(patterns, converter, context);
  }

  void populateConversionPattern(PPHLOToRingTypeConverter &converter,
                                 RewritePatternSet &patterns) {
    auto *context = patterns.getContext();

    addTypeOnlyConversion<
#include "libspu/dialect/ring/transforms/non_ring_op_list.h.inc"
        ,
#define GET_OP_LIST
#include "libspu/dialect/pphlo/IR/ops.cc.inc"
        >(patterns, converter, context);
  }

 public:
  DecayPublicFxp(const DecayPublicFxp &) = default;
  DecayPublicFxp() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    PPHLOToRingTypeConverter converter;

    target.addLegalDialect<pphlo::PPHloDialect, ring::RingDialect,
                           mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                           mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                           mlir::math::MathDialect,
                           mlir::linalg::LinalgDialect>();

    target.addDynamicallyLegalOp<
#include "libspu/dialect/ring/transforms/non_ring_op_list.h.inc"
        ,
#define GET_OP_LIST
#include "libspu/dialect/pphlo/IR/ops.cc.inc"
        >([&](Operation *op) { return converter.isLegal(op); });

    target.addDynamicallyLegalOp<pphlo::CustomCallOp>(
        [&](pphlo::CustomCallOp op) {
          return op.getCallTargetName() == PARTIAL_CONVERT ||
                 op.getCallTargetName() == PARTIAL_BITCONVERT ||
                 converter.isLegal(op);
        });

    populateConversionPattern(converter, patterns);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class CustomCallRestore : public OpRewritePattern<pphlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(pphlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    // PARTIAL_BITCONVERT -> bitcast
    if (op.getCallTargetName() == PARTIAL_BITCONVERT) {
      rewriter.replaceOpWithNewOp<pphlo::BitcastConvertOp>(
          op, op->getResultTypes()[0], op->getOperand(0));
      return success();
    }

    if (op.getCallTargetName() == PARTIAL_CONVERT) {
      rewriter.replaceOpWithNewOp<pphlo::ConvertOp>(op, op->getResultTypes()[0],
                                                    op->getOperand(0));
      return success();
    }

    return failure();
  }
};

struct IntrinsicCleanup : public ring::IntrinsicCleanupBase<IntrinsicCleanup> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->add<CustomCallRestore>(ctx);
  }
};

}  // namespace

namespace ring {

std::unique_ptr<OperationPass<func::FuncOp>> createDecayPublicFixedpoint() {
  return std::make_unique<DecayPublicFxp>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createIntrinsicCleanup() {
  return std::make_unique<IntrinsicCleanup>();
}

}  // namespace ring
}  // namespace mlir::spu
