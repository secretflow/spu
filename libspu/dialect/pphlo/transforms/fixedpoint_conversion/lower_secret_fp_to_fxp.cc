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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/type_converter.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {
namespace {

template <typename OP, class Enable = void>
class FxpOpConverter : public OpConversionPattern<OP> {
 public:
  FxpOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<OP>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      OP op, typename OP::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> result_types;

    if (failed(this->typeConverter->convertTypes(op->getResultTypes(),
                                                 result_types))) {
      return failure();
    }

    llvm::SmallVector<TypeConverter::SignatureConversion> sig_converters;

    for (int64_t idx = 0; idx < op->getNumRegions(); ++idx) {
      auto &entry_block = op->getRegion(idx).front();

      sig_converters.emplace_back(entry_block.getNumArguments());

      for (const auto &arg : entry_block.getArguments()) {
        auto arg_t = this->getTypeConverter()->convertType(arg.getType());
        sig_converters.back().addInputs(arg.getArgNumber(), arg_t);
      }
    }

    llvm::SmallVector<Value> operands = llvm::to_vector(adaptor.getOperands());

    // For any op without region, make it all based on fixedpoint
    if (op->getNumRegions() == 0) {
      const auto *stc =
          static_cast<const SecretFloatConverter *>(this->typeConverter);
      for (auto &o : operands) {
        o = convertFloatToFixed(
            rewriter, op->getLoc(), o,
            stc->toFixedPointIfPossible(mlir::cast<ShapedType>(o.getType())));
      }
    }

    auto new_op = rewriter.create<OP>(op.getLoc(), result_types, operands,
                                      op->getAttrs());

    for (int64_t idx = 0; idx < op->getNumRegions(); ++idx) {
      // Copy over the operations inside the region.
      rewriter.inlineRegionBefore(op->getRegion(idx), new_op->getRegion(idx),
                                  new_op->getRegion(idx).end());

      if (failed(rewriter.convertRegionTypes(&new_op->getRegion(idx),
                                             *this->getTypeConverter(),
                                             &sig_converters[idx]))) {
        return failure();
      }
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <typename OP>
class FxpOpConverter<
    OP,
    typename std::enable_if_t<
        std::is_same_v<OP, MulOp> || std::is_same_v<OP, DotOp> ||
        std::is_same_v<OP, DotGeneralOp> || std::is_same_v<OP, ConvolutionOp>>>
    : public OpConversionPattern<OP> {
 public:
  FxpOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<OP>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      OP op, typename FxpOpConverter::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());

    auto lhs_t = tools.getBaseType(adaptor.getLhs().getType());
    auto rhs_t = tools.getBaseType(adaptor.getRhs().getType());

    size_t operand_fxp_bits = 0;
    if (auto fxp_t = mlir::dyn_cast<FixedPointType>(lhs_t)) {
      operand_fxp_bits += fxp_t.getFraction();
    }
    if (auto fxp_t = mlir::dyn_cast<FixedPointType>(rhs_t)) {
      operand_fxp_bits += fxp_t.getFraction();
    }

    if (mlir::isa<FloatType>(lhs_t) || mlir::isa<FloatType>(rhs_t)) {
      operand_fxp_bits *= 2;
    }

    auto result_t = this->typeConverter->convertType(op->getResultTypes()[0]);
    auto result_fxp_type =
        mlir::dyn_cast<FixedPointType>(tools.getBaseType(result_t));

    if (result_fxp_type.getFraction() == operand_fxp_bits) {
      // No need to do truncation
      rewriter.replaceOpWithNewOp<OP>(op, result_t, adaptor.getOperands(),
                                      op->getAttrs());
    } else {
      auto mul_result_t = tools.replaceBaseType(
          result_t,
          FixedPointType::get(op->getContext(), result_fxp_type.getWidth(),
                              operand_fxp_bits));
      // mul
      auto mul_op = rewriter.create<OP>(op->getLoc(), mul_result_t,
                                        adaptor.getOperands(), op->getAttrs());
      // trunc
      rewriter.replaceOpWithNewOp<TruncOp>(op, result_t, mul_op.getResult());
    }

    return success();
  }
};

template <>
class FxpOpConverter<mlir::func::ReturnOp>
    : public OpConversionPattern<mlir::func::ReturnOp> {
 public:
  FxpOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<mlir::func::ReturnOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      ::mlir::func::ReturnOp op, ::mlir::func::ReturnOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    rewriter.modifyOpInPlace(
        op, [&]() { operation->setOperands(adaptor.getOperands()); });
    return success();
  }
};

template <>
class FxpOpConverter<CaseOp> : public OpConversionPattern<CaseOp> {
 public:
  FxpOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<CaseOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      CaseOp op, CaseOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return emitOptionalError(op->getLoc(),
                             "Should not have secret float based case");
  }
};

class ArithSelectConverter : public OpConversionPattern<arith::SelectOp> {
 public:
  ArithSelectConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<arith::SelectOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      arith::SelectOp op, arith::SelectOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, adaptor.getCondition(),
                                                 adaptor.getTrueValue(),
                                                 adaptor.getFalseValue());
    return success();
  }
};

class TensorDimConverter : public OpConversionPattern<tensor::DimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::DimOp op, tensor::DimOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, adaptor.getSource(),
                                               adaptor.getIndex());
    return success();
  }
};

class FuncOpConverter : public OpConversionPattern<::mlir::func::FuncOp> {
 public:
  FuncOpConverter(TypeConverter &type_converter, MLIRContext *context)
      : OpConversionPattern<::mlir::func::FuncOp>(type_converter, context) {}

  LogicalResult matchAndRewrite(
      ::mlir::func::FuncOp op, ::mlir::func::FuncOpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);

    auto functionType = op.getFunctionType();
    auto &region = op.getBody();

    // Convert non-entry blocks
    for (Block &block :
         llvm::make_early_inc_range(llvm::drop_begin(region, 1))) {
      TypeConverter::SignatureConversion conversion(
          /*numOrigInputs=*/block.getNumArguments());
      for (BlockArgument blockArgument : block.getArguments()) {
        auto idx = blockArgument.getArgNumber();
        auto convertedType =
            typeConverter->convertType(blockArgument.getType());

        conversion.addInputs(idx, convertedType);
      }
      rewriter.applySignatureConversion(&block, conversion, getTypeConverter());
    }

    // Convert function arguments using the provided TypeConverter.
    TypeConverter::SignatureConversion conversion(functionType.getNumInputs());
    for (const auto &blockarg : llvm::enumerate(op.getBody().getArguments())) {
      auto convertedType =
          typeConverter->convertType(blockarg.value().getType());
      conversion.addInputs(blockarg.index(), convertedType);
    }

    // If the SignatureConversion doesn't apply, bail out.
    if (failed(rewriter.convertRegionTypes(&region, *getTypeConverter(),
                                           &conversion))) {
      rewriter.cancelOpModification(op);
      return failure();
    }

    // Update the signature of the function.
    SmallVector<Type, 2> newResultTypes;
    if (failed(typeConverter->convertTypes(functionType.getResults(),
                                           newResultTypes))) {
      rewriter.cancelOpModification(op);
      return failure();
    }

    // Update return types
    op.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                        newResultTypes));
    rewriter.finalizeOpModification(op);

    return success();
  }
};

struct LowerSecretFloatToFxp
    : public LowerSecretFloatToFixedPointBase<LowerSecretFloatToFxp> {
 private:
  template <typename OpT>
  static void addPatterns(MLIRContext *context, SecretFloatConverter &converter,
                          RewritePatternSet &patterns) {
    patterns.insert<FxpOpConverter<OpT>>(converter, context);
  }

  template <typename OpT, typename OpT2, typename... OpTs>
  static void addPatterns(MLIRContext *context, SecretFloatConverter &converter,
                          RewritePatternSet &patterns) {
    addPatterns<OpT>(context, converter, patterns);
    addPatterns<OpT2, OpTs...>(context, converter, patterns);
  }

  void populateFixedpointConversionPattern(SecretFloatConverter &converter,
                                           RewritePatternSet &patterns) {
    auto *context = patterns.getContext();

    patterns.insert<FuncOpConverter, ArithSelectConverter, TensorDimConverter>(
        converter, context);

    addPatterns<
#define GET_OP_LIST
#include "libspu/dialect/pphlo/IR/ops.cc.inc"
        , mlir::func::ReturnOp>(context, converter, patterns);
  }

  FxpWidthConfig getFxpWidthConfig() const {
    return {f16_width_.getValue(), f16_fraction_bits_.getValue(),
            f32_width_.getValue(), f32_fraction_bits_.getValue(),
            f64_width_.getValue(), f64_fraction_bits_.getValue()};
  }

 public:
  LowerSecretFloatToFxp(const LowerSecretFloatToFxp &) = default;
  LowerSecretFloatToFxp() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    SecretFloatConverter converter(getFxpWidthConfig());

    target
        .addLegalDialect<PPHloDialect,
                         // Public compute may already in following dialects
                         mlir::arith::ArithDialect, mlir::math::MathDialect,
                         mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    target.addDynamicallyLegalOp<
#define GET_OP_LIST
#include "libspu/dialect/pphlo/IR/ops.cc.inc"
        , mlir::func::ReturnOp>(
        [&](Operation *op) { return converter.isLegal(op); });

    // arith select can work on secret when cond is a public.
    target.addDynamicallyLegalOp<arith::SelectOp, tensor::DimOp>(
        [&](Operation *op) { return converter.isLegal(op); });

    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });

    populateFixedpointConversionPattern(converter, patterns);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerSecretFloatToFxp() {
  return std::make_unique<LowerSecretFloatToFxp>();
}

}  // namespace mlir::spu::pphlo
