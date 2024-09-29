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

#include "libspu/core/half.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/type_converter.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"
#include "libspu/dialect/utils/lowering_intrinsic.h"

namespace mlir::spu::pphlo {
namespace {

template <typename OP>
struct FxpOpConverter : public OpConversionPattern<OP> {
  using OpConversionPattern<OP>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OP op, typename FxpOpConverter::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> result_types(op->getNumResults());
    for (size_t idx = 0; idx < op->getNumResults(); ++idx) {
      result_types[idx] =
          this->typeConverter->convertType(op->getResultTypes()[idx]);
    }

    auto new_op = rewriter.create<OP>(op.getLoc(), result_types,
                                      adaptor.getOperands(), op->getAttrs());

    for (int64_t idx = 0; idx < op->getNumRegions(); ++idx) {
      // Copy over the operations inside the region.
      rewriter.inlineRegionBefore(op->getRegion(idx), new_op->getRegion(idx),
                                  new_op->getRegion(idx).end());

      if (failed(rewriter.convertRegionTypes(&new_op->getRegion(idx),
                                             *this->getTypeConverter()))) {
        return failure();
      }
    }

    rewriter.replaceOp(op, new_op);

    return success();
  }
};

template <>
struct FxpOpConverter<CaseOp> : public OpConversionPattern<CaseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CaseOp op, CaseOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return emitOptionalError(op->getLoc(), "Should not see pphlo.case here");
  }
};

template <>
struct FxpOpConverter<EpsilonOp> : public OpConversionPattern<EpsilonOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EpsilonOp op, EpsilonOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto el_type = getElementTypeOrSelf(op.getType());

    auto result_type = typeConverter->convertType(op.getType());

    TypeTools tools(op->getContext());

    auto fxp_bits = tools.getFxpBits(result_type);

    double eps = 1.0 / std::pow(2, fxp_bits);

    FloatAttr eps_attr;

    // Add a real epsilon, if real eps < fxp eps, this eps will be dropped.
    switch (el_type.getIntOrFloatBitWidth()) {
      case 16: {
        eps += std::numeric_limits<half_float::half>::epsilon();
        eps_attr = rewriter.getF16FloatAttr(eps);
        break;
      }
      case 32: {
        eps += std::numeric_limits<float>::epsilon();
        eps_attr = rewriter.getF32FloatAttr(eps);
        break;
      }
      case 64: {
        eps += std::numeric_limits<double>::epsilon();
        eps_attr = rewriter.getF64FloatAttr(eps);
        break;
      }
      default: {
        return emitOptionalError(op->getLoc(), "Unhandled float type");
      }
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op,
        SplatElementsAttr::get(mlir::cast<ShapedType>(op.getType()), eps_attr));

    return success();
  }
};

struct ReturnConverter : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  static bool isLegal(func::ReturnOp op) {
    for (auto operand : op->getOperands()) {
      if (mlir::isa<FloatType>(getElementTypeOrSelf(operand.getType()))) {
        if (auto *parent = operand.getDefiningOp()) {
          if (mlir::isa<PPHloDialect>(parent->getDialect())) {
            if (auto callOp = mlir::dyn_cast<pphlo::CustomCallOp>(parent)) {
              if (callOp.getAllowFloat()) {
                continue;
              }
            }
            return false;
          }
        }
      }
    }
    return true;
  }

  LogicalResult matchAndRewrite(
      func::ReturnOp op, func::ReturnOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());

    auto loc = op->getLoc();
    llvm::SmallVector<Value> operands(op->getNumOperands());

    for (auto ins : llvm::enumerate(adaptor.getOperands())) {
      auto idx = ins.index();
      if (tools.isFixedPointType(ins.value().getType())) {
        operands[idx] = rewriter
                            .create<pphlo::CustomCallOp>(
                                loc, op->getOperandTypes()[idx], ins.value(),
                                DECODE_FROM_FXP, false, true)
                            ->getResult(0);
      } else {
        operands[idx] = adaptor.getOperands()[idx];
      }
    }

    rewriter.modifyOpInPlace(
        op, [&]() { op.getOperation()->setOperands(operands); });

    return success();
  }
};

struct LowerPPHloFloatInputs
    : public LowerPPHloFloatInputsBase<LowerPPHloFloatInputs> {
 private:
  template <typename OpT>
  static void addPatterns(MLIRContext *context, FloatConverter &converter,
                          RewritePatternSet &patterns) {
    patterns.insert<FxpOpConverter<OpT>>(converter, context);
  }

  template <typename OpT, typename OpT2, typename... OpTs>
  static void addPatterns(MLIRContext *context, FloatConverter &converter,
                          RewritePatternSet &patterns) {
    addPatterns<OpT>(context, converter, patterns);
    addPatterns<OpT2, OpTs...>(context, converter, patterns);
  }

  void populateFixedpointConversionPattern(FloatConverter &converter,
                                           RewritePatternSet &patterns) {
    auto *context = patterns.getContext();

    addPatterns<
#define GET_OP_LIST
#include "libspu/dialect/pphlo/IR/ops.cc.inc"
        >(context, converter, patterns);

    patterns.insert<ReturnConverter>(converter, context);
  }

  FxpWidthConfig getFxpWidthConfig() const {
    return {f16_width_.getValue(), f16_fraction_bits_.getValue(),
            f32_width_.getValue(), f32_fraction_bits_.getValue(),
            f64_width_.getValue(), f64_fraction_bits_.getValue()};
  }

 public:
  LowerPPHloFloatInputs(const LowerPPHloFloatInputs &) = default;
  LowerPPHloFloatInputs() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    FloatConverter converter(getFxpWidthConfig());

    target
        .addLegalDialect<PPHloDialect, mlir::arith::ArithDialect,
                         mlir::math::MathDialect, mlir::tensor::TensorDialect,
                         mlir::scf::SCFDialect, mlir::linalg::LinalgDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp>();

    target.addDynamicallyLegalOp<
#define GET_OP_LIST
#include "libspu/dialect/pphlo/IR/ops.cc.inc"
        >([&](Operation *op) { return converter.isLegal(op); });

    target.addDynamicallyLegalOp<CustomCallOp>([&](CustomCallOp op) {
      return op.getAllowFloat() || converter.isLegal(op);
    });

    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return ReturnConverter::isLegal(op); });

    populateFixedpointConversionPattern(converter, patterns);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerPPHloFloatInputs() {
  return std::make_unique<LowerPPHloFloatInputs>();
}

}  // namespace mlir::spu::pphlo
