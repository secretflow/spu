// Copyright 2025 Ant Group Co., Ltd.
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "libspu/compiler/transforms/stablehlo/pass_details.h"

namespace mlir::spu::stablehlo {

namespace {

// Base class for expanding complex binary operations
template <typename OpTy>
struct ExpandComplexBinaryOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    // Check if operands are complex tensors
    auto lhsType = mlir::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<RankedTensorType>(rhs.getType());

    if (!lhsType || !rhsType ||
        !mlir::isa<ComplexType>(lhsType.getElementType()) ||
        !mlir::isa<ComplexType>(rhsType.getElementType())) {
      return failure();
    }

    // Extract real and imaginary parts
    auto realLhs = rewriter.create<mlir::stablehlo::RealOp>(op.getLoc(), lhs);
    auto imagLhs = rewriter.create<mlir::stablehlo::ImagOp>(op.getLoc(), lhs);
    auto realRhs = rewriter.create<mlir::stablehlo::RealOp>(op.getLoc(), rhs);
    auto imagRhs = rewriter.create<mlir::stablehlo::ImagOp>(op.getLoc(), rhs);

    return rewriteComplex(op, rewriter, realLhs, imagLhs, realRhs, imagRhs);
  }

  virtual LogicalResult rewriteComplex(OpTy op, PatternRewriter &rewriter,
                                       Value realLhs, Value imagLhs,
                                       Value realRhs, Value imagRhs) const = 0;
};

// Pattern to expand complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
struct ExpandComplexAdd : public ExpandComplexBinaryOp<mlir::stablehlo::AddOp> {
  using ExpandComplexBinaryOp<mlir::stablehlo::AddOp>::ExpandComplexBinaryOp;

  LogicalResult rewriteComplex(mlir::stablehlo::AddOp op,
                               PatternRewriter &rewriter, Value realLhs,
                               Value imagLhs, Value realRhs,
                               Value imagRhs) const override {
    // Add real parts and imaginary parts separately
    auto realSum =
        rewriter.create<mlir::stablehlo::AddOp>(op.getLoc(), realLhs, realRhs);
    auto imagSum =
        rewriter.create<mlir::stablehlo::AddOp>(op.getLoc(), imagLhs, imagRhs);

    // Create complex result as the final representation before dialect
    // conversion
    auto complexResult = rewriter.create<mlir::stablehlo::ComplexOp>(
        op.getLoc(), op.getType(), realSum, imagSum);

    rewriter.replaceOp(op, complexResult);
    return success();
  }
};

// Pattern to expand complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
struct ExpandComplexSub
    : public ExpandComplexBinaryOp<mlir::stablehlo::SubtractOp> {
  using ExpandComplexBinaryOp<
      mlir::stablehlo::SubtractOp>::ExpandComplexBinaryOp;

  LogicalResult rewriteComplex(mlir::stablehlo::SubtractOp op,
                               PatternRewriter &rewriter, Value realLhs,
                               Value imagLhs, Value realRhs,
                               Value imagRhs) const override {
    // Subtract real parts and imaginary parts separately
    auto realDiff = rewriter.create<mlir::stablehlo::SubtractOp>(
        op.getLoc(), realLhs, realRhs);
    auto imagDiff = rewriter.create<mlir::stablehlo::SubtractOp>(
        op.getLoc(), imagLhs, imagRhs);

    // Create complex result - this will be handled by the StableHLO complex
    // expander
    auto complexResult = rewriter.create<mlir::stablehlo::ComplexOp>(
        op.getLoc(), op.getType(), realDiff, imagDiff);

    rewriter.replaceOp(op, complexResult);
    return success();
  }
};

// Pattern to expand complex multiplication: (a+bi) * (c+di) = (ac-bd) +
// (ad+bc)i
struct ExpandComplexMul : public ExpandComplexBinaryOp<mlir::stablehlo::MulOp> {
  using ExpandComplexBinaryOp<mlir::stablehlo::MulOp>::ExpandComplexBinaryOp;

  LogicalResult rewriteComplex(mlir::stablehlo::MulOp op,
                               PatternRewriter &rewriter, Value realLhs,
                               Value imagLhs, Value realRhs,
                               Value imagRhs) const override {
    // Compute real part: ac - bd
    auto ac =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), realLhs, realRhs);
    auto bd =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), imagLhs, imagRhs);
    auto realPart =
        rewriter.create<mlir::stablehlo::SubtractOp>(op.getLoc(), ac, bd);

    // Compute imaginary part: ad + bc
    auto ad =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), realLhs, imagRhs);
    auto bc =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), imagLhs, realRhs);
    auto imagPart =
        rewriter.create<mlir::stablehlo::AddOp>(op.getLoc(), ad, bc);

    // Create complex result - this will be handled by the StableHLO complex
    // expander
    auto complexResult = rewriter.create<mlir::stablehlo::ComplexOp>(
        op.getLoc(), op.getType(), realPart, imagPart);

    rewriter.replaceOp(op, complexResult);
    return success();
  }
};

// Pattern to expand complex division: (a+bi) / (c+di) = (ac+bd)/(c²+d²) +
// (bc-ad)/(c²+d²)i
struct ExpandComplexDiv : public ExpandComplexBinaryOp<mlir::stablehlo::DivOp> {
  using ExpandComplexBinaryOp<mlir::stablehlo::DivOp>::ExpandComplexBinaryOp;

  LogicalResult rewriteComplex(mlir::stablehlo::DivOp op,
                               PatternRewriter &rewriter, Value realLhs,
                               Value imagLhs, Value realRhs,
                               Value imagRhs) const override {
    // Compute denominator: c² + d²
    auto cSquared =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), realRhs, realRhs);
    auto dSquared =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), imagRhs, imagRhs);
    auto denominator = rewriter.create<mlir::stablehlo::AddOp>(
        op.getLoc(), cSquared, dSquared);

    // Compute real part: (ac + bd) / (c² + d²)
    auto ac =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), realLhs, realRhs);
    auto bd =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), imagLhs, imagRhs);
    auto acPlusBd =
        rewriter.create<mlir::stablehlo::AddOp>(op.getLoc(), ac, bd);
    auto realPart = rewriter.create<mlir::stablehlo::DivOp>(
        op.getLoc(), acPlusBd, denominator);

    // Compute imaginary part: (bc - ad) / (c² + d²)
    auto bc =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), imagLhs, realRhs);
    auto ad =
        rewriter.create<mlir::stablehlo::MulOp>(op.getLoc(), realLhs, imagRhs);
    auto bcMinusAd =
        rewriter.create<mlir::stablehlo::SubtractOp>(op.getLoc(), bc, ad);
    auto imagPart = rewriter.create<mlir::stablehlo::DivOp>(
        op.getLoc(), bcMinusAd, denominator);

    // Create complex result - this will be handled by the StableHLO complex
    // expander
    auto complexResult = rewriter.create<mlir::stablehlo::ComplexOp>(
        op.getLoc(), op.getType(), realPart, imagPart);

    rewriter.replaceOp(op, complexResult);
    return success();
  }
};

}  // namespace

// Custom pass that applies complex expansion patterns
struct ExpandComplexOpsPass : public Pass {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandComplexOpsPass)

  ExpandComplexOpsPass() : Pass(mlir::TypeID::get<ExpandComplexOpsPass>()) {}

  StringRef getName() const override { return "spu-expand-complex-ops"; }

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<ExpandComplexOpsPass>();
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    return opName.getStringRef() == "func.func";
  }

  void runOnOperation() override {
    auto func = cast<mlir::func::FuncOp>(getOperation());
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandComplexAdd, ExpandComplexSub, ExpandComplexMul,
                 ExpandComplexDiv>(&getContext());

    GreedyRewriteConfig config;
    config.enableFolding();
    (void)applyPatternsGreedily(func, std::move(patterns), config);
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::stablehlo::StablehloDialect>();
  }
};

std::unique_ptr<Pass> createExpandComplexOpsPass() {
  return std::make_unique<ExpandComplexOpsPass>();
}

}  // namespace mlir::spu::stablehlo