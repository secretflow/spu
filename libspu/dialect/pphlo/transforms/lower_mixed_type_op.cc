// Copyright 2022 Ant Group Co., Ltd.
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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"
#include "libspu/dialect/pphlo/transforms/passes.h"

namespace mlir::spu::pphlo {

namespace {

// Idea here:
// %1 = convert(%0) int -> fxp
// %3 = mul/dot(%1, %2) fxp, fxp -> fxp
// Can be rewrite into
// %3 = mul/dot(%0, %2) int, fxp -> fxp // Save one truncation
template <typename OpT>
struct FxpIntMulTruncationRemover : public OpRewritePattern<OpT> {
 private:
  TypeTools typetools_;

  bool isLegitConvert(ConvertOp op) const {
    if (op == nullptr) {
      return true;
    }

    // Only int->fxp conversion is considered legit
    return typetools_.isFloatType(op.getType()) &&
           typetools_.isIntType(op.getOperand().getType());
  }

 public:
  explicit FxpIntMulTruncationRemover(MLIRContext *context)
      : OpRewritePattern<OpT>(context), typetools_(context) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    if (!typetools_.isFloatType(op.getType())) {
      return failure();  // Must be an op result in fp type
    }

    auto lhs_convert = lhs.template getDefiningOp<ConvertOp>();
    auto rhs_convert = rhs.template getDefiningOp<ConvertOp>();

    if (((lhs_convert != nullptr && rhs_convert == nullptr) ||
         (lhs_convert == nullptr && rhs_convert != nullptr)) &&
        (isLegitConvert(lhs_convert) && isLegitConvert(rhs_convert))) {
      llvm::SmallVector<mlir::Value, 2> operands(2);
      operands[0] = lhs_convert == nullptr ? lhs : lhs_convert.getOperand();
      operands[1] = rhs_convert == nullptr ? rhs : rhs_convert.getOperand();
      rewriter.replaceOpWithNewOp<OpT>(op, op.getType(), operands,
                                       op->getAttrs());
      return success();
    }
    return failure();
  }
};

struct LowerMixedTypeOp : public LowerMixedTypeOpBase<LowerMixedTypeOp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<FxpIntMulTruncationRemover<MulOp>,
                     FxpIntMulTruncationRemover<DotOp>,
                     FxpIntMulTruncationRemover<DotGeneralOp>>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerMixedTypeOpPass() {
  return std::make_unique<LowerMixedTypeOp>();
}

}  // namespace mlir::spu::pphlo
