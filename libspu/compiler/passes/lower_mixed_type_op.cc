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

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/compiler/passes/passes.h"
#include "libspu/dialect/pphlo_ops.h"
#include "libspu/dialect/pphlo_types.h"

namespace mlir::pphlo {

namespace {

// Idea here:
// %1 = convert(%0) int -> fxp
// %3 = mul/dot(%1, %2) fxp, fxp -> fxp
// Can be rewrite into
// %3 = mul/dot(%0, %2) int, fxp -> fxp // Save one truncation
template <typename OpT>
struct FxpIntMulTruncationRemover : public OpRewritePattern<OpT> {
public:
  explicit FxpIntMulTruncationRemover(MLIRContext *context)
      : OpRewritePattern<OpT>(context) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    TypeTools tools;

    if (!tools.getExpressedType(op.getType())
             .template isa<::mlir::FloatType>()) {
      return failure(); // Must be an op result in fp type
    }

    auto lhs_convert = lhs.template getDefiningOp<mlir::pphlo::ConvertOp>();
    auto rhs_convert = rhs.template getDefiningOp<mlir::pphlo::ConvertOp>();

    if ((lhs_convert != nullptr && rhs_convert == nullptr) ||
        (lhs_convert == nullptr && rhs_convert != nullptr)) {
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
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerMixedTypeOpPass() {
  return std::make_unique<LowerMixedTypeOp>();
}

} // namespace mlir::pphlo
