// Copyright 2023 Ant Group Co., Ltd.
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

#include <iostream>
#include <limits>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "spdlog/spdlog.h"

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

struct DivRewriter : public OpRewritePattern<DivOp> {
  explicit DivRewriter(MLIRContext *context)
      : OpRewritePattern<DivOp>(context) {}

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    // Pattern 1:
    // y/sqrt(x + eps)
    auto denominator = op.getRhs();
    if (auto sqrt = denominator.getDefiningOp<SqrtOp>()) {
      auto newRsqrt = rewriter.create<RsqrtOp>(
          denominator.getLoc(), denominator.getType(), sqrt.getOperand());
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getLhs(),
                                         newRsqrt);
      return success();
    } else {
      // Pattern 2:
      // y/(k*sqrt(x + eps)) -> y/k*rsqrt(x+eps)
      if (auto mulOp = denominator.getDefiningOp<MulOp>()) {
        auto sqrtOp = mulOp.getRhs().getDefiningOp<SqrtOp>();
        auto k = mulOp.getLhs();
        if (sqrtOp == nullptr) {
          sqrtOp = mulOp.getLhs().getDefiningOp<SqrtOp>();
          k = mulOp.getRhs();
        }
        if (sqrtOp) {
          // y/k
          auto newDiv = rewriter.create<DivOp>(
              op.getLoc(), op->getResultTypes(), op.getLhs(), k);
          // rsqrt(x+eps)
          auto newRsqrt = rewriter.create<RsqrtOp>(
              op->getLoc(), sqrtOp->getResultTypes(), sqrtOp->getOperand(0));
          // y/k*rsqrt(x+eps)
          rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newDiv,
                                             newRsqrt);
          return success();
        }
      }
    }
    return failure();
  }
};

struct RewriteDivSqrtPatterns
    : public RewriteDivSqrtPatternsBase<RewriteDivSqrtPatterns> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<DivRewriter>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRewriteDivSqrtPatterns() {
  return std::make_unique<RewriteDivSqrtPatterns>();
}

} // namespace mlir::pphlo
