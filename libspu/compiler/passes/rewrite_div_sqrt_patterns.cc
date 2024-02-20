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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/dialect/pphlo/ops.h"

namespace mlir::spu::pphlo {

namespace {

struct DivRewriter : public OpRewritePattern<DivOp> {
private:
  Operation *rewriteSqrtIfPossible(PatternRewriter &rewriter,
                                   Operation *op) const {
    if (op == nullptr || op->getNumOperands() != 1) {
      return nullptr;
    }

    if (mlir::isa<SqrtOp>(op)) {
      return rewriter.create<RsqrtOp>(op->getLoc(), op->getResultTypes(),
                                      op->getOperand(0));
    }

    if (auto bcastOp = mlir::dyn_cast<BroadcastOp>(op)) {
      if (auto *inner = rewriteSqrtIfPossible(
              rewriter, bcastOp.getOperand().getDefiningOp())) {
        return rewriter.create<BroadcastOp>(
            op->getLoc(), bcastOp->getResultTypes(), inner->getResult(0),
            bcastOp.getBroadcastDimensions());
      }
      return nullptr;
    }

    return nullptr;
  }

public:
  explicit DivRewriter(MLIRContext *context)
      : OpRewritePattern<DivOp>(context) {}

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    // Pattern 1:
    // y/sqrt(x) -> y*rsqrt(x)
    auto denominator = op.getRhs();
    if (auto *newop =
            rewriteSqrtIfPossible(rewriter, denominator.getDefiningOp())) {
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getLhs(),
                                         newop->getResult(0));
      return success();
    } else {
      // Pattern 2:
      // y/(k*sqrt(x)) -> y/k*rsqrt(x)
      if (auto mulOp = denominator.getDefiningOp<MulOp>()) {
        Value k;
        Operation *rsqrt =
            rewriteSqrtIfPossible(rewriter, mulOp.getLhs().getDefiningOp());
        if (rsqrt != nullptr) {
          k = mulOp.getRhs();
        } else {
          k = mulOp.getLhs();
          rsqrt =
              rewriteSqrtIfPossible(rewriter, mulOp.getRhs().getDefiningOp());
        }

        // No 1/sqrt -> rsqrt rewrite, bailout
        if (rsqrt == nullptr) {
          return failure();
        }

        auto newDiv = rewriter.create<DivOp>(op.getLoc(), op->getResultTypes(),
                                             op.getLhs(), k);
        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newDiv,
                                           rsqrt->getResult(0));

        return success();
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

} // namespace mlir::spu::pphlo
