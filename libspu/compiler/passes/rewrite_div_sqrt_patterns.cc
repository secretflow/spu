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
    // y/sqrt(x)
    auto denominator = op.getRhs();
    if (auto sqrt = denominator.getDefiningOp<SqrtOp>()) {
      auto rsqrt = rewriter.create<RsqrtOp>(
          denominator.getLoc(), denominator.getType(), sqrt.getOperand());
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getLhs(), rsqrt);
      return success();
    }
    // Pattern 2:
    // y/(k*sqrt(x))
    if (auto mul = denominator.getDefiningOp<MulOp>()) {
      auto sqrt = mul.getLhs().getDefiningOp<SqrtOp>();
      auto k = mul.getRhs();
      // Try rhs
      if (sqrt == nullptr) {
        sqrt = mul.getRhs().getDefiningOp<SqrtOp>();
        k = mul.getLhs();
      }
      // Still not...bailout
      if (sqrt == nullptr) {
        return failure();
      }

      auto div =
          rewriter.create<DivOp>(op.getLoc(), op.getType(), op.getLhs(), k);
      auto rsqrt = rewriter.create<RsqrtOp>(sqrt->getLoc(), sqrt.getType(),
                                            sqrt->getOperands()[0]);
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), div, rsqrt);
      return success();
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
