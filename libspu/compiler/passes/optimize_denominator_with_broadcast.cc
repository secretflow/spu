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
#include "libspu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

struct DivRewriter : public OpRewritePattern<DivOp> {
  explicit DivRewriter(MLIRContext *context)
      : OpRewritePattern<DivOp>(context) {}

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    auto denominator = op.getRhs();

    if (auto bcst = denominator.getDefiningOp<BroadcastOp>()) {
      // denominator is bcast
      // x/bcast(y) -> x*bcast(1/y)
      auto bcasted_v = bcst.getOperand();
      // 1/y
      auto reci_v = rewriter.create<ReciprocalOp>(
          op->getLoc(), bcasted_v.getType(), bcasted_v);
      // bcast(1/y)
      auto bcsted_reci_v = rewriter.create<BroadcastOp>(
          op->getLoc(), bcst.getType(), reci_v, bcst.getBroadcastDimensions());
      // x*bcast(1/y)
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getLhs(),
                                         bcsted_reci_v);
      return success();
    }
    return failure();
  }
};

struct OptimizeDenominatorWithBcast
    : public OptimizeDenominatorWithBcastBase<OptimizeDenominatorWithBcast> {
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

std::unique_ptr<OperationPass<func::FuncOp>>
createOptimizeDenominatorWithBroadcast() {
  return std::make_unique<OptimizeDenominatorWithBcast>();
}

} // namespace mlir::pphlo
