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

#include "spu/compiler/passes/pass_details.h"
#include "spu/compiler/passes/passes.h"
#include "spu/dialect/pphlo_ops.h"
#include "spu/dialect/pphlo_types.h"

namespace mlir::pphlo {

namespace {

// Idea here:
//   select(p, x, y)
// into
//   p' = prefer_a(p)
//   select(p', x, y)
// Rational:
// If the predicate is used by multiple select, explicit doing a to_a op can
// reduce the cost of to_a
struct SelectConversion : public OpRewritePattern<SelectOp> {
public:
  explicit SelectConversion(MLIRContext *context)
      : OpRewritePattern<SelectOp>(context) {}

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    // Only do this for certain select...
    if (op.pred().getDefiningOp<PreferAOp>() != nullptr) {
      // This select pred has already been optimized, bailout here
      return failure();
    }

    auto pref_a = rewriter.create<PreferAOp>(op->getLoc(), op.pred().getType(),
                                             op.pred());
    rewriter.replaceOpWithNewOp<SelectOp>(op, op->getResultTypes(), pref_a,
                                          op.on_true(), op.on_false());
    return success();
  }
};

struct OptimizeSelect : public OptimizeSelectBase<OptimizeSelect> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  void populateOwningPatterns(RewritePatternSet *patterns,
                              MLIRContext *ctx) const {
    patterns->insert<SelectConversion>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeSelectPass() {
  return std::make_unique<OptimizeSelect>();
}

} // namespace mlir::pphlo
