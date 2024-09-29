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

#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"
#include "libspu/dialect/pphlo/transforms/passes.h"

namespace mlir::spu::pphlo {

namespace {

/// Returns true if 'val' is a splat of zero, false otherwise.
// static bool isSplatZero(DenseElementsAttr val) {
//   auto type = val.getElementType();
//   if (llvm::isa<FloatType>(type)) {
//     return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
//   }
//   if (llvm::isa<IntegerType>(type)) {
//     return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
//   }
//   return false;
// }

// Pattern 1
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
                                PatternRewriter &rewrite) const override {
    // Pattern 1:
    auto pred = op.getPred();
    // Only do this for certain select...
    if (auto pref_a = pred.getDefiningOp<CustomCallOp>()) {
      if (pref_a.getCallTargetName() == PREFER_A) {
        // This select pred has already been optimized, bailout here
        return failure();
      }
    }

    // If this pred has only one use...do not rewrite, with mula1b is faster
    if (pred.hasOneUse()) {
      return failure();
    }

    auto number_of_selects = 0;
    for (auto &use : pred.getUses()) {
      if (mlir::isa<SelectOp>(use.getOwner())) {
        ++number_of_selects;
      }
    }

    // Although this value is used by multiple operations, there is still a
    // single select
    if (number_of_selects == 1) {
      return failure();
    }

    OpBuilder builder(op);
    // set insertion point
    auto new_loc = op->getLoc();
    if (mlir::isa<mlir::BlockArgument>(pred)) {
      builder.setInsertionPointToStart(op->getBlock());
    } else {
      builder.setInsertionPoint(pred.getDefiningOp()->getNextNode());
      new_loc = pred.getDefiningOp()->getLoc();
    }
    auto pref_a =
        builder.create<CustomCallOp>(new_loc, pred.getType(), pred, PREFER_A);

    // Only replace select usage
    pred.replaceUsesWithIf(pref_a->getResult(0), [](OpOperand &use) {
      return mlir::isa<SelectOp>(use.getOwner());
    });

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
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<SelectConversion>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeSelectPass() {
  return std::make_unique<OptimizeSelect>();
}

}  // namespace mlir::spu::pphlo
