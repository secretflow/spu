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
#include "libspu/compiler/passes/passes.h"
#include "libspu/dialect/pphlo/ops.h"

namespace mlir::spu::pphlo {

namespace {

struct SortConversion : public OpRewritePattern<SortOp> {
public:
  explicit SortConversion(MLIRContext *context)
      : OpRewritePattern<SortOp>(context) {}

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    auto &comp = op.getComparator();
    if (op->getNumOperands() == 1) {
      // When there is only one operand, stable or not seems irrelevant
      op.setIsStable(false);
    }

    // If has a single instruction comparator, check if it's a simple sort.
    if (comp.hasOneBlock() &&
        llvm::hasSingleElement(comp.front().without_terminator())) {
      auto &inst = comp.front().front();
      // Single instruction comparator.
      if (mlir::isa<pphlo::LessOp>(inst) || mlir::isa<pphlo::GreaterOp>(inst)) {
        SortDirectionAttr direction;
        if (mlir::isa<pphlo::GreaterOp>(inst)) {
          // descent
          direction =
              SortDirectionAttr::get(op->getContext(), SortDirection::DES);
        } else {
          // ascent
          direction =
              SortDirectionAttr::get(op->getContext(), SortDirection::ASC);
        }
        auto lhs_idx =
            inst.getOperand(0).dyn_cast<mlir::BlockArgument>().getArgNumber();
        auto rhs_idx =
            inst.getOperand(1).dyn_cast<mlir::BlockArgument>().getArgNumber();
        // FIXME: If the comparator is using operands other than the first one,
        // we should just reorder operands instead of bailout
        if (lhs_idx != 0 || rhs_idx != 1) {
          return failure();
        }

        rewriter.replaceOpWithNewOp<pphlo::SimpleSortOp>(
            op, op.getResultTypes(), op.getOperands(), op.getDimensionAttr(),
            rewriter.getI64IntegerAttr(1), direction);
        return success();
      }
    }
    return failure();
  }
};

struct SortLowering : public SortLoweringBase<SortLowering> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<SortConversion>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSortLowering() {
  return std::make_unique<SortLowering>();
}

} // namespace mlir::spu::pphlo
