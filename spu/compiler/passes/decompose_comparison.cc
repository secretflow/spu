// Copyright 2021 Ant Group Co., Ltd.
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

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spu/compiler/passes/pass_details.h"
#include "spu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

// != -> 1 - equal
// >= -> 1 - less
// <= -> 1 - greater

template <typename CompTy, typename LowerTy>
struct CompareOpConverter : public OpRewritePattern<CompTy> {
  explicit CompareOpConverter(MLIRContext *context)
      : OpRewritePattern<CompTy>(context) {}

  LogicalResult matchAndRewrite(CompTy op,
                                PatternRewriter &rewriter) const override {
    OpBuilder builder(op);

    auto eq_op =
        builder.create<LowerTy>(op.getLoc(), op.getType(), op.getOperands());

    rewriter.replaceOpWithNewOp<NotOp>(op, op.getType(), eq_op);

    return success();
  }
};

struct DecomposeComparison
    : public DecomposeComparisonBase<DecomposeComparison> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  void populateOwningPatterns(RewritePatternSet *patterns,
                              MLIRContext *ctx) const {
    patterns->insert<CompareOpConverter<NotEqualOp, EqualOp>,
                     CompareOpConverter<GreaterEqualOp, LessOp>,
                     CompareOpConverter<LessEqualOp, GreaterOp>>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeComparisonPass() {
  return std::make_unique<DecomposeComparison>();
}

} // namespace mlir::pphlo
