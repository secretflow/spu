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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

struct CastConverter : public OpRewritePattern<UnrealizedConversionCastOp> {
  explicit CastConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder builder(op);

    auto to_type = op.getResultTypes()[0];

    rewriter.replaceOpWithNewOp<ConvertOp>(op, to_type, op->getOperands());

    return success();
  }
};

struct LowerConversionCast
    : public LowerConversionCastBase<LowerConversionCast> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<CastConverter>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerConversionCastPass() {
  return std::make_unique<LowerConversionCast>();
}

} // namespace mlir::pphlo
