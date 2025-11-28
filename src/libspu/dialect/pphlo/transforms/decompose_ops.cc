// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {

namespace {

#include "libspu/dialect/pphlo/transforms/decompose_patterns.cc.inc"

// https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add
// Boolean add is logical or
struct BooleanAddDecompose : public OpRewritePattern<AddOp> {
 private:
  TypeTools tool_;

 public:
  explicit BooleanAddDecompose(MLIRContext *context)
      : OpRewritePattern<AddOp>(context), tool_(context) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto el_type = mlir::dyn_cast<IntegerType>(tool_.getBaseType(op.getType()));

    if (!el_type || el_type.getWidth() > 1) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), op.getLhs(),
                                      op.getRhs());

    return success();
  }
};

// Rewrites `pphlo.case` to a nested `pphlo.if`.
struct PublicCaseToNestedIf : public OpRewritePattern<pphlo::CaseOp> {
  using OpRewritePattern<pphlo::CaseOp>::OpRewritePattern;

  void inlinePPhloRegionIntoNewRegion(PatternRewriter &rewriter, Region &region,
                                      Region &ring) const {
    // Remove an existing block, then move the region over.
    if (!ring.empty()) {
      rewriter.eraseBlock(&ring.back());
    }
    rewriter.inlineRegionBefore(region, ring, ring.end());
  }

  // Recursively create if/else ops to handle each possible value in a case op.
  pphlo::IfOp createNestedCases(int currentIdx, pphlo::CaseOp op,
                                PatternRewriter &outerBuilder) const {
    Location loc = op.getLoc();
    Value idxValue = op.getIndex();
    auto finalIdx = op.getBranches().size() - 2;
    // Determine if the current index matches the case index.
    auto scalarType = idxValue.getType();
    auto shapedType = mlir::cast<ShapedType>(scalarType);
    auto constAttr = DenseElementsAttr::get(
        shapedType, {mlir::cast<mlir::Attribute>(
                        outerBuilder.getI32IntegerAttr(currentIdx))});
    Value currentIdxVal =
        outerBuilder.create<pphlo::ConstantOp>(loc, constAttr);
    auto pphloIf = outerBuilder.create<pphlo::IfOp>(
        loc, op.getResultTypes(),
        outerBuilder.create<pphlo::EqualOp>(loc, idxValue, currentIdxVal));
    inlinePPhloRegionIntoNewRegion(outerBuilder, op.getBranches()[currentIdx],
                                   pphloIf.getTrueBranch());
    int nextIdx = currentIdx + 1;
    // Don't recurse for the final default block.
    if (currentIdx == static_cast<int64_t>(finalIdx)) {
      inlinePPhloRegionIntoNewRegion(outerBuilder, op.getBranches()[nextIdx],
                                     pphloIf.getFalseBranch());
    } else {
      PatternRewriter::InsertionGuard guard(outerBuilder);
      outerBuilder.setInsertionPointToEnd(&pphloIf.getFalseBranch().back());
      auto innerIf = createNestedCases(nextIdx, op, outerBuilder);
      outerBuilder.create<pphlo::ReturnOp>(op.getLoc(), innerIf.getResults());
    }
    return pphloIf;
  }

  LogicalResult matchAndRewrite(pphlo::CaseOp op,
                                PatternRewriter &rewriter) const override {
    // Inline the op if there is only a default block.
    if (op.getBranches().size() == 1) {
      Block &block = op.getBranches().front().front();
      auto results = block.getTerminator()->getOperands();
      // Remove the mhlo.return terminator, then inline the block.
      rewriter.eraseOp(block.getTerminator());
      rewriter.inlineBlockBefore(/*source=*/&block, /*dest=*/op.getOperation(),
                                 /*argValues=*/{});
      rewriter.replaceOp(op, results);
      return success();
    }

    TypeTools tools(op->getContext());
    if (tools.isSecretType(op.getIndex().getType())) {
      // Leave it to secret cf inline
      return failure();
    }
    // Begin recursion with case 0.
    rewriter.replaceOp(op, createNestedCases(0, op, rewriter).getResults());
    return success();
  }
};

struct DecomposeOps : public DecomposeOpsBase<DecomposeOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    GreedyRewriteConfig config;
    config.enableFolding();
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    populateWithGenerated(*patterns);
    patterns->add<BooleanAddDecompose, PublicCaseToNestedIf>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeOps() {
  return std::make_unique<DecomposeOps>();
}

}  // namespace mlir::spu::pphlo
