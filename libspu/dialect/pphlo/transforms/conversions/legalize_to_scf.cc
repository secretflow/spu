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

// Base mlir headers
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// depending dialects
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "libspu/device/intrinsic_table.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {
namespace {

Value extractTensorValue(OpBuilder &b, Value tensor) {
  auto loc = tensor.getLoc();
  if (mlir::cast<TensorType>(tensor.getType()).hasRank() &&
      mlir::cast<TensorType>(tensor.getType()).getRank() != 0) {
    tensor = b.create<tensor::CollapseShapeOp>(
        loc, tensor, SmallVector<ReassociationIndices>());
  }
  return b.create<tensor::ExtractOp>(loc, tensor, ValueRange());
}

void inlineRegionIntoSCFRegion(PatternRewriter &rewriter, Region &ring,
                               Region &scf) {
  // Remove an existing block, then move the region over.
  if (!scf.empty()) {
    rewriter.eraseBlock(&scf.back());
  }
  rewriter.inlineRegionBefore(ring, scf, scf.end());
  // Fix up the terminator.
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&scf.back());
  auto *terminator = scf.back().getTerminator();
  rewriter.replaceOpWithNewOp<scf::YieldOp>(terminator,
                                            terminator->getOperands());
}

// Rewrites `pphlo.if` to `scf.if`.
struct IfConverter : public OpRewritePattern<pphlo::IfOp> {
  using OpRewritePattern<pphlo::IfOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(pphlo::IfOp op,
                                PatternRewriter &rewriter) const override {
    auto scfIf = rewriter.create<scf::IfOp>(
        op.getLoc(), op.getResultTypes(),
        extractTensorValue(rewriter, op.getCondition()),
        /*withElseRegion=*/true);
    inlineRegionIntoSCFRegion(rewriter, op.getTrueBranch(),
                              scfIf.getThenRegion());
    inlineRegionIntoSCFRegion(rewriter, op.getFalseBranch(),
                              scfIf.getElseRegion());
    rewriter.replaceOp(op, scfIf.getResults());
    return success();
  }
};

struct ScfForBounds {
  Value lb;
  Value ub;
  Value step;
  unsigned indexArgIndex;
};

std::optional<ScfForBounds> extractForBounds(const TypeTools &tools,
                                             pphlo::WhileOp op) {
  auto &cond = op.getCond().front();
  auto &body = op.getBody().front();
  if (cond.getOperations().size() != 2) {
    return std::nullopt;
  }

  auto matchBbArg = [](Value v, Block &block) -> std::optional<unsigned> {
    if (!mlir::isa<BlockArgument>(v) || v.getParentBlock() != &block) {
      return std::nullopt;
    }
    return mlir::cast<BlockArgument>(v).getArgNumber();
  };

  auto compare = llvm::dyn_cast<pphlo::LessOp>(cond.front());
  // If the rhs of the comapare is defined outside the block, it's a constant
  // within the loop.
  if (!compare || compare.getRhs().getParentBlock() == &cond ||
      !getElementTypeOrSelf(compare.getLhs().getType())
           .isSignlessIntOrIndex() ||
      !tools.isPublicType(compare.getType())) {
    return std::nullopt;
  }

  auto iterArg = matchBbArg(compare.getLhs(), cond);
  if (!iterArg) {
    return std::nullopt;
  }

  auto add = llvm::dyn_cast_or_null<pphlo::AddOp>(
      body.getTerminator()->getOperand(*iterArg).getDefiningOp());
  if (!add || matchBbArg(add.getLhs(), body) != iterArg ||
      add.getRhs().getParentBlock() == &body ||
      tools.isPublicType(add.getType())) {
    return std::nullopt;
  }

  ScfForBounds bounds;
  bounds.ub = compare.getRhs();
  bounds.step = add.getRhs();
  bounds.lb = op->getOperand(*iterArg);
  bounds.indexArgIndex = *iterArg;
  return bounds;
}

// Rewrites `pphlo.while` to `scf.while`.
struct WhileConverter : public OpRewritePattern<pphlo::WhileOp> {
  using OpRewritePattern<pphlo::WhileOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(pphlo::WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    TypeTools tools(op->getContext());

    if (auto bounds = extractForBounds(tools, op)) {
      auto newForOp = rewriter.create<scf::ForOp>(
          loc, extractTensorValue(rewriter, bounds->lb),
          extractTensorValue(rewriter, bounds->ub),
          extractTensorValue(rewriter, bounds->step), op.getOperands());

      rewriter.setInsertionPointToEnd(newForOp.getBody());
      // Inline while body, and only replace the mhlo.return with an scf.yield.
      inlineRegionIntoSCFRegion(rewriter, op.getBody(), newForOp.getRegion());
      auto indexArg = newForOp.getRegion().insertArgument(
          unsigned{0}, newForOp.getLowerBound().getType(), loc);
      auto oldIndexArg =
          newForOp.getRegion().getArgument(1 + bounds->indexArgIndex);
      rewriter.setInsertionPointToStart(&newForOp.getRegion().front());
      auto indexArgTensor = rewriter.create<tensor::FromElementsOp>(
          loc, oldIndexArg.getType(), indexArg);
      oldIndexArg.replaceAllUsesWith(indexArgTensor);

      rewriter.replaceOp(op, newForOp.getResults());
      return success();
    }

    auto newWhileOp = rewriter.create<scf::WhileOp>(loc, op.getResultTypes(),
                                                    op.getOperands());

    // Inline while condition. The block is the same, except the boolean result
    // needs to be extracted and used with an scf.condition.
    rewriter.inlineRegionBefore(op.getCond(), newWhileOp.getBefore(),
                                newWhileOp.getBefore().end());
    auto conditionReturn =
        cast<pphlo::ReturnOp>(newWhileOp.getBefore().front().getTerminator());
    rewriter.setInsertionPointToEnd(&newWhileOp.getBefore().front());

    auto cond_return = conditionReturn->getOperand(0);

    if (!tools.isPublicType(cond_return.getType())) {
      cond_return =
          rewriter
              .create<pphlo::CustomCallOp>(
                  loc, TypeRange{tools.getExpressedType(cond_return.getType())},
                  ValueRange{cond_return}, TRY_REVEAL_COND)
              ->getResult(0);
    }

    Value i1 = extractTensorValue(rewriter, cond_return);
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        conditionReturn, i1, newWhileOp.getBeforeArguments());

    // Inline while body, and only replace the mhlo.return with an scf.yield.
    inlineRegionIntoSCFRegion(rewriter, op.getBody(), newWhileOp.getAfter());

    rewriter.replaceOp(op, newWhileOp.getResults());
    return success();
  }
};

// Rewrites `pphlo.case` to a nested `scf.if`.
struct CaseConverter : public OpRewritePattern<pphlo::CaseOp> {
  using OpRewritePattern<pphlo::CaseOp>::OpRewritePattern;

  // Recursively create if/else ops to handle each possible value in a case op.
  scf::IfOp createNestedCases(int currentIdx, pphlo::CaseOp op,
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
    Value currentIdxVal = outerBuilder.create<arith::ConstantOp>(
        loc, idxValue.getType(), constAttr);
    auto scfIf = outerBuilder.create<scf::IfOp>(
        loc, op.getResultTypes(),
        extractTensorValue(outerBuilder, outerBuilder.create<arith::CmpIOp>(
                                             loc, arith::CmpIPredicate::eq,
                                             idxValue, currentIdxVal)));
    inlineRegionIntoSCFRegion(outerBuilder, op.getBranches()[currentIdx],
                              scfIf.getThenRegion());
    int nextIdx = currentIdx + 1;
    // Don't recurse for the final default block.
    if (currentIdx == static_cast<int64_t>(finalIdx)) {
      inlineRegionIntoSCFRegion(outerBuilder, op.getBranches()[nextIdx],
                                scfIf.getElseRegion());
    } else {
      PatternRewriter::InsertionGuard guard(outerBuilder);
      outerBuilder.setInsertionPointToEnd(&scfIf.getElseRegion().back());
      auto innerIf = createNestedCases(nextIdx, op, outerBuilder);
      outerBuilder.create<scf::YieldOp>(op.getLoc(), innerIf.getResults());
    }
    return scfIf;
  }

  LogicalResult matchAndRewrite(pphlo::CaseOp op,
                                PatternRewriter &rewriter) const override {
    // Inline the op if there is only a default block.
    if (op.getBranches().size() == 1) {
      Block &block = op.getBranches().front().front();
      auto results = block.getTerminator()->getOperands();
      // Remove the pphlo.return terminator, then inline the block.
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

struct LegalizeToSCF : public LegalizeToSCFBase<LegalizeToSCF> {
 private:
  void populateRewritePatterns(RewritePatternSet &patterns) {
    auto *context = patterns.getContext();

    // Controlflow
    patterns.insert<IfConverter, WhileConverter, CaseConverter>(context);
  }

 public:
  LegalizeToSCF(const LegalizeToSCF &) = default;
  LegalizeToSCF() = default;

  void runOnOperation() override {
    auto &context = getContext();

    RewritePatternSet patterns(&context);

    populateRewritePatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToSCF() {
  return std::make_unique<LegalizeToSCF>();
}

}  // namespace mlir::spu::pphlo
