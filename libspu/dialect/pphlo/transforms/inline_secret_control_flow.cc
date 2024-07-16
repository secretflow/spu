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
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

namespace mlir::spu::pphlo {

namespace {

class CaseConverter : public OpRewritePattern<CaseOp> {
 private:
  TypeTools tools_;

  Value reshapeOrBroadcast(PatternRewriter &rewriter, Location loc, Value in,
                           RankedTensorType target_type) const {
    auto in_type = mlir::dyn_cast<RankedTensorType>(in.getType());
    auto broadcasted_mask_type =
        RankedTensorType::get(target_type.getShape(), in_type.getElementType());
    if (target_type.getNumElements() == in_type.getNumElements()) {
      return rewriter.create<ReshapeOp>(loc, broadcasted_mask_type, in);
    } else {
      return rewriter.create<BroadcastOp>(
          loc, broadcasted_mask_type, in,
          llvm::SmallVector<int64_t>(target_type.getRank(), 0));
    }
  }

  // Basic algorithm here:
  // %out = case(%idx) {
  //  b0^ { yield r0 }
  //  b1^ { yield r1 }
  //  ...
  //  bn^ { yield rn }
  // }
  // r0, r1, r2, ..., rn represent results of each case region,
  // %out represents results of branch, where branch id == %idx
  // 1. Compute all branches and get r0...rn
  // 2. Generate a mask m = equal(%idx, [0, n]), where only branch id == %idx
  // should be one
  // 3. Compute mr0 = m[0]*r0, mr1 = m[1]*r1, ..., mrn = m[n]*rn
  // 4. Accumulate mrs, %out = sum(mr0, mr1, ..., mrn)
  void inlineRegionIntoParent(CaseOp &op, PatternRewriter &rewriter) const {
    auto *blockBeforeCase = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();
    auto *blockAfterCase = rewriter.splitBlock(blockBeforeCase, initPosition);

    // Collect all blocks
    llvm::SmallVector<Block *> blocks_to_work;
    for (auto &r : op->getRegions()) {
      blocks_to_work.emplace_back(&r.front());
      rewriter.inlineRegionBefore(r, blockAfterCase);
    }

    Value index = op.getIndex();
    int64_t num_cases = op->getNumRegions();
    auto index_type = op.getIndex().getType();
    auto index_base = tools_.getBaseType(index_type);

    // Clamp index to [0, num_cases)
    auto zero_const = rewriter.create<ConstantOp>(
        op->getLoc(),
        rewriter.getZeroAttr(RankedTensorType::get({}, index_base)));
    auto num_cases_const = rewriter.create<ConstantOp>(
        op->getLoc(),
        DenseIntElementsAttr::get(RankedTensorType::get({}, index_base),
                                  static_cast<int32_t>(num_cases - 1)));
    index = rewriter.create<ClampOp>(op->getLoc(), index_type, zero_const,
                                     index, num_cases_const);

    // Reconnect all results.
    // build mask
    auto iota = rewriter.create<IotaOp>(
        op->getLoc(), RankedTensorType::get({num_cases}, index_base), 0);
    auto index_reshaped = rewriter.create<ReshapeOp>(
        op->getLoc(),
        RankedTensorType::get({1}, getElementTypeOrSelf(index_type)), index);
    auto index_brocasted = rewriter.create<BroadcastOp>(
        op->getLoc(),
        RankedTensorType::get({num_cases}, getElementTypeOrSelf(index_type)),
        index_reshaped, llvm::ArrayRef<int64_t>{0});
    auto masks = rewriter.create<EqualOp>(op->getLoc(), iota, index_brocasted);

    llvm::SmallVector<Value> result_masks;
    auto mask_slice_type =
        RankedTensorType::get({1}, getElementTypeOrSelf(masks));
    for (int64_t region_id = 0; region_id < op.getNumRegions(); ++region_id) {
      auto m = rewriter.create<SliceOp>(op->getLoc(), mask_slice_type, masks,
                                        llvm::ArrayRef<int64_t>{region_id},
                                        llvm::ArrayRef<int64_t>{region_id + 1},
                                        llvm::ArrayRef<int64_t>{1});
      result_masks.emplace_back(m);
    }

    std::vector<Value> rets(op->getNumResults());

    // First case
    auto &first_return = blocks_to_work[0]->back();
    for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
      auto m = reshapeOrBroadcast(
          rewriter, op->getLoc(), result_masks[0],
          mlir::dyn_cast<RankedTensorType>(op.getResultTypes()[idx]));
      rets[idx] =
          rewriter.create<MulOp>(op->getLoc(), op->getResultTypes()[idx],
                                 first_return.getOperand(idx), m);
    }

    // Other cases
    for (int64_t branch_idx = 1; branch_idx < num_cases; ++branch_idx) {
      auto &branch_return = blocks_to_work[branch_idx]->back();
      for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
        auto m = reshapeOrBroadcast(
            rewriter, op->getLoc(), result_masks[branch_idx],
            mlir::dyn_cast<RankedTensorType>(op.getResultTypes()[idx]));
        m = rewriter.create<MulOp>(op->getLoc(), op->getResultTypes()[idx],
                                   branch_return.getOperand(idx), m);
        rets[idx] = rewriter.create<AddOp>(
            op->getLoc(), op->getResultTypes()[idx], rets[idx], m);
      }
    }

    // Replace results
    for (int64_t idx = 0; idx < op->getNumResults(); ++idx) {
      rewriter.replaceAllUsesWith(op->getResults()[idx], rets[idx]);
    }

    // Erase all returns
    for (auto *b : blocks_to_work) {
      rewriter.eraseOp(&b->back());
    }

    // Merge all blocks
    for (auto *b : blocks_to_work) {
      rewriter.mergeBlocks(b, blockBeforeCase);
    }
    rewriter.mergeBlocks(blockAfterCase, blockBeforeCase);
  }

 public:
  explicit CaseConverter(MLIRContext *context)
      : OpRewritePattern(context), tools_(context) {}

  LogicalResult matchAndRewrite(CaseOp op,
                                PatternRewriter &rewriter) const override {
    auto index = op.getIndex();

    if (tools_.getTypeVisibility(index.getType()) == Visibility::PUBLIC) {
      return failure();
    }

    inlineRegionIntoParent(op, rewriter);
    return success();
  }
};

class IfConverter : public OpRewritePattern<IfOp> {
 private:
  // Basic algorithm
  // %out = if(%pred) {
  //   trueBranch  { yield r0 }
  //   falseBranch { yield r1 }
  // }
  // With oblivious execution:
  // %out = select(%pred, r0, r1)
  void inlineRegionIntoParent(IfOp &op, PatternRewriter &rewriter) const {
    auto *blockBeforeIf = rewriter.getInsertionBlock();
    auto &trueBlock = op.getTrueBranch().front();
    auto &falseBlock = op.getFalseBranch().front();
    auto initPosition = rewriter.getInsertionPoint();
    auto *blockAfterIf = rewriter.splitBlock(blockBeforeIf, initPosition);

    // Remove the IfOp and returns.
    auto &trueReturnOp = trueBlock.back();
    auto &falseReturnOp = falseBlock.back();
    rewriter.inlineRegionBefore(op.getTrueBranch(), blockAfterIf);
    rewriter.inlineRegionBefore(op.getFalseBranch(), blockAfterIf);
    for (const auto &[idx, ret] : llvm::enumerate(op->getResults())) {
      auto s = rewriter.create<SelectOp>(
          op->getLoc(), op.getResultTypes()[idx], op.getCondition(),
          trueReturnOp.getOperands()[idx], falseReturnOp.getOperands()[idx]);
      rewriter.replaceAllUsesWith(op->getResult(idx), s);
    }
    rewriter.eraseOp(&trueReturnOp);
    rewriter.eraseOp(&falseReturnOp);

    rewriter.mergeBlocks(&trueBlock, blockBeforeIf);
    rewriter.mergeBlocks(&falseBlock, blockBeforeIf);
    rewriter.mergeBlocks(blockAfterIf, blockBeforeIf);
  }

 public:
  explicit IfConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    TypeTools tools(op->getContext());
    auto pred = op.getCondition();

    if (tools.getTypeVisibility(pred.getType()) == Visibility::PUBLIC) {
      return failure();
    }

    inlineRegionIntoParent(op, rewriter);
    return success();
  }
};

struct InlineSecretControlFlow
    : public InlineSecretControlFlowBase<InlineSecretControlFlow> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<IfConverter, CaseConverter>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInlineSecretControlFlow() {
  return std::make_unique<InlineSecretControlFlow>();
}

}  // namespace mlir::spu::pphlo
