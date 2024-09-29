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
#include "libspu/dialect/pphlo/transforms/passes.h"

namespace mlir::spu::pphlo {

namespace {

struct DotGeneralExpander : public OpRewritePattern<DotGeneralOp> {
 public:
  explicit DotGeneralExpander(MLIRContext *context)
      : OpRewritePattern<DotGeneralOp>(context) {}

  LogicalResult matchAndRewrite(DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs_type = mlir::dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhs_type = mlir::dyn_cast<RankedTensorType>(op.getRhs().getType());

    auto dnum = op.getDotDimensionNumbersAttr();
    if (dnum.getLhsBatchingDimensions().empty() &&
        dnum.getLhsContractingDimensions().empty() &&
        dnum.getRhsBatchingDimensions().empty() &&
        dnum.getRhsContractingDimensions().empty()) {
      // No batch, no contracing...dim
      // Aggregate all other dims
      // lhs shape -> (numel, 1)
      // rhs shape -> (1, numel)
      auto reshaped_lhs = rewriter.create<ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get({lhs_type.getNumElements(), 1},
                                lhs_type.getElementType()),
          op.getLhs());
      auto reshaped_rhs = rewriter.create<ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get({1, rhs_type.getNumElements()},
                                rhs_type.getElementType()),
          op.getRhs());
      // dot reshaped_lhs, reshaped_rhs
      auto dot =
          rewriter.create<DotOp>(op->getLoc(), reshaped_lhs, reshaped_rhs);
      // reshape back to result shape
      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getResult().getType(), dot);
      return success();
    }

    if (dnum.getLhsContractingDimensions().size() != 1 ||
        dnum.getRhsContractingDimensions().size() != 1 ||
        dnum.getLhsBatchingDimensions()[0] != 0 ||
        dnum.getLhsContractingDimensions()[0] != 2) {
      return failure();
    }

    if (lhs_type.getRank() != rhs_type.getRank() || lhs_type.getRank() != 3) {
      return failure();
    }

    if (lhs_type.getElementType() != rhs_type.getElementType()) {
      return failure();
    }

    int64_t num_batch =
        mlir::dyn_cast<RankedTensorType>(op.getLhs().getType()).getShape()[0];

    llvm::SmallVector<Value> results(num_batch);
    llvm::SmallVector<int64_t, 3> lhs_slice_begin(3, 0);
    llvm::SmallVector<int64_t, 3> lhs_slice_end(lhs_type.getShape().begin(),
                                                lhs_type.getShape().end());
    llvm::SmallVector<int64_t, 3> rhs_slice_begin(3, 0);
    llvm::SmallVector<int64_t, 3> rhs_slice_end(rhs_type.getShape().begin(),
                                                rhs_type.getShape().end());
    llvm::SmallVector<int64_t, 3> strides(3, 1);

    auto lhs_slice_type = RankedTensorType::get(
        {1, lhs_type.getShape()[1], lhs_type.getShape()[2]},
        lhs_type.getElementType());
    auto rhs_slice_type = RankedTensorType::get(
        {1, rhs_type.getShape()[1], rhs_type.getShape()[2]},
        rhs_type.getElementType());

    auto lhs_dot_type =
        RankedTensorType::get({lhs_type.getShape()[1], lhs_type.getShape()[2]},
                              lhs_type.getElementType());
    auto rhs_dot_type =
        RankedTensorType::get({rhs_type.getShape()[1], rhs_type.getShape()[2]},
                              rhs_type.getElementType());

    for (int64_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
      lhs_slice_begin[0] = batch_idx;
      lhs_slice_end[0] = batch_idx + 1;
      rhs_slice_begin[0] = batch_idx;
      rhs_slice_end[0] = batch_idx + 1;
      // Slice lhs & rhs
      auto slice_lhs =
          rewriter.create<SliceOp>(op->getLoc(), lhs_slice_type, op.getLhs(),
                                   lhs_slice_begin, lhs_slice_end, strides);
      auto slice_rhs =
          rewriter.create<SliceOp>(op->getLoc(), rhs_slice_type, op.getRhs(),
                                   rhs_slice_begin, rhs_slice_end, strides);
      // Reshape
      auto reshaped_lhs =
          rewriter.create<ReshapeOp>(op->getLoc(), lhs_dot_type, slice_lhs);
      auto reshaped_rhs =
          rewriter.create<ReshapeOp>(op->getLoc(), rhs_dot_type, slice_rhs);
      // Dot
      auto dot =
          rewriter.create<DotOp>(op->getLoc(), reshaped_lhs, reshaped_rhs);
      auto dot_type = mlir::dyn_cast<RankedTensorType>(dot.getType());
      // Unsqueeze
      results[batch_idx] = rewriter.create<ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get(
              {1, dot_type.getShape()[0], dot_type.getShape()[1]},
              dot_type.getElementType()),
          dot);
    }

    auto concat = rewriter.create<ConcatenateOp>(op->getLoc(), results, 0);
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getResult().getType(),
                                           concat);
    return success();
  }
};

struct DotGeneralToDot : public DotGeneralToDotBase<DotGeneralToDot> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<DotGeneralExpander>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDotGeneralToDot() {
  return std::make_unique<DotGeneralToDot>();
}

}  // namespace mlir::spu::pphlo
