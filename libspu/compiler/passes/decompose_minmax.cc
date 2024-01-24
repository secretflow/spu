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

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/dialect/pphlo/ops.h"

namespace mlir::spu::pphlo {

namespace {

// MaxOp -> select(greater(x, y), x, y)
// MinOp -> select(less(x,y), x, y)
template <typename InOp, typename RetOp>
struct MinMaxOpConverter : public OpRewritePattern<InOp> {
private:
  TypeTools typetools_;

public:
  explicit MinMaxOpConverter(MLIRContext *context)
      : OpRewritePattern<InOp>(context), typetools_(context) {}

  LogicalResult matchAndRewrite(InOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder builder(op);

    auto ret_type = op.getType().template dyn_cast<mlir::RankedTensorType>();
    auto ret_vis = typetools_.getTypeVisibility(op.getType());
    auto gt_ret = RankedTensorType::get(
        ret_type.getShape(),
        typetools_.getType(mlir::IntegerType::get(op->getContext(), 1),
                           ret_vis));

    auto gt = builder.create<RetOp>(op->getLoc(), gt_ret, op.getOperands());

    rewriter.replaceOpWithNewOp<SelectOp>(op, ret_type, gt.getResult(),
                                          op.getOperand(0), op.getOperand(1));

    return success();
  }
};

struct DecomposeMinMax : public DecomposeMinMaxBase<DecomposeMinMax> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<MinMaxOpConverter<MaxOp, GreaterOp>,
                     MinMaxOpConverter<MinOp, LessOp>>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeMinMaxPass() {
  return std::make_unique<DecomposeMinMax>();
}

} // namespace mlir::spu::pphlo
