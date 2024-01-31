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

// Idea here:
// %2 = convert(%0)
// %3 = reshape(%2)
// mul(%1, %3)
// Can be rewrite into
// %2 = reshape(%0)
// %3 = convert(%2)
// mul(%1, %3)
// Makes mixed_mul/dot optimization easier
template <typename OpT>
struct TypeAgnosticOpConverter : public OpRewritePattern<OpT> {
public:
  explicit TypeAgnosticOpConverter(MLIRContext *context)
      : OpRewritePattern<OpT>(context) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();
    auto parentConvert = operand.template getDefiningOp<ConvertOp>();
    if (parentConvert == nullptr) {
      return failure();
    }

    const auto &from_type = parentConvert.getOperand()
                                .getType()
                                .template dyn_cast<RankedTensorType>();
    const auto &to_type =
        op.getResult().getType().template dyn_cast<RankedTensorType>();

    OpBuilder builder(op);

    auto new_reshape = builder.create<OpT>(
        op->getLoc(),
        RankedTensorType::get(to_type.getShape(), from_type.getElementType()),
        parentConvert.getOperand(), op->getAttrs());

    rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getType(), new_reshape);

    return success();
  }
};

struct ConvertPushDown : public ConvertPushDownBase<ConvertPushDown> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<TypeAgnosticOpConverter<ReshapeOp>,
                     TypeAgnosticOpConverter<TransposeOp>,
                     TypeAgnosticOpConverter<SliceOp>>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertPushDownPass() {
  return std::make_unique<ConvertPushDown>();
}

} // namespace mlir::spu::pphlo
