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
#include "libspu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

// Convert the following pattern %s1 = mul(%p0, %s0); %s2 = mul(%p1, %s1) into
// %p2 = mul(%p0, %p1); %s1 = mul(%p2, %s1)
struct MulConverter : public OpRewritePattern<MulOp> {
private:
  TypeTools tools_;
  bool isMulSP(MulOp op) const {
    auto lhs_vis = tools_.getTypeVisibility(op.getLhs().getType());
    auto rhs_vis = tools_.getTypeVisibility(op.getRhs().getType());

    return lhs_vis != rhs_vis;
  }

  std::pair<mlir::Value, mlir::Value>
  getSecretAndPublicOperand(MulOp op) const {
    auto lhs_vis = tools_.getTypeVisibility(op.getLhs().getType());

    auto secret_operand =
        lhs_vis == Visibility::VIS_SECRET ? op.getLhs() : op.getRhs();
    auto public_operand =
        lhs_vis == Visibility::VIS_SECRET ? op.getRhs() : op.getLhs();

    return {secret_operand, public_operand};
  }

public:
  explicit MulConverter(MLIRContext *context) : OpRewritePattern(context) {
    setHasBoundedRewriteRecursion(false);
  }

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {

    // If lhs and rhs has the same visibility, bailout
    if (!isMulSP(op)) {
      return failure();
    }

    auto [curr_secret_operand, curr_public_operand] =
        getSecretAndPublicOperand(op);
    // %s must yield from a mul and that mul has only one use
    if (auto prev_mul = curr_secret_operand.getDefiningOp<MulOp>()) {
      if (isMulSP(prev_mul)) {
        auto [prev_secret_operand, prev_public_operand] =
            getSecretAndPublicOperand(prev_mul);
        OpBuilder builder(op);
        auto mul_pp =
            builder.create<MulOp>(op.getLoc(), prev_public_operand.getType(),
                                  prev_public_operand, curr_public_operand);

        rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), mul_pp.getResult(),
                                           prev_secret_operand);
        return success();
      }
    }

    return failure();
  }
};

struct ReduceTruncation : public ReduceTruncBase<ReduceTruncation> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<MulConverter>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createReduceTruncationPass() {
  return std::make_unique<ReduceTruncation>();
}

} // namespace mlir::pphlo
