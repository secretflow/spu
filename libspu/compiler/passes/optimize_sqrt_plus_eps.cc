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

#include <iostream>
#include <limits>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "spdlog/spdlog.h"

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

template <typename T>
bool isSmallerThanEps(T v) {
  return std::numeric_limits<T>::epsilon() >= v;
}

struct SqrtRewriter : public OpRewritePattern<AddOp> {
  explicit SqrtRewriter(MLIRContext *context)
      : OpRewritePattern<AddOp>(context) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // Pattern
    // (sqrt(x) + small_const)
    // Into
    // sqrt(x + eps)
    auto added_const = op.getRhs().getDefiningOp<ConstantOp>();
    if (!added_const) {
      return failure();
    }

    auto added_sqrt = op.getLhs().getDefiningOp<SqrtOp>();
    if (!added_sqrt) {
      return failure();
    }

    auto const_value = added_const.getValue().dyn_cast<DenseFPElementsAttr>();
    if (!const_value) {
      return failure();
    }

    if (const_value.getElementType().isF32()) {
      if (!isSmallerThanEps(const_value.getSplatValue<float>())) {
        return failure();
      }
    } else {
      SPU_ENFORCE(const_value.getElementType().isF64());
      if (!isSmallerThanEps(const_value.getSplatValue<double>())) {
        return failure();
      }
    }

    auto eps = rewriter.create<EpsilonOp>(added_sqrt->getLoc(),
                                          added_const->getResultTypes());
    auto add = rewriter.create<AddOp>(added_sqrt.getLoc(),
                                      added_sqrt->getResultTypes(),
                                      added_sqrt->getOperand(0), eps);
    rewriter.replaceOpWithNewOp<SqrtOp>(op, op->getResultTypes(), add);

    return success();
  }
};

struct OptimizeSqrtPlusEps
    : public OptimizeSqrtPlusEpsBase<OptimizeSqrtPlusEps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<SqrtRewriter>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeSqrtPlusEps() {
  return std::make_unique<OptimizeSqrtPlusEps>();
}

} // namespace mlir::pphlo
