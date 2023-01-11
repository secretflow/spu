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

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "spdlog/spdlog.h"

#include "spu/compiler/passes/pass_details.h"
#include "spu/dialect/pphlo_ops.h"

namespace mlir::pphlo {

namespace {

template <typename T>
bool isSmallerThanEps(T v) {
  return std::numeric_limits<T>::epsilon() >= v;
}

struct SqrtRewriter : public OpRewritePattern<DivOp> {
  explicit SqrtRewriter(MLIRContext *context)
      : OpRewritePattern<DivOp>(context) {}

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    // Pattern
    // y/(sqrt(x) + small_const)
    // Into
    // y*rsqrt(x + small_const)
    auto rhs = op.rhs();
    auto rhs_add = rhs.getDefiningOp<AddOp>();
    if (!rhs_add) {
      return failure();
    }

    auto added_const = rhs_add.rhs().getDefiningOp<ConstantOp>();
    if (!added_const) {
      return failure();
    }

    auto added_sqrt = rhs_add.lhs().getDefiningOp<SqrtOp>();
    if (!added_sqrt) {
      return failure();
    }

    auto const_value = added_const.value().dyn_cast<DenseFPElementsAttr>();
    if (!const_value) {
      return failure();
    }

    if (const_value.getElementType().isF32()) {
      if (!isSmallerThanEps(const_value.getSplatValue<float>())) {
        return failure();
      }
    } else {
      YACL_ENFORCE(const_value.getElementType().isF64());
      if (!isSmallerThanEps(const_value.getSplatValue<double>())) {
        return failure();
      }
    }

    auto eps = rewriter.create<EpsilonOp>(added_sqrt->getLoc(),
                                          added_const->getResultTypes());
    auto add = rewriter.create<AddOp>(added_sqrt.getLoc(),
                                      added_sqrt->getResultTypes(),
                                      added_sqrt->getOperand(0), eps);
    auto rsqrt = rewriter.create<RsqrtOp>(added_sqrt.getLoc(),
                                          added_sqrt->getResultTypes(), add);
    rewriter.replaceOpWithNewOp<MulOp>(op, op->getResultTypes(), op.lhs(),
                                       rsqrt);

    return success();
  }
};

struct OptimizeSqrtToRsqrt
    : public OptimizeSqrtToRsqrtBase<OptimizeSqrtToRsqrt> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  void populateOwningPatterns(RewritePatternSet *patterns,
                              MLIRContext *ctx) const {
    patterns->insert<SqrtRewriter>(ctx);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeSqrtToRsqrtPass() {
  return std::make_unique<OptimizeSqrtToRsqrt>();
}

} // namespace mlir::pphlo
