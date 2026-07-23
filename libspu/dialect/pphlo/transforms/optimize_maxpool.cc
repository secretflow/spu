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

#include <llvm/ADT/STLExtras.h>

#include <functional>
#include <numeric>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/IR/types.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"
#include "libspu/dialect/pphlo/transforms/passes.h"

namespace mlir::spu::pphlo {

namespace {

struct SelectAndScatterConverter : public OpRewritePattern<SelectAndScatterOp> {
 private:
  TypeTools typetools_;

  Value rewriteReduceWindow(ReduceWindowOp op,
                            PatternRewriter &rewriter) const {
    auto window_size = std::accumulate(op.getWindowDimensions().begin(),
                                       op.getWindowDimensions().end(), 1,
                                       std::multiplies<int64_t>());

    auto current_ret_type =
        mlir::dyn_cast<RankedTensorType>(op.getResult(0).getType());

    std::vector<int64_t> index_result_shape = current_ret_type.getShape();
    index_result_shape.emplace_back(window_size);

    auto current_ret_vis = typetools_.getTypeVisibility(current_ret_type);

    auto index_result_type = RankedTensorType::get(
        index_result_shape,
        typetools_.getType(rewriter.getI1Type(), current_ret_vis));

    OpBuilder builder(op);
    builder.setInsertionPoint(op.getOperation());
    auto argmax = builder.create<ArgMaxOp>(
        op->getLoc(), SmallVector<Type>{current_ret_type, index_result_type},
        op.getInputs()[0], op.getWindowDimensions(),
        DenseI64ArrayAttr::get(op->getContext(),
                               op.getWindowStrides().value_or(std::nullopt)),
        DenseI64ArrayAttr::get(op->getContext(),
                               op.getWindowDilations().value_or(std::nullopt)));

    op->getResult(0).replaceAllUsesWith(argmax->getResult(0));

    return argmax->getResult(1);
  }

  static bool isSingleRegion(Region &r) {
    if (r.hasOneBlock()) {
      return llvm::hasSingleElement(r.front().without_terminator());
    }
    return false;
  }

 public:
  explicit SelectAndScatterConverter(MLIRContext *context)
      : OpRewritePattern(context), typetools_(context) {}

  LogicalResult matchAndRewrite(SelectAndScatterOp op,
                                PatternRewriter &rewriter) const override {
    // Select and scatter region should be a single element region
    if (!isSingleRegion(op.getScatter()) || !isSingleRegion(op.getSelect())) {
      return failure();
    }

    // Select should be a GE
    // Scatter should be an add
    if (!mlir::isa<pphlo::GreaterEqualOp>(op.getSelect().front().front()) ||
        !mlir::isa<pphlo::AddOp>(op.getScatter().front().front())) {
      return failure();
    }

    // Find previous reduce window
    auto input = op.getOperand();
    auto uses = input.getUses();

    LogicalResult status = failure();
    Value selected_indices;
    bool rewritten = false;

    for (const auto &u : uses) {
      if (auto previous_reduce_window =
              mlir::dyn_cast<ReduceWindowOp>(u.getOwner())) {
        if (previous_reduce_window.getInputs().size() != 1) {
          continue;
        }
        if (!isSingleRegion(previous_reduce_window.getBody())) {
          continue;
        }
        if (!mlir::isa<pphlo::MaxOp>(
                previous_reduce_window.getBody().front().front())) {
          continue;
        }

        // Check window dimension
        // Check windows strides
        // Check padding
        if (op.getWindowDimensions() !=
            previous_reduce_window.getWindowDimensions()) {
          continue;
        }

        if (op.getWindowStrides() !=
            previous_reduce_window.getWindowStrides()) {
          continue;
        }

        // Make sure no dilation
        auto window_dilation = previous_reduce_window.getWindowDilations();
        if (window_dilation.has_value() &&
            !llvm::all_of(*window_dilation, [](int64_t v) { return v == 1; })) {
          continue;
        }

        selected_indices =
            rewriteReduceWindow(previous_reduce_window, rewriter);
        rewritten = true;

        break;
      }
    }

    if (!rewritten) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<pphlo::MaxPoolScatterOp>(
        op, op->getResultTypes()[0], selected_indices, op.getSource(),
        DenseI64ArrayAttr::get(op->getContext(), op.getWindowDimensions()),
        DenseI64ArrayAttr::get(op->getContext(),
                               op.getWindowStrides().value_or(std::nullopt)));

    return status;
  }
};

struct OptimizeMaxPooling : public OptimizeMaxPoolingBase<OptimizeMaxPooling> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<SelectAndScatterConverter>(ctx);
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeMaxPoolingPass() {
  return std::make_unique<OptimizeMaxPooling>();
}

}  // namespace mlir::spu::pphlo
