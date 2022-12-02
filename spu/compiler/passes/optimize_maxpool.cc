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

#include "spu/compiler/passes/pass_details.h"
#include "spu/compiler/passes/passes.h"
#include "spu/dialect/pphlo_base_enums.h"
#include "spu/dialect/pphlo_ops.h"
#include "spu/dialect/pphlo_types.h"

namespace mlir::pphlo {

namespace {

struct SelectAndScatterConverter : public OpRewritePattern<SelectAndScatterOp> {
private:
  TypeTools tools_;

  Value rewriteReduceWindow(ReduceWindowOp op,
                            PatternRewriter &rewriter) const {
    auto window_size =
        std::accumulate(op.window_dimensions().getValues<int64_t>().begin(),
                        op.window_dimensions().getValues<int64_t>().end(), 1,
                        std::multiplies<int64_t>());

    auto current_ret_type =
        op.getResult(0).getType().dyn_cast<RankedTensorType>();

    std::vector<int64_t> index_result_shape = current_ret_type.getShape();
    index_result_shape.emplace_back(window_size);

    auto current_ret_vis = tools_.getTypeVisibility(current_ret_type);

    auto index_result_type = RankedTensorType::get(
        index_result_shape,
        tools_.getTypeWithVisibility(rewriter.getI1Type(), current_ret_vis));

    OpBuilder builder(op);
    builder.setInsertionPoint(op.getOperation());
    auto argmax = builder.create<ArgMaxOp>(
        op->getLoc(), SmallVector<Type>{current_ret_type, index_result_type},
        op.inputs()[0], op.window_dimensions(),
        op.window_strides().value_or(nullptr),
        op.base_dilations().value_or(nullptr),
        op.window_dilations().value_or(nullptr),
        op.padding().value_or(nullptr));

    op->getResult(0).replaceAllUsesWith(argmax->getResult(0));

    return argmax->getResult(1);
  }

  bool isSingleRegion(Region &r) const {
    if (r.hasOneBlock()) {
      return llvm::hasSingleElement(r.front().without_terminator());
    }
    return false;
  }

public:
  explicit SelectAndScatterConverter(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(SelectAndScatterOp op,
                                PatternRewriter &rewriter) const override {

    // Select and scatter region should be a single element region
    if (!isSingleRegion(op.scatter()) || !isSingleRegion(op.select())) {
      return failure();
    }

    // Select should be a GE
    // Scatter should be an add
    if (!mlir::isa<pphlo::GreaterEqualOp>(op.select().front().front()) ||
        !mlir::isa<pphlo::AddOp>(op.scatter().front().front())) {
      return failure();
    }

    // Find previous reduce window
    auto input = op.operand();
    auto uses = input.getUses();

    LogicalResult status = failure();
    Value selected_indicies;

    auto isAllOne = [](const DenseIntElementsAttr &attr) {
      return attr.isSplat() && attr.getSplatValue<int64_t>() == 1;
    };

    for (const auto &u : uses) {
      if (auto previous_reduce_window =
              mlir::dyn_cast<ReduceWindowOp>(u.getOwner())) {
        if (previous_reduce_window.inputs().size() != 1) {
          continue;
        }
        if (!isSingleRegion(previous_reduce_window.body())) {
          continue;
        }
        if (!mlir::isa<pphlo::MaxOp>(
                previous_reduce_window.body().front().front())) {
          continue;
        }

        // Check window dimension
        // Check windows strides
        // Check padding
        if (op.window_dimensions() !=
            previous_reduce_window.window_dimensions()) {
          continue;
        }

        if (op.window_strides() != previous_reduce_window.window_strides()) {
          continue;
        }

        if (op.padding() != previous_reduce_window.padding()) {
          continue;
        }

        // Make sure no dialation
        auto window_dilation = previous_reduce_window.window_dilations();
        auto base_dilation = previous_reduce_window.base_dilations();
        if (window_dilation.has_value() && !isAllOne(*window_dilation)) {
          continue;
        }
        if (base_dilation.has_value() && !isAllOne(*base_dilation)) {
          continue;
        }

        selected_indicies =
            rewriteReduceWindow(previous_reduce_window, rewriter);

        break;
      }
    }

    rewriter.replaceOpWithNewOp<pphlo::MaxPoolScatterOp>(
        op, op->getResultTypes()[0], selected_indicies, op.source(),
        op.window_dimensions(), op.window_strides().value_or(nullptr),
        op.padding().value_or(nullptr));

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
  void populateOwningPatterns(RewritePatternSet *patterns,
                              MLIRContext *ctx) const {
    patterns->insert<SelectAndScatterConverter>(ctx);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeMaxPoolingPass() {
  return std::make_unique<OptimizeMaxPooling>();
}

} // namespace mlir::pphlo
