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
  // Rewrite reduce body from a unary max reduce to a binary GE reduce,
  // Which returns both max value and onehot for max location
  void rewriteReduceBody(Region &r, PatternRewriter &rewriter) const {
    auto comp = mlir::dyn_cast<pphlo::MaxOp>(r.front().front());
    YASL_ENFORCE(comp);

    auto builder = OpBuilder::atBlockBegin(&r.front());

    auto comp_ret = comp->getResultTypes()[0];
    auto comp_vis = tools_.getTypeVisibility(comp_ret);
    auto index_ret_t = tools_.getTypeWithVisibility(
        RankedTensorType::get({}, rewriter.getI8Type()), comp_vis);
    auto ge_ret_t = tools_.getTypeWithVisibility(
        RankedTensorType::get({}, rewriter.getI1Type()), comp_vis);
    auto ge = builder.create<pphlo::GreaterEqualOp>(comp->getLoc(), ge_ret_t,
                                                    comp.lhs(), comp.rhs());

    auto select1 = builder.create<pphlo::SelectOp>(
        comp->getLoc(), TypeRange{comp_ret, index_ret_t}, ge,
        ValueRange{comp.lhs(), r.getArgument(1)},
        ValueRange{comp.rhs(), r.getArgument(3)});

    auto *operation = r.front().getTerminator();
    rewriter.updateRootInPlace(
        operation, [&]() { operation->setOperands(select1->getResults()); });
  }

  Value rewriteReduceWindow(ReduceWindowOp op,
                            PatternRewriter &rewriter) const {
    auto pub_mask_type = tools_.getTypeWithVisibility(rewriter.getI8Type(),
                                                      Visibility::VIS_PUBLIC);

    std::vector<int64_t> window_shape(
        op.window_dimensions().getValues<int64_t>().begin(),
        op.window_dimensions().getValues<int64_t>().end());
    auto window_size = std::accumulate(window_shape.begin(), window_shape.end(),
                                       1, std::multiplies<int64_t>());

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op.getOperation());

    auto init_value = builder.create<pphlo::ConstantOp>(
        op->getLoc(),
        DenseElementsAttr::get(RankedTensorType::get({}, rewriter.getI8Type()),
                               rewriter.getI8IntegerAttr(-1)));

    // Build a window mask as eye(n), where n = window size
    std::vector<Attribute> mask(window_size * window_size,
                                rewriter.getI8IntegerAttr(0));
    for (int64_t idx = 0; idx < window_size; ++idx) {
      mask[idx * window_size + idx] = rewriter.getI8IntegerAttr(1);
    }
    auto mask_const = builder.create<pphlo::ConstantOp>(
        op->getLoc(),
        DenseElementsAttr::get(RankedTensorType::get({window_size, window_size},
                                                     rewriter.getI8Type()),
                               mask));

    // Rewrite reduce window from max to argmax
    llvm::SmallVector<Value, 4> operands;
    operands.emplace_back(op.inputs()[0]);
    operands.emplace_back(mask_const);
    operands.emplace_back(op.init_values()[0]);
    operands.emplace_back(init_value);

    auto current_ret_type =
        op->getResultTypes()[0].dyn_cast<RankedTensorType>();
    auto current_ret_vis = tools_.getTypeVisibility(current_ret_type);

    std::vector<int64_t> index_result_shape = current_ret_type.getShape();
    index_result_shape.emplace_back(window_size);

    auto index_result_type = RankedTensorType::get(
        index_result_shape,
        tools_.getTypeWithVisibility(rewriter.getI8Type(), current_ret_vis));

    auto new_reduce_window = builder.create<pphlo::ReduceWindowOp>(
        op->getLoc(), SmallVector<Type>{current_ret_type, index_result_type},
        operands, op->getAttrs());

    new_reduce_window.last_operand_is_window_maskAttr(
        BoolAttr::get(op->getContext(), true));
    new_reduce_window.ignore_init_valueAttr(
        BoolAttr::get(op->getContext(), true));

    rewriter.inlineRegionBefore(op.body(), new_reduce_window.body(),
                                new_reduce_window.body().end());

    new_reduce_window.body().insertArgument(
        1, RankedTensorType::get({}, pub_mask_type), op->getLoc());
    new_reduce_window.body().addArgument(
        RankedTensorType::get({}, pub_mask_type), op->getLoc());

    op->getResult(0).replaceAllUsesWith(new_reduce_window->getResult(0));

    // Rewrite body
    rewriteReduceBody(new_reduce_window.body(), rewriter);

    return new_reduce_window->getResults()[1];
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
        if (window_dilation.hasValue() && !isAllOne(*window_dilation)) {
          continue;
        }
        if (base_dilation.hasValue() && !isAllOne(*base_dilation)) {
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
