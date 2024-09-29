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

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"

// #define DEBUG_RANGE

#ifdef DEBUG_RANGE
#include "spdlog/spdlog.h"

#include "libspu/dialect/utils/utils.h"

#define RANGE_DEBUG(...) __VA_ARGS__
#else
#define RANGE_DEBUG(...) static_cast<void>(0)
#endif

namespace mlir::spu::pphlo {

namespace {

IntegerValueRange getMaxRange(Value value) {
  TypeTools tools(value.getContext());
  unsigned width = 0;
  if (tools.isFloatType(value.getType())) {
    width = 2 * getElementTypeOrSelf(value.getType()).getIntOrFloatBitWidth();
  } else {
    width = tools.getIntOrFxpWidth(value.getType());
  }

  if (width == 0) {
    return {};
  }

  APInt umin = APInt::getMinValue(width);
  APInt umax = APInt::getMaxValue(width);
  APInt smin = width != 1 ? APInt::getSignedMinValue(width) : umin;
  APInt smax = width != 1 ? APInt::getSignedMaxValue(width) : umax;
  return IntegerValueRange{ConstantIntRanges{umin, umax, smin, smax}};
}

}  // namespace

/// This lattice element represents the fxp value range of an SSA value.
/// When this lattice is updated, it automatically updates the constant value
/// of the SSA value (if the range can be narrowed to one).
class IntegerValueRangeLattice
    : public ::mlir::dataflow::Lattice<IntegerValueRange> {
 public:
  using Lattice::Lattice;
};

class FxpRangeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<IntegerValueRangeLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// At an entry point, we cannot reason about integer value ranges.
  void setToEntryState(IntegerValueRangeLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(getMaxRange(lattice->getPoint())));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  void visitOperation(Operation *op,
                      ArrayRef<const IntegerValueRangeLattice *> operands,
                      ArrayRef<IntegerValueRangeLattice *> results) override {
    RANGE_DEBUG(SPDLOG_INFO("infer range on {}", mlirObjectToString(*op)));
    auto inferrable = dyn_cast<InferFxpRangeInterface>(op);
    if (!inferrable) {
      RANGE_DEBUG(SPDLOG_INFO("not inferrable"));
      return setAllToEntryStates(results);
    }

    llvm::SmallVector<ConstantIntRanges> argRanges;
    argRanges.reserve(operands.size());

    for (const IntegerValueRangeLattice *lattice : operands) {
      if (!lattice->getValue().isUninitialized()) {
        argRanges.push_back(lattice->getValue().getValue());
      }
    }

    auto joinCallback = [&](Value v, const ConstantIntRanges &intRanges) {
      ConstantIntRanges attrs(intRanges);
      auto result = dyn_cast<OpResult>(v);
      if (!result) {
        return;
      }
      RANGE_DEBUG(SPDLOG_INFO("Inferred range {}", mlirObjectToString(attrs)));
      assert(llvm::is_contained(op->getResults(), result));

      IntegerValueRangeLattice *lattice = results[result.getResultNumber()];
      IntegerValueRange oldRange = lattice->getValue();
      ChangeResult changed = lattice->join(attrs);

      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<::mlir::OpTrait::IsTerminator>();
      });
      if (isYieldedResult && !oldRange.isUninitialized() &&
          !(lattice->getValue() == oldRange)) {
        changed |= lattice->join(getMaxRange(v));
      }
      propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultRanges(argRanges, joinCallback);
  }

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow. This
  /// function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<IntegerValueRangeLattice *> argLattices,
      unsigned firstIndex) override {
    if (auto inferrable = dyn_cast<InferFxpRangeInterface>(op)) {
      llvm::SmallVector<ConstantIntRanges> argRanges;
      argRanges.reserve(op->getOperands().size());

      for (Value value : op->getOperands()) {
        if (!getLatticeElementFor(op, value)->getValue().isUninitialized()) {
          argRanges.push_back(
              getLatticeElementFor(op, value)->getValue().getValue());
        }
      }

      auto joinCallback = [&](Value v, const ConstantIntRanges &intRanges) {
        IntegerValueRange attrs(intRanges);
        auto arg = dyn_cast<BlockArgument>(v);
        if (!arg) {
          return;
        }
        if (!llvm::is_contained(successor.getSuccessor()->getArguments(),
                                arg)) {
          return;
        }

        IntegerValueRangeLattice *lattice = argLattices[arg.getArgNumber()];
        IntegerValueRange oldRange = lattice->getValue();

        ChangeResult changed = lattice->join(attrs);

        // Catch loop results with loop variant bounds and conservatively make
        // them [-inf, inf] so we don't circle around infinitely often (because
        // the dataflow analysis in MLIR doesn't attempt to work out trip counts
        // and often can't).
        bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
          return op->hasTrait<::mlir::OpTrait::IsTerminator>();
        });
        if (isYieldedValue && !oldRange.isUninitialized() &&
            !(lattice->getValue() == oldRange)) {
          changed |= lattice->join(getMaxRange(v));
        }
        propagateIfChanged(lattice, changed);
      };

      inferrable.inferResultRanges(argRanges, joinCallback);
      return;
    }

    return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
        op, successor, argLattices, firstIndex);
  }
};

namespace {

/// Succeeds when a value is statically non-negative in that it has a lower
/// bound on its value (if it is treated as signed) and that bound is
/// non-negative.
LogicalResult staticallySatisfy(
    DataFlowSolver &solver, Value v,
    std::function<bool(const ConstantIntRanges &)> &&foo) {
  const auto *result = solver.lookupState<IntegerValueRangeLattice>(v);
  if ((result == nullptr) || result->getValue().isUninitialized()) {
    return failure();
  }
  const ConstantIntRanges &range = result->getValue().getValue();
  return success(foo(range));
}

struct AbsPattern : OpRewritePattern<SignOp> {
 private:
  DataFlowSolver *solver_;
  TypeTools tools_;

 public:
  explicit AbsPattern(MLIRContext *context, DataFlowSolver *solver)
      : OpRewritePattern<SignOp>(context), solver_(solver), tools_(context) {}

  LogicalResult matchAndRewrite(SignOp op,
                                PatternRewriter &rewriter) const override {
    auto shape = dyn_cast<RankedTensorType>(op.getType()).getShape();
    if (succeeded(staticallySatisfy(*solver_, op.getResult(),
                                    [](const ConstantIntRanges &range) {
                                      return range.smin().getSExtValue() > 0;
                                    }))) {
      // Known positive, sign to const 1.
      auto c = rewriter.create<arith::ConstantOp>(
          op->getLoc(), rewriter.getOneAttr(RankedTensorType::get(
                            shape, rewriter.getI32Type())));
      rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getType(), c);
    } else if (succeeded(
                   staticallySatisfy(*solver_, op.getResult(),
                                     [](const ConstantIntRanges &range) {
                                       return range.smax().getSExtValue() < 0;
                                     }))) {
      // Known negative, sign to const -1
      auto c = rewriter.create<arith::ConstantOp>(
          op->getLoc(), DenseIntElementsAttr::get(
                            RankedTensorType::get(shape, rewriter.getI32Type()),
                            static_cast<int32_t>(-1)));
      rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getType(), c);
    }

    return success();
  }
};

struct RangeOptimization : public RangeOptimizationBase<RangeOptimization> {
 private:
  std::shared_ptr<DataFlowSolver> solver_;

  void populateOwningPatterns(RewritePatternSet *patterns, MLIRContext *ctx) {
    patterns->add<AbsPattern>(ctx, solver_.get());
  }

 public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    solver_ = std::make_shared<DataFlowSolver>();

    solver_->load<dataflow::DeadCodeAnalysis>();
    solver_->load<FxpRangeAnalysis>();
    if (failed(solver_->initializeAndRun(op))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    populateOwningPatterns(&patterns, ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRangeOptimization() {
  return std::make_unique<RangeOptimization>();
}

}  // namespace mlir::spu::pphlo
