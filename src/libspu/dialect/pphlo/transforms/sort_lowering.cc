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

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/dialect/pphlo/transforms/pass_details.h"
#include "libspu/dialect/pphlo/transforms/passes.h"

namespace mlir::spu::pphlo {

namespace {

struct SortConversion : public OpRewritePattern<SortOp> {
 public:
  explicit SortConversion(MLIRContext *context)
      : OpRewritePattern<SortOp>(context) {}

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    auto &comp = op.getComparator();
    if (op->getNumOperands() == 1) {
      // When there is only one operand, stable or not seems irrelevant
      op.setIsStable(false);
    }

    // If has a single instruction comparator, check if it's a simple sort.
    if (comp.hasOneBlock() &&
        llvm::hasSingleElement(comp.front().without_terminator())) {
      auto &inst = comp.front().front();
      // Single instruction comparator.
      if (mlir::isa<pphlo::LessOp>(inst) || mlir::isa<pphlo::GreaterOp>(inst)) {
        SortDirectionAttr direction;
        if (mlir::isa<pphlo::GreaterOp>(inst)) {
          // descent
          direction =
              SortDirectionAttr::get(op->getContext(), SortDirection::DES);
        } else {
          // ascent
          direction =
              SortDirectionAttr::get(op->getContext(), SortDirection::ASC);
        }
        auto lhs_idx = mlir::dyn_cast<mlir::BlockArgument>(inst.getOperand(0))
                           .getArgNumber();
        auto rhs_idx = mlir::dyn_cast<mlir::BlockArgument>(inst.getOperand(1))
                           .getArgNumber();
        // FIXME: If the comparator is using operands other than the first one,
        // we should just reorder operands instead of bailout
        if (lhs_idx != 0 || rhs_idx != 1) {
          return failure();
        }

        rewriter.replaceOpWithNewOp<pphlo::SimpleSortOp>(
            op, op.getResultTypes(), op.getOperands(), op.getDimensionAttr(),
            rewriter.getI64IntegerAttr(1), direction);
        return success();
      }
    }

    // pattern for jax.lax.sort lowering
    if (comp.hasOneBlock()) {
      auto &first_inst = comp.front().front();
      bool match_less = matchPattern(&first_inst, m_Op<pphlo::LessOp>());
      bool match_greater = matchPattern(&first_inst, m_Op<pphlo::GreaterOp>());
      if (match_less || match_greater) {
        SortDirectionAttr direction;

        if (match_greater) {
          // descent
          direction =
              SortDirectionAttr::get(op->getContext(), SortDirection::DES);
        } else {
          // ascent
          direction =
              SortDirectionAttr::get(op->getContext(), SortDirection::ASC);
        }

        size_t key_nums = 0;
        const auto comp_name = first_inst.getName().getStringRef();
        // save the result for each instruction for following check.
        std::vector<mlir::Value> results;
        for (auto &instr : comp.front().without_terminator()) {
          if (matchPattern(&instr, m_Op(comp_name))) {
            key_nums++;
          }
          results.push_back(instr.getResult(0));
        }

        // idx of and/or blocks
        size_t lhs_idx = 2 * key_nums - 3;
        size_t rhs_idx = 2 * key_nums - 2;
        for (auto [i, instr] :
             llvm::enumerate(comp.front().without_terminator())) {
          if (i <= 2 * key_nums - 2) {
            auto lhs_arg =
                mlir::dyn_cast<mlir::BlockArgument>(instr.getOperand(0));
            auto rhs_arg =
                mlir::dyn_cast<mlir::BlockArgument>(instr.getOperand(1));

            if (lhs_arg == nullptr || rhs_arg == nullptr) {
              return failure();
            }

            auto lhs_idx = lhs_arg.getArgNumber();
            auto rhs_idx = rhs_arg.getArgNumber();

            // less + equal blocks
            if ((i & 1) == 0 && matchPattern(&instr, m_Op(comp_name))) {
              if (lhs_idx != i || rhs_idx != (i + 1)) {
                return failure();
              }
            }
            // equal op
            if ((i & 1) == 1 && matchPattern(&instr, m_Op<pphlo::EqualOp>())) {
              if (lhs_idx != (i - 1) || rhs_idx != i) {
                return failure();
              }
            }
          } else {
            // check the operands of and/or
            auto lhs = instr.getOperand(0);
            auto rhs = instr.getOperand(1);
            bool pass = (lhs == results[lhs_idx] && rhs == results[rhs_idx]) ||
                        (lhs == results[rhs_idx] && rhs == results[lhs_idx]);

            // and blocks
            if ((i & 1) == 1 && matchPattern(&instr, m_Op<pphlo::AndOp>())) {
              if (!pass) {
                return failure();
              }
            }

            // or blocks
            if ((i & 1) == 0 && matchPattern(&instr, m_Op<pphlo::OrOp>())) {
              if (!pass) {
                return failure();
              }
            }

            lhs_idx--;
            rhs_idx++;
          }
        }

        rewriter.replaceOpWithNewOp<pphlo::SimpleSortOp>(
            op, op.getResultTypes(), op.getOperands(), op.getDimensionAttr(),
            rewriter.getI64IntegerAttr(key_nums), direction);
        return success();
      }
    }

    return failure();
  }
};

struct SortLowering : public SortLoweringBase<SortLowering> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    GreedyRewriteConfig config;
    config.enableFolding();
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

 private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<SortConversion>(ctx);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSortLowering() {
  return std::make_unique<SortLowering>();
}

}  // namespace mlir::spu::pphlo
