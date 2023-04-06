// Copyright 2021 Ant Group Co., Ltd.
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

#include "libspu/compiler/passes/visibility_inference.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "libspu/core/prelude.h"
#include "libspu/dialect/pphlo_base_enums.h"

namespace mlir::pphlo {

void VisibilityInference::inferFunc(func::FuncOp &func) {
  for (auto &blk : func) {
    inferBlock(blk);
  }
}

void VisibilityInference::inferRegion(Region &r) {
  for (auto &blk : r) {
    inferBlock(blk);
  }
}

void VisibilityInference::inferBlock(Block &blk) {
  for (auto &op : blk) {
    inferOperation(op);
  }
}

void VisibilityInference::inferIf(Operation &op) {
  auto ifOp = llvm::dyn_cast<stablehlo::IfOp>(op);

  llvm::SmallVector<Visibility, 2> input_vis;
  for (const auto &operand : op.getOperands()) {
    input_vis.emplace_back(ValueVis_.getValueVisibility(operand));
  }

  // Infer true branch
  for (const auto &blkarg : ifOp.getTrueBranch().getArguments()) {
    ValueVis_.setValueVisibility(blkarg, input_vis[blkarg.getArgNumber()]);
  }
  inferRegion(ifOp.getTrueBranch());

  // Infer false branch
  for (const auto &blkarg : ifOp.getFalseBranch().getArguments()) {
    ValueVis_.setValueVisibility(blkarg, input_vis[blkarg.getArgNumber()]);
  }
  inferRegion(ifOp.getFalseBranch());

  // Infer result visibility
  auto &true_return = ifOp.getTrueBranch().back().back();
  auto &false_return = ifOp.getFalseBranch().back().back();
  SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(true_return));
  SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(false_return));

  // Cond vis
  auto cond_vis = ValueVis_.getValueVisibility(ifOp.getPred());

  for (const auto &ret : llvm::enumerate(ifOp->getResults())) {
    SmallVector<Visibility, 2> vis;

    // Always push cond into consideration
    vis.emplace_back(cond_vis);

    // Get true branch result vis
    vis.emplace_back(
        ValueVis_.getValueVisibility(true_return.getOperand(ret.index())));
    // Get false branch result vis
    vis.emplace_back(
        ValueVis_.getValueVisibility(false_return.getOperand(ret.index())));

    ValueVis_.setValueVisibility(ret.value(),
                                 TypeTools::inferResultVisibility(vis));
  }
}

void VisibilityInference::inferCase(Operation &op) {
  auto caseOp = llvm::dyn_cast<stablehlo::CaseOp>(op);

  // Collect
  llvm::SmallVector<Visibility, 2> input_vis;
  llvm::SmallVector<Operation *, 2> returns;
  for (const auto &operand : caseOp->getOperands()) {
    input_vis.emplace_back(ValueVis_.getValueVisibility(operand));
  }

  // Infer each branch
  for (auto &region : caseOp.getBranches()) {
    for (const auto &blkarg : region.getArguments()) {
      ValueVis_.setValueVisibility(blkarg, input_vis[blkarg.getArgNumber()]);
    }
    inferRegion(region);
    auto *ret = &region.back().back();
    SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(ret));
    returns.emplace_back(ret);
  }

  // Index vis
  auto index_vis = ValueVis_.getValueVisibility(caseOp.getIndex());

  // Infer result visibility
  for (const auto &ret_enu : llvm::enumerate(caseOp->getResults())) {
    SmallVector<Visibility, 2> vis;

    vis.emplace_back(index_vis);

    for (auto *ret : returns) {
      vis.emplace_back(
          ValueVis_.getValueVisibility(ret->getOperand(ret_enu.index())));
    }

    ValueVis_.setValueVisibility(ret_enu.value(),
                                 TypeTools::inferResultVisibility(vis));
  }
}

void VisibilityInference::inferWhile(Operation &op) {
  auto whileOp = llvm::dyn_cast<stablehlo::WhileOp>(op);

  // Initial body visibility
  SmallVector<Visibility> input_vis(op.getNumOperands());
  SmallVector<Visibility> result_vis(op.getNumOperands());

  for (int64_t idx = 0; idx < op.getNumOperands(); ++idx) {
    input_vis[idx] = ValueVis_.getValueVisibility(whileOp->getOperand(idx));
  }

  bool converge = false;
  do {
    // Push visibility to block args
    for (const auto &blkarg : whileOp.getBody().getArguments()) {
      ValueVis_.setValueVisibility(blkarg, input_vis[blkarg.getArgNumber()]);
    }

    // Infer body region
    inferRegion(whileOp.getBody());

    // Get result visibility
    auto &body_return = *whileOp.getBody().front().getTerminator();
    SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(body_return));

    // Update visibility
    for (int64_t idx = 0; idx < body_return.getNumOperands(); ++idx) {
      result_vis[idx] =
          ValueVis_.getValueVisibility(body_return.getOperand(idx));
    }

    converge = (input_vis == result_vis);
    input_vis.swap(result_vis);
  } while (!converge);

  for (int64_t idx = 0; idx < op.getNumOperands(); ++idx) {
    ValueVis_.setValueVisibility(whileOp.getBody().getArgument(idx),
                                 input_vis[idx]);
    ValueVis_.setValueVisibility(whileOp.getCond().getArgument(idx),
                                 input_vis[idx]);
  }

  inferRegion(whileOp.getCond());

  // Update result visibility
  for (int64_t idx = 0; idx < op.getNumResults(); ++idx) {
    ValueVis_.setValueVisibility(op.getResult(idx), input_vis[idx]);
  }
}

void VisibilityInference::inferSort(Operation &op) {
  auto sortOp = llvm::dyn_cast<stablehlo::SortOp>(op);

  // Push inputs to body region
  for (const auto &in : llvm::enumerate(op.getOperands())) {
    auto inputVis = ValueVis_.getValueVisibility(in.value());
    ValueVis_.setValueVisibility(
        sortOp.getComparator().getArgument(2 * in.index()), inputVis);
    ValueVis_.setValueVisibility(
        sortOp.getComparator().getArgument(2 * in.index() + 1), inputVis);

    // Sort does not change result vis
    ValueVis_.setValueVisibility(op.getResult(in.index()), inputVis);
  }

  inferRegion(sortOp.getComparator());

  // Get comparator visibility
  auto &comp_ret = *sortOp.getComparator().front().getTerminator();
  SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(comp_ret));

  if (ValueVis_.getValueVisibility(comp_ret.getOperand(0)) ==
      Visibility::VIS_SECRET) {
    // If comparator result is secret, all results are secrets
    for (const auto &in : llvm::enumerate(op.getOperands())) {
      ValueVis_.setValueVisibility(
          sortOp.getComparator().getArgument(2 * in.index()),
          Visibility::VIS_SECRET);
      ValueVis_.setValueVisibility(
          sortOp.getComparator().getArgument(2 * in.index() + 1),
          Visibility::VIS_SECRET);

      // Sort does not change result vis
      ValueVis_.setValueVisibility(op.getResult(in.index()),
                                   Visibility::VIS_SECRET);
    }

    inferRegion(sortOp.getComparator());
  }
}

void VisibilityInference::inferSelectAndScatter(Operation &op) {
  auto selectAndScatterOp = llvm::dyn_cast<stablehlo::SelectAndScatterOp>(op);

  auto op_vis = ValueVis_.getValueVisibility(selectAndScatterOp.getOperand());
  auto source_vis =
      ValueVis_.getValueVisibility(selectAndScatterOp.getSource());
  auto init_vis =
      ValueVis_.getValueVisibility(selectAndScatterOp.getInitValue());

  // init and operand must have the same visibility
  auto promoted_init_op_vis =
      TypeTools::inferResultVisibility({op_vis, init_vis});

  // Select region
  {
    ValueVis_.setValueVisibility(selectAndScatterOp.getSelect().getArgument(0),
                                 promoted_init_op_vis);
    ValueVis_.setValueVisibility(selectAndScatterOp.getSelect().getArgument(1),
                                 promoted_init_op_vis);
    inferRegion(selectAndScatterOp.getSelect());
  }
  // Scatter region
  {
    ValueVis_.setValueVisibility(selectAndScatterOp.getScatter().getArgument(0),
                                 source_vis);
    ValueVis_.setValueVisibility(selectAndScatterOp.getScatter().getArgument(1),
                                 promoted_init_op_vis);
    inferRegion(selectAndScatterOp.getScatter());
  }

  // Result visibility should be same as scatter result
  // body return
  auto &scatter_return = selectAndScatterOp.getScatter().back().back();
  SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(scatter_return));
  SPU_ENFORCE(
      llvm::dyn_cast<stablehlo::ReturnOp>(scatter_return)->getNumOperands() ==
      1);

  ValueVis_.setValueVisibility(
      selectAndScatterOp.getResult(),
      ValueVis_.getValueVisibility(scatter_return.getOperand(0)));
}

void VisibilityInference::inferOperation(Operation &op) {
  if (llvm::isa<stablehlo::ReduceOp>(op)) {
    inferReduce<stablehlo::ReduceOp>(op);
  } else if (llvm::isa<stablehlo::ReduceWindowOp>(op)) {
    inferReduce<stablehlo::ReduceWindowOp>(op);
  } else if (llvm::isa<stablehlo::WhileOp>(op)) {
    inferWhile(op);
  } else if (llvm::isa<stablehlo::IfOp>(op)) {
    inferIf(op);
  } else if (llvm::isa<stablehlo::CaseOp>(op)) {
    inferCase(op);
  } else if (llvm::isa<stablehlo::ConstantOp>(op)) {
    // Constant always returns public
    ValueVis_.setValueVisibility(op.getResult(0), Visibility::VIS_PUBLIC);
  } else if (llvm::isa<stablehlo::SortOp>(op)) {
    inferSort(op);
  } else if (llvm::isa<stablehlo::GatherOp>(op)) {
    // For gather op, if either operand or indices is a secret, result is a
    // secret
    auto operand_vis = ValueVis_.getValueVisibility(op.getOperand(0));
    auto indices_vis = ValueVis_.getValueVisibility(op.getOperand(1));
    ValueVis_.setValueVisibility(
        op.getResult(0),
        TypeTools::inferResultVisibility({operand_vis, indices_vis}));
  } else if (llvm::isa<stablehlo::SelectAndScatterOp>(op)) {
    inferSelectAndScatter(op);
  } else if (op.getNumResults() == 1) {
    SmallVector<Visibility, 2> operand_vis;
    for (auto operand : op.getOperands()) {
      operand_vis.emplace_back(ValueVis_.getValueVisibility(operand));
    }
    auto ret_vis = TypeTools::inferResultVisibility(operand_vis);
    ValueVis_.setValueVisibility(op.getResult(0), ret_vis);
  } else if (llvm::isa<mlir::func::ReturnOp>(op) ||
             llvm::isa<stablehlo::ReturnOp>(op)) {
    // Do nothing
  } else {
    std::string dump;
    llvm::raw_string_ostream debug_s(dump);
    debug_s << "Unhandled op: ";
    op.print(debug_s);
    llvm_unreachable(debug_s.str().c_str());
  }
}
} // namespace mlir::pphlo