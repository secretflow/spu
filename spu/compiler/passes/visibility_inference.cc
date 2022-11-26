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

#include "spu/compiler/passes/visibility_inference.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "yacl/base/exception.h"

#include "spu/dialect/pphlo_base_enums.h"

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
  auto ifOp = llvm::dyn_cast<mhlo::IfOp>(op);

  // Infer true branch
  for (const auto &blkarg : ifOp.getTrueBranch().getArguments()) {
    ValueVis_.setValueVisibility(
        blkarg, ValueVis_.getValueVisibility(
                    ifOp.getTrueBranch().getArgument(blkarg.getArgNumber())));
  }
  inferRegion(ifOp.getTrueBranch());

  // Infer false branch
  for (const auto &blkarg : ifOp.getFalseBranch().getArguments()) {
    ValueVis_.setValueVisibility(
        blkarg, ValueVis_.getValueVisibility(
                    ifOp.getFalseBranch().getArgument(blkarg.getArgNumber())));
  }
  inferRegion(ifOp.getFalseBranch());

  // Infer result visibility
  auto &true_return = ifOp.getTrueBranch().back().back();
  auto &false_return = ifOp.getFalseBranch().back().back();
  YACL_ENFORCE(llvm::isa<mhlo::ReturnOp>(true_return));
  YACL_ENFORCE(llvm::isa<mhlo::ReturnOp>(false_return));
  YACL_ENFORCE(true_return.getNumOperands() == false_return.getNumOperands());
  YACL_ENFORCE(true_return.getNumOperands() == ifOp->getNumResults());

  for (const auto &ret : llvm::enumerate(ifOp->getResults())) {
    SmallVector<Visibility, 2> vis;

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

void VisibilityInference::inferWhile(Operation &op) {
  auto whileOp = llvm::dyn_cast<mhlo::WhileOp>(op);

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
    YACL_ENFORCE(llvm::isa<mhlo::ReturnOp>(body_return));

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
  auto sortOp = llvm::dyn_cast<mhlo::SortOp>(op);

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
}

void VisibilityInference::inferSelectAndScatter(Operation &op) {
  auto selectAndScatterOp = llvm::dyn_cast<mhlo::SelectAndScatterOp>(op);

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
  YACL_ENFORCE(llvm::isa<mhlo::ReturnOp>(scatter_return));
  YACL_ENFORCE(
      llvm::dyn_cast<mhlo::ReturnOp>(scatter_return)->getNumOperands() == 1);

  ValueVis_.setValueVisibility(
      selectAndScatterOp.getResult(),
      ValueVis_.getValueVisibility(scatter_return.getOperand(0)));
}

void VisibilityInference::inferOperation(Operation &op) {
  if (llvm::isa<mhlo::ReduceOp>(op)) {
    inferReduce<mhlo::ReduceOp>(op);
  } else if (llvm::isa<mhlo::ReduceWindowOp>(op)) {
    inferReduce<mhlo::ReduceWindowOp>(op);
  } else if (llvm::isa<mhlo::WhileOp>(op)) {
    inferWhile(op);
  } else if (llvm::isa<mhlo::IfOp>(op)) {
    inferIf(op);
  } else if (llvm::isa<mhlo::ConstantOp>(op)) {
    // Constant always returns public
    ValueVis_.setValueVisibility(op.getResult(0), Visibility::VIS_PUBLIC);
  } else if (llvm::isa<mhlo::SortOp>(op)) {
    inferSort(op);
  } else if (llvm::isa<mhlo::GatherOp>(op)) {
    // For gather op, visibility should be the same as first operand
    ValueVis_.setValueVisibility(
        op.getResult(0), ValueVis_.getValueVisibility(op.getOperand(0)));
  } else if (llvm::isa<mhlo::SelectAndScatterOp>(op)) {
    inferSelectAndScatter(op);
  } else if (op.getNumResults() == 1) {
    SmallVector<Visibility, 2> operand_vis;
    for (auto operand : op.getOperands()) {
      operand_vis.emplace_back(ValueVis_.getValueVisibility(operand));
    }
    auto ret_vis = TypeTools::inferResultVisibility(operand_vis);
    ValueVis_.setValueVisibility(op.getResult(0), ret_vis);
  } else if (llvm::isa<mlir::func::ReturnOp>(op) ||
             llvm::isa<mhlo::ReturnOp>(op)) {
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