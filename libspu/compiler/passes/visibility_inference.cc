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

namespace mlir::spu::pphlo {

void VisibilityInference::infer(func::FuncOp &func) {
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

// %result = "stablehlo.if"(%pred) ({
//   "stablehlo.return"(%result_true_branch) : (tensor<i32>) -> ()
// }, {
//   "stablehlo.return"(%result_false_branch) : (tensor<i32>) -> ()
// }) : (tensor<i1>) -> tensor<i32>
void VisibilityInference::inferIf(Operation &op) {
  auto ifOp = llvm::dyn_cast<stablehlo::IfOp>(op);

  auto pred_vis = value_vis_.getValueVisibility(ifOp.getPred());

  // C1 input_types(true_branch) = input_types(false_branch) = []
  SPU_ENFORCE(ifOp.getTrueBranch().getNumArguments() == 0 &&
              ifOp.getFalseBranch().getNumArguments() == 0);
  // Infer true and false branch
  inferRegion(ifOp.getTrueBranch());
  inferRegion(ifOp.getFalseBranch());

  // Infer result visibility
  auto &true_return = ifOp.getTrueBranch().back().back();
  auto &false_return = ifOp.getFalseBranch().back().back();
  SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(true_return));
  SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(false_return));

  // Cond vis
  SmallVector<Visibility, 3> ret_vis(3);
  SmallVector<Visibility> return_vis;

  ret_vis[0] = pred_vis;
  for (const auto &ret : llvm::enumerate(ifOp->getResults())) {
    // Get true branch result vis
    ret_vis[1] =
        value_vis_.getValueVisibility(true_return.getOperand(ret.index()));
    // Get false branch result vis
    ret_vis[2] =
        value_vis_.getValueVisibility(false_return.getOperand(ret.index()));

    auto expected_vis = tools_.computeCommonVisibility(ret_vis);

    value_vis_.setValueVisibility(ret.value(), expected_vis);

    // Force return from both braches to be same type
    return_vis.emplace_back(expected_vis);
  }

  value_vis_.setOperationInputVisibility(&true_return, return_vis);
  value_vis_.setOperationInputVisibility(&false_return, return_vis);
}

// %result0, %result1 = "stablehlo.case"(%index) ({
//   "stablehlo.return"(%result_branch0, %result_branch0) : (tensor<2xi64>,
//   tensor<2xi64>) -> ()
// }, {
//   "stablehlo.return"(%result_branch1, %result_branch1) : (tensor<2xi64>,
//   tensor<2xi64>) -> ()
// }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
void VisibilityInference::inferCase(Operation &op) {
  auto caseOp = llvm::dyn_cast<stablehlo::CaseOp>(op);

  // Collect
  llvm::SmallVector<Operation *, 3> returns;

  // Infer each branch
  for (auto &region : caseOp.getBranches()) {
    // C2
    SPU_ENFORCE(region.getNumArguments() == 0);
    inferRegion(region);
    auto *ret = &region.back().back();
    SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(ret));
    returns.emplace_back(ret);
  }

  // Index vis
  SmallVector<Visibility> result_vis(returns.size() + 1);
  SmallVector<Visibility> return_vis(caseOp->getNumResults());
  result_vis[0] = value_vis_.getValueVisibility(caseOp.getIndex());

  // Infer result visibility
  for (const auto &ret_enu : llvm::enumerate(caseOp->getResults())) {
    for (size_t idx = 0; idx < returns.size(); ++idx) {
      result_vis[idx + 1] = value_vis_.getValueVisibility(
          returns[idx]->getOperand(ret_enu.index()));
    }

    auto expected_vis = tools_.computeCommonVisibility(result_vis);
    value_vis_.setValueVisibility(ret_enu.value(), expected_vis);

    return_vis[ret_enu.index()] = expected_vis;
  }

  for (const auto &rt : returns) {
    value_vis_.setOperationInputVisibility(rt, return_vis);
  }
}

void VisibilityInference::inferWhile(Operation &op) {
  auto whileOp = llvm::dyn_cast<stablehlo::WhileOp>(op);

  // Initial body visibility
  SmallVector<Visibility> input_vis(op.getNumOperands());
  SmallVector<Visibility> result_vis(op.getNumOperands());

  for (int64_t idx = 0; idx < op.getNumOperands(); ++idx) {
    input_vis[idx] = value_vis_.getValueVisibility(whileOp->getOperand(idx));
  }

  bool converge = false;
  do {
    // Push visibility to block args
    for (const auto &blkarg : whileOp.getBody().getArguments()) {
      value_vis_.setValueVisibility(blkarg, input_vis[blkarg.getArgNumber()]);
    }

    // Infer body region
    inferRegion(whileOp.getBody());

    // Get result visibility
    auto &body_return = *whileOp.getBody().front().getTerminator();
    SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(body_return));

    // Update visibility
    for (int64_t idx = 0; idx < body_return.getNumOperands(); ++idx) {
      result_vis[idx] =
          value_vis_.getValueVisibility(body_return.getOperand(idx));
    }

    converge = (input_vis == result_vis);
    input_vis.swap(result_vis);
  } while (!converge);

  for (int64_t idx = 0; idx < op.getNumOperands(); ++idx) {
    value_vis_.setValueVisibility(whileOp.getBody().getArgument(idx),
                                  input_vis[idx]);
    value_vis_.setValueVisibility(whileOp.getCond().getArgument(idx),
                                  input_vis[idx]);
  }

  inferRegion(whileOp.getCond());

  // Update result visibility
  for (int64_t idx = 0; idx < op.getNumResults(); ++idx) {
    value_vis_.setValueVisibility(op.getResult(idx), input_vis[idx]);
  }

  value_vis_.setOperationInputVisibility(&op, std::move(input_vis));
}

void VisibilityInference::inferSort(Operation &op) {
  auto sortOp = llvm::dyn_cast<stablehlo::SortOp>(op);

  // Push inputs to body region
  for (const auto &in : llvm::enumerate(op.getOperands())) {
    auto inputVis = value_vis_.getValueVisibility(in.value());
    value_vis_.setValueVisibility(
        sortOp.getComparator().getArgument(2 * in.index()), inputVis);
    value_vis_.setValueVisibility(
        sortOp.getComparator().getArgument(2 * in.index() + 1), inputVis);

    // Sort does not change result vis
    value_vis_.setValueVisibility(op.getResult(in.index()), inputVis);
  }

  inferRegion(sortOp.getComparator());

  // Get comparator visibility
  auto &comp_ret = *sortOp.getComparator().front().getTerminator();
  SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(comp_ret));

  auto comp_ret_vis = value_vis_.getValueVisibility(comp_ret.getOperand(0));

  // If comparator result is secret, all results are secrets
  for (int64_t idx = 0; idx < op.getNumResults(); ++idx) {
    auto in_vis = value_vis_.getValueVisibility(op.getOperand(idx));
    value_vis_.setValueVisibility(
        op.getResult(idx),
        tools_.computeCommonVisibility({comp_ret_vis, in_vis}));
  }
}

void VisibilityInference::inferSelectAndScatter(Operation &op) {
  auto selectAndScatterOp = llvm::dyn_cast<stablehlo::SelectAndScatterOp>(op);

  auto op_vis = value_vis_.getValueVisibility(selectAndScatterOp.getOperand());
  auto source_vis =
      value_vis_.getValueVisibility(selectAndScatterOp.getSource());
  auto init_vis =
      value_vis_.getValueVisibility(selectAndScatterOp.getInitValue());

  // init and operand must have the same visibility
  auto promoted_init_op_vis =
      tools_.computeCommonVisibility({op_vis, init_vis});

  // Select region
  {
    value_vis_.setValueVisibility(selectAndScatterOp.getSelect().getArgument(0),
                                  promoted_init_op_vis);
    value_vis_.setValueVisibility(selectAndScatterOp.getSelect().getArgument(1),
                                  promoted_init_op_vis);
    inferRegion(selectAndScatterOp.getSelect());
  }
  // Scatter region
  {
    value_vis_.setValueVisibility(
        selectAndScatterOp.getScatter().getArgument(0), source_vis);
    value_vis_.setValueVisibility(
        selectAndScatterOp.getScatter().getArgument(1), promoted_init_op_vis);
    inferRegion(selectAndScatterOp.getScatter());
  }

  // Result visibility should be same as scatter result
  // body return
  auto &scatter_return = selectAndScatterOp.getScatter().back().back();
  SPU_ENFORCE(llvm::isa<stablehlo::ReturnOp>(scatter_return));
  SPU_ENFORCE(
      llvm::dyn_cast<stablehlo::ReturnOp>(scatter_return)->getNumOperands() ==
      1);

  value_vis_.setValueVisibility(
      selectAndScatterOp.getResult(),
      value_vis_.getValueVisibility(scatter_return.getOperand(0)));

  value_vis_.setOperationInputVisibility(
      &op, llvm::SmallVector<Visibility>{promoted_init_op_vis, source_vis,
                                         promoted_init_op_vis});
}

void VisibilityInference::inferIntrinsic(Operation &op) {
  [[maybe_unused]] auto c_op = llvm::dyn_cast<stablehlo::CustomCallOp>(op);
  // if (c_op.getCallTargetName() == "bla") {
  //    ... special inference rule
  //    return;
  // }

  if (c_op.getCallTargetName() == "mhlo.topk") {
    auto input_vis = value_vis_.getValueVisibility(op.getOperand(0));
    value_vis_.setValueVisibility(c_op->getResult(0), input_vis);
    value_vis_.setValueVisibility(c_op.getResult(1), input_vis);
    return;
  }

  // Default rule
  if (op.getNumResults() == 1) {
    SmallVector<Visibility, 2> operand_vis;
    for (auto operand : op.getOperands()) {
      operand_vis.emplace_back(value_vis_.getValueVisibility(operand));
    }
    auto ret_vis = tools_.computeCommonVisibility(operand_vis);
    value_vis_.setValueVisibility(op.getResult(0), ret_vis);
  } else {
    SPU_ENFORCE(op.getNumResults() == op.getNumOperands(),
                "Default intrinsic inference can only handle single output or "
                "#output matches #input");

    for (int64_t idx = 0; idx < op.getNumResults(); ++idx) {
      value_vis_.setValueVisibility(
          op.getResult(idx), value_vis_.getValueVisibility(op.getOperand(idx)));
    }
  }
}

void VisibilityInference::inferOperation(Operation &op) {
  if (llvm::isa<stablehlo::ConstantOp, stablehlo::IotaOp>(op)) {
    // Constant/Iota always returns public
    value_vis_.setValueVisibility(op.getResult(0), Visibility::PUBLIC);
  } else if (llvm::isa<stablehlo::ReduceOp>(op)) {
    inferReduce<stablehlo::ReduceOp>(op);
  } else if (llvm::isa<stablehlo::ReduceWindowOp>(op)) {
    inferReduce<stablehlo::ReduceWindowOp>(op);
  } else if (llvm::isa<stablehlo::WhileOp>(op)) {
    inferWhile(op);
  } else if (llvm::isa<stablehlo::IfOp>(op)) {
    inferIf(op);
  } else if (llvm::isa<stablehlo::CaseOp>(op)) {
    inferCase(op);
  } else if (llvm::isa<stablehlo::SortOp>(op)) {
    inferSort(op);
  } else if (llvm::isa<stablehlo::SelectAndScatterOp>(op)) {
    inferSelectAndScatter(op);
  } else if (llvm::isa<stablehlo::CustomCallOp>(op)) {
    inferIntrinsic(op);
  } else if (llvm::isa<stablehlo::PadOp, stablehlo::ConcatenateOp>(op)) {
    SmallVector<Visibility, 2> operand_vis;
    for (auto operand : op.getOperands()) {
      operand_vis.emplace_back(value_vis_.getValueVisibility(operand));
    }
    auto ret_vis = tools_.computeCommonVisibility(operand_vis);
    value_vis_.setValueVisibility(op.getResult(0), ret_vis);
    value_vis_.setOperationInputVisibility(
        &op, llvm::SmallVector<Visibility>(op.getNumOperands(), ret_vis));
  } else if (op.getNumResults() == 1) {
    SmallVector<Visibility, 2> operand_vis;
    for (auto operand : op.getOperands()) {
      operand_vis.emplace_back(value_vis_.getValueVisibility(operand));
    }
    auto ret_vis = tools_.computeCommonVisibility(operand_vis);
    value_vis_.setValueVisibility(op.getResult(0), ret_vis);
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
} // namespace mlir::spu::pphlo