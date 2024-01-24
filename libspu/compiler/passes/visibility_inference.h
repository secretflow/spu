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

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "libspu/compiler/passes/value_visibility_map.h"
#include "libspu/core/prelude.h"

namespace mlir::spu::pphlo {

class VisibilityInference {
public:
  explicit VisibilityInference(MLIRContext *context,
                               ValueVisibilityMap &value_vis)
      : value_vis_(value_vis), tools_(context) {}

  void infer(func::FuncOp &func);

private:
  void inferRegion(Region &region);
  void inferBlock(Block &blk);
  void inferOperation(Operation &op);
  void inferWhile(Operation &op);
  void inferIf(Operation &op);
  void inferCase(Operation &op);
  void inferSort(Operation &op);
  void inferSelectAndScatter(Operation &op);
  void inferIntrinsic(Operation &op);

  template <class T>
  void inferReduce(Operation &op) {
    auto reduceOp = llvm::dyn_cast<T>(op);

    size_t num_results = op.getNumResults();
    llvm::SmallVector<Visibility> input_vis(num_results * 2);
    for (size_t idx = 0; idx < num_results; ++idx) {
      auto inputVis =
          value_vis_.getValueVisibility(reduceOp.getOperands()[idx]);
      auto initVis =
          value_vis_.getValueVisibility(reduceOp.getInitValues()[idx]);

      auto promoted_vis = tools_.computeCommonVisibility({inputVis, initVis});
      input_vis[idx] = promoted_vis;
      input_vis[idx + num_results] = promoted_vis;

      // Set region block arg visibility
      value_vis_.setValueVisibility(reduceOp.getBody().getArgument(idx),
                                    promoted_vis);
      value_vis_.setValueVisibility(
          reduceOp.getBody().getArgument(num_results + idx), promoted_vis);
    }

    // ret0 = reduce(init0, val0)
    // Push inputs to body region
    inferRegion(reduceOp.getBody());

    // Get body return
    bool reinfer = false;
    auto *terminator = reduceOp.getBody().back().getTerminator();
    SPU_ENFORCE(terminator &&
                terminator->getNumOperands() == reduceOp->getNumResults());
    for (size_t idx = 0; idx < reduceOp->getNumResults(); ++idx) {
      auto result_vis =
          value_vis_.getValueVisibility(terminator->getOperand(idx));
      if (result_vis != input_vis[idx]) {
        reinfer = true;
        input_vis[idx] = result_vis;
        input_vis[idx + num_results] = result_vis;

        value_vis_.setValueVisibility(reduceOp.getBody().getArgument(idx),
                                      result_vis);
        value_vis_.setValueVisibility(
            reduceOp.getBody().getArgument(num_results + idx), result_vis);
      }
    }

    if (reinfer) {
      // ret0 = reduce(init0, val0)
      // Push inputs to body region
      inferRegion(reduceOp.getBody());
    }

    for (size_t idx = 0; idx < reduceOp->getNumResults(); ++idx) {
      auto result_vis =
          value_vis_.getValueVisibility(terminator->getOperand(idx));
      value_vis_.setValueVisibility(reduceOp->getResult(idx), result_vis);
    }

    value_vis_.setOperationInputVisibility(&op, std::move(input_vis));
  }

  ValueVisibilityMap &value_vis_;
  TypeTools tools_;
};

} // namespace mlir::spu::pphlo
