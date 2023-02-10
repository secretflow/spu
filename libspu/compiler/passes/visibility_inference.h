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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "libspu/compiler/passes/value_visibility_map.h"
#include "libspu/dialect/pphlo_types.h"

namespace mlir::pphlo {

class VisibilityInference {
public:
  explicit VisibilityInference(ValueVisibilityMap &ValueVis)
      : ValueVis_(ValueVis) {}

  void inferFunc(func::FuncOp &func);
  void inferRegion(Region &region);
  void inferBlock(Block &blk);
  void inferOperation(Operation &op);

private:
  void inferWhile(Operation &op);
  void inferIf(Operation &op);
  void inferCase(Operation &op);
  void inferSort(Operation &op);
  void inferSelectAndScatter(Operation &op);

  template <class T>
  void inferReduce(Operation &op) {
    auto reduceOp = llvm::dyn_cast<T>(op);

    size_t num_results = op.getNumResults();
    std::vector<Visibility> input_vis;
    for (size_t idx = 0; idx < num_results; ++idx) {
      auto inputVis = ValueVis_.getValueVisibility(reduceOp.getOperands()[idx]);
      auto initVis =
          ValueVis_.getValueVisibility(reduceOp.getInitValues()[idx]);

      auto promoted_vis = TypeTools::inferResultVisibility({inputVis, initVis});
      input_vis.emplace_back(promoted_vis);

      ValueVis_.setValueVisibility(reduceOp.getBody().getArgument(idx),
                                   promoted_vis);
      ValueVis_.setValueVisibility(
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
    std::vector<Visibility> ret_vis;
    for (size_t idx = 0; idx < reduceOp->getNumResults(); ++idx) {
      auto resultVis =
          ValueVis_.getValueVisibility(terminator->getOperand(idx));
      ValueVis_.setValueVisibility(reduceOp->getResult(idx), resultVis);
      ret_vis.emplace_back(resultVis);
      if (resultVis != input_vis[idx]) {
        reinfer = true;
      }
    }

    if (reinfer) {
      for (size_t idx = 0; idx < num_results; ++idx) {
        ValueVis_.setValueVisibility(reduceOp.getBody().getArgument(idx),
                                     ret_vis[idx]);
        ValueVis_.setValueVisibility(
            reduceOp.getBody().getArgument(num_results + idx), ret_vis[idx]);
      }

      // ret0 = reduce(init0, val0)
      // Push inputs to body region
      inferRegion(reduceOp.getBody());
    }
  }

  ValueVisibilityMap &ValueVis_;
};

} // namespace mlir::pphlo
