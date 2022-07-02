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

#include "spu/compiler/passes/value_visibility_map.h"

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
  void inferReduce(Operation &op);
  void inferReduceWindow(Operation &op);
  void inferWhile(Operation &op);
  void inferIf(Operation &op);
  void inferSort(Operation &op);
  void inferSelectAndScatter(Operation &op);

  ValueVisibilityMap &ValueVis_;
};

} // namespace mlir::pphlo
