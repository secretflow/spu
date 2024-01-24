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

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"

#include "libspu/dialect/pphlo/types.h"

namespace mlir::spu::pphlo {

class ValueVisibilityMap {
private:
  llvm::DenseMap<Value, Visibility> value_vis_;
  llvm::DenseMap<Operation *, llvm::SmallVector<Visibility>> op_in_vis_;
  llvm::SmallVector<Visibility> input_vis_;
  llvm::SmallVector<Visibility> output_vis_;

public:
  Visibility getValueVisibility(const Value &v) const;
  std::optional<llvm::ArrayRef<Visibility>>
  getOperationInputVisibility(Operation *op) const {
    auto iter = op_in_vis_.find(op);
    if (iter == op_in_vis_.end()) {
      return std::nullopt;
    }
    return iter->getSecond();
  }

  Visibility getInputsVisibility(int64_t idx) const { return input_vis_[idx]; }
  Visibility getOutputVisibility(int64_t idx) const { return output_vis_[idx]; }

  void appendInputVisibility(Visibility vis) { input_vis_.emplace_back(vis); }
  void appendOutputVisibility(Visibility vis) { output_vis_.emplace_back(vis); }

  void setValueVisibility(const Value &val, Visibility vis);
  void setOperationInputVisibility(Operation *op,
                                   llvm::SmallVector<Visibility> &&vis);
  void setOperationInputVisibility(Operation *op,
                                   llvm::ArrayRef<Visibility> vis);
};

} // namespace mlir::spu::pphlo
