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

#include "libspu/dialect/pphlo/transforms/legalize_from_stablehlo/value_visibility_map.h"

#include "libspu/core/prelude.h"

namespace mlir::spu::pphlo {

Visibility ValueVisibilityMap::getValueVisibility(const Value &v) const {
  const auto &iter = value_vis_.find(v);
  SPU_ENFORCE(iter != value_vis_.end());
  return iter->second;
}

void ValueVisibilityMap::setValueVisibility(const Value &val, Visibility vis) {
  value_vis_[val] = vis;
}

void ValueVisibilityMap::setOperationInputVisibility(
    Operation *op, llvm::SmallVector<Visibility> &&vis) {
  op_in_vis_[op] = std::move(vis);
}

void ValueVisibilityMap::setOperationInputVisibility(
    Operation *op, llvm::ArrayRef<Visibility> vis) {
  op_in_vis_[op] = llvm::SmallVector<Visibility>(vis);
}

}  // namespace mlir::spu::pphlo
