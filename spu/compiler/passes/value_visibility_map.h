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

#include "spu/dialect/pphlo_base_enums.h"

namespace mlir::pphlo {

class ValueVisibilityMap {
private:
  llvm::DenseMap<Value, Visibility> storage;

public:
  Visibility getValueVisibility(const Value &v) const;

  void setValueVisibility(const Value &val, Visibility vis);
};

} // namespace mlir::pphlo
