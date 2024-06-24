// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/dialect/utils/utils.h"

#include "llvm/ADT/Twine.h"
#include "spdlog/spdlog.h"

namespace mlir::spu {

mlir::func::FuncOp get_entrypoint(ModuleOp op) {
  // Get the main function
  auto entry_func = op.lookupSymbol<mlir::func::FuncOp>("main");
  if (!entry_func) {
    auto funcs = op.getOps<func::FuncOp>();
    if (std::distance(funcs.begin(), funcs.end()) == 1) {
      entry_func = *funcs.begin();
    }
  }

  return entry_func;
}

}  // namespace mlir::spu
