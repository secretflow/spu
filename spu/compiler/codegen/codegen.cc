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

#include "spu/compiler/codegen/codegen.h"

#include <string>

#include "llvm/Support/raw_os_ostream.h"

namespace spu::compiler {

std::string CodeGen::doit(mlir::ModuleOp module) {
  std::string ir_dump;
  llvm::raw_string_ostream stream(ir_dump);
  module.print(stream);

  return stream.str();
}

} // namespace spu::compiler
