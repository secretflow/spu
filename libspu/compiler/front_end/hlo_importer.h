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

#include <string>

#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace spu::compiler {

class CompilationContext;

class HloImporter final {
public:
  // clang-format off
  explicit HloImporter(CompilationContext *context) : context_(context) {};
  // clang-format on

  /// Load a xla module and returns a mlir-hlo module
  mlir::OwningOpRef<mlir::ModuleOp>
  parseXlaModuleFromString(const std::string &content);

private:
  CompilationContext *context_;
};

} // namespace spu::compiler
