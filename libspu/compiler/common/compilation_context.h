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

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include "libspu/spu.pb.h"

namespace mlir {
class PassManager;
}

namespace spu::compiler {

class CompilationContext {
public:
  CompilationContext();

  ~CompilationContext();

  mlir::MLIRContext *getMLIRContext() { return &context_; }

  /// Setup pretty print for a pass manager
  void setupPrettyPrintConfigurations(mlir::PassManager *pm);

  void setCompilerOptions(const std::string &serialized_copts);

  const CompilerOptions &getCompilerOptions() const { return options_; }

  bool hasPrettyPrintEnabled() const { return options_.enable_pretty_print(); }

  XLAPrettyPrintKind getXlaPrettyPrintKind() const {
    return options_.xla_pp_kind();
  }

  std::filesystem::path getPrettyPrintDir() const;

private:
  std::unique_ptr<mlir::PassManager::IRPrinterConfig>
  getIRPrinterConfig() const;

  mlir::MLIRContext context_;
  std::unique_ptr<mlir::PassManager::IRPrinterConfig> pp_config_;

  CompilerOptions options_;
};

} // namespace spu::compiler
