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

namespace mlir {
class PassManager;
}

namespace spu::compiler {

class CompilationContext {
public:
  CompilationContext();

  ~CompilationContext();

  mlir::MLIRContext *getMLIRContext() { return &context_; }

  /// Enable pretty print and set folder
  /// If dir does not exist, this api will create new folder
  void enablePrettyPrintWithDir(std::string_view dir);

  /// Setup pretty print for a pass manager
  void setupPrettyPrintConfigurations(mlir::PassManager *pm);

  void setInputVisibilityString(const std::string &vis) { input_vis_ = vis; }

  std::string getInputVisibilityString() const { return input_vis_; }

  bool hasPrettyPrintEnabled() const { return pp_config_ != nullptr; }

  std::filesystem::path getPrettyPrintDir() const;

private:
  std::unique_ptr<mlir::PassManager::IRPrinterConfig>
  getIRPrinterConfig() const;

  mlir::MLIRContext context_;
  std::unique_ptr<mlir::PassManager::IRPrinterConfig> pp_config_;

  std::string input_vis_;
};

} // namespace spu::compiler
