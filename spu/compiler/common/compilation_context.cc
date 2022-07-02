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

#include "spu/compiler/common/compilation_context.h"

#include "llvm/Support/ErrorHandling.h"
#include "yasl/base/exception.h"

#include "spu/compiler/common/ir_printer_config.h"

namespace {

void SPUErrorHandler(void * /*use_data*/, const char *reason,
                     bool /*gen_crash_diag*/) {
  YASL_THROW(reason);
}

} // namespace

namespace spu::compiler {

CompilationContext::CompilationContext() {
  // Set an error handler
  llvm::remove_fatal_error_handler();
  llvm::install_fatal_error_handler(SPUErrorHandler);
}

CompilationContext::~CompilationContext() {
  llvm::remove_fatal_error_handler();
}

std::unique_ptr<mlir::PassManager::IRPrinterConfig>
CompilationContext::getIRPrinterConfig() const {
  if (pp_config_ == nullptr) {
    return nullptr;
  }
  auto config = static_cast<mlir::pphlo::IRPrinterConfig *>(pp_config_.get());
  return std::make_unique<mlir::pphlo::IRPrinterConfig>(*config);
}

void CompilationContext::enablePrettyPrintWithDir(std::string_view dir) {
  pp_config_ = std::make_unique<mlir::pphlo::IRPrinterConfig>(dir);
}

void CompilationContext::setupPrettyPrintConfigurations(mlir::PassManager *pm) {
  if (hasPrettyPrintEnabled()) {
    getMLIRContext()->disableMultithreading();
    pm->enableIRPrinting(getIRPrinterConfig());
  }
}

std::filesystem::path CompilationContext::getPrettyPrintDir() const {
  YASL_ENFORCE(hasPrettyPrintEnabled());
  return static_cast<const mlir::pphlo::IRPrinterConfig *>(pp_config_.get())
      ->GetPrettyPrintDir();
}

} // namespace spu::compiler
