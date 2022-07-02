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

#include "spu/compiler/common/ir_printer_config.h"

#include <chrono>

#include "fmt/chrono.h"
#include "fmt/format.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"
#include "spdlog/spdlog.h"

namespace mlir::pphlo {

/// PP counter, this is a session persistent
static std::int64_t pp_cnt = 0;

IRPrinterConfig::IRPrinterConfig(std::string_view pp_dir)
    : PassManager::IRPrinterConfig(/*printModuleScope*/ true,
                                   /*printAfterOnlyOnChange*/ true),
      pp_dir_(pp_dir) {
  auto current_time =
      fmt::format("{:%Y-%m-%d-%H:%M:%S}", std::chrono::system_clock::now());

  pp_dir_ /= current_time;

  std::error_code ec;
  if (std::filesystem::create_directories(pp_dir_, ec) == false) {
    spdlog::error("Failed to create pp folder, error = {}", ec.message());
  }
}

void IRPrinterConfig::printBeforeIfEnabled(Pass *pass, Operation *operation,
                                           PrintCallbackFn printCallback) {
  std::filesystem::path file_name =
      pp_dir_ / genFileName(pass->getName(), "before");
  std::error_code ec;
  llvm::raw_fd_ostream f(file_name.c_str(), ec, llvm::sys::fs::CD_CreateNew);
  if (ec.value() != 0) {
    spdlog::error("Open file {} failed, error = {}", file_name.c_str(),
                  ec.message());
  }
  printCallback(f);
}

void IRPrinterConfig::printAfterIfEnabled(Pass *pass, Operation *operation,
                                          PrintCallbackFn printCallback) {
  std::filesystem::path file_name =
      pp_dir_ / genFileName(pass->getName(), "after");
  std::error_code ec;
  llvm::raw_fd_ostream f(file_name.c_str(), ec, llvm::sys::fs::CD_CreateNew);
  if (ec.value() != 0) {
    spdlog::error("Open file {} failed, error = {}", file_name.c_str(),
                  ec.message());
  }
  printCallback(f);
}

std::string IRPrinterConfig::genFileName(StringRef passName,
                                         StringRef stage) const {
  return fmt::format("{}-{}-{}.mlir", pp_cnt++, passName, stage);
}

} // namespace mlir::pphlo