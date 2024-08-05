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
#include <string>
#include <string_view>

#include "mlir/Pass/PassManager.h"

namespace mlir::pphlo {

class IRPrinterConfig final : public PassManager::IRPrinterConfig {
public:
  /*explicit*/ IRPrinterConfig(std::string_view pp_dir); // NOLINT

  IRPrinterConfig(const IRPrinterConfig &) = default;

  void printBeforeIfEnabled(Pass *pass, Operation *operation,
                            PrintCallbackFn print_callback) override;

  void printAfterIfEnabled(Pass *pass, Operation *operation,
                           PrintCallbackFn print_callback) override;

  std::filesystem::path GetPrettyPrintDir() const { return pp_dir_; }

private:
  static std::string genFileName(StringRef pass_name, StringRef stage);
  std::filesystem::path pp_dir_;
};

} // namespace mlir::pphlo
