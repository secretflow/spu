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

#include <chrono>
#include <optional>
#include <unordered_map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "spu/device/executor.h"
#include "spu/hal/context.h"
#include "spu/hal/value.h"

namespace spu::device {

class Frame;

class PPHloExecutor : public Executor {
public:
  explicit PPHloExecutor(HalContext *ctx);

  ~PPHloExecutor() override;

  std::vector<hal::Value> run(const std::string &code,
                              const std::vector<hal::Value> &inputs) override;

  mlir::OwningOpRef<mlir::ModuleOp> parseSourceString(const std::string &code);

  using timepoint_t =
      std::optional<std::chrono::high_resolution_clock::time_point>;

  timepoint_t profileStart() {
    if (hctx_->rt_config().enable_pphlo_profile()) {
      return std::chrono::high_resolution_clock::now();
    }
    return {};
  }

  void profileEnd(llvm::StringRef identifier, timepoint_t &time_point) {
    if (time_point.has_value()) {
      auto e = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
          e - *time_point);
      auto &record = op_profile_records_[identifier.str()];
      record.count++;
      record.time += duration;
    }
  }

private:
  std::vector<hal::Value> executeFunc(mlir::func::FuncOp &fcn,
                                      llvm::ArrayRef<hal::Value> inputs);

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

} // namespace spu::device
