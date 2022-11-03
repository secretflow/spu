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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "spu/device/executor.h"
#include "spu/kernel/value.h"

namespace spu::device::pphlo {

class PPHloExecutor : public Executor {
public:
  explicit PPHloExecutor(HalContext *ctx);

  ~PPHloExecutor() override;

  std::vector<spu::Value> run(const std::string &code,
                              const std::vector<spu::Value> &inputs) override;

  mlir::OwningOpRef<mlir::ModuleOp> parseSourceString(const std::string &code);

private:
  std::vector<spu::Value> executeFunc(mlir::func::FuncOp &fcn,
                                      llvm::ArrayRef<spu::Value> inputs);

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

} // namespace spu::device::pphlo
