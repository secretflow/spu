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

#include "spu/device/pphlo/executor.h"

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "yacl/base/exception.h"

#include "spu/device/frame.h"
#include "spu/device/pphlo/region_executor.h"
#include "spu/dialect/pphlo_dialect.h"
#include "spu/kernel/context.h"
#include "spu/kernel/value.h"

namespace spu::device::pphlo {

namespace {

std::mutex ErrorHandlerMutex;

void SPUErrorHandler(void * /*use_data*/, const char *reason,
                     bool /*gen_crash_diag*/) {
  YACL_THROW(reason);
}

} // namespace

std::vector<spu::Value>
PPHloExecutor::executeFunc(mlir::func::FuncOp &fcn,
                           llvm::ArrayRef<spu::Value> inputs) {
  Frame callFrame;
  RegionExecutor executor(getContext(), &callFrame, op_profiler_);
  return executor.executeRegion(fcn.getBody(), inputs);
}

PPHloExecutor::PPHloExecutor(HalContext *ctx) : Executor(ctx) {
  // Set an error handler
  {
    std::lock_guard<std::mutex> guard(ErrorHandlerMutex);
    llvm::remove_fatal_error_handler();
    llvm::install_fatal_error_handler(SPUErrorHandler);
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::pphlo::PPHloDialect, mlir::func::FuncDialect>();
  mlir_context_ = std::make_unique<mlir::MLIRContext>(registry);

  int64_t tr_mask = 0;
  if (ctx->rt_config().enable_action_trace()) {
    tr_mask |= TR_LOG;
  }

  if (ctx->rt_config().enable_hal_profile()) {
    tr_mask |= TR_HLO | TR_HAL | TR_MPC;
    tr_mask |= TR_REC;
  }

  getTracer(GET_CTX_NAME(ctx))->setMask(tr_mask);
  getTracer(GET_CTX_NAME(ctx))->clearRecords();
}

PPHloExecutor::~PPHloExecutor() {
  std::lock_guard<std::mutex> guard(ErrorHandlerMutex);
  llvm::remove_fatal_error_handler();
}

std::vector<spu::Value>
PPHloExecutor::run(const std::string &code,
                   const std::vector<spu::Value> &inputs) {
  auto moduleOpRef =
      mlir::parseSourceString<mlir::ModuleOp>(code, mlir_context_.get());

  if (hctx_->rt_config().enable_pphlo_trace()) {
    SPDLOG_INFO("Executing module {}",
                moduleOpRef->getName().value_or("Unnamed"));
  }

  auto entry_function = moduleOpRef->lookupSymbol<mlir::func::FuncOp>("main");
  YACL_ENFORCE(entry_function);

  return executeFunc(entry_function, inputs);
}

mlir::OwningOpRef<mlir::ModuleOp>
PPHloExecutor::parseSourceString(const std::string &code) {
  auto moduleOp =
      mlir::parseSourceString<mlir::ModuleOp>(code, mlir_context_.get());
  return moduleOp;
}

} // namespace spu::device::pphlo
