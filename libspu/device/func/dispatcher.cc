// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/device/func/dispatcher.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "libspu/core/trace.h"  // IWYU pragma: keep
#include "libspu/device/executor.h"
#include "libspu/device/intrinsics/dispatcher.h"
#include "libspu/device/utils/utils.h"   // IWYU pragma: keep
#include "libspu/dialect/utils/utils.h"  // IWYU pragma: keep

namespace spu::device {

namespace {

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::func::CallOp &op, const ExecutionOptions &opts) {
  // First we get the callee
  auto *module = mlir::SymbolTable::getNearestSymbolTable(op);

  [[maybe_unused]] auto opFunc =
      mlir::dyn_cast_or_null<mlir::SymbolOpInterface>(
          mlir::SymbolTable::lookupSymbolIn(module, op.getCallee()));

  SPU_DEBUG_ONLY_ENFORCE(opFunc, "Unable to find callee function");
  SPU_DEBUG_ONLY_ENFORCE(opFunc.isDeclaration(), "Expect a prototype");

  std::vector<MemRef> inputs(op->getNumOperands());
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    inputs[idx] = sscope->lookupValue(operand);
  }

  const auto &extra = executor->getExtraIntrinsicHandler();
  if (extra.has_value() && (*extra)(sctx, op.getOperation(), inputs)) {
    return;
  }

  auto results = dispatcher(executor, sctx, op, inputs);

  SPU_ENFORCE(results.size() == op->getNumResults(),
              "Intrinsic kernel returns wrong number of results");

  for (auto [result_value, actual_result] :
       llvm::zip(op->getResults(), results)) {
    sscope->addValue(result_value, actual_result);
  }
}

}  // namespace

#include "libspu/device/utils/dispatch_template.cc.inc"

void dispatch(mlir::func::FuncDialect *, SPUContext *sctx, SymbolScope *sscope,
              mlir::Operation &op, OpExecutor *executor,
              const ExecutionOptions &opts) {
  dispatchOp<mlir::func::CallOp>(executor, sctx, sscope, op, opts);
}

}  // namespace spu::device
