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

#include "libspu/device/scf/dispatcher.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "libspu/core/trace.h"           // IWYU pragma: keep
#include "libspu/device/utils/utils.h"   // IWYU pragma: keep
#include "libspu/dialect/utils/utils.h"  // IWYU pragma: keep
#include "libspu/kernel/hal/public_helper.h"

namespace spu::device {

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::scf::IfOp &op, const ExecutionOptions &) {
  bool v = kernel::hal::getScalarValue<bool>(
      sctx, sscope->lookupValue(op.getCondition()));

  std::vector<spu::MemRef> results;
  if (v) {
    results = runRegion(executor, sctx, sscope, op.getThenRegion(), {});
  } else {
    results = runRegion(executor, sctx, sscope, op.getThenRegion(), {});
  }

  // Copy output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    sscope->addValue(ret.value(), results[ret.index()]);
  }
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::scf::WhileOp &op, const ExecutionOptions &) {
  // First inputs vectors
  std::vector<spu::MemRef> inputs;
  inputs.reserve(op->getNumOperands());

  // Prepare inputs
  for (const auto operand : op->getOperands()) {
    inputs.emplace_back(sscope->lookupValue(operand));
  }

  auto cond_eval = [&]() {
    auto ret = runRegion(executor, sctx, sscope, op.getBefore(), inputs);
    // scf.condition

    // first operand is condition
    auto b = kernel::hal::getScalarValue<bool>(sctx, ret[0]);

    // Forward rest to after
    inputs.resize(ret.size() - 1);
    for (size_t idx = 1; idx < ret.size(); ++idx) {
      inputs[idx - 1] = ret[idx];
    }

    return b;
  };

  while (cond_eval()) {
    inputs = runRegion(executor, sctx, sscope, op.getAfter(), inputs);
  }

  // Copy output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    sscope->addValue(ret.value(), inputs[ret.index()]);
  }
}

void execute(OpExecutor *executor, SPUContext *sctx, SymbolScope *sscope,
             mlir::scf::ForOp &op, const ExecutionOptions &) {
  auto lb_v = sscope->lookupValue(op.getLowerBound());
  auto ub_v = sscope->lookupValue(op.getUpperBound());
  auto step_v = sscope->lookupValue(op.getStep());

  auto lb = kernel::hal::getScalarValue<int64_t>(sctx, lb_v);
  auto ub = kernel::hal::getScalarValue<int64_t>(sctx, ub_v);
  auto step = kernel::hal::getScalarValue<int64_t>(sctx, step_v);

  std::vector<MemRef> iter_args;

  // Create loop induction var
  iter_args.emplace_back(lb_v.eltype(), Shape());

  for (auto i : op.getInitArgs()) {
    iter_args.emplace_back(sscope->lookupValue(i));
  }

  auto *idx = static_cast<int64_t *>(iter_args.front().data());

  for (*idx = lb; *idx < ub; *idx += step) {
    // forward results to next iter
    for (const auto &ret : llvm::enumerate(
             runRegion(executor, sctx, sscope, op.getRegion(), iter_args))) {
      iter_args[ret.index() + 1] = ret.value();
    }
  }

  for (const auto &ret : llvm::enumerate(op->getResults())) {
    sscope->addValue(ret.value(), iter_args[ret.index() + 1]);
  }
}

#include "libspu/device/utils/dispatch_template.cc.inc"

void dispatch(mlir::scf::SCFDialect *, SPUContext *sctx, SymbolScope *sscope,
              mlir::Operation &op, OpExecutor *executor,
              const ExecutionOptions &opts) {
  dispatchOp<mlir::scf::WhileOp,  //
             mlir::scf::IfOp,     //
             mlir::scf::ForOp     //
             >(executor, sctx, sscope, op, opts);
}

}  // namespace spu::device