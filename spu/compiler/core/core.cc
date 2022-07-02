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

#include "spu/compiler/core/core.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "yasl/base/exception.h"

#include "spu/compiler/common/compilation_context.h"
#include "spu/compiler/passes/passes.h"

namespace spu::compiler {

Core::Core(CompilationContext *ctx) : ctx_(ctx) {}

void Core::doit(mlir::ModuleOp module) {
  mlir::PassManager pm(ctx_->getMLIRContext());
  buildPipeline(&pm);

  ctx_->setupPrettyPrintConfigurations(&pm);

  auto ret = pm.run(module);

  if (ret.failed()) {
    YASL_THROW("Run core pipeline failed");
  }
}

void Core::buildPipeline(mlir::PassManager *pm) {
  {
    // lowering
    auto &optPM = pm->nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::pphlo::createDecomposeComparisonPass());
    optPM.addPass(mlir::pphlo::createDecomposeSqrtPass());
  }
  {
    auto &optPM = pm->nest<mlir::func::FuncOp>();
    // Cleanup
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }
}

} // namespace spu::compiler
