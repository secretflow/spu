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

#include "libspu/compiler/core/core.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/passes/passes.h"
#include "libspu/core/prelude.h"

namespace spu::compiler {

Core::Core(CompilationContext *ctx) : ctx_(ctx) {}

void Core::doit(mlir::ModuleOp module) {
  mlir::PassManager pm(ctx_->getMLIRContext());
  buildPipeline(&pm);

  ctx_->setupPrettyPrintConfigurations(&pm);

  auto ret = pm.run(module);

  if (ret.failed()) {
    SPU_THROW("Run core pipeline failed");
  }
}

void Core::buildPipeline(mlir::PassManager *pm) {
  const auto &options = ctx_->getCompilerOptions();

  // lowering
  auto &optPM = pm->nest<mlir::func::FuncOp>();
  if (!options.disable_maxpooling_optimization()) {
    // Need a cse before maxpooling
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::spu::pphlo::createOptimizeMaxPoolingPass());
  }
  optPM.addPass(mlir::spu::pphlo::createDecomposeComparisonPass());
  optPM.addPass(mlir::spu::pphlo::createDecomposeMinMaxPass());
  optPM.addPass(mlir::spu::pphlo::createSortLowering());

  if (!options.disable_partial_sort_optimization()) {
    optPM.addPass(mlir::spu::pphlo::createPartialSortToTopK());
  }

  if (!options.disable_sqrt_plus_epsilon_rewrite()) {
    optPM.addPass(mlir::spu::pphlo::createOptimizeSqrtPlusEps());
  }

  if (!options.disable_div_sqrt_rewrite()) {
    optPM.addPass(mlir::spu::pphlo::createRewriteDivSqrtPatterns());
  }

  if (options.enable_optimize_denominator_with_broadcast()) {
    optPM.addPass(mlir::spu::pphlo::createOptimizeDenominatorWithBroadcast());
  }

  optPM.addPass(mlir::createCSEPass());

  optPM.addPass(mlir::spu::pphlo::createConvertPushDownPass());

  if (!options.disable_reduce_truncation_optimization()) {
    optPM.addPass(mlir::spu::pphlo::createReduceTruncationPass());
  }

  if (!options.disallow_mix_types_opts()) {
    optPM.addPass(mlir::spu::pphlo::createLowerMixedTypeOpPass());
  }

  optPM.addPass(mlir::createCanonicalizerPass());

  if (!options.disable_select_optimization()) {
    optPM.addPass(mlir::spu::pphlo::createOptimizeSelectPass());
  }

  optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
  optPM.addPass(mlir::createCSEPass());

  if (!options.disable_deallocation_insertion()) {
    optPM.addPass(mlir::spu::pphlo::createInsertDeallocationOp());
  }
}

} // namespace spu::compiler
