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

#include "spu/compiler/front_end/fe.h"

#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "yasl/base/exception.h"

#include "spu/compiler/common/compilation_context.h"
#include "spu/compiler/front_end/hlo_importer.h"
#include "spu/compiler/passes/passes.h"
#include "spu/dialect/pphlo_dialect.h"

namespace spu::compiler {

FE::FE(CompilationContext *ctx) : ctx_(ctx) {
  ctx_->getMLIRContext()->loadDialect<mlir::pphlo::PPHloDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp> FE::doit(const std::string &input) {
  // Import hlo
  HloImporter importer(ctx_);
  auto module = importer.parseXlaModuleFromString(input);

  // Run pipeline
  mlir::PassManager pm(ctx_->getMLIRContext());
  buildFrontEndPipeline(&pm);

  ctx_->setupPrettyPrintConfigurations(&pm);

  auto ret = pm.run(module.get());

  if (ret.failed()) {
    YASL_THROW("Run front end pipeline failed");
  }

  return module;
}

void FE::buildFrontEndPipeline(mlir::PassManager *pm) {
  {
    // mhlo side
    pm->addPass(mlir::createInlinerPass());
    pm->addPass(mlir::mhlo::CreateExpandHloTuplesPass());
    auto &optPM = pm->nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::mhlo::createLegalizeEinsumToDotGeneralPass());
    optPM.addPass(mlir::mhlo::createLegalizeGeneralDotPass());
    optPM.addPass(mlir::mhlo::createSinkConstantsToControlFlowPass());
    optPM.addPass(mlir::mhlo::createLowerComplexPass());
    optPM.addPass(mlir::mhlo::createFlattenTuplePass());
    optPM.addPass(mlir::mhlo::createLegalizeTrigonometricToApproximationPass());
    optPM.addPass(mlir::mhlo::createBroadcastPropagationPass());
  }
  {
    // Cleanup
    auto &optPM = pm->nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createSCCPPass());
    optPM.addPass(mlir::createCSEPass());
  }
  {
    // Dialect conversion
    auto vis_str = ctx_->getInputVisibilityString();
    if (vis_str.empty()) {
      pm->addPass(mlir::pphlo::createLegalizeToPPHloPass());
    } else {
      pm->addPass(mlir::pphlo::createLegalizeToPPHloPass(vis_str));
    }
    auto &optPM = pm->nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::pphlo::createLowerConversionCastPass());
  }
}

} // namespace spu::compiler
