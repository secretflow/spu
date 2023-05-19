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

#include "libspu/compiler/front_end/fe.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/front_end/hlo_importer.h"
#include "libspu/compiler/passes/passes.h"
#include "libspu/core/prelude.h"
#include "libspu/dialect/pphlo_dialect.h"

namespace spu::compiler {

FE::FE(CompilationContext *ctx) : ctx_(ctx) {
  ctx_->getMLIRContext()
      ->loadDialect<mlir::pphlo::PPHloDialect, mlir::mhlo::MhloDialect,
                    mlir::stablehlo::StablehloDialect,
                    mlir::func::FuncDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp> FE::doit(const std::string &source_str) {
  CompilationSource source;
  source.ParseFromString(source_str);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  switch (source.ir_type()) {
  case spu::SourceIRType::XLA: {
    HloImporter importer(ctx_);
    module = importer.parseXlaModuleFromString(source.ir_txt());
    break;
  }
  case spu::SourceIRType::MLIR_HLO: {
    module = mlir::parseSourceString<mlir::ModuleOp>(source.ir_txt(),
                                                     ctx_->getMLIRContext());
    break;
  }
  default: {
    SPU_THROW("Unsupported input IR type = {}", source.ir_type());
  }
  }

  std::string input_vis_str;
  {
    std::vector<std::string> input_vis;
    for (const auto &v : source.input_visibility()) {
      input_vis.emplace_back(Visibility_Name(v));
    }
    input_vis_str = fmt::format("input_vis_list={}", fmt::join(input_vis, ","));
  }

  // Run pipeline
  mlir::PassManager pm(ctx_->getMLIRContext());
  buildFrontEndPipeline(&pm, input_vis_str);

  ctx_->setupPrettyPrintConfigurations(&pm);

  auto ret = pm.run(module.get());

  if (ret.failed()) {
    SPU_THROW("Run front end pipeline failed");
  }

  return module;
}

void FE::buildFrontEndPipeline(mlir::PassManager *pm, const std::string &args) {

  // mhlo side
  {
    pm->addPass(mlir::createInlinerPass());
    pm->addPass(mlir::mhlo::createExpandHloTuplesPass());

    auto &optPM = pm->nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::mhlo::createLegalizeEinsumToDotGeneralPass());
    optPM.addPass(mlir::mhlo::createLegalizeGeneralDotPass());
    optPM.addPass(mlir::mhlo::createSinkConstantsToControlFlowPass());
    optPM.addPass(mlir::mhlo::createLowerComplexPass());
    optPM.addPass(mlir::mhlo::createFlattenTuplePass());
    optPM.addPass(mlir::mhlo::createLegalizeTrigonometricToApproximationPass());
    optPM.addPass(mlir::mhlo::createBroadcastPropagationPass());

    // Convert to stablehlo
    pm->addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  }

  // stablehlo now
  // Dialect conversion
  {
    auto l = mlir::pphlo::createLegalizeToPPHloPass();
    if (!args.empty()) {
      SPU_ENFORCE(l->initializeOptions(args).succeeded());
    }
    pm->addPass(std::move(l));
  }
  auto &optPM = pm->nest<mlir::func::FuncOp>();
  optPM.addPass(mlir::pphlo::createLowerConversionCastPass());
}

} // namespace spu::compiler
