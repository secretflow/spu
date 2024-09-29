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

#include "fmt/ranges.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/compiler/utils/utils.h"
#include "libspu/core/prelude.h"
#include "libspu/dialect/pphlo/transforms/passes.h"
#include "libspu/dialect/ring/transforms/passes.h"

namespace spu::compiler {

namespace {

std::string buildFxpWidthOptionString(const CompilerOptions &options) {
  std::vector<std::string> opt_strs;
  if (options.ring_width() != 0 && options.ring_fxp_bits() != 0) {
    opt_strs.emplace_back(fmt::format("f16_width={}", options.ring_width()));
    opt_strs.emplace_back(fmt::format("f32_width={}", options.ring_width()));
    opt_strs.emplace_back(fmt::format("f64_width={}", options.ring_width()));
    opt_strs.emplace_back(
        fmt::format("f16_fxp_bits={}", options.ring_fxp_bits()));
    opt_strs.emplace_back(
        fmt::format("f32_fxp_bits={}", options.ring_fxp_bits()));
    opt_strs.emplace_back(
        fmt::format("f64_fxp_bits={}", options.ring_fxp_bits()));
  }
  return fmt::format("{}", fmt::join(opt_strs, " "));
}

std::string buildFxpApproximationOptionString(const CompilerOptions &options) {
  std::vector<std::string> opt_strs;
  opt_strs.emplace_back(fmt::format("lower_accuracy_rsqrt={}",
                                    options.enable_lower_accuracy_rsqrt()));

  if (options.fxp_div_goldschmidt_iters() != 0) {
    opt_strs.emplace_back(
        fmt::format("divide_iter={}", options.fxp_div_goldschmidt_iters()));
  }

  if (options.fxp_exp_mode() != EXP_DEFAULT) {
    switch (options.fxp_exp_mode()) {
    case EXP_PADE: {
      opt_strs.emplace_back("exp_mode=pade");
      break;
    }
    case EXP_TAYLOR: {
      opt_strs.emplace_back("exp_mode=taylor");
      break;
    }
    default:
      llvm_unreachable("should not hit");
    }
  }

  if (options.fxp_exp_iters() != 0) {
    opt_strs.emplace_back(fmt::format("exp_iter={}", options.fxp_exp_iters()));
  }

  if (options.fxp_log_mode() != LOG_DEFAULT) {
    switch (options.fxp_log_mode()) {
    case LOG_NEWTON: {
      opt_strs.emplace_back("log_mode=newton");
      break;
    }
    case LOG_PADE: {
      opt_strs.emplace_back("log_mode=pade");
      break;
    }
    case LOG_MINMAX: {
      opt_strs.emplace_back("log_mode=minmax");
      break;
    }
    default:
      llvm_unreachable("should not hit");
    }
  }

  if (options.fxp_log_iters() != 0) {
    opt_strs.emplace_back(fmt::format("log_iter={}", options.fxp_log_iters()));
  }

  if (options.fxp_log_orders() != 0) {
    opt_strs.emplace_back(
        fmt::format("log_order={}", options.fxp_log_orders()));
  }

  if (options.sigmoid_mode() != SIGMOID_DEFAULT) {
    switch (options.sigmoid_mode()) {
    case SIGMOID_MM1: {
      opt_strs.emplace_back("sigmoid_mode=mm1");
      break;
    }
    case SIGMOID_SEG3: {
      opt_strs.emplace_back("sigmoid_mode=seg3");
      break;
    }
    case SIGMOID_REAL: {
      opt_strs.emplace_back("sigmoid_mode=real");
      break;
    }
    default:
      llvm_unreachable("should not hit");
    }
  }

  if (options.sine_cosine_iters() != 0) {
    opt_strs.emplace_back(
        fmt::format("sin_cos_iter={}", options.sine_cosine_iters()));
  }

  return fmt::format("{}", fmt::join(opt_strs, " "));
}

} // namespace

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

void Core::buildPublicAndFixedPointPipeline(mlir::PassManager *pm) {
  const auto &options = ctx_->getCompilerOptions();
  auto &optPM = pm->nest<mlir::func::FuncOp>();

  // Now convert all pure public based computations to arith/math dialect
  optPM.addPass(mlir::spu::pphlo::createLegalizeToArith());

  // Now decompose some secret based ops
  optPM.addPass(mlir::spu::pphlo::createSecretDecomposeOps());

  // Now lower secret float to fixedpoint
  auto option_str = buildFxpWidthOptionString(options);
  {
    auto l = mlir::spu::pphlo::createLowerSecretFloatToFxp();
    if (failed(l->initializeOptions(option_str,
                                    mlir::spu::argparser_error_handler))) {
      (void)mlir::emitOptionalError(std::nullopt, "Failed to init fxp options");
    }
    optPM.addPass(std::move(l));
  }

  // Expand all fixedpoint based nonlinear approximations
  {
    auto l = mlir::spu::pphlo::createExpandFixedPointApprox();
    auto option_str = buildFxpApproximationOptionString(options);
    if (failed(l->initializeOptions(option_str,
                                    mlir::spu::argparser_error_handler))) {
      (void)mlir::emitOptionalError(std::nullopt, "Failed to init fxp options");
    }
    optPM.addPass(std::move(l));
  }

  optPM.addPass(mlir::spu::pphlo::createLegalizeToTensor());
  optPM.addPass(mlir::spu::pphlo::createLegalizeToSCF());
  optPM.addPass(mlir::spu::pphlo::createLegalizeToLinalg());

  // There might be some mixed float/fixepoint ops created during
  // approximations, lower them here
  {
    auto l = mlir::spu::pphlo::createLowerPPHloFloatInputs();
    if (failed(l->initializeOptions(option_str,
                                    mlir::spu::argparser_error_handler))) {
      (void)mlir::emitOptionalError(std::nullopt, "Failed to init fxp options");
    }
    optPM.addPass(std::move(l));
  }

  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
}

void Core::buildFixedPointPipeline(mlir::PassManager *pm) {
  const auto &options = ctx_->getCompilerOptions();

  // lowering
  {
    auto &optPM = pm->nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::spu::pphlo::createSortLowering());

    if (!options.disable_partial_sort_optimization()) {
      optPM.addPass(mlir::spu::pphlo::createPartialSortToTopK());
    }

    optPM.addPass(mlir::spu::pphlo::createInlineSecretControlFlow());

    if (!options.disable_reduce_truncation_optimization()) {
      optPM.addPass(mlir::spu::pphlo::createReduceTruncationPass());
    }

    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // First decompose patterns apply to both secret and public operations
    optPM.addPass(mlir::spu::pphlo::createGeneralDecomposeOps());

    // This pass requires float info, so run it before fxp lowering
    optPM.addPass(mlir::spu::pphlo::createRewriteSignbitPatterns());

    // Make life easier
    optPM.addPass(mlir::spu::pphlo::createDotGeneralToDot());

    // Some opts to help accuracy
    if (!options.disable_sqrt_plus_epsilon_rewrite()) {
      optPM.addPass(mlir::spu::pphlo::createOptimizeSqrtPlusEps());
    }

    if (!options.disable_div_sqrt_rewrite()) {
      optPM.addPass(mlir::spu::pphlo::createRewriteDivSqrtPatterns());
    }

    if (options.enable_optimize_denominator_with_broadcast()) {
      optPM.addPass(mlir::spu::pphlo::createOptimizeDenominatorWithBroadcast());
    }
  }

  buildPublicAndFixedPointPipeline(pm);

  {
    auto &optPM = pm->nest<mlir::func::FuncOp>();

    optPM.addPass(mlir::spu::pphlo::createSecretDecomposeOps());

    if (!options.disable_range_based_optimization()) {
      // optPM.addPass(mlir::spu::pphlo::createRangeOptimization());
      //
      // optPM.addPass(mlir::createCanonicalizerPass());
      // optPM.addPass(mlir::createCSEPass());
    }

    if (!options.disallow_mix_types_opts()) {
      optPM.addPass(mlir::spu::pphlo::createLowerMixedTypeOpPass());
    }

    optPM.addPass(mlir::spu::pphlo::createConvertPushDownPass());

    if (!options.disable_select_optimization()) {
      optPM.addPass(mlir::spu::pphlo::createOptimizeSelectPass());
    }

    optPM.addPass(mlir::spu::pphlo::createGeneralDecomposeOps());

    // Final cleanup
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }
}

void Core::buildRingPipeline(mlir::PassManager *pm) {

  {
    auto &optPM = pm->nest<mlir::func::FuncOp>();

    // Lower to ring mini driver
    // Lower public fixedpoint to integer first
    optPM.addPass(mlir::spu::ring::createDecayPublicFixedpoint());

    // Restore some lowering intrinsics
    optPM.addPass(mlir::spu::ring::createIntrinsicCleanup());

    optPM.addPass(mlir::spu::pphlo::createLegalizeToArith());
  }

  // Predecompose
  pm->addPass(mlir::spu::ring::createPreRingLoweringOpDecompose());

  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::spu::pphlo::createLegalizeToArith());

  pm->addPass(mlir::spu::ring::createLegalizeToRing());

  pm->addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}

void Core::buildPipeline(mlir::PassManager *pm) {
  buildFixedPointPipeline(pm);
  buildRingPipeline(pm);
}

} // namespace spu::compiler
