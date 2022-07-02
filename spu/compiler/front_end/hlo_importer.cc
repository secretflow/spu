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

#include "spu/compiler/front_end/hlo_importer.h"

#include "spdlog/spdlog.h"
#include "tensorflow/compiler/mlir/xla/hlo_module_importer.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/conditional_to_select.h"
#include "tensorflow/compiler/xla/service/convolution_group_converter.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/operand_upcaster.h"
#include "tensorflow/compiler/xla/service/qr_expander.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/result_caster.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "yasl/base/exception.h"

#include "spu/compiler/common/compilation_context.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"

namespace xla {
void runHloPasses(xla::HloModule *module) {
  HloPassPipeline pipeline("HLO passes");
  pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                            /*allow_mixed_precision=*/false);

  pipeline.AddPass<OperandUpcaster>();
  pipeline.AddPass<ResultCaster>();

  // Remove zero-sized HLO from the input so that other passes don't have to
  // handle it.
  pipeline.AddPass<ZeroSizedHloElimination>();

  pipeline.AddPass<ConditionalToSelect>();
  pipeline.AddPass<MapInliner>();

  pipeline.AddPass<CholeskyExpander>(); // Eliminate chol
  pipeline.AddPass<QrExpander>();       // Eliminate qr
  pipeline.AddPass<TriangularSolveExpander>();

  // Inline computations with a single call site.
  pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  pipeline.AddPass<BatchDotSimplification>();
  pipeline.AddPass<DotDecomposer>(); // Simplify dot
  // Convert BF16 operations to F32 operations so that the SPU backend can
  // support BF16 operations without directly implementing a BF16 lowering for
  // most ops.
  BFloat16Support bf16;
  pipeline.AddPass<BFloat16Normalization>(&bf16);
  // After canonicalization, there may be more batch dots that can be
  // simplified.
  pipeline.AddPass<BatchDotSimplification>();
  auto cost_model = [](HloInstruction *) {
    // No cost module for SPU.
    return false;
  };
  pipeline.AddPass<ConvolutionGroupConverter>(
      /*should_expand=*/[](HloInstruction *) { return true; }, cost_model,
      /*convert_batch_groups_only=*/true);
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);
  pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);

  // Run the following passes to a fixed point.
  [&pipeline =
       pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
    pipeline.AddInvariantCheckerDebug<HloVerifier>(
        /*layout_sensitive=*/false,
        /*allow_mixed_precision=*/false);

    AlgebraicSimplifierOptions options;
    options.set_enable_dot_strength_reduction(true);
    pipeline.AddPass<AlgebraicSimplifier>(options);
    pipeline.AddPass<SortSimplifier>();
    pipeline.AddPass<HloDCE>();

    // BatchNormExpander can create zero-sized ops, so zero-sized HLO
    // elimination has to come after that pass.
    pipeline.AddPass<ZeroSizedHloElimination>();

    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<WhileLoopSimplifier>();

    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<ReshapeMover>();
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<ConditionalSimplifier>();
  }();
  pipeline.AddPass<BitcastDtypesExpander>();

  auto status = pipeline.Run(module).status();

  YASL_ENFORCE(status.ok());
}
} // namespace xla

namespace spu::compiler {

mlir::OwningOpRef<mlir::ModuleOp>
HloImporter::parseXlaModuleFromString(const std::string &content) {
  // Stage 1: Load hlo_module
  xla::HloModuleProto hlo_module;
  if (!hlo_module.ParseFromString(content)) {
    // If parse as HloModuleProto fails, try HloProto.
    xla::HloProto hlo_proto;
    if (!hlo_proto.ParseFromString(content)) {
      YASL_THROW("Failed to parse hlo module from string");
    }
    hlo_module = hlo_proto.hlo_module();
  }

  xla::DebugOptions debug_options;

  if (context_->hasPrettyPrintEnabled()) {
    debug_options.set_xla_dump_hlo_pass_re(".*");
    debug_options.set_xla_dump_to(context_->getPrettyPrintDir().string());
    debug_options.set_xla_dump_hlo_as_text(true);
    debug_options.set_xla_detailed_logging_and_dumping(true);
  }

  auto module_config =
      xla::HloModule::CreateModuleConfigFromProto(hlo_module, debug_options);
  if (!module_config.status().ok()) {
    YASL_THROW(module_config.status().error_message());
  }

  auto module = xla::HloModule::CreateFromProto(hlo_module, *module_config);
  if (!module.status().ok()) {
    YASL_THROW(module.status().error_message());
  }

  xla::runHloPasses((*module).get());

  // Stage 2: Ask mlir hlo to convert xla module into mlir
  // Create importer
  auto mlir_hlo = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(
      mlir::UnknownLoc::get(context_->getMLIRContext())));
  xla::HloModuleImporter importer(mlir_hlo.get());

  auto status = importer.Import(**module);
  if (!status.ok()) {
    YASL_THROW(status.error_message());
  }

  return mlir_hlo;
}

} // namespace spu::compiler