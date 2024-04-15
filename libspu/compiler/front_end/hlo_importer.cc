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

#include "libspu/compiler/front_end/hlo_importer.h"

#include "xla/service/algebraic_simplifier.h"
#include "xla/service/batch_dot_simplification.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/call_inliner.h"
#include "xla/service/cholesky_expander.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/conditional_to_select.h"
#include "xla/service/convolution_4d_expander.h"
#include "xla/service/convolution_group_converter.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/eigh_expander.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gather_simplifier.h"
#include "xla/service/gpu/dot_dimension_sorter.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/map_inliner.h"
#include "xla/service/operand_upcaster.h"
#include "xla/service/qr_expander.h"
#include "xla/service/real_imag_expander.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/result_caster.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/slice_sinker.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/translate/hlo_to_mhlo/hlo_module_importer.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/core/prelude.h"

#include "xla/service/hlo.pb.h"

namespace xla {
void runHloPasses(xla::HloModule *module) {

  // Simplifier options
  AlgebraicSimplifierOptions options;
  // For MPC, dot is way faster than reduce
  options.set_enable_dot_strength_reduction(false);
  // We do not handle nan, so just use faster minmax
  options.set_minmax_propagate_nan(false);
  // Transpose and reshape is cheep for us
  options.set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);

  HloPassPipeline pipeline("optimization");
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
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<TriangularSolveExpander>();

  // Convert BF16 operations to F32 operations so that the SPU backend can
  // support BF16 operations without directly implementing a BF16 lowering for
  // most ops.
  FloatSupport bf16_support(BF16);
  pipeline.AddPass<FloatNormalization>(&bf16_support);

  // Inline computations with a single call site.
  pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  pipeline.AddPass<gpu::DotDimensionSorter>();
  pipeline.AddPass<BatchDotSimplification>();
  pipeline.AddPass<DotDecomposer>(); // Simplify dot

  pipeline.AddPass<Convolution4DExpander>();

  // After canonicalization, there may be more batch dots that can be
  // simplified.
  pipeline.AddPass<BatchDotSimplification>();
  auto cost_model = [](HloInstruction *) {
    // No cost module for SPU.
    return false;
  };
  pipeline.AddPass<ConvolutionGroupConverter>(
      /*should_expand=*/[](HloInstruction *) { return true; }, cost_model,
      /*convert_batch_groups_only=*/false);
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);

  // Run the following passes to a fixed point.
  [&, &pipeline =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
    pipeline.AddInvariantCheckerDebug<HloVerifier>(
        /*layout_sensitive=*/false,
        /*allow_mixed_precision=*/false);

    pipeline.AddPass<GatherSimplifier>();
    pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
    pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);
    pipeline.AddPass<AlgebraicSimplifier>(options);
    pipeline.AddPass<BitcastDtypesExpander>();
    // AlgebraicSimplifier may add contracting dimensions to a dot.
    pipeline.AddPass<gpu::DotDimensionSorter>();
    pipeline.AddPass<DotDecomposer>();

    pipeline.AddPass<SortSimplifier>();
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<WhileLoopSimplifier>();
    pipeline.AddPass<SliceSinker>();
    pipeline.AddPass<ReshapeMover>();
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<ConditionalSimplifier>();
    pipeline.AddPass<RealImagExpander>();
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    pipeline.AddPass<HloDCE>();
  }();

  auto status = pipeline.Run(module).status();

  SPU_ENFORCE(status.ok());
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
      // Try human-readable format
      if (!google::protobuf::TextFormat::ParseFromString(content, &hlo_proto)) {
        SPU_THROW("Failed to parse hlo module from string {}", content);
      }
    }
    hlo_module = hlo_proto.hlo_module();
  }

  xla::DebugOptions debug_options;

  if (context_->hasPrettyPrintEnabled()) {
    debug_options.set_xla_dump_hlo_pass_re(".*");
    debug_options.set_xla_dump_to(context_->getPrettyPrintDir().string());
    switch (context_->getXlaPrettyPrintKind()) {
    case spu::XLAPrettyPrintKind::DOT: {
      debug_options.set_xla_dump_hlo_as_dot(true);
      break;
    }
    case spu::XLAPrettyPrintKind::HTML: {
      debug_options.set_xla_dump_hlo_as_html(true);
      break;
    }
    default: {
      debug_options.set_xla_dump_hlo_as_text(true);
      break;
    }
    }
    debug_options.set_xla_enable_dumping(true);
  }

  auto module_config =
      xla::HloModule::CreateModuleConfigFromProto(hlo_module, debug_options);
  if (!module_config.status().ok()) {
    SPU_THROW(module_config.status().message());
  }

  auto module = xla::HloModule::CreateFromProto(hlo_module, *module_config);
  if (!module.status().ok()) {
    SPU_THROW(module.status().message());
  }

  xla::runHloPasses((*module).get());

  // Stage 2: Ask mlir hlo to convert xla module into mlir
  // Create importer
  auto mlir_hlo = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(
      mlir::UnknownLoc::get(context_->getMLIRContext())));
  xla::HloModuleImporter importer(mlir_hlo.get());

  auto status = importer.Import(**module);
  if (!status.ok()) {
    SPU_THROW(status.message());
  }

  return mlir_hlo;
}

} // namespace spu::compiler
