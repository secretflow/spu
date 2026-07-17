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
#include "xla/service/batchnorm_expander.h"
#include "xla/service/call_inliner.h"
#include "xla/service/conditional_to_select.h"
#include "xla/service/convolution_group_converter.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gpu/transforms/dot_dimension_sorter.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/map_inliner.h"
#include "xla/service/operand_upcaster.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/result_caster.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/translate/hlo_to_mhlo/hlo_module_importer.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/core/prelude.h"

#include "xla/service/hlo.pb.h"
#include "xla/service/indexed_array_analysis.h"
#include "xla/service/all_reduce_contiguous.h"
#include "xla/service/dynamic_dimension_simplifier.h"
#include "xla/service/stable_sort_expander.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/all_reduce_folder.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/service/convolution_4d_expander.h"
#include "xla/service/add_original_value.h"
#include "xla/service/fusion_constant_sinking.h"
#include "xla/service/all_gather_broadcast_reorder.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/p2p_schedule_preparation.h"
#include "xla/service/gather_simplifier.h"
#include "xla/service/rng_expander.h"
#include "xla/service/select_and_scatter_expander.h"
#include "xla/service/real_imag_expander.h"
#include "xla/service/eigh_expander.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/service/convolution_pred_expander.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/cholesky_expander.h"
#include "xla/service/stochastic_convert_decomposer.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/qr_expander.h"
#include "xla/service/slice_sinker.h"
#include "xla/service/batch_dot_simplification.h"
#include "xla/service/convert_operand_folding.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/collective_quantizer.h"
#include "xla/service/dot_dimension_merger.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/reduce_scatter_reassociate.h"

namespace xla {
void runHloPasses(xla::HloModule *module,
                  const spu::CompilerOptions &compiler_options) {
  // Simplifier options
  AlgebraicSimplifierOptions options;
  // For MPC, dot is way faster than reduce
  options.set_enable_dot_strength_reduction(false);
  // We do not handle nan, so just use faster minmax
  options.set_minmax_propagate_nan(false);
  // Transpose and reshape is cheep for us
  options.set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);
  // End of simplifier options

  HloPassPipeline pipeline("optimization");
  pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                            /*allow_mixed_precision=*/false);

  // Start to modify Flags to enable/disable passes
  if (compiler_options.able_allreducefolder_0()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_0()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_0()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_0()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_0()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_0()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_0()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_0()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_0()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_0()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_0()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_0()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_0()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_0()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_0()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_0()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_0()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_0()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_0()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_0()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_0()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_0()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_0()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_0()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_0()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_0()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_0()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_0()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_0()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_0()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_0()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_0()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_0()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_0()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_0()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_0()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_operandupcaster()) {
    pipeline.AddPass<OperandUpcaster>();
  }
  if (compiler_options.able_allreducefolder_1()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_1()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_1()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_1()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_1()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_1()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_1()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_1()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_1()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_1()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_1()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_1()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_1()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_1()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_1()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_1()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_1()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_1()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_1()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_1()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_1()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_1()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_1()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_1()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_1()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_1()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_1()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_1()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_1()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_1()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_1()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_1()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_1()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_1()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_1()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_1()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_resultcaster()) {
    pipeline.AddPass<ResultCaster>();
  }
  if (compiler_options.able_allreducefolder_2()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_2()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_2()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_2()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_2()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_2()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_2()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_2()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_2()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_2()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_2()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_2()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_2()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_2()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_2()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_2()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_2()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_2()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_2()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_2()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_2()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_2()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_2()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_2()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_2()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_2()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_2()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_2()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_2()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_2()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_2()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_2()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_2()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_2()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_2()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_2()) {
    pipeline.AddPass<StableSortExpander>();
  }

  // Remove zero-sized HLO from the input so that other passes don't have to
  // handle it.
  if (!compiler_options.disable_zerosizedhloelimination()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_allreducefolder_3()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_3()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_3()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_3()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_3()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_3()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_3()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_3()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_3()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_3()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_3()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_3()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_3()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_3()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_3()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_3()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_3()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_3()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_3()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_3()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_3()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_3()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_3()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_3()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_3()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_3()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_3()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_3()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_3()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_3()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_3()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_3()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_3()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_3()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_3()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_3()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_conditionaltoselect()) {
    pipeline.AddPass<ConditionalToSelect>();
  }
  if (compiler_options.able_allreducefolder_4()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_4()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_4()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_4()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_4()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_4()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_4()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_4()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_4()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_4()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_4()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_4()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_4()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_4()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_4()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_4()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_4()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_4()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_4()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_4()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_4()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_4()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_4()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_4()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_4()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_4()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_4()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_4()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_4()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_4()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_4()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_4()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_4()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_4()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_4()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_4()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_mapinliner()) {
    pipeline.AddPass<MapInliner>();
  }
  if (compiler_options.able_allreducefolder_5()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_5()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_5()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_5()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_5()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_5()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_5()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_5()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_5()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_5()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_5()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_5()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_5()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_5()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_5()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_5()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_5()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_5()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_5()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_5()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_5()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_5()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_5()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_5()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_5()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_5()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_5()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_5()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_5()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_5()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_5()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_5()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_5()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_5()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_5()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_5()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_choleskyexpander()) {
    pipeline.AddPass<CholeskyExpander>(); // Eliminate chol
  }
  if (compiler_options.able_allreducefolder_6()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_6()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_6()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_6()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_6()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_6()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_6()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_6()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_6()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_6()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_6()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_6()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_6()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_6()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_6()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_6()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_6()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_6()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_6()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_6()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_6()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_6()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_6()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_6()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_6()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_6()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_6()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_6()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_6()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_6()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_6()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_6()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_6()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_6()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_6()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_6()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_qrexpander()) {
    pipeline.AddPass<QrExpander>();       // Eliminate qr
  }
  if (compiler_options.able_allreducefolder_7()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_7()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_7()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_7()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_7()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_7()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_7()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_7()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_7()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_7()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_7()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_7()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_7()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_7()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_7()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_7()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_7()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_7()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_7()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_7()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_7()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_7()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_7()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_7()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_7()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_7()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_7()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_7()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_7()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_7()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_7()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_7()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_7()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_7()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_7()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_7()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_eighexpander()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_allreducefolder_8()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_8()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_8()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_8()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_8()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_8()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_8()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_8()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_8()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_8()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_8()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_8()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_8()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_8()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_8()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_8()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_8()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_8()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_8()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_8()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_8()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_8()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_8()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_8()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_8()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_8()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_8()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_8()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_8()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_8()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_8()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_8()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_8()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_8()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_8()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_8()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_triangularsolveexpander()) {
    pipeline.AddPass<TriangularSolveExpander>();
  }
  if (compiler_options.able_allreducefolder_9()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_9()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_9()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_9()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_9()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_9()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_9()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_9()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_9()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_9()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_9()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_9()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_9()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_9()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_9()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_9()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_9()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_9()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_9()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_9()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_9()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_9()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_9()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_9()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_9()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_9()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_9()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_9()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_9()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_9()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_9()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_9()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_9()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_9()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_9()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_9()) {
    pipeline.AddPass<StableSortExpander>();
  }
  // Convert BF16 operations to F32 operations so that the SPU backend can
  // support BF16 operations without directly implementing a BF16 lowering for
  // most ops.
  FloatSupport bf16_support(BF16);
  if (!compiler_options.disable_floatnormalization()) {
    pipeline.AddPass<FloatNormalization>(&bf16_support);
  }
  if (compiler_options.able_allreducefolder_10()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_10()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_10()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_10()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_10()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_10()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_10()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_10()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_10()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_10()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_10()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_10()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_10()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_10()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_10()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_10()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_10()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_10()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_10()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_10()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_10()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_10()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_10()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_10()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_10()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_10()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_10()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_10()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_10()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_10()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_10()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_10()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_10()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_10()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_10()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_10()) {
    pipeline.AddPass<StableSortExpander>();
  }
  // Inline computations with a single call site.
  if (!compiler_options.disable_callinliner()) {
    pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  }
  if (compiler_options.able_allreducefolder_11()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_11()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_11()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_11()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_11()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_11()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_11()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_11()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_11()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_11()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_11()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_11()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_11()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_11()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_11()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_11()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_11()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_11()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_11()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_11()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_11()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_11()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_11()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_11()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_11()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_11()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_11()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_11()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_11()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_11()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_11()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_11()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_11()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_11()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_11()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_11()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_gpu_dotdimensionsorter()) {
    pipeline.AddPass<gpu::DotDimensionSorter>();
  }
  if (compiler_options.able_allreducefolder_12()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_12()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_12()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_12()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_12()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_12()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_12()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_12()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_12()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_12()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_12()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_12()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_12()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_12()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_12()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_12()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_12()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_12()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_12()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_12()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_12()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_12()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_12()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_12()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_12()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_12()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_12()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_12()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_12()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_12()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_12()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_12()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_12()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_12()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_12()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_12()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_batchdotsimplification()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_allreducefolder_13()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_13()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_13()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_13()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_13()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_13()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_13()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_13()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_13()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_13()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_13()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_13()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_13()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_13()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_13()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_13()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_13()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_13()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_13()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_13()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_13()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_13()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_13()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_13()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_13()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_13()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_13()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_13()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_13()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_13()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_13()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_13()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_13()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_13()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_13()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_13()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_dotdecomposer()) {
    pipeline.AddPass<DotDecomposer>(); // Simplify dot
  }
  if (compiler_options.able_allreducefolder_14()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_14()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_14()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_14()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_14()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_14()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_14()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_14()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_14()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_14()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_14()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_14()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_14()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_14()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_14()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_14()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_14()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_14()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_14()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_14()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_14()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_14()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_14()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_14()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_14()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_14()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_14()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_14()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_14()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_14()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_14()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_14()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_14()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_14()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_14()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_14()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_convolution4dexpander()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_allreducefolder_15()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_15()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_15()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_15()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_15()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_15()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_15()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_15()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_15()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_15()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_15()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_15()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_15()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_15()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_15()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_15()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_15()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_15()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_15()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_15()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_15()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_15()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_15()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_15()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_15()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_15()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_15()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_15()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_15()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_15()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_15()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_15()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_15()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_15()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_15()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_15()) {
    pipeline.AddPass<StableSortExpander>();
  }
  // After canonicalization, there may be more batch dots that can be
  // simplified.
  if (!compiler_options.disable_batchdotsimplification_1()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_allreducefolder_16()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_16()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_16()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_16()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_16()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_16()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_16()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_16()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_16()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_16()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_16()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_16()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_16()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_16()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_16()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_16()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_16()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_16()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_16()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_16()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_16()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_16()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_16()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_16()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_16()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_16()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_16()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_16()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_16()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_16()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_16()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_16()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_16()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_16()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_16()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_16()) {
    pipeline.AddPass<StableSortExpander>();
  }
  auto cost_model = [](HloInstruction *) {
    // No cost module for SPU.
    return false;
  };
  if (!compiler_options.disable_convolutiongroupconverter()) {
    pipeline.AddPass<ConvolutionGroupConverter>(
        /*should_expand=*/[](HloInstruction *) { return true; }, cost_model,
        /*convert_batch_groups_only=*/false);
  }
  if (compiler_options.able_allreducefolder_17()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_17()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_17()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_17()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_17()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_17()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_17()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_17()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_17()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_17()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_17()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_17()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_17()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_17()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_17()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_17()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_17()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_17()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_17()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_17()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_17()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_17()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_17()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_17()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_17()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_17()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_17()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_17()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_17()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_17()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_17()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_17()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_17()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_17()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_17()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_17()) {
    pipeline.AddPass<StableSortExpander>();
  }
  if (!compiler_options.disable_batchnormexpander()) {
    pipeline.AddPass<BatchNormExpander>(
        /*rewrite_training_op=*/true,
        /*rewrite_inference_op=*/true,
        /*rewrite_grad_op=*/true);
  }
  if (compiler_options.able_allreducefolder_18()) {
    pipeline.AddPass<AllReduceFolder>();
  }
  if (compiler_options.able_indexedarrayanalysisprinterpass_18()) {
    pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  }
  if (compiler_options.able_collectivequantizer_18()) {
    pipeline.AddPass<CollectiveQuantizer>();
  }
  if (compiler_options.able_sortsimplifier_18()) {
    pipeline.AddPass<SortSimplifier>();
  }
  if (compiler_options.able_fusionconstantsinking_18()) {
    pipeline.AddPass<FusionConstantSinking>();
  }
  if (compiler_options.able_dotdimensionmerger_18()) {
    pipeline.AddPass<DotDimensionMerger>();
  }
  if (compiler_options.able_p2pschedulepreparation_18()) {
    pipeline.AddPass<P2PSchedulePreparation>();
  }
  if (compiler_options.able_batchdotsimplification_18()) {
    pipeline.AddPass<BatchDotSimplification>();
  }
  if (compiler_options.able_shardingremover_18()) {
    pipeline.AddPass<ShardingRemover>();
  }
  if (compiler_options.able_flattencallgraph_18()) {
    pipeline.AddPass<FlattenCallGraph>();
  }
  if (compiler_options.able_slicesinker_18()) {
    pipeline.AddPass<SliceSinker>();
  }
  if (compiler_options.able_allgatherbroadcastreorder_18()) {
    pipeline.AddPass<AllGatherBroadcastReorder>();
  }
  if (compiler_options.able_allreducecontiguous_18()) {
    pipeline.AddPass<AllReduceContiguous>();
  }
  if (compiler_options.able_conditionalsimplifier_18()) {
    pipeline.AddPass<ConditionalSimplifier>();
  }
  if (compiler_options.able_stochasticconvertdecomposer_18()) {
    pipeline.AddPass<StochasticConvertDecomposer>();
  }
  if (compiler_options.able_reducescatterreassociate_18()) {
    pipeline.AddPass<ReduceScatterReassociate>();
  }
  if (compiler_options.able_conditionalcanonicalizer_18()) {
    pipeline.AddPass<ConditionalCanonicalizer>();
  }
  if (compiler_options.able_zerosizedhloelimination_18()) {
    pipeline.AddPass<ZeroSizedHloElimination>();
  }
  if (compiler_options.able_reshapedecomposer_18()) {
    pipeline.AddPass<ReshapeDecomposer>();
  }
  if (compiler_options.able_addoriginalvalue_18()) {
    pipeline.AddPass<AddOriginalValue>();
  }
  if (compiler_options.able_dynamicdimensionsimplifier_18()) {
    pipeline.AddPass<DynamicDimensionSimplifier>();
  }
  if (compiler_options.able_dotdecomposer_18()) {
    pipeline.AddPass<DotDecomposer>();
  }
  if (compiler_options.able_hloconstantfolding_18()) {
    pipeline.AddPass<HloConstantFolding>();
  }
  if (compiler_options.able_rngexpander_18()) {
    pipeline.AddPass<RngExpander>();
  }
  if (compiler_options.able_convolutionpredexpander_18()) {
    pipeline.AddPass<ConvolutionPredExpander>();
  }
  if (compiler_options.able_convertoperandfolding_18()) {
    pipeline.AddPass<ConvertOperandFolding>();
  }
  if (compiler_options.able_choleskyexpander_18()) {
    pipeline.AddPass<CholeskyExpander>();
  }
  if (compiler_options.able_bitcastdtypesexpander_18()) {
    pipeline.AddPass<BitcastDtypesExpander>();
  }
  if (compiler_options.able_realimagexpander_18()) {
    pipeline.AddPass<RealImagExpander>();
  }
  if (compiler_options.able_convolution4dexpander_18()) {
    pipeline.AddPass<Convolution4DExpander>();
  }
  if (compiler_options.able_qrexpander_18()) {
    pipeline.AddPass<QrExpander>();
  }
  if (compiler_options.able_eighexpander_18()) {
    pipeline.AddPass<EighExpander>();
  }
  if (compiler_options.able_selectandscatterexpander_18()) {
    pipeline.AddPass<SelectAndScatterExpander>();
  }
  if (compiler_options.able_scattersimplifier_18()) {
    pipeline.AddPass<ScatterSimplifier>();
  }
  if (compiler_options.able_gathersimplifier_18()) {
    pipeline.AddPass<GatherSimplifier>();
  }
  if (compiler_options.able_stablesortexpander_18()) {
    pipeline.AddPass<StableSortExpander>();
  }
  // Run the following passes to a fixed point.
  [&, &pipeline =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
    pipeline.AddInvariantCheckerDebug<HloVerifier>(
        /*layout_sensitive=*/false,
        /*allow_mixed_precision=*/false);
    if (compiler_options.able_allreducefolder_19()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_19()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_19()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_19()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_19()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_19()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_19()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_19()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_19()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_19()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_19()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_19()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_19()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_19()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_19()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_19()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_19()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_19()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_19()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_19()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_19()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_19()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_19()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_19()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_19()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_19()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_19()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_19()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_19()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_19()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_19()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_19()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_19()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_19()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_19()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_19()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_gathersimplifier()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_allreducefolder_20()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_20()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_20()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_20()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_20()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_20()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_20()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_20()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_20()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_20()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_20()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_20()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_20()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_20()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_20()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_20()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_20()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_20()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_20()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_20()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_20()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_20()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_20()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_20()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_20()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_20()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_20()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_20()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_20()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_20()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_20()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_20()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_20()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_20()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_20()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_20()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_gatherexpander()) {
      pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
    }
    if (compiler_options.able_allreducefolder_21()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_21()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_21()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_21()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_21()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_21()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_21()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_21()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_21()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_21()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_21()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_21()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_21()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_21()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_21()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_21()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_21()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_21()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_21()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_21()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_21()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_21()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_21()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_21()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_21()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_21()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_21()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_21()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_21()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_21()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_21()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_21()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_21()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_21()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_21()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_21()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_scatterexpander()) {
      pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);
    }
    if (compiler_options.able_allreducefolder_22()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_22()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_22()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_22()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_22()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_22()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_22()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_22()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_22()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_22()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_22()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_22()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_22()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_22()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_22()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_22()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_22()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_22()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_22()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_22()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_22()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_22()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_22()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_22()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_22()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_22()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_22()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_22()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_22()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_22()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_22()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_22()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_22()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_22()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_22()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_22()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_algebraicsimplifier()) {
      pipeline.AddPass<AlgebraicSimplifier>(options);
    }
    if (compiler_options.able_allreducefolder_23()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_23()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_23()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_23()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_23()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_23()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_23()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_23()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_23()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_23()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_23()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_23()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_23()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_23()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_23()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_23()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_23()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_23()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_23()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_23()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_23()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_23()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_23()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_23()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_23()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_23()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_23()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_23()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_23()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_23()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_23()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_23()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_23()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_23()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_23()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_23()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_bitcastdtypesexpander()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_allreducefolder_24()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_24()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_24()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_24()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_24()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_24()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_24()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_24()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_24()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_24()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_24()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_24()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_24()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_24()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_24()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_24()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_24()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_24()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_24()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_24()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_24()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_24()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_24()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_24()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_24()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_24()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_24()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_24()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_24()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_24()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_24()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_24()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_24()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_24()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_24()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_24()) {
      pipeline.AddPass<StableSortExpander>();
    }
    // AlgebraicSimplifier may add contracting dimensions to a dot.
    if (!compiler_options.disable_gpu_dotdimensionsorter_1()) {
      pipeline.AddPass<gpu::DotDimensionSorter>();
    }
    if (compiler_options.able_allreducefolder_25()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_25()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_25()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_25()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_25()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_25()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_25()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_25()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_25()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_25()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_25()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_25()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_25()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_25()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_25()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_25()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_25()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_25()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_25()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_25()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_25()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_25()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_25()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_25()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_25()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_25()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_25()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_25()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_25()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_25()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_25()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_25()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_25()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_25()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_25()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_25()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_dotdecomposer_1()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_allreducefolder_26()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_26()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_26()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_26()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_26()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_26()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_26()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_26()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_26()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_26()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_26()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_26()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_26()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_26()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_26()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_26()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_26()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_26()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_26()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_26()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_26()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_26()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_26()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_26()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_26()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_26()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_26()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_26()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_26()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_26()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_26()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_26()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_26()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_26()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_26()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_26()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_sortsimplifier()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_allreducefolder_27()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_27()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_27()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_27()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_27()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_27()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_27()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_27()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_27()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_27()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_27()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_27()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_27()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_27()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_27()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_27()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_27()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_27()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_27()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_27()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_27()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_27()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_27()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_27()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_27()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_27()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_27()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_27()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_27()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_27()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_27()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_27()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_27()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_27()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_27()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_27()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_tuplesimplifier()) {
      pipeline.AddPass<TupleSimplifier>();
    }
    if (compiler_options.able_allreducefolder_28()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_28()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_28()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_28()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_28()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_28()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_28()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_28()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_28()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_28()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_28()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_28()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_28()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_28()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_28()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_28()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_28()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_28()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_28()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_28()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_28()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_28()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_28()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_28()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_28()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_28()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_28()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_28()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_28()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_28()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_28()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_28()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_28()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_28()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_28()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_28()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_whileloopsimplifier()) {
      pipeline.AddPass<WhileLoopSimplifier>();
    }
    if (compiler_options.able_allreducefolder_29()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_29()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_29()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_29()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_29()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_29()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_29()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_29()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_29()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_29()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_29()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_29()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_29()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_29()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_29()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_29()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_29()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_29()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_29()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_29()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_29()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_29()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_29()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_29()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_29()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_29()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_29()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_29()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_29()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_29()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_29()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_29()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_29()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_29()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_29()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_29()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_slicesinker()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allreducefolder_30()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_30()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_30()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_30()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_30()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_30()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_30()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_30()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_30()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_30()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_30()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_30()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_30()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_30()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_30()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_30()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_30()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_30()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_30()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_30()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_30()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_30()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_30()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_30()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_30()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_30()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_30()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_30()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_30()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_30()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_30()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_30()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_30()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_30()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_30()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_30()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_reshapemover()) {
      pipeline.AddPass<ReshapeMover>();
    }
    if (compiler_options.able_allreducefolder_31()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_31()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_31()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_31()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_31()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_31()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_31()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_31()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_31()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_31()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_31()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_31()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_31()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_31()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_31()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_31()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_31()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_31()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_31()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_31()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_31()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_31()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_31()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_31()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_31()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_31()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_31()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_31()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_31()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_31()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_31()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_31()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_31()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_31()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_31()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_31()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_hloconstantfolding()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_allreducefolder_32()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_32()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_32()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_32()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_32()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_32()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_32()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_32()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_32()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_32()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_32()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_32()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_32()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_32()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_32()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_32()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_32()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_32()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_32()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_32()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_32()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_32()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_32()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_32()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_32()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_32()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_32()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_32()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_32()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_32()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_32()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_32()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_32()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_32()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_32()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_32()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_conditionalsimplifier()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_allreducefolder_33()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_33()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_33()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_33()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_33()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_33()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_33()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_33()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_33()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_33()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_33()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_33()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_33()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_33()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_33()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_33()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_33()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_33()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_33()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_33()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_33()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_33()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_33()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_33()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_33()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_33()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_33()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_33()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_33()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_33()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_33()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_33()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_33()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_33()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_33()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_33()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_realimagexpander()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_allreducefolder_34()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_34()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_34()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_34()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_34()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_34()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_34()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_34()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_34()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_34()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_34()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_34()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_34()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_34()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_34()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_34()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_34()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_34()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_34()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_34()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_34()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_34()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_34()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_34()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_34()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_34()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_34()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_34()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_34()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_34()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_34()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_34()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_34()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_34()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_34()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_34()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_hlocse()) {
      pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    }
    if (compiler_options.able_allreducefolder_35()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_35()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_35()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_35()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_35()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_35()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_35()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_35()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_35()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_35()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_35()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_35()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_35()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_35()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_35()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_35()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_35()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_35()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_35()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_35()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_35()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_35()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_35()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_35()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_35()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_35()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_35()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_35()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_35()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_35()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_35()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_35()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_35()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_35()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_35()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_35()) {
      pipeline.AddPass<StableSortExpander>();
    }
    if (!compiler_options.disable_hlodce()) {
      pipeline.AddPass<HloDCE>();
    }
    if (compiler_options.able_allreducefolder_36()) {
      pipeline.AddPass<AllReduceFolder>();
    }
    if (compiler_options.able_indexedarrayanalysisprinterpass_36()) {
      pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
    }
    if (compiler_options.able_collectivequantizer_36()) {
      pipeline.AddPass<CollectiveQuantizer>();
    }
    if (compiler_options.able_sortsimplifier_36()) {
      pipeline.AddPass<SortSimplifier>();
    }
    if (compiler_options.able_fusionconstantsinking_36()) {
      pipeline.AddPass<FusionConstantSinking>();
    }
    if (compiler_options.able_dotdimensionmerger_36()) {
      pipeline.AddPass<DotDimensionMerger>();
    }
    if (compiler_options.able_p2pschedulepreparation_36()) {
      pipeline.AddPass<P2PSchedulePreparation>();
    }
    if (compiler_options.able_batchdotsimplification_36()) {
      pipeline.AddPass<BatchDotSimplification>();
    }
    if (compiler_options.able_shardingremover_36()) {
      pipeline.AddPass<ShardingRemover>();
    }
    if (compiler_options.able_flattencallgraph_36()) {
      pipeline.AddPass<FlattenCallGraph>();
    }
    if (compiler_options.able_slicesinker_36()) {
      pipeline.AddPass<SliceSinker>();
    }
    if (compiler_options.able_allgatherbroadcastreorder_36()) {
      pipeline.AddPass<AllGatherBroadcastReorder>();
    }
    if (compiler_options.able_allreducecontiguous_36()) {
      pipeline.AddPass<AllReduceContiguous>();
    }
    if (compiler_options.able_conditionalsimplifier_36()) {
      pipeline.AddPass<ConditionalSimplifier>();
    }
    if (compiler_options.able_stochasticconvertdecomposer_36()) {
      pipeline.AddPass<StochasticConvertDecomposer>();
    }
    if (compiler_options.able_reducescatterreassociate_36()) {
      pipeline.AddPass<ReduceScatterReassociate>();
    }
    if (compiler_options.able_conditionalcanonicalizer_36()) {
      pipeline.AddPass<ConditionalCanonicalizer>();
    }
    if (compiler_options.able_zerosizedhloelimination_36()) {
      pipeline.AddPass<ZeroSizedHloElimination>();
    }
    if (compiler_options.able_reshapedecomposer_36()) {
      pipeline.AddPass<ReshapeDecomposer>();
    }
    if (compiler_options.able_addoriginalvalue_36()) {
      pipeline.AddPass<AddOriginalValue>();
    }
    if (compiler_options.able_dynamicdimensionsimplifier_36()) {
      pipeline.AddPass<DynamicDimensionSimplifier>();
    }
    if (compiler_options.able_dotdecomposer_36()) {
      pipeline.AddPass<DotDecomposer>();
    }
    if (compiler_options.able_hloconstantfolding_36()) {
      pipeline.AddPass<HloConstantFolding>();
    }
    if (compiler_options.able_rngexpander_36()) {
      pipeline.AddPass<RngExpander>();
    }
    if (compiler_options.able_convolutionpredexpander_36()) {
      pipeline.AddPass<ConvolutionPredExpander>();
    }
    if (compiler_options.able_convertoperandfolding_36()) {
      pipeline.AddPass<ConvertOperandFolding>();
    }
    if (compiler_options.able_choleskyexpander_36()) {
      pipeline.AddPass<CholeskyExpander>();
    }
    if (compiler_options.able_bitcastdtypesexpander_36()) {
      pipeline.AddPass<BitcastDtypesExpander>();
    }
    if (compiler_options.able_realimagexpander_36()) {
      pipeline.AddPass<RealImagExpander>();
    }
    if (compiler_options.able_convolution4dexpander_36()) {
      pipeline.AddPass<Convolution4DExpander>();
    }
    if (compiler_options.able_qrexpander_36()) {
      pipeline.AddPass<QrExpander>();
    }
    if (compiler_options.able_eighexpander_36()) {
      pipeline.AddPass<EighExpander>();
    }
    if (compiler_options.able_selectandscatterexpander_36()) {
      pipeline.AddPass<SelectAndScatterExpander>();
    }
    if (compiler_options.able_scattersimplifier_36()) {
      pipeline.AddPass<ScatterSimplifier>();
    }
    if (compiler_options.able_gathersimplifier_36()) {
      pipeline.AddPass<GatherSimplifier>();
    }
    if (compiler_options.able_stablesortexpander_36()) {
      pipeline.AddPass<StableSortExpander>();
    }
  }();
  // End of modifying Flags to enable/disable passes

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
    SPU_THROW("{}", module_config.status().message());
  }

  auto module = xla::HloModule::CreateFromProto(hlo_module, *module_config);
  if (!module.status().ok()) {
    SPU_THROW("{}", module.status().message());
  }

  xla::runHloPasses((*module).get(), context_->getCompilerOptions());

  // Stage 2: Ask mlir hlo to convert xla module into mlir
  // Create importer
  auto mlir_hlo = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(
      mlir::UnknownLoc::get(context_->getMLIRContext())));
  xla::HloModuleImporter importer(mlir_hlo.get());

  auto status = importer.Import(**module);
  if (!status.ok()) {
    SPU_THROW("{}", status.message());
  }

  return mlir_hlo;
}

} // namespace spu::compiler

