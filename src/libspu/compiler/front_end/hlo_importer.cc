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

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/bitcast_dtypes_expander.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/convolution_4d_expander.h"
#include "xla/hlo/transforms/expanders/dot_decomposer.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/hlo/transforms/expanders/real_imag_expander.h"
#include "xla/hlo/transforms/operand_upcaster.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/batch_dot_simplification.h"
#include "xla/hlo/transforms/simplifiers/convolution_group_converter.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/result_caster.h"
#include "xla/hlo/transforms/simplifiers/slice_sinker.h"
#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_module_importer.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/call_inliner.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/conditional_to_select.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/gpu/transforms/dot_dimension_sorter.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/map_inliner.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/shape_util.h"

#include "libspu/compiler/common/compilation_context.h"
#include "libspu/core/prelude.h"

#include "xla/service/hlo.pb.h"

namespace xla {

// we port gather simplifier from XLA, and handle the batched gather case by
// ourself.
class GeneralGatherSimplifier : public OpExpanderPass {
public:
  absl::string_view name() const override {
    return "general_gather_simplifier";
  }

  static bool IsSimplifiedGather(const HloGatherInstruction *gather);

protected:
  bool InstructionMatchesPattern(HloInstruction *inst) override;

  absl::StatusOr<HloInstruction *>
  ExpandInstruction(HloInstruction *inst) override;
};

absl::StatusOr<HloInstruction *>
GeneralGatherSimplifier::ExpandInstruction(HloInstruction *inst) {
  auto *gather = DynCast<HloGatherInstruction>(inst);

  // If any slice size is 0, we can just return a constant zero.
  if (absl::c_linear_search(gather->gather_slice_sizes(), 0)) {
    auto *zero = gather->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(gather->shape().element_type())));
    return gather->AddInstruction(
        HloInstruction::CreateBroadcast(gather->shape(), zero, {}));
  }

  const auto &dims = gather->gather_dimension_numbers();

  // We break the logic into two separate branches for clarity:
  // We will use the `HloPassFix` in the pipeline, after the first
  // simplification, the gather may become a non-batched gather, which can be
  // further simplified in the next iteration.
  if (!dims.operand_batching_dims().empty() ||
      !dims.start_indices_batching_dims().empty()) {
    // Handle batched gather by making batch dims explicit in start_indices.
    // This effectively converts a batched gather into a regular gather.

    // 1. Prepare operands
    auto *operand = gather->operands()[0];
    auto *start_indices = gather->operands()[1];
    auto index_vector_dim = dims.index_vector_dim();
    auto start_indices_rank = start_indices->shape().dimensions().size();
    auto original_index_vector_dim = index_vector_dim;
    bool transposed = false;

    // Ensure start_indices has a vector dimension.
    // If index_vector_dim is equal to rank, it means it's implicitly 1 (scalar
    // indices).
    // We need to make it explicit.
    if (index_vector_dim == static_cast<int64_t>(start_indices_rank)) {
      std::vector<int64_t> new_dims(start_indices->shape().dimensions().begin(),
                                    start_indices->shape().dimensions().end());
      new_dims.push_back(1);
      auto new_shape =
          ShapeUtil::MakeShape(start_indices->shape().element_type(), new_dims);
      start_indices = gather->AddInstruction(
          HloInstruction::CreateReshape(new_shape, start_indices));
      start_indices_rank++;
      // For scalar case, original_index_vector_dim was effectively 'rank',
      // and we added a dim at 'rank'. So indices are preserved.
      // We can treat it as not transposed for index mapping purposes.
    } else if (index_vector_dim !=
               static_cast<int64_t>(start_indices_rank) - 1) {
      // Move index_vector_dim to the end for easier concatenation.
      std::vector<int64_t> perm;
      for (int64_t i = 0; i < static_cast<int64_t>(start_indices_rank); ++i) {
        if (i != index_vector_dim) {
          perm.push_back(i);
        }
      }
      perm.push_back(index_vector_dim);
      start_indices = gather->AddInstruction(HloInstruction::CreateTranspose(
          start_indices->shape(), start_indices, perm));
      index_vector_dim = start_indices_rank - 1;
      transposed = true;
    }

    // 2. Generate batch indices
    std::vector<HloInstruction *> new_index_components;

    // The batching dims in start_indices correspond to the batching dims in
    // operand. We need to create an Iota for each batch dim and broadcast it to
    // match start_indices.

    // We iterate through operand_batching_dims. For each, there must be a
    // corresponding dimension in start_indices_batching_dims (they are paired).
    for (int64_t i = 0;
         i < static_cast<int64_t>(dims.operand_batching_dims().size()); ++i) {
      int64_t batch_dim_idx = dims.start_indices_batching_dims(i);

      // Adjust batch_dim_idx if we transposed start_indices
      if (transposed) {
        if (batch_dim_idx > original_index_vector_dim) {
          batch_dim_idx--;
        } else if (batch_dim_idx == original_index_vector_dim) {
          batch_dim_idx = start_indices_rank - 1;
        }
      }

      int64_t batch_dim_size = start_indices->shape().dimensions(batch_dim_idx);

      // Create Iota for this batch dimension
      // We need to broadcast this iota to the shape of start_indices (excluding
      // vector dim). Since we moved vector dim to the end, we can just
      // broadcast to [dims without vector dim].
      std::vector<int64_t> broadcast_dims = {batch_dim_idx};
      Shape component_shape = start_indices->shape();
      // The component will be concatenated along the vector dim, so its vector
      // dim size is 1.
      std::vector<int64_t> component_dims(component_shape.dimensions().begin(),
                                          component_shape.dimensions().end());
      component_dims[start_indices_rank - 1] = 1;
      Shape final_component_shape = ShapeUtil::MakeShape(
          start_indices->shape().element_type(), component_dims);

      // Create Iota [batch_size]
      Shape iota_shape = ShapeUtil::MakeShape(
          start_indices->shape().element_type(), {batch_dim_size});
      auto *iota =
          gather->AddInstruction(HloInstruction::CreateIota(iota_shape, 0));

      // Broadcast to full shape
      auto *broadcasted_iota =
          gather->AddInstruction(HloInstruction::CreateBroadcast(
              final_component_shape, iota, broadcast_dims));

      new_index_components.push_back(broadcasted_iota);
    }

    // 3. Concatenate batch indices with original indices
    new_index_components.push_back(start_indices);
    // Calculate the expected output shape for concatenation
    std::vector<int64_t> concat_output_dims(
        start_indices->shape().dimensions().begin(),
        start_indices->shape().dimensions().end());
    int64_t concat_dim_size = 0;
    for (const auto *comp : new_index_components) {
      concat_dim_size += comp->shape().dimensions(start_indices_rank - 1);
    }
    concat_output_dims[start_indices_rank - 1] = concat_dim_size;
    Shape concat_output_shape = ShapeUtil::MakeShape(
        start_indices->shape().element_type(), concat_output_dims);

    auto *new_start_indices =
        gather->AddInstruction(HloInstruction::CreateConcatenate(
            concat_output_shape, new_index_components, start_indices_rank - 1));

    // 4. Update GatherDimensionNumbers
    auto new_dims = dims;
    new_dims.clear_operand_batching_dims();
    new_dims.clear_start_indices_batching_dims();

    // Prepend operand batching dims to start_index_map
    // Because we prepended the batch indices to the start_indices tensor.
    std::vector<int64_t> new_start_index_map(
        dims.operand_batching_dims().begin(),
        dims.operand_batching_dims().end());
    new_start_index_map.insert(new_start_index_map.end(),
                               dims.start_index_map().begin(),
                               dims.start_index_map().end());

    new_dims.clear_start_index_map();
    for (auto dim : new_start_index_map) {
      new_dims.add_start_index_map(dim);
    }

    // Since we slice size=1 on batching dims, we must collapse them to avoid
    // having an explicit dimension of size 1 in the output data part.
    std::vector<int64_t> new_collapsed_slice_dims(
        dims.collapsed_slice_dims().begin(), dims.collapsed_slice_dims().end());
    new_collapsed_slice_dims.insert(new_collapsed_slice_dims.end(),
                                    dims.operand_batching_dims().begin(),
                                    dims.operand_batching_dims().end());
    std::sort(new_collapsed_slice_dims.begin(), new_collapsed_slice_dims.end());

    new_dims.clear_collapsed_slice_dims();
    for (auto dim : new_collapsed_slice_dims) {
      new_dims.add_collapsed_slice_dims(dim);
    }

    // Update index_vector_dim (we ensured it's the last one)
    new_dims.set_index_vector_dim(start_indices_rank - 1);

    // 5. Update Slice Sizes
    std::vector<int64_t> new_slice_sizes(gather->gather_slice_sizes().begin(),
                                         gather->gather_slice_sizes().end());
    for (auto dim : dims.operand_batching_dims()) {
      SPU_ENFORCE(new_slice_sizes[dim] == 1, "dim {} should be 1, but got {}",
                  dim, new_slice_sizes[dim]);
    }

    // 6. Create new Gather
    auto *new_gather = gather->AddInstruction(HloInstruction::CreateGather(
        gather->shape(), operand, new_start_indices, new_dims, new_slice_sizes,
        gather->indices_are_sorted()));

    // Replace the old gather with the new one
    return new_gather;
  }

  // the raw branch of gather simplifier from XLA
  int operand_rank =
      dims.collapsed_slice_dims().size() + dims.offset_dims().size();

  // Make the operand conform to start_index_map.
  auto [operand_permutation, operand_permutation_inverse] =
      MakeOperandStartIndexPermutations(dims.start_index_map(), operand_rank);
  auto *operand = gather->operands()[0];
  auto *start_indices = gather->operands()[1];
  TF_ASSIGN_OR_RETURN(operand, MaybeTranspose(operand, operand_permutation));
  TF_ASSIGN_OR_RETURN(
      start_indices,
      TransformStartIndices(start_indices, dims.index_vector_dim()));

  // Permute the slice sizes according to start_index_map and compute the new
  // output shape for the Gather op.
  auto slice_sizes = Permute(gather->gather_slice_sizes(), operand_permutation);
  std::vector<int64_t> output_dims = {start_indices->shape().dimensions(0)};
  absl::c_copy(slice_sizes, std::back_inserter(output_dims));
  Shape output_shape =
      ShapeUtil::MakeShape(operand->shape().element_type(), output_dims);

  std::vector<int64_t> offset_dims(operand_rank);
  absl::c_iota(offset_dims, 1);
  std::vector<int64_t> start_index_map(dims.start_index_map().size());
  absl::c_iota(start_index_map, 0);

  auto *result = gather->AddInstruction(HloInstruction::CreateGather(
      output_shape, operand, start_indices,
      HloGatherInstruction::MakeGatherDimNumbers(
          offset_dims,
          /*collapsed_slice_dims=*/{}, start_index_map, /*index_vector_dim=*/1),
      slice_sizes, gather->indices_are_sorted()));

  // Undo the start_index_map transpose.
  std::vector<int64_t> output_permutation(1 + // start index dimension.
                                          operand_rank);
  absl::c_transform(operand_permutation_inverse, output_permutation.begin() + 1,
                    [](int64_t dim) { return dim + 1; });
  TF_ASSIGN_OR_RETURN(result, MaybeTranspose(result, output_permutation));

  // Collapse the requested slice dimensions.
  if (!dims.collapsed_slice_dims().empty()) {
    std::vector<int64_t> collapsed_slice_dims(
        dims.collapsed_slice_dims().size());
    absl::c_transform(dims.collapsed_slice_dims(), collapsed_slice_dims.begin(),
                      [](int64_t dim) { return dim + 1; });
    TF_ASSIGN_OR_RETURN(result,
                        ElideDegenerateDims(result, collapsed_slice_dims));
  }

  // Expand the start index dimensions.
  auto original_start_index_dims = gather->operands()[1]->shape().dimensions();
  std::vector<int64_t> start_indices_dims;
  for (uint32_t i = 0; i < original_start_index_dims.size(); ++i) {
    if (i != dims.index_vector_dim()) {
      start_indices_dims.push_back(original_start_index_dims[i]);
    }
  }
  if (start_indices_dims.size() > 1) {
    TF_ASSIGN_OR_RETURN(result,
                        ExpandFirstDimIntoNDims(result, start_indices_dims));
  } else if (start_indices_dims.empty()) {
    TF_ASSIGN_OR_RETURN(result, ElideDegenerateDims(result, {0}));
  }

  // Move the offset dims to the final locations.
  std::vector<int64_t> output_perm;
  auto output_rank = static_cast<int64_t>(start_indices_dims.size() +
                                          dims.offset_dims().size());
  output_perm.reserve(output_rank);
  auto offset_dim_index = static_cast<int64_t>(start_indices_dims.size());
  int64_t start_index_dim_index = 0;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (absl::c_linear_search(dims.offset_dims(), i)) {
      output_perm.push_back(offset_dim_index++);
    } else {
      output_perm.push_back(start_index_dim_index++);
    }
  }
  return MaybeTranspose(result, output_perm);
}

bool GeneralGatherSimplifier::IsSimplifiedGather(
    const HloGatherInstruction *gather) {
  auto *start_indices = gather->operands()[1];
  const auto &dims = gather->gather_dimension_numbers();
  return start_indices->shape().dimensions().size() == 2 &&
         dims.index_vector_dim() == 1 &&
         IsIdentityPermutation(dims.start_index_map()) &&
         dims.collapsed_slice_dims().empty() &&
         *dims.offset_dims().begin() == 1 &&
         *dims.offset_dims().rbegin() == dims.offset_dims().size();
}

bool GeneralGatherSimplifier::InstructionMatchesPattern(HloInstruction *inst) {
  auto *gather = DynCast<HloGatherInstruction>(inst);
  return (gather != nullptr) && !IsSimplifiedGather(gather);
}

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
  pipeline.AddPass<ConvolutionGroupConverter>(
      /*should_expand*/ [](HloInstruction *) { return true; },
      /*is_cost_viable*/ [](HloInstruction *) { return true; },
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

    // use our general gather simplifier to handle complex gather
    pipeline.AddPass<GeneralGatherSimplifier>();
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
    SPU_THROW("{}", module_config.status().message());
  }

  auto module = xla::HloModule::CreateFromProto(hlo_module, *module_config);
  if (!module.status().ok()) {
    SPU_THROW("{}", module.status().message());
  }

  xla::runHloPasses((*module).get());

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
