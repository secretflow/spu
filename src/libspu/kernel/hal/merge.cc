#include "libspu/kernel/hal/merge.h"

#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {

std::vector<spu::Value> merge(SPUContext *ctx,
                              absl::Span<const spu::Value> inputs,
                              int64_t sort_dim, bool is_stable,
                              const hal::CompFn &comparator_body,
                              Visibility comparator_ret_vis) {
  std::vector<spu::Value> inputs_vec(inputs.begin(), inputs.end());

  const auto &shape = inputs_vec[0].shape();
  const int64_t split_idx = shape[sort_dim];

  spu::Value combined = hal::concatenate(ctx, inputs_vec, sort_dim);
  std::vector<spu::Value> single_input_vec = {combined};

  auto sort_fn = [&](absl::Span<const spu::Value> input) {
    return hal::merge1d(ctx, input, split_idx, comparator_body,
                        comparator_ret_vis, is_stable);
  };
  return hal::permute(ctx, single_input_vec, sort_dim, sort_fn);
}

std::vector<spu::Value> merge_with_payloads(
    SPUContext *ctx, absl::Span<const spu::Value> inputs,
    absl::Span<const spu::Value> payloads, int64_t sort_dim, bool is_stable,
    const hal::CompFn &comparator_body) {
  SPU_ENFORCE_EQ(inputs.size(), payloads.size(),
                 "Number of input tensors and payload tensors must match");

  std::vector<spu::Value> inputs_vec(inputs.begin(), inputs.end());
  const auto &shape = inputs_vec[0].shape();
  const int64_t split_idx = shape[sort_dim];
  spu::Value combined_values = hal::concatenate(ctx, inputs_vec, sort_dim);

  std::vector<spu::Value> payloads_vec(payloads.begin(), payloads.end());
  spu::Value combined_payloads = hal::concatenate(ctx, payloads_vec, sort_dim);

  std::vector<spu::Value> packed_input = {combined_values, combined_payloads};

  auto sort_fn = [&](absl::Span<const spu::Value> sliced_inputs) {
    return merge1d_with_payloads(ctx, sliced_inputs, split_idx, comparator_body,
                                 is_stable);
  };
  return hal::permute(ctx, packed_input, sort_dim, sort_fn);
}

}  // namespace spu::kernel::hal