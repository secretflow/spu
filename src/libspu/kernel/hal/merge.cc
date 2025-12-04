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
  spu::Value combined = hal::concatenate(ctx, inputs_vec, sort_dim);
  std::vector<spu::Value> single_input_vec = {combined};

  auto sort_fn = [&](absl::Span<const spu::Value> input) {
    return hal::merge1d(ctx, input, comparator_body, comparator_ret_vis,
                        is_stable);
  };
  return hal::permute(ctx, single_input_vec, sort_dim, sort_fn);
}

}  // namespace spu::kernel::hal