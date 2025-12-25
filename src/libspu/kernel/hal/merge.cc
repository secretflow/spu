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

  // split_idx 是 x1 的大小
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
                 "Number of input tensors and valid tensors must match");

  // 1. 拼接 Values
  std::vector<spu::Value> inputs_vec(inputs.begin(), inputs.end());
  const auto &shape = inputs_vec[0].shape();
  const int64_t split_idx = shape[sort_dim];
  spu::Value combined_values = hal::concatenate(ctx, inputs_vec, sort_dim);

  // 2. 拼接 payloads
  std::vector<spu::Value> payloads_vec(payloads.begin(), payloads.end());
  spu::Value combined_payloads = hal::concatenate(ctx, payloads_vec, sort_dim);

  // 3. 打包成对，准备降维
  // packed_input 包含两个大 Tensor：[0]=Values, [1]=payloads
  std::vector<spu::Value> packed_input = {combined_values, combined_payloads};

  // 4. 定义 1D 处理回调
  auto sort_fn = [&](absl::Span<const spu::Value> sliced_inputs) {
    // sliced_inputs[0] 是切片后的 Value
    // sliced_inputs[1] 是切片后的 Valid
    return merge1d_with_payloads(ctx, sliced_inputs, split_idx, comparator_body,
                                 is_stable);
  };

  // 5. 执行 Permute
  // permute 会自动遍历除了 sort_dim 以外的维度，将 sort_dim 这一维切出来传给
  // sort_fn
  return hal::permute(ctx, packed_input, sort_dim, sort_fn);
}

}  // namespace spu::kernel::hal