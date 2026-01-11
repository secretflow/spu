#include "libspu/kernel/hal/merge.h"

#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
namespace spu::kernel::hal {

// keys is placed as: [lhs_key1, rhs_key1, lhs_key2, rhs_key2, ...]
std::vector<spu::Value> merge(SPUContext *ctx,
                              absl::Span<const spu::Value> keys,
                              absl::Span<const spu::Value> payloads,
                              int64_t sort_dim, bool is_stable,
                              const hal::CompFn &comparator_body,
                              Visibility comparator_ret_vis) {
  const auto &kshape = keys[0].shape();
  SPU_ENFORCE(kshape[0] % 2 == 0,
              "The number of key arrays must be even, got={}", kshape[0]);
  spu::Shape new_shape = {kshape[0] / 2, kshape[1] * 2};
  spu::Value keys_concated = hal::reshape(ctx, keys[0], new_shape);
  const int64_t split_idx = kshape[sort_dim];

  std::vector<spu::Value> input_concated;
  bool with_payloads;
  if (payloads.empty()) {
    input_concated = {keys_concated};
    with_payloads = false;
  } else {
    const auto &pshape = payloads[0].shape();
    SPU_ENFORCE(pshape[0] % 2 == 0,
                "The number of payload arrays must be even, got={}", pshape[0]);
    SPU_ENFORCE(kshape[0] == pshape[0],
                "The numbers of key arrays and payload arrays must be equal, "
                "got keys={}, "
                "payloads={}",
                kshape[0], pshape[0]);
    spu::Shape new_shape = {pshape[0] / 2, pshape[1] * 2};
    spu::Value payloads_concated = hal::reshape(ctx, payloads[0], new_shape);
    input_concated = {keys_concated, payloads_concated};
    with_payloads = true;
  }

  auto sort_fn = [&](absl::Span<const spu::Value> input) {
    return hal::merge1d(ctx, input, with_payloads, split_idx, comparator_body,
                        comparator_ret_vis, is_stable);
  };
  return hal::permute(ctx, input_concated, sort_dim, sort_fn);
}

}  // namespace spu::kernel::hal