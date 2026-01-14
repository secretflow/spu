#include "libspu/kernel/hal/merge.h"

#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/test_util.h"
namespace spu::kernel::hal {

// keys is placed as: [lhs_key1, rhs_key1, lhs_key2, rhs_key2, ...]
// payloads is placed as: [lhs_payload1, rhs_payload1, lhs_payload2,
// rhs_payload2, ...]
std::vector<spu::Value> merge(SPUContext *ctx,
                              absl::Span<const spu::Value> keys,
                              absl::Span<const spu::Value> payloads,
                              int64_t sort_dim, bool is_stable,
                              SortDirection direction,
                              Visibility comparator_ret_vis) {
  // Sanity check
  SPU_ENFORCE(!keys.empty(), "Keys must not be empty");
  const auto &kshape = keys[0].shape();
  SPU_ENFORCE(kshape[0] % 2 == 0,
              "The number of key arrays must be even, got={}", kshape[0]);
  SPU_ENFORCE(sort_dim >= 0 && sort_dim < kshape.ndim(),
              "Invalid sort_dim={}, must be in range [0, {})", sort_dim,
              kshape.ndim());

  // Concatenate lhs_key and rhs_key to get [key1, key2, ...]
  spu::Shape new_shape = {kshape[0] / 2, kshape[1] * 2};
  spu::Value keys_concated = hal::reshape(ctx, keys[0], new_shape);
  const int64_t split_idx = kshape[sort_dim];

  std::vector<spu::Value> input_concated;
  bool with_payloads;
  if (payloads.empty()) {
    input_concated = {keys_concated};
    with_payloads = false;
  } else {
    // Process payloads
    const auto &pshape = payloads[0].shape();
    SPU_ENFORCE(pshape[0] % 2 == 0,
                "The number of payload arrays must be even, got={}", pshape[0]);
    SPU_ENFORCE(kshape == pshape,
                "Keys and payloads must have the same shape, "
                "got key_shape={}, payload_shape={}",
                kshape, pshape);

    // Concatenate lhs_payload and rhs_payload to get [payload1, payload2, ...]
    spu::Shape new_shape = {pshape[0] / 2, pshape[1] * 2};
    spu::Value payloads_concated = hal::reshape(ctx, payloads[0], new_shape);
    input_concated = {keys_concated, payloads_concated};
    with_payloads = true;
  }

  auto sort_fn = [&](absl::Span<const spu::Value> input) {
    return hal::merge1d(ctx, input, with_payloads, split_idx, direction,
                        comparator_ret_vis, is_stable);
  };
  return hal::permute(ctx, input_concated, sort_dim, sort_fn);
}

}  // namespace spu::kernel::hal