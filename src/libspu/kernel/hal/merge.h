#pragma once

#include "libspu/kernel/hal/permute.h"

namespace spu::kernel::hal {
std::vector<spu::Value> merge(SPUContext *ctx,
                              absl::Span<const spu::Value> keys,
                              absl::Span<const spu::Value> payloads,
                              int64_t sort_dim, bool is_stable,
                              SortDirection direction,
                              Visibility comparator_ret_vis);

}  // namespace spu::kernel::hal
