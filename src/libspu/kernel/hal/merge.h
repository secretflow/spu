#pragma once

#include "libspu/kernel/hal/permute.h"

namespace spu::kernel::hal {
std::vector<spu::Value> merge(SPUContext* ctx,
                              absl::Span<const spu::Value> inputs,
                              int64_t sort_dim, bool is_stable,
                              const hal::CompFn& comparator_body,
                              Visibility comparator_ret_vis);

std::vector<spu::Value> merge_with_payloads(
    SPUContext* ctx, absl::Span<const spu::Value> inputs,
    absl::Span<const spu::Value> payloads, int64_t sort_dim, bool is_stable,
    const hal::CompFn& comparator_body);

}  // namespace spu::kernel::hal
