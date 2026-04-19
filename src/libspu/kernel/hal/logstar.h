#pragma once

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/permute.h"

namespace spu::kernel::hal {
spu::Value duplicate_brent_kung(SPUContext* ctx, const spu::Value& x,
                                const spu::Value& g_in);

std::pair<std::vector<spu::Value>, int64_t> extract_ordered(
    SPUContext* ctx, const spu::Value& x_in, const spu::Value& conditions);

spu::Value LogstarRecursive(SPUContext* ctx, SortDirection direction,
                            const spu::Value& x, const spu::Value& y);

spu::Value logstar(SPUContext* ctx, SortDirection direction,
                   const spu::Value& x, const spu::Value& y);

}  // namespace spu::kernel::hal