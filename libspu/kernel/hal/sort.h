#pragma once

#include "absl/types/span.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {

using CompFn = std::function<spu::Value(absl::Span<const spu::Value>)>;

std::vector<spu::Value> sort1d(SPUContext *ctx,
                               absl::Span<spu::Value const> inputs,
                               const CompFn &cmp, Visibility comparator_ret_vis,
                               bool is_stable);

}  // namespace spu::kernel::hal