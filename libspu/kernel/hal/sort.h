#pragma once

#include "absl/types/span.h"

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {

using CompFn = std::function<spu::Value(absl::Span<const spu::Value>)>;

// simple sort direction
enum class SortDirection {
  Ascending,
  Descending,
};

// general sort1d with comparator
std::vector<spu::Value> sort1d(SPUContext *ctx,
                               absl::Span<spu::Value const> inputs,
                               const CompFn &cmp, Visibility comparator_ret_vis,
                               bool is_stable);

// simple sort1d without comparator
std::vector<spu::Value> simple_sort1d(SPUContext *ctx,
                                      absl::Span<spu::Value const> inputs,
                                      SortDirection direction);

}  // namespace spu::kernel::hal