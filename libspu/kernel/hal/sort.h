#pragma once

#include "absl/types/span.h"

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

using CompFn = std::function<spu::Value(absl::Span<const spu::Value>)>;

// sort direction for sorters without comparators
enum class SortDirection {
  Ascending,
  Descending,
};

// general sort1d with comparator
std::vector<spu::Value> sort1d(SPUContext *ctx,
                               absl::Span<spu::Value const> inputs,
                               const CompFn &cmp, Visibility comparator_ret_vis,
                               bool is_stable);

// simple sort1d.
//
// Inputs:
//  - inputs: a vector of 1-D operands to be sorted
//  - direction: sorting order
//  - num_keys: the number of operands to treat as keys (count from index 0)
//  - valid_bits: indicates the numeric range of keys for performance hint
std::vector<spu::Value> simple_sort1d(SPUContext *ctx,
                                      absl::Span<spu::Value const> inputs,
                                      SortDirection direction, int64_t num_keys,
                                      int64_t valid_bits);

}  // namespace spu::kernel::hal