#include "libspu/kernel/hal/sort.h"

#include <algorithm>

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

#include "libspu/spu.pb.h"

namespace spu::kernel::hal {
namespace {

Value Permute1D(SPUContext *, const Value &x, const Index &indices) {
  SPU_ENFORCE(x.shape().size() == 1);
  return Value(x.data().linear_gather(indices), x.dtype());
}

using SequenceT = std::vector<std::pair<Index, Index>>;

void CmpSwap(SPUContext *ctx, const CompFn &comparator_body,
             std::vector<spu::Value> &values_to_sort, const Index &lhs_indices,
             const Index &rhs_indices) {
  size_t num_operands = values_to_sort.size();

  std::vector<spu::Value> values;
  values.reserve(2 * num_operands);
  for (size_t i = 0; i < num_operands; ++i) {
    values.emplace_back(values_to_sort[i].data().linear_gather(lhs_indices),
                        values_to_sort[i].dtype());
    values.emplace_back(values_to_sort[i].data().linear_gather(rhs_indices),
                        values_to_sort[i].dtype());
  }

  spu::Value predicate = comparator_body(values);
  predicate = hal::_prefer_a(ctx, predicate);

  for (size_t i = 0; i < num_operands; ++i) {
    auto fst = values[2 * i];
    auto sec = values[2 * i + 1];

    auto greater = spu::kernel::hal::select(ctx, predicate, fst, sec);
    auto less = spu::kernel::hal::select(ctx, predicate, sec, fst);

    values_to_sort[i].data().linear_scatter(greater.data(), lhs_indices);
    values_to_sort[i].data().linear_scatter(less.data(), rhs_indices);
  }
}

// Bitonic sort sequence for arbitrary size
// Ref:
// https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
inline int GreatestPowerOfTwoLessThan(int64_t n) {
  int64_t k = 1;
  while (k < n) {
    k = k << 1;
  }
  return k >> 1;
}

void MergeSequence(SequenceT &seq, int64_t lo, int64_t n, bool forward,
                   int64_t &depth) {
  if (n > 1) {
    auto m = GreatestPowerOfTwoLessThan(n);
    if (static_cast<int64_t>(seq.size()) - 1 < depth) {
      seq.resize(depth + 1);
    }
    for (auto i = lo; i < lo + n - m; ++i) {
      if (forward) {
        seq[depth].first.emplace_back(i);
        seq[depth].second.emplace_back(i + m);
      } else {
        seq[depth].first.emplace_back(i + m);
        seq[depth].second.emplace_back(i);
      }
    }
    ++depth;

    int64_t lower_depth = depth;
    MergeSequence(seq, lo, m, forward, lower_depth);

    int64_t upper_depth = depth;
    MergeSequence(seq, lo + m, n - m, forward, upper_depth);

    depth = std::max(lower_depth, upper_depth);
  }
}

void SortSequence(SequenceT &seq, int64_t lo, int64_t n, bool forward,
                  int64_t &depth) {
  if (n > 1) {
    int64_t m = n / 2;
    int64_t lower_depth = depth;

    SortSequence(seq, lo, m, !forward, lower_depth);

    int64_t upper_depth = depth;
    SortSequence(seq, lo + m, n - m, forward, upper_depth);

    depth = std::max(lower_depth, upper_depth);

    MergeSequence(seq, lo, n, forward, ++depth);
  }
}

void BuildCmpSwapSequence(SequenceT &seq, int64_t numel) {
  int64_t depth = 0;
  SortSequence(seq, 0, numel, true, depth);
}

void BitonicSort(SPUContext *ctx, const CompFn &comparator_body,
                 std::vector<spu::Value> &values_to_sort) {
  // Build a sorting network...
  SequenceT sequence;
  BuildCmpSwapSequence(sequence, values_to_sort.front().numel());

  for (const auto &seq : sequence) {
    if (seq.first.empty()) {
      continue;  // Skip empty sequence
    }
    CmpSwap(ctx, comparator_body, values_to_sort, seq.first, seq.second);
  }
}

}  // namespace

std::vector<spu::Value> sort1d(SPUContext *ctx,
                               absl::Span<spu::Value const> inputs,
                               const CompFn &cmp, Visibility comparator_ret_vis,
                               bool is_stable) {
  // sanity check.
  SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
  SPU_ENFORCE(inputs[0].shape().ndim() == 1,
              "Inputs should be 1-d but actually have {} dimensions",
              inputs[0].shape().ndim());
  SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                          [&inputs](const spu::Value v) {
                            return v.shape() == inputs[0].shape();
                          }),
              "Inputs shape mismatched");

  std::vector<spu::Value> ret;
  if (comparator_ret_vis == VIS_PUBLIC) {
    Index indices_to_sort(inputs[0].numel());
    std::iota(indices_to_sort.begin(), indices_to_sort.end(), 0);
    auto comparator = [&cmp, &inputs, &ctx](int64_t a, int64_t b) {
      std::vector<spu::Value> values;
      values.reserve(2 * inputs.size());
      for (int64_t i = 0; i < static_cast<int64_t>(inputs.size()); ++i) {
        values.push_back(hal::slice(ctx, inputs[i], {a}, {a + 1}));
        values.push_back(hal::slice(ctx, inputs[i], {b}, {b + 1}));
      }
      spu::Value cmp_ret = cmp(values);
      return getBooleanValue(ctx, cmp_ret);
    };

    if (is_stable) {
      std::stable_sort(indices_to_sort.begin(), indices_to_sort.end(),
                       comparator);
    } else {
      std::sort(indices_to_sort.begin(), indices_to_sort.end(), comparator);
    }

    ret.reserve(inputs.size());
    for (int64_t i = 0; i < static_cast<int64_t>(inputs.size()); ++i) {
      ret.push_back(Permute1D(ctx, inputs[i], indices_to_sort));
    }
  } else {
    SPU_ENFORCE(!is_stable,
                "Stable sort is unsupported if comparator return is secret.");

    // make a copy for inplace sort
    for (auto const &input : inputs) {
      ret.push_back(input.clone());
    }
    BitonicSort(ctx, cmp, ret);
  }

  return ret;
}

std::vector<spu::Value> simple_sort1d(SPUContext *ctx,
                                      absl::Span<spu::Value const> inputs,
                                      SortDirection direction) {
  // Fall back to generic sort
  SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
  if (inputs[0].isPublic() || !ctx->hasKernel("sort_a")) {
    auto ret = sort1d(
        ctx, inputs,
        [&](absl::Span<const spu::Value> cmp_inputs) {
          if (direction == SortDirection::Ascending) {
            return hal::less(ctx, cmp_inputs[0], cmp_inputs[1]);
          }
          if (direction == SortDirection::Descending) {
            return hal::greater(ctx, cmp_inputs[0], cmp_inputs[1]);
          }
          SPU_THROW("Should not reach here");
        },
        inputs[0].vtype(), false);
    return ret;
  } else {
    auto ret = _sort_s(ctx, inputs);
    if (direction == SortDirection::Descending) {
      std::reverse(ret.begin(), ret.end());
    }
    return ret;
  }
}

}  // namespace spu::kernel::hal