// Copyright 2022 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sort.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/utils.h"

#include "libspu/spu.pb.h"

namespace spu::kernel::hlo {
namespace {

Value permute1D(SPUContext *ctx, const Value &x,
                absl::Span<const int64_t> indices) {
  SPU_ENFORCE(x.shape().size() == 1);
  return Value(x.data().linear_gather(indices), x.dtype());
}

void SliceCopy(spu::Value &dst, const spu::Value &src,
               std::vector<int64_t> dst_indices, size_t dim) {
  auto copy_size = src.shape()[0];
  for (int64_t idx = 0; idx < copy_size; ++idx) {
    dst_indices[dim] = idx;
    dst.data().update_slice(src.data().slice_scalar_at({idx}), dst_indices);
  }
}

// Sort will be inplace, so always make a copy here.
std::vector<spu::Value> GetValuesToSort(SPUContext *ctx,
                                        absl::Span<const spu::Value> inputs,
                                        const std::vector<int64_t> &indices,
                                        int64_t sort_dim,
                                        int64_t sort_dim_elements,
                                        int64_t num_operands) {
  std::vector<int64_t> limit_indices(indices.begin(), indices.end());
  std::for_each(limit_indices.begin(), limit_indices.end(),
                [](int64_t &index) { ++index; });
  limit_indices[sort_dim] = sort_dim_elements;
  std::vector<spu::Value> values_to_sort;
  values_to_sort.reserve(num_operands);
  for (int64_t i = 0; i < num_operands; ++i) {
    auto value_to_sort = hal::reshape(
        ctx, hal::slice(ctx, inputs[i], indices, limit_indices, {}),
        {sort_dim_elements});
    // So reshape is free...make a copy
    if (value_to_sort.data().buf()->data() == inputs[i].data().buf()->data()) {
      values_to_sort.emplace_back(value_to_sort.clone());
    } else {
      values_to_sort.emplace_back(value_to_sort);
    }
  }
  return values_to_sort;
}

using SequenceT =
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>;

void CmpSwap(SPUContext *ctx, const CompFn &comparator_body,
             std::vector<spu::Value> &values_to_sort,
             absl::Span<const int64_t> lhs_indices,
             absl::Span<const int64_t> rhs_indices) {
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

std::vector<spu::Value> Sort(SPUContext *ctx,
                             absl::Span<const spu::Value> inputs,
                             int64_t sort_dim, bool is_stable,
                             const CompFn &comparator_body,
                             Visibility comparator_ret_vis) {
  int64_t num_operands = inputs.size();
  auto key_shape = inputs[0].shape();
  auto rank = key_shape.size();
  std::vector<spu::Value> results;
  results.reserve(num_operands);
  for (int64_t i = 0; i < num_operands; ++i) {
    results.emplace_back(
        NdArrayRef(inputs[i].data().eltype(), inputs[i].shape()),
        inputs[i].dtype());
  }
  std::vector<int64_t> zero_base(rank, 0);
  std::vector<int64_t> increment(rank, 1);
  int64_t sort_dim_elements = key_shape[sort_dim];
  SPU_ENFORCE(
      sort_dim >= 0 && sort_dim < static_cast<int64_t>(increment.size()),
      "Unexpected out-of-bound sort dimension {}"
      " accessing increment of size {} ",
      sort_dim, increment.size());
  increment[sort_dim] = sort_dim_elements;

  if (comparator_ret_vis == VIS_PUBLIC) {
    // Iterate through each dimension except 'sort_dim'.
    forEachIndex(key_shape, zero_base, key_shape, increment,
                 [&](const std::vector<int64_t> &indices) {
                   // Extract a slice from each operand literal that corresponds
                   // to exactly the row in dimension 'sort_dim'.
                   std::vector<spu::Value> values_to_sort =
                       GetValuesToSort(ctx, inputs, indices, sort_dim,
                                       sort_dim_elements, num_operands);

                   std::vector<int64_t> indices_to_sort(sort_dim_elements);
                   std::iota(indices_to_sort.begin(), indices_to_sort.end(), 0);
                   auto comparator = [&comparator_body, &num_operands, &ctx,
                                      &values_to_sort](int64_t a, int64_t b) {
                     std::vector<spu::Value> values;
                     values.reserve(2 * num_operands);
                     for (int64_t i = 0; i < num_operands; ++i) {
                       values.push_back(
                           hal::slice_scalar_at(ctx, values_to_sort[i], {a}));
                       values.push_back(
                           hal::slice_scalar_at(ctx, values_to_sort[i], {b}));
                     }
                     spu::Value ret = comparator_body(values);
                     return getBooleanValue(ctx, ret);
                   };

                   if (is_stable) {
                     std::stable_sort(indices_to_sort.begin(),
                                      indices_to_sort.end(), comparator);
                   } else {
                     std::sort(indices_to_sort.begin(), indices_to_sort.end(),
                               comparator);
                   }

                   std::vector<int64_t> start_indices(rank, 0);
                   for (int64_t i = 0; i < num_operands; ++i) {
                     auto sorted_value =
                         permute1D(ctx, values_to_sort[i], indices_to_sort);
                     SliceCopy(results[i], sorted_value, indices, sort_dim);
                   }
                 });
  } else {
    SPU_ENFORCE(!is_stable,
                "Stable sort is unsupported if comparator return is secret.");

    // Iterate through each dimension except 'sort_dim'.
    forEachIndex(
        key_shape, zero_base, key_shape, increment,
        [&](const std::vector<int64_t> &indices) {
          std::vector<spu::Value> values_to_sort = GetValuesToSort(
              ctx, inputs, indices, sort_dim, sort_dim_elements, num_operands);

          BitonicSort(ctx, comparator_body, values_to_sort);

          for (int64_t i = 0; i < num_operands; ++i) {
            SliceCopy(results[i], values_to_sort.at(i), indices, sort_dim);
          }
        });
  }

  return results;
}

}  // namespace spu::kernel::hlo
