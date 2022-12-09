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

#include "absl/numeric/bits.h"

#include "spu/kernel/hal/concat.h"
#include "spu/kernel/hal/permute_util.h"
#include "spu/kernel/hal/polymorphic.h"
#include "spu/kernel/hlo/casting.h"
#include "spu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {
namespace {

void SliceCopy(spu::Value &dst, const spu::Value &src,
               std::vector<int64_t> dst_indices, size_t dim) {
  auto copy_size = src.shape()[0];
  for (int64_t idx = 0; idx < copy_size; ++idx) {
    dst_indices[dim] = idx;
    dst.copyElementFrom(src, {idx}, dst_indices);
  }
}

std::vector<spu::Value> GetValuesToSort(HalContext *ctx,
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
    values_to_sort.push_back(std::move(value_to_sort));
  }
  return values_to_sort;
}

// Refers to implementation of emp-tool:
// https://github.com/emp-toolkit/emp-tool/blob/b07a7d9ab3053a3e16991751402742d418377f63/emp-tool/circuits/number.h
void CmpSwap(HalContext *ctx, const CompFn &comparator_body,
             std::vector<spu::Value> *values_to_sort, int64_t x_start_indices,
             int64_t y_start_indices, int64_t n) {
  size_t num_operands = values_to_sort->size();

  std::vector<spu::Value> values;
  values.reserve(2 * num_operands);
  for (size_t i = 0; i < num_operands; ++i) {
    values.push_back(hal::slice(ctx, values_to_sort->at(i), {x_start_indices},
                                {x_start_indices + n}, {1}));
    values.push_back(hal::slice(ctx, values_to_sort->at(i), {y_start_indices},
                                {y_start_indices + n}, {1}));
  }
  spu::Value predicate = comparator_body(values);

  for (size_t i = 0; i < num_operands; ++i) {
    auto fst = hal::slice(ctx, values_to_sort->at(i), {x_start_indices},
                          {x_start_indices + n}, {1});
    auto sec = hal::slice(ctx, values_to_sort->at(i), {y_start_indices},
                          {y_start_indices + n}, {1});

    auto greater = spu::kernel::hal::select(ctx, predicate, fst, sec);
    auto less = spu::kernel::hal::select(ctx, predicate, sec, fst);

    values_to_sort->at(i).copyElementFrom(
        greater, {}, {static_cast<int64_t>(x_start_indices)},
        static_cast<int64_t>(n * values_to_sort->front().elsize()));
    values_to_sort->at(i).copyElementFrom(
        less, {}, {static_cast<int64_t>(y_start_indices)},
        static_cast<int64_t>(n * values_to_sort->front().elsize()));
  }
}

void GenerateBitonicMergeIndex(size_t n,
                               std::vector<std::vector<size_t>> *indices) {
  YACL_ENFORCE(absl::has_single_bit(n));
  size_t stage = absl::bit_width(n) - 1;

  for (int i = static_cast<int>(stage); i > 0; i--) {
    std::vector<size_t> fst;
    std::vector<size_t> sec;

    for (size_t j = 0; j < n; j++) {
      if (((j >> (i - 1)) & 1) == 0) {
        fst.emplace_back(j);
      } else {
        sec.emplace_back(j);
      }
    }

    fst.insert(fst.end(), sec.begin(), sec.end());
    indices->emplace_back(fst);
  }
}

void GenerateBitonicSortIndex(size_t n,
                              std::vector<std::vector<size_t>> *indices) {
  YACL_ENFORCE(absl::has_single_bit(n));
  size_t stage = absl::bit_width(n) - 1;

  for (int stage_idx = 0; stage_idx < static_cast<int>(stage - 1);
       stage_idx++) {
    for (int substage_idx = static_cast<int>(stage_idx); substage_idx > -1;
         substage_idx--) {
      std::vector<size_t> fst;
      std::vector<size_t> sec;
      for (size_t i = 0; i < n; i++) {
        bool asc_flag = ((i >> (stage_idx + 1)) & 1) == 0;
        bool fst_flag = ((i >> substage_idx) & 1) == 0;

        if (asc_flag ^ fst_flag) {
          sec.emplace_back(i);
        } else {
          fst.emplace_back(i);
        }
      }

      fst.insert(fst.end(), sec.begin(), sec.end());
      indices->emplace_back(fst);
    }
  }
}

std::vector<spu::Value> BitonicSort(
    HalContext *ctx, const CompFn &comparator_body,
    const std::vector<spu::Value> &values_to_sort, size_t n) {
  YACL_ENFORCE(absl::has_single_bit(n));

  std::vector<std::vector<size_t>> indices;
  GenerateBitonicSortIndex(n, &indices);
  GenerateBitonicMergeIndex(n, &indices);

  std::vector<spu::Value> target = values_to_sort;

  for (const auto &index : indices) {
    // permute
    std::vector<spu::Value> permuted_values;

    for (auto v : target) {
      permuted_values.emplace_back(hal::permute(ctx, v, 0, xt::adapt(index)));
    }

    // cmp and swap
    CmpSwap(ctx, comparator_body, &permuted_values, 0,
            static_cast<int64_t>(n / 2), static_cast<int64_t>(n / 2));

    // inverse permute
    std::vector<size_t> inverse_permutation(index.size());
    std::iota(inverse_permutation.begin(), inverse_permutation.end(), 0);
    std::sort(inverse_permutation.begin(), inverse_permutation.end(),
              [&index](int left, int right) -> bool {
                return index[left] < index[right];
              });

    target.clear();

    for (auto v : permuted_values) {
      target.emplace_back(
          hal::permute(ctx, v, 0, xt::adapt(inverse_permutation)));
    }
  }

  return target;
}

spu::Value Repeat(HalContext *ctx, const spu::Value &in, int64_t repeats) {
  const auto &origin_shape = in.shape();

  auto shape_1 = origin_shape;
  shape_1.insert(shape_1.begin(), 1);
  auto shape_2 = shape_1;
  shape_2[0] = repeats;
  auto shape_3 = origin_shape;
  shape_3[0] = origin_shape[0] * repeats;

  return hal::reshape(
      ctx, hal::broadcast_to(ctx, hal::reshape(ctx, in, shape_1), shape_2),
      shape_3);
}

std::vector<spu::Value> GetPaddingOfBitonicSort(
    HalContext *ctx, const CompFn &comparator_body,
    const std::vector<spu::Value> &values_to_sort) {
  std::vector<spu::Value> values = values_to_sort;
  int64_t n = values[0].numel();
  size_t num_operands = values_to_sort.size();

  while (n > 1) {
    size_t n_bit_floor = absl::bit_floor(static_cast<size_t>(n));

    std::vector<spu::Value> cmp_values;
    cmp_values.reserve(2 * num_operands);

    std::vector<spu::Value> new_values;
    new_values.reserve(num_operands);
    for (size_t i = 0; i < num_operands; ++i) {
      cmp_values.push_back(hal::slice(ctx, values.at(i), {0},
                                      {static_cast<int64_t>(n_bit_floor / 2)},
                                      {1}));
      cmp_values.push_back(
          hal::slice(ctx, values.at(i), {static_cast<int64_t>(n_bit_floor / 2)},
                     {static_cast<int64_t>(n_bit_floor)}, {1}));
    }

    spu::Value predicate = comparator_body(cmp_values);

    for (size_t i = 0; i < num_operands; ++i) {
      auto fst = hal::slice(ctx, values.at(i), {0},
                            {static_cast<int64_t>(n_bit_floor / 2)}, {1});
      auto sec =
          hal::slice(ctx, values.at(i), {static_cast<int64_t>(n_bit_floor / 2)},
                     {static_cast<int64_t>(n_bit_floor)}, {1});

      auto greater = spu::kernel::hal::select(ctx, predicate, sec, fst);

      if (absl::has_single_bit(static_cast<size_t>(n))) {
        new_values.push_back(greater);
      } else {
        auto tail = hal::slice(ctx, values.at(i),
                               {static_cast<int64_t>(n_bit_floor)}, {n}, {1});
        new_values.push_back(hal::concatenate(ctx, {greater, tail}, 0));
      }
    }

    values = new_values;
    n = values[0].numel();
  }

  return values;
}

}  // namespace

std::vector<spu::Value> Sort(HalContext *ctx,
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
  YACL_ENFORCE(
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
                       values.push_back(values_to_sort[i].getElementAt(a));
                       values.push_back(values_to_sort[i].getElementAt(b));
                     }
                     spu::Value ret = comparator_body(values);
                     return getConditionValue(ctx, ret);
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
                     auto sorted_value = hal::permute(
                         ctx, values_to_sort[i], 0, xt::adapt(indices_to_sort));
                     SliceCopy(results[i], sorted_value, indices, sort_dim);
                   }
                 });
  } else {
    // Iterate through each dimension except 'sort_dim'.
    forEachIndex(
        key_shape, zero_base, key_shape, increment,
        [&](const std::vector<int64_t> &indices) {
          std::vector<spu::Value> values_to_sort = GetValuesToSort(
              ctx, inputs, indices, sort_dim, sort_dim_elements, num_operands);

          int64_t original_n = values_to_sort[0].numel();
          if (original_n > 1) {
            if (absl::has_single_bit(static_cast<size_t>(original_n))) {
              values_to_sort =
                  BitonicSort(ctx, comparator_body, values_to_sort, original_n);

              for (int64_t i = 0; i < num_operands; ++i) {
                SliceCopy(results[i], values_to_sort[i], indices, sort_dim);
              }
            } else {
              auto padding =
                  GetPaddingOfBitonicSort(ctx, comparator_body, values_to_sort);

              int64_t padding_length =
                  absl::bit_ceil(static_cast<size_t>(original_n)) - original_n;

              for (int64_t i = 0; i < num_operands; ++i) {
                values_to_sort[i] =
                    hal::concatenate(ctx,
                                     {values_to_sort[i],
                                      Repeat(ctx, padding[i], padding_length)},
                                     0);
              }

              values_to_sort = BitonicSort(ctx, comparator_body, values_to_sort,
                                           values_to_sort[0].numel());

              for (int64_t i = 0; i < num_operands; ++i) {
                auto v = hal::slice(ctx, values_to_sort.at(i), {0},
                                    {original_n}, {1});

                SliceCopy(results[i], v, indices, sort_dim);
              }
            }
          } else {
            for (int64_t i = 0; i < num_operands; ++i) {
              SliceCopy(results[i], values_to_sort[i], indices, sort_dim);
            }
          }
        });
  }

  return results;
}

}  // namespace spu::kernel::hlo
