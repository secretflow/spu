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

#include <xtensor/xtensor_forward.hpp>

#include "absl/numeric/bits.h"
#include "xtensor/xeval.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xsort.hpp"

#include "libspu/core/xt_helper.h"
#include "libspu/kernel/hal/concat.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/utils.h"

namespace spu::kernel::hlo {
namespace {

// FIXME: Remove these hacky templates. Use the same stride arithmetic
// techniques with shape_ops.
template <size_t S>
struct element_t_s {
  std::array<std::byte, S> buf;
  // xtensor uses operator+ to compute type promotion rule of container element
  // So we provides a empty + to make it happy
  element_t_s operator+(const element_t_s & /*unused*/) { return *this; }
};

#define __CASE_SIZE(SIZE, ...)                       \
  case (SIZE): {                                     \
    using element_t = element_t_s<SIZE>;             \
    [[maybe_unused]] constexpr size_t _kSize = SIZE; \
    return __VA_ARGS__();                            \
  }

#define DISPATCH_ALL_ELSIZE(SIZE, ...)                        \
  [&] {                                                       \
    switch (SIZE) {                                           \
      __CASE_SIZE(1, __VA_ARGS__)                             \
      __CASE_SIZE(2, __VA_ARGS__)                             \
      __CASE_SIZE(4, __VA_ARGS__)                             \
      __CASE_SIZE(8, __VA_ARGS__)                             \
      __CASE_SIZE(16, __VA_ARGS__)                            \
      __CASE_SIZE(32, __VA_ARGS__)                            \
      default:                                                \
        SPU_THROW("un-implemented for elment_size={}", SIZE); \
    }                                                         \
  }()

Value permute(HalContext *ctx, const Value &x, size_t axis,
              const xt::xarray<size_t> &permutation) {
  const size_t dimension = x.shape().size();

  const auto &x_data = x.data();
  if (dimension == 1) {
    Value result({x_data.eltype(), x.shape()}, x.dtype());

    for (int64_t i = 0; i < x.numel(); i++) {
      result.copyElementFrom(x, {static_cast<int64_t>(permutation(i))}, {i});
    }

    return result;
  }

  if (axis < dimension - 1) {
    xt::dynamic_shape<std::size_t> perm;
    xt::dynamic_shape<std::size_t> reverse_perm;
    std::tie(perm, reverse_perm) = xt::detail::get_permutations(
        permutation.dimension(), axis, permutation.layout());

    auto permutation_t = xt::eval(xt::transpose(permutation, perm));

    const auto &x_data = x.data();
    return DISPATCH_ALL_ELSIZE(x_data.elsize(), [&]() -> Value {
      auto x_t = xt::eval(xt::transpose(xt_adapt<element_t>(x_data), perm));
      std::vector<int64_t> ret_shape{x_t.shape().begin(), x_t.shape().end()};
      NdArrayRef ret(x_data.eltype(), ret_shape);
      xt_mutable_adapt<element_t>(ret) = xt::empty<element_t>(ret_shape);

      std::size_t n_iters =
          std::accumulate(ret_shape.begin(), ret_shape.end() - 1,
                          std::size_t(1), std::multiplies<>());
      std::ptrdiff_t data_secondary_stride = ret_shape.back();
      auto x_ptr = x_t.data();
      auto permutation_ptr = permutation_t.data();
      auto ret_ptr = static_cast<element_t *>(ret.data());

      for (std::size_t i = 0; i < n_iters; i++, x_ptr += data_secondary_stride,
                       permutation_ptr += data_secondary_stride,
                       ret_ptr += data_secondary_stride) {
        for (std::ptrdiff_t j = 0; j < data_secondary_stride; j++) {
          std::memcpy(
              ret_ptr + j,
              x_ptr + static_cast<std::ptrdiff_t>(*(permutation_ptr + j)),
              sizeof(element_t));
        }
      }

      return hal::transpose(ctx, Value(ret, x.dtype()),
                            absl::MakeSpan((const int64_t *)reverse_perm.data(),
                                           reverse_perm.size()));
    });
  }

  return DISPATCH_ALL_ELSIZE(x.data().elsize(), [&]() -> Value {
    auto ret_shape = x.shape();
    NdArrayRef ret(x.data().eltype(), ret_shape);
    xt_mutable_adapt<element_t>(ret) = xt::empty<element_t>(ret_shape);

    std::size_t n_iters =
        std::accumulate(ret_shape.begin(), ret_shape.end() - 1, std::size_t(1),
                        std::multiplies<>());
    std::ptrdiff_t data_secondary_stride = ret_shape[axis];
    auto x_ptr = static_cast<const element_t *>(x.data().data());
    auto permutation_ptr = permutation.data();
    auto ret_ptr = static_cast<element_t *>(ret.data());

    for (std::size_t i = 0; i < n_iters; i++, x_ptr += data_secondary_stride,
                     permutation_ptr += data_secondary_stride,
                     ret_ptr += data_secondary_stride) {
      for (std::ptrdiff_t j = 0; j < data_secondary_stride; j++) {
        std::memcpy(ret_ptr + j,
                    x_ptr + static_cast<std::ptrdiff_t>(*(permutation_ptr + j)),
                    sizeof(element_t));
      }
    }

    return Value(ret, x.dtype());
  });
}

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
  SPU_ENFORCE(absl::has_single_bit(n));
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
  SPU_ENFORCE(absl::has_single_bit(n));
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

std::pair<std::vector<size_t>,
          std::vector<std::pair<size_t, std::vector<Value>>>>
PrepProcessPadding(HalContext *ctx,
                   const std::vector<spu::Value> &values_to_sort, size_t n,
                   const std::vector<size_t> &permute_index,
                   const std::vector<size_t> &padding_indices) {
  std::vector<size_t> new_padding_indices;
  std::vector<std::pair<size_t, std::vector<Value>>> value_records;

  std::unordered_set<size_t> padding_indices_set(padding_indices.begin(),
                                                 padding_indices.end());

  size_t num_operands = values_to_sort.size();

  if (!padding_indices.empty()) {
    for (size_t i = 0; i < n / 2; i++) {
      size_t fst_idx = permute_index.at(i);
      size_t sec_idx = permute_index.at(i + n / 2);
      bool fst_is_padding =
          padding_indices_set.find(fst_idx) != padding_indices_set.end();
      bool sec_is_padding =
          padding_indices_set.find(sec_idx) != padding_indices_set.end();

      if (fst_is_padding && sec_is_padding) {
        new_padding_indices.push_back(fst_idx);
        new_padding_indices.push_back(sec_idx);
      } else if (fst_is_padding || sec_is_padding) {
        new_padding_indices.push_back(sec_idx);

        std::vector<Value> fst_values;
        std::vector<Value> sec_values;

        for (size_t i = 0; i < num_operands; ++i) {
          fst_values.push_back(hal::slice(
              ctx, values_to_sort.at(i), {static_cast<int64_t>(fst_idx)},
              {static_cast<int64_t>(fst_idx + 1)}, {1}));
          sec_values.push_back(hal::slice(
              ctx, values_to_sort.at(i), {static_cast<int64_t>(sec_idx)},
              {static_cast<int64_t>(sec_idx + 1)}, {1}));
        }

        if (fst_is_padding) {
          value_records.emplace_back(fst_idx, sec_values);
          value_records.emplace_back(sec_idx, fst_values);
        } else {
          value_records.emplace_back(fst_idx, fst_values);
          value_records.emplace_back(sec_idx, sec_values);
        }
      }
    }
  }

  SPU_ENFORCE_EQ(padding_indices.size(), new_padding_indices.size());

  return {new_padding_indices, value_records};
}

void PostProcessPadding(
    HalContext *ctx,
    const std::vector<std::pair<size_t, std::vector<Value>>> &value_records,
    std::vector<spu::Value> *values_to_sort) {
  size_t num_operands = values_to_sort->size();

  for (const auto &index_value_pair : value_records) {
    for (size_t i = 0; i < num_operands; ++i) {
      values_to_sort->at(i).copyElementFrom(
          index_value_pair.second.at(i), {},
          {static_cast<int64_t>(index_value_pair.first)},
          static_cast<int64_t>(values_to_sort->front().elsize()));
    }
  }
}

std::vector<spu::Value> BitonicSort(
    HalContext *ctx, const CompFn &comparator_body,
    const std::vector<spu::Value> &values_to_sort, size_t n,
    const std::vector<size_t> &init_padding_indices) {
  SPU_ENFORCE(absl::has_single_bit(n));

  std::vector<std::vector<size_t>> indices;
  GenerateBitonicSortIndex(n, &indices);
  GenerateBitonicMergeIndex(n, &indices);

  std::vector<spu::Value> target = values_to_sort;

  std::vector<size_t> padding_indices = init_padding_indices;

  for (const auto &index : indices) {
    auto [new_padding_indices, value_records] =
        PrepProcessPadding(ctx, target, n, index, padding_indices);

    padding_indices = new_padding_indices;

    // permute
    std::vector<spu::Value> permuted_values;

    permuted_values.reserve(target.size());
    for (const auto &v : target) {
      permuted_values.emplace_back(permute(ctx, v, 0, xt::adapt(index)));
    }

    // cmp and swap
    // TODO(junfeng): We should avoid doing CmpSwap with paddings here to save
    // costs.
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

    for (const auto &v : permuted_values) {
      target.emplace_back(permute(ctx, v, 0, xt::adapt(inverse_permutation)));
    }

    PostProcessPadding(ctx, value_records, &target);
  }

  return target;
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
                       values.push_back(values_to_sort[i].getElementAt(a));
                       values.push_back(values_to_sort[i].getElementAt(b));
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
                     auto sorted_value = permute(ctx, values_to_sort[i], 0,
                                                 xt::adapt(indices_to_sort));
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

          int64_t original_n = values_to_sort[0].numel();
          if (original_n > 1) {
            int64_t padding_length =
                absl::bit_ceil(static_cast<size_t>(original_n)) - original_n;

            if (padding_length > 0) {
              for (int64_t i = 0; i < num_operands; ++i) {
                auto padding_shape = values_to_sort[i].shape();
                padding_shape[0] = padding_length;

                auto padding =
                    hal::zeros(ctx, values_to_sort[i].vtype(),
                               values_to_sort[i].dtype(), padding_shape);

                values_to_sort[i] =
                    hal::concatenate(ctx, {values_to_sort[i], padding}, 0);
              }
            }

            std::vector<size_t> init_padding_indices(padding_length);
            std::iota(init_padding_indices.begin(), init_padding_indices.end(),
                      original_n);

            auto sorted_value =
                BitonicSort(ctx, comparator_body, values_to_sort,
                            values_to_sort[0].numel(), init_padding_indices);

            for (int64_t i = 0; i < num_operands; ++i) {
              if (padding_length > 0) {
                auto v =
                    hal::slice(ctx, sorted_value.at(i), {0}, {original_n}, {1});

                SliceCopy(results[i], v, indices, sort_dim);
              } else {
                SliceCopy(results[i], sorted_value.at(i), indices, sort_dim);
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
