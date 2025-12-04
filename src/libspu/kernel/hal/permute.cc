// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/kernel/hal/permute.h"

#include <algorithm>

#include "libspu/core/bit_utils.h"
#include "libspu/core/context.h"
#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/random.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/utils.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/spu.h"

namespace spu::kernel::hal {

namespace internal {

inline FieldType _get_field_from_n(SPUContext *ctx, size_t n) {
  const auto protocol = ctx->config().protocol;
  // only semi2k support flexible field selection now
  if (protocol != ProtocolKind::SEMI2K) {
    return FieldType::FT_INVALID;
  }

  if (n <= (static_cast<uint64_t>(1) << 8)) {
    return FieldType::FM8;
  } else if (n <= (static_cast<uint64_t>(1) << 16)) {
    return FieldType::FM16;
  } else if (n <= (static_cast<uint64_t>(1) << 32)) {
    return FieldType::FM32;
  } else {
    return FieldType::FM64;
  }
}

inline int64_t _get_owner(const Value &x) {
  return x.storage_type().as<Private>()->owner();
}

inline bool _has_same_owner(const Value &x, const Value &y) {
  return _get_owner(x) == _get_owner(y);
}

hal::CompFn _get_cmp_func(SPUContext *ctx, int64_t num_keys,
                          SortDirection direction, bool append_rand = false) {
  hal::CompFn comp_fn = [ctx, num_keys, direction, append_rand](
                            absl::Span<const spu::Value> values) -> spu::Value {
    auto scalar_cmp = [direction](spu::SPUContext *ctx, const spu::Value &lhs,
                                  const spu::Value &rhs) {
      if (direction == SortDirection::Ascending) {
        return hal::less(ctx, lhs, rhs);
      }
      return hal::greater(ctx, lhs, rhs);
    };

    spu::Value k1 = hal::constant(ctx, true, DT_I1, values[0].shape());
    spu::Value pre_equal = k1;
    spu::Value result = scalar_cmp(ctx, values[0], values[1]);
    // the idea here is that if the two values of the last key is equal,
    // than we compare the two values of the current key, and iteratively to
    // update the result which indicates whether to swap values
    int64_t idx;
    for (idx = 2; idx < num_keys * 2; idx += 2) {
      pre_equal = hal::bitwise_and(
          ctx, pre_equal, hal::equal(ctx, values[idx - 2], values[idx - 1]));
      auto current = scalar_cmp(ctx, values[idx], values[idx + 1]);
      current = hal::bitwise_and(ctx, pre_equal, current);
      result = hal::bitwise_or(ctx, result, current);
    }

    // append rand value to avoid the same key "pitfall" in partition-based
    // algorithms (e.g. quick-sort, quick-select).
    if (append_rand) {
      // must use secret bits here, otherwise some infos will leak
      auto rand_bits = hal::random(ctx, VIS_SECRET, DT_I1, values[0].shape());

      // equal has better performance for aby3
      // cmp+andbb has better performance for semi2k now
      pre_equal = hal::bitwise_and(
          ctx, pre_equal, hal::equal(ctx, values[idx - 2], values[idx - 1]));
      auto current = hal::bitwise_and(ctx, pre_equal, rand_bits);
      result = hal::bitwise_or(ctx, result, current);
    }

    return result;
  };

  return comp_fn;
}

bool _has_efficient_shuffle(SPUContext *ctx) {
  const auto prot = ctx->config().protocol;

  // semi2k and aby3 have highly efficient constant round implementation.
  return prot == ProtocolKind::SEMI2K || prot == ProtocolKind::ABY3;
}

bool _check_method_require(SPUContext *ctx, RuntimeConfig::SortMethod method) {
  bool pass = false;
  switch (method) {
    case RuntimeConfig::SORT_RADIX:
      pass = ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am") &&
             ctx->hasKernel("perm_ap") && ctx->hasKernel("inv_perm_am") &&
             ctx->hasKernel("inv_perm_ap");
      break;
    case RuntimeConfig::SORT_QUICK:
      // quick sort only requires small subsets of shuffle kernels, but need
      // rand_b kernel to avoid calling of a2b.
      pass = ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am") &&
             ctx->hasKernel("rand_b");
      break;
    case RuntimeConfig::SORT_NETWORK:
      // sort network is a general method which can be used for all MPC
      // protocols.
      pass = true;
      break;
    default:
      SPU_THROW("Should not reach here");
  }

  return pass;
}

RuntimeConfig::SortMethod select_sort_method(
    SPUContext *ctx, RuntimeConfig::SortMethod preferred_method) {
  SPU_ENFORCE(preferred_method != RuntimeConfig::SORT_DEFAULT);

  // if the preferred method is not supported, fall back to sorting network now.
  const RuntimeConfig::SortMethod fallback_method = RuntimeConfig::SORT_NETWORK;

  switch (preferred_method) {
    case RuntimeConfig::SORT_RADIX:
      if (internal::_check_method_require(ctx, RuntimeConfig::SORT_RADIX)) {
        return preferred_method;
      }
      break;

    case RuntimeConfig::SORT_QUICK:
      if (internal::_check_method_require(ctx, RuntimeConfig::SORT_QUICK)) {
        return preferred_method;
      }
      break;

    case RuntimeConfig::SORT_NETWORK:
      // always true now.
      if (internal::_check_method_require(ctx, RuntimeConfig::SORT_NETWORK)) {
        return preferred_method;
      }
      SPU_THROW("should not reach here");
      break;

    default:
      SPU_THROW("should not reach here");
  }

  return fallback_method;
}

std::vector<spu::Value> fallback_sort1d(SPUContext *ctx,
                                        absl::Span<spu::Value const> inputs,
                                        int64_t num_keys,
                                        SortDirection direction) {
  auto comp_fn = _get_cmp_func(ctx, num_keys, direction);
  Visibility vis = std::all_of(inputs.begin(), inputs.begin() + num_keys,
                               [](const spu::Value &v) { return v.isPublic(); })
                       ? VIS_PUBLIC
                       : VIS_SECRET;
  // currently, general sort1d only supports odd-even sorting network which is
  // an unstable sort method.
  auto ret = sort1d(ctx, inputs, comp_fn, vis, false);
  return ret;
}

void _hint_nbits(const Value &a, size_t nbits) {
  if (a.storage_type().isa<BShare>()) {
    const_cast<Type &>(a.storage_type()).as<BShare>()->setNbits(nbits);
  }
}

// generate inverse permutation
Index _inverse_index(const Index &p) {
  Index q(p.size());
  const auto n = static_cast<int64_t>(p.size());
  for (int64_t i = 0; i < n; ++i) {
    q[p[i]] = i;
  }
  return q;
}

spu::Value _2s(SPUContext *ctx, const Value &x) {
  if (x.isPublic()) {
    return _p2s(ctx, x);
  } else if (x.isPrivate()) {
    return _v2s(ctx, x);
  }
  return x;
}

Value _permute_1d(SPUContext *, const Value &x, const Index &indices) {
  SPU_ENFORCE(x.shape().size() == 1);
  return Value(x.data().linear_gather(indices), x.dtype());
}

Value _prefix_sum(SPUContext *ctx, const Value &x) {
  SPU_ENFORCE(x.shape().ndim() == 2U && x.shape()[0] == 1,
              "x should be 1-row matrix");
  auto x_v = hal::reshape(ctx, x, {x.numel()});
  auto ret = hal::associative_scan(hal::_add, ctx, x_v);
  return hal::reshape(ctx, ret, {1, x.numel()});
}

void _cmp_swap(SPUContext *ctx, const CompFn &comparator_body,
               absl::Span<spu::Value> values_to_sort, const Index &lhs_indices,
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
    const auto &fst = values[2 * i];
    const auto &sec = values[2 * i + 1];

    auto greater = select(ctx, predicate, fst, sec);
    auto less = sub(ctx, add(ctx, fst, sec), greater);

    values_to_sort[i].data().linear_scatter(greater.data(), lhs_indices);
    values_to_sort[i].data().linear_scatter(less.data(), rhs_indices);
  }
}

// Secure Odd-even mergesort
// Ref:
// https://hwlang.de/algorithmen/sortieren/networks/oemen.htm
std::vector<spu::Value> odd_even_merge_sort(
    SPUContext *ctx, const CompFn &comparator_body,
    absl::Span<spu::Value const> inputs) {
  // make a copy for inplace sort
  std::vector<spu::Value> ret;
  for (auto const &input : inputs) {
    spu::Value casted;
    if (!input.isSecret()) {
      // we can not linear_scatter a secret value to a public operand
      casted = _2s(ctx, input.clone()).setDtype(input.dtype());
    } else {
      casted = input.clone();
    }
    // we can not linear_scatter an ashare value to a bshare operand
    casted = _prefer_a(ctx, casted);
    ret.emplace_back(std::move(casted));
  }

  // sort by per network layer for memory optimizations, sorting N elements
  // needs log2(N) stages, and the i_th stage has i layers, which means the
  // same latency cost as BitonicSort but less _cmp_swap unit.
  const auto n = inputs.front().numel();
  for (int64_t max_gap_in_stage = 1; max_gap_in_stage < n;
       max_gap_in_stage += max_gap_in_stage) {
    for (int64_t step = max_gap_in_stage; step > 0; step /= 2) {
      // collect index pairs that can be computed parallelly.
      Index lhs_indices;
      Index rhs_indices;

      for (int64_t j = step % max_gap_in_stage; j + step < n;
           j += step + step) {
        for (int64_t i = 0; i < step; i++) {
          auto lhs_idx = i + j;
          auto rhs_idx = i + j + step;

          if (rhs_idx >= n) break;

          auto range = max_gap_in_stage + max_gap_in_stage;
          if (lhs_idx / range == rhs_idx / range) {
            lhs_indices.emplace_back(lhs_idx);
            rhs_indices.emplace_back(rhs_idx);
          }
        }
      }

      _cmp_swap(ctx, comparator_body, absl::MakeSpan(ret), lhs_indices,
                rhs_indices);
    }
  }

  return ret;
}

// Secure Odd-even merge
// Ref:
// https://hwlang.de/algorithmen/sortieren/networks/oemen.htm
std::vector<spu::Value> odd_even_merge(SPUContext *ctx,
                                       const CompFn &comparator_body,
                                       absl::Span<spu::Value const> inputs) {
  // make a copy for inplace merge
  std::vector<spu::Value> ret;
  for (auto const &input : inputs) {
    spu::Value casted;
    if (!input.isSecret()) {
      // we can not linear_scatter a secret value to a public operand
      casted = _2s(ctx, input.clone()).setDtype(input.dtype());
    } else {
      casted = input.clone();
    }
    // we can not linear_scatter an ashare value to a bshare operand
    casted = _prefer_a(ctx, casted);
    ret.emplace_back(std::move(casted));
  }

  // merge by per network layer for memory optimizations.
  const auto n = inputs.front().numel();
  int64_t max_gap_in_stage = n / 2;
  for (int64_t step = max_gap_in_stage; step > 0; step /= 2) {
    Index lhs_indices, rhs_indices;

    for (int64_t j = step % max_gap_in_stage; j + step < n; j += step + step) {
      for (int64_t i = 0; i < step; ++i) {
        auto lhs_idx = i + j;
        auto rhs_idx = i + j + step;
        if (rhs_idx >= n) break;

        auto range = max_gap_in_stage * 2;
        if (lhs_idx / range == rhs_idx / range) {
          lhs_indices.emplace_back(lhs_idx);
          rhs_indices.emplace_back(rhs_idx);
        }
      }
    }

    if (ctx->lctx()->Rank() == 0) {
      // 打印 lhs_indices 的大小，所有参与方都会打印
      std::cout << "Number of comparisons in each stage: " << lhs_indices.size()
                << std::endl;
    }
    _cmp_swap(ctx, comparator_body, absl::MakeSpan(ret), lhs_indices,
              rhs_indices);
  }
  return ret;
}

void Swap(absl::Span<spu::Value> arr, const Index &lhs_indices,
          const Index &rhs_indices) {
  if (lhs_indices.empty() ||
      (lhs_indices.size() == 1 && rhs_indices.size() == 1 &&
       lhs_indices[0] == rhs_indices[0])) {
    return;
  }

  const auto num_operands = arr.size();

  for (size_t i = 0; i < num_operands; ++i) {
    auto lhs_arr = arr[i].data().linear_gather(lhs_indices);
    auto rhs_arr = arr[i].data().linear_gather(rhs_indices);

    arr[i].data().linear_scatter(lhs_arr, rhs_indices);
    arr[i].data().linear_scatter(rhs_arr, lhs_indices);
  }
}

void CompSwapSingle(SPUContext *ctx, const CompFn &comparator_body,
                    absl::Span<spu::Value> arr, int64_t lo, int64_t hi,
                    const TopKConfig &config) {
  if (lo == hi) {
    return;
  }
  // const auto num_operands = arr.size();
  std::vector<Value> values;

  values.emplace_back(slice_scalar_at(ctx, arr[0], {lo}));
  values.emplace_back(slice_scalar_at(ctx, arr[0], {hi}));
  if (config.confusion) {
    values.emplace_back(slice_scalar_at(ctx, arr[1], {lo}));
    values.emplace_back(slice_scalar_at(ctx, arr[1], {hi}));
  }

  auto predicate = comparator_body(values);
  auto _predicate = getBooleanValue(ctx, hal::reveal(ctx, predicate));

  if (!_predicate) {
    Swap(arr, {lo}, {hi});
  }
}

void HandleSmallArray(SPUContext *ctx, const CompFn &comparator_body,
                      absl::Span<spu::Value> arr, int64_t lo, int64_t hi,
                      const TopKConfig &config) {
  if (hi == lo + 1) {
    CompSwapSingle(ctx, comparator_body, arr, lo, hi, config);
  }
}

std::vector<Value> _construct_cmp_values(
    SPUContext *ctx, const std::vector<std::pair<int64_t, int64_t>> &intervals,
    absl::Span<spu::Value const> arr, const int64_t quick_sort_thres,
    const int64_t num_keys) {
  int64_t lo;
  int64_t hi;
  int64_t left;
  int64_t right;

  std::vector<std::vector<Value>> cmp_values(2 * num_keys);
  for (auto &values : cmp_values) {
    values.reserve(intervals.size());
  }

  for (const auto &interval : intervals) {
    std::tie(lo, hi) = interval;

    if (hi - lo <= quick_sort_thres) {
      continue;
    }

    left = lo + 1;
    right = hi;

    for (int64_t i = 0; i < num_keys; i++) {
      // pivot
      cmp_values[2 * i].push_back(broadcast_to(
          ctx, slice_scalar_at(ctx, arr[i], {lo}), {right - left + 1}));
      // others
      cmp_values[2 * i + 1].push_back(slice(ctx, arr[i], {left}, {right + 1}));
    }
  }

  // no need to quick sort
  if (cmp_values[0].empty()) {
    return {};
  }

  std::vector<Value> ret;
  ret.reserve(2 * num_keys);

  for (int64_t i = 0; i < 2 * num_keys; i++) {
    ret.push_back(concatenate(ctx, cmp_values[i], 0));
  }

  return ret;
}

bool Partition(SPUContext *ctx, const int64_t num_keys,
               const CompFn &comparator_body, absl::Span<spu::Value> arr,
               std::vector<std::pair<int64_t, int64_t>> &intervals) {
  if (intervals.empty()) {
    return false;
  }

  int64_t quick_sort_thres = ctx->config().quick_sort_threshold;

  int64_t lo;  // left end of current interval
  int64_t hi;  // right end of current interval

  int64_t left;   // location of left pointer
  int64_t right;  // location of right pointer
  int64_t mid;    // location of pivot element after partition

  auto values =
      _construct_cmp_values(ctx, intervals, arr, quick_sort_thres, num_keys);

  if (values.empty()) {
    return false;
  }

  auto predicate = comparator_body(values);
  auto _predicate = dump_public_as<bool>(ctx, hal::reveal(ctx, predicate));

  Index lhs_indices;
  Index rhs_indices;
  Index pivot_indices;
  Index mid_indices;
  // save partition output, i.e. (lo, mid, hi), where mid is the location of
  // pivot after partition.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> pos;
  // save the intervals that do not need quick sort anymore.
  std::vector<std::pair<int64_t, int64_t>> pass_vec;

  int64_t length = 0;
  for (auto item : intervals) {
    std::tie(lo, hi) = item;

    if (hi - lo <= quick_sort_thres) {
      pass_vec.emplace_back(lo, hi);
      continue;
    }

    left = lo + 1;
    right = hi;

    auto offset = left;
    // use two pointer for partition
    for (;;) {
      while (right >= left && !_predicate[left - offset + length]) {
        left++;
      }
      while (right >= left && _predicate[right - offset + length]) {
        right--;
      }
      if (right < left) {
        break;
      }

      lhs_indices.emplace_back(left);
      rhs_indices.emplace_back(right);

      left++;
      right--;
    }
    length += (hi - lo);

    pivot_indices.emplace_back(lo);
    mid_indices.emplace_back(right);
    pos.emplace_back(lo, right, hi);
  }
  Swap(arr, lhs_indices, rhs_indices);
  // swap the pivot
  Swap(arr, pivot_indices, mid_indices);

  intervals.swap(pass_vec);
  intervals.reserve(2 * intervals.size());

  while (!pos.empty()) {
    std::tie(lo, mid, hi) = pos.back();
    pos.pop_back();
    if (lo < mid) {
      intervals.emplace_back(lo, mid - 1);
    }
    if (mid < hi) {
      intervals.emplace_back(mid + 1, hi);
    }
  }

  return true;
}

// this algorithm is mainly adopted from odd-even mergesort, but we can reveal
// the comparison because of shuffling
void mergesort(SPUContext *ctx, const CompFn &comparator_body,
               absl::Span<spu::Value> arr,
               std::vector<std::pair<int64_t, int64_t>> &intervals) {
  const auto N = arr.front().numel();
  int64_t logn = Log2Ceil(N);
  // max depth for odd-even merge network
  int64_t depth = ((logn + 1) * logn) / 2;

  std::vector<Index> lhs_indices(depth);
  std::vector<Index> rhs_indices(depth);

  int64_t lo;
  int64_t hi;
  for (auto item : intervals) {
    std::tie(lo, hi) = item;
    if (hi - lo <= 0) {
      continue;
    }

    int64_t n = hi - lo + 1;
    int64_t cnt = 0;
    for (int64_t max_gap_in_stage = 1; max_gap_in_stage < n;
         max_gap_in_stage += max_gap_in_stage) {
      for (int64_t step = max_gap_in_stage; step > 0; step /= 2) {
        for (int64_t j = step % max_gap_in_stage; j + step < n;
             j += step + step) {
          auto range = max_gap_in_stage + max_gap_in_stage;

          for (int64_t i = 0; i < step; i++) {
            auto lhs_idx = i + j;
            auto rhs_idx = i + j + step;

            if (rhs_idx >= n) {
              break;
            }

            if (lhs_idx / range == rhs_idx / range) {
              lhs_indices[cnt].emplace_back(lhs_idx + lo);
              rhs_indices[cnt].emplace_back(rhs_idx + lo);
            }
          }
        }
        cnt += 1;
      }
    }
  }

  size_t num_operands = arr.size();
  for (size_t i = 0; i < lhs_indices.size(); i++) {
    if (lhs_indices[i].empty()) {
      continue;
    }

    Index lhs_indice;
    Index rhs_indice;

    std::vector<spu::Value> values;
    values.reserve(2 * num_operands);

    for (size_t j = 0; j < num_operands; ++j) {
      values.emplace_back(arr[j].data().linear_gather(lhs_indices[i]),
                          arr[j].dtype());
      values.emplace_back(arr[j].data().linear_gather(rhs_indices[i]),
                          arr[j].dtype());
    }
    auto predicate = comparator_body(values);
    auto _predicate = dump_public_as<bool>(ctx, hal::reveal(ctx, predicate));
    for (size_t k = 0; k < _predicate.size(); k++) {
      if (!_predicate[k]) {
        lhs_indice.emplace_back(lhs_indices[i][k]);
        rhs_indice.emplace_back(rhs_indices[i][k]);
      }
    }
    Swap(arr, lhs_indice, rhs_indice);
  }
}

std::vector<spu::Value> QuickMergesort(SPUContext *ctx, const int64_t num_keys,
                                       const CompFn &quick_comp,
                                       const CompFn &merge_comp,
                                       absl::Span<spu::Value const> inputs) {
  // we do not need to copy or _2s here because of the secret shuffling.
  std::vector<spu::Value> ret(inputs.begin(), inputs.end());

  const auto n = inputs.front().numel();
  std::vector<std::pair<int64_t, int64_t>> intervals;
  intervals.emplace_back(0, n - 1);
  int64_t quicksort_num = 0;
  // set max depth to avoid infinite loop
  int64_t depth = 1000;
  bool need_quick_sort = true;

  while (!intervals.empty()) {
    need_quick_sort =
        Partition(ctx, num_keys, quick_comp, absl::MakeSpan(ret), intervals);
    quicksort_num += 1;

    if (!need_quick_sort || (quicksort_num == depth)) {
      break;
    }
  }

  if (intervals.empty()) {
    return ret;
  }

  mergesort(ctx, merge_comp, absl::MakeSpan(ret), intervals);

  return ret;
}

std::vector<spu::Value> PrepareSort(SPUContext *ctx,
                                    absl::Span<spu::Value const> inputs) {
  std::vector<spu::Value> inp;
  inp.reserve(inputs.size());

  auto rand_perm = _rand_perm_s(ctx, inputs.front().shape());
  // use a random permutation to break link of values, such that the following
  // comparison can be revealed without loss of information.
  for (const auto &input : inputs) {
    inp.emplace_back(std::move(
        _perm_ss(ctx, _2s(ctx, input), rand_perm).setDtype(input.dtype())));
  }

  return inp;
}

std::vector<spu::Value> quick_sort(SPUContext *ctx,
                                   absl::Span<spu::Value const> inputs,
                                   int64_t num_keys, SortDirection direction) {
  auto inp = PrepareSort(ctx, inputs);
  // quick sort will append extra random key
  auto quick_comp = _get_cmp_func(ctx, num_keys, direction, true);
  // in merge sort stage, only normal keys are used for comparison
  auto merge_comp = _get_cmp_func(ctx, num_keys, direction);
  auto ret = QuickMergesort(ctx, num_keys, quick_comp, merge_comp,
                            absl::MakeSpan(inp));
  return ret;
}

void TwoWayPartition(SPUContext *ctx, const CompFn &comparator_body,
                     absl::Span<spu::Value> arr, int64_t lo, int64_t hi,
                     const TopKConfig &config,
                     std::vector<std::pair<int64_t, int64_t>> &intervals) {
  // Just use first element as pivot, so left=lo+1
  auto left = lo + 1;
  auto right = hi;

  // collect and do comparison once
  // const auto num_operands = arr.size();
  std::vector<Value> values;
  // arr contains: value, random_value, index
  // values: pivot_value, rest_value, pivot_rand, rest_rand
  values.push_back(broadcast_to(ctx, slice_scalar_at(ctx, arr[0], {lo}),
                                {right - left + 1}));
  values.push_back(slice(ctx, arr[0], {left}, {right + 1}));
  if (config.confusion) {
    values.push_back(broadcast_to(ctx, slice_scalar_at(ctx, arr[1], {lo}),
                                  {right - left + 1}));
    values.push_back(slice(ctx, arr[1], {left}, {right + 1}));
  }

  auto predicate = comparator_body(values);
  auto _predicate = dump_public_as<bool>(ctx, hal::reveal(ctx, predicate));

  auto offset = left;
  Index lhs_indices;
  Index rhs_indices;

  // use two pointer for partition
  for (;;) {
    while (right >= left && !_predicate[left - offset]) {
      left++;
    }
    while (right >= left && _predicate[right - offset]) {
      right--;
    }
    if (right < left) {
      break;
    }

    lhs_indices.emplace_back(left);
    rhs_indices.emplace_back(right);

    left++;
    right--;
  }
  // do all non-overlaping swap
  Swap(arr, lhs_indices, rhs_indices);
  // swap the pivot
  Swap(arr, {lo}, {right});

  if (config.k_lo - 1 < right && right < config.k_hi - 1) {
    intervals.emplace_back(lo, right - 1);
    intervals.emplace_back(left, hi);
    return;
  }

  if (right >= config.k_hi - 1) {
    hi = right - 1;
  }
  if (right <= config.k_lo - 1) {
    lo = left;
  }
  intervals.emplace_back(lo, hi);
}

std::vector<spu::Value> QuickSelectTopk(SPUContext *ctx,
                                        const CompFn &comparator_body,
                                        absl::Span<spu::Value> input,
                                        const TopKConfig &config) {
  const auto n = input.front().numel();
  int64_t lo;
  int64_t hi;

  // save value and index
  std::vector<Value> out;

  // to support multiple ks, maintain all intervals to search
  std::vector<std::pair<int64_t, int64_t>> intervals;

  // first seach the whole interval
  intervals.emplace_back(0, n - 1);

  while (!intervals.empty()) {
    std::tie(lo, hi) = intervals.back();
    intervals.pop_back();

    if (hi <= lo + 1) {
      // exit loop when interval<=2
      HandleSmallArray(ctx, comparator_body, input, lo, hi, config);
    } else {
      TwoWayPartition(ctx, comparator_body, input, lo, hi, config, intervals);
    }
  }

  out.push_back(slice(ctx, input.front(), {0}, {config.k_hi}));
  if (!config.value_only) {
    out.push_back(slice(ctx, input.back(), {0}, {config.k_hi}));
  }
  return out;
}

std::vector<spu::Value> PrepareInput(SPUContext *ctx, const Value &input,
                                     const TopKConfig &config) {
  std::vector<spu::Value> inp;

  // shuffle with random permutation to break link of values
  auto rand_perm = _rand_perm_s(ctx, input.shape());
  inp.push_back(_perm_ss(ctx, input, rand_perm).setDtype(input.dtype()));

  // we concate random value to hide the data-dependant running pattern
  // for quick select;
  // consider an extreme case where all values are identical, two-way partition
  // will run very slowly. If running multiple times and finding that it
  // consistently takes a long time, it can be reasonably inferred that there is
  // a significant amount of duplicate data in the original dataset. However,
  // with the addition of randomness, we can essentially assume that all data
  // points are unique, which would lead to a more stable runtime.
  if (config.confusion) {
    inp.push_back(hal::random(ctx, Visibility::VIS_SECRET, DataType::DT_F64,
                              input.shape()));
  }

  if (!config.value_only) {
    auto dt =
        ctx->config().field == FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
    // shuffle index with the same permutation as values
    inp.push_back(
        _perm_ss(ctx, _p2s(ctx, hal::iota(ctx, dt, input.numel())), rand_perm)
            .setDtype(dt));
  }

  return inp;
}

// Ref: https://eprint.iacr.org/2019/695.pdf
// Algorithm 13 Optimized inverse application of a permutation
//
// The steps are as follows:
//   1) secure shuffle <perm> as <sp>
//   2) secure shuffle <x> as <sx>
//   3) reveal securely shuffled <sp> as m
//   4) inverse permute <sx> by m and return
std::pair<std::vector<spu::Value>, spu::Value> _opt_apply_inv_perm_ss(
    SPUContext *ctx, absl::Span<spu::Value const> x, const spu::Value &perm,
    const spu::Value &random_perm) {
  // 1. <SP> = secure shuffle <perm>
  auto sp = hal::_perm_ss(ctx, perm, random_perm);

  // 2. <SX> = secure shuffle <x>
  std::vector<spu::Value> sx;
  for (size_t i = 0; i < x.size(); ++i) {
    sx.emplace_back(hal::_perm_ss(ctx, x[i], random_perm));
  }

  // 3. M = reveal(<SP>)
  auto m = _s2p(ctx, sp);
  SPU_ENFORCE_EQ(m.shape().ndim(), 1U, "perm should be 1-d tensor");

  // 4. <T> = SP(<SX>)
  std::vector<spu::Value> v;

  for (size_t i = 0; i < sx.size(); ++i) {
    auto t = hal::_inv_perm_sp(ctx, sx[i], m);
    v.emplace_back(std::move(t));
  }

  return {v, m};
}

// Process two bit vectors in one loop
// Reference: https://eprint.iacr.org/2019/695.pdf (5.2 Optimizations)
//
// perm = _gen_inv_perm_by_bv(x, y)
//   input: bit vector x, bit vector y
//          bit vector y is more significant than x
//   output: shared inverse permutation
//
// We can generate inverse permutation by two bit vectors in one loop.
// It needs one extra mul op and 2 times memory to store intermediate data
// than _gen_inv_perm_by_bv. But the number of invocations of
// permutation-related protocols such as SecureInvPerm or Compose will be
// reduced to half.
//
// If we process three bit vectors in one loop, it needs at least four extra
// mul ops and 2^2 times data to store intermediate data. The number of
// invocations of permutation-related protocols such as SecureInvPerm or
// Compose will be reduced to 1/3. It's latency friendly but not bandwidth
// friendly.
//
// Example:
//   1) x = [0, 1], y = [1, 0]
//   2) rev_x = [1, 0], rev_y = [0, 1]
//   3) f0 = rev_x * rev_y = [0, 0]
//      f1 = x * rev_y = [0, 1]
//      f2 = rev_x * y = [1, 0]
//      f3 = x * y = [0, 0]
//      f =  [f0, f1, f2, f3] = [0, 0, 0, 1, 1, 0, 0, 0]
//   4) s[i] = s[i - 1] + f[i], s[0] = f[0]
//      s = [0, 0, 0, 1, 2, 2, 2, 2]
//   5) fs = f * s
//      fs = [0, 0, 0, 1, 2, 0, 0, 0]
//   6) split fs to four vector
//      fsv[0] = [0, 0]
//      fsv[1] = [0, 1]
//      fsv[2] = [2, 0]
//      fsv[3] = [0, 0]
//   7) r = fsv[0] + fsv[1] + fsv[2] + fsv[3]
//      r = [2, 1]
//   8) get res by sub r by one
//      res = [1, 0]
spu::Value _gen_inv_perm_by_bv(SPUContext *ctx, const spu::Value &x,
                               const spu::Value &y) {
  SPU_ENFORCE(x.shape() == y.shape(), "x and y should has the same shape");
  SPU_ENFORCE(x.shape().ndim() == 1, "x and y should be 1-d");
  const auto field = x.storage_type().as<Ring2k>()->field();

  const auto k1 = _constant(ctx, 1U, x.shape(), field);
  auto rev_x = _sub(ctx, k1, x);
  auto rev_y = _sub(ctx, k1, y);
  auto f0 = _mul(ctx, rev_x, rev_y);
  auto f1 = _sub(ctx, rev_y, f0);
  auto f2 = _sub(ctx, rev_x, f0);
  auto f3 = _sub(ctx, y, f2);

  const auto numel = x.numel();
  auto f = concatenate(ctx,
                       {unsqueeze(ctx, f0), unsqueeze(ctx, f1),
                        unsqueeze(ctx, f2), unsqueeze(ctx, f3)},
                       1);

  // calculate prefix sum
  auto ps = _prefix_sum(ctx, f);

  // mul f and s
  auto fs = _mul(ctx, f, ps);

  auto fs0 = slice(ctx, fs, {0, 0}, {1, numel}, {});
  auto fs1 = slice(ctx, fs, {0, numel}, {1, 2 * numel}, {});
  auto fs2 = slice(ctx, fs, {0, 2 * numel}, {1, 3 * numel}, {});
  auto fs3 = slice(ctx, fs, {0, 3 * numel}, {1, 4 * numel}, {});

  // calculate result
  auto s01 = _add(ctx, fs0, fs1);
  auto s23 = _add(ctx, fs2, fs3);
  auto r = _add(ctx, s01, s23);
  auto res = _sub(ctx, reshape(ctx, r, x.shape()), k1);

  return res;
}

// Generate perm by bit vector
//   input: bit vector generated by bit decomposition
//   output: shared inverse permutation
//
// Example:
//   1) x = [1, 0, 1, 0, 0]
//   2) rev_x = [0, 1, 0, 1, 1]
//   3) f = [rev_x, x]
//      f = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
//   4) s[i] = s[i - 1] + f[i], s[0] = f[0]
//      s = [0, 1, 1, 2, 3, 4, 4, 5, 5, 5]
//   5) fs = f * s
//      fs = [0, 1, 0, 2, 3, 4, 0, 5, 0, 0]
//   6) split fs to two vector
//      fsv[0] = [0, 1, 0, 2, 3]
//      fsv[1] = [4, 0, 5, 0, 0]
//   7) r = fsv[0] + fsv[1]
//      r = [4, 1, 5, 2, 3]
//   8) get res by sub r by one
//      res = [3, 0, 4, 1, 2]
spu::Value _gen_inv_perm_by_bv(SPUContext *ctx, const spu::Value &x) {
  SPU_ENFORCE(x.shape().ndim() == 1, "x should be 1-d");
  const auto field = x.storage_type().as<Ring2k>()->field();

  const auto k1 = _constant(ctx, 1U, x.shape(), field);
  auto rev_x = _sub(ctx, k1, x);

  const auto numel = x.numel();
  auto f = concatenate(ctx, {unsqueeze(ctx, rev_x), unsqueeze(ctx, x)}, 1);

  // calculate prefix sum
  auto ps = _prefix_sum(ctx, f);

  // mul f and s
  auto fs = _mul(ctx, f, ps);

  auto fs0 = slice(ctx, fs, {0, 0}, {1, numel}, {});
  auto fs1 = slice(ctx, fs, {0, numel}, {1, 2 * numel}, {});

  // calculate result
  auto r = _add(ctx, fs0, fs1);
  auto res = _sub(ctx, reshape(ctx, r, x.shape()), k1);
  return res;
}

// Ref: https://eprint.iacr.org/2019/695.pdf
// Algorithm 14: Optimized composition of two permutations
//
// Compose is actually a special case of apply_perm where both inputs are
// permutations.
//
// The input is a shared inverse permutation <perm>, a public permutation
// shuffled_perm generated by _opt_apply_inv_perm_ss, and a secret permutation
// share random_perm for secure unshuffle.
//
// The steps are as follows:
//   1) permute <perm> by shuffled_perm as <sm>
//   2) secure unshuffle <sm> and return results
spu::Value _opt_apply_perm_ss(SPUContext *ctx, const spu::Value &perm,
                              const spu::Value &shuffled_perm,
                              const spu::Value &random_perm) {
  auto sm = hal::_perm_sp(ctx, perm, shuffled_perm);
  // this is actually shuffle
  auto res = hal::_inv_perm_ss(ctx, sm, random_perm);
  return res;
}

std::vector<spu::Value> _bit_decompose(SPUContext *ctx, const spu::Value &x,
                                       int64_t valid_bits) {
  auto x_bshare = _prefer_b_bits(ctx, x, valid_bits);
  size_t nbits = valid_bits != -1
                     ? static_cast<size_t>(valid_bits)
                     : x_bshare.storage_type().as<BShare>()->nbits();
  _hint_nbits(x_bshare, nbits);
  if (ctx->hasKernel("b2a_disassemble")) {
    const auto perm_field = internal::_get_field_from_n(ctx, x.numel());
    auto ret = dynDispatch<std::vector<spu::Value>>(ctx, "b2a_disassemble",
                                                    x_bshare, perm_field);
    return ret;
  }

  const auto k1 = _constant(ctx, 1U, x.shape());
  std::vector<spu::Value> rets_b;
  rets_b.reserve(nbits);

  for (size_t bit = 0; bit < nbits; ++bit) {
    auto x_bshare_shift =
        right_shift_logical(ctx, x_bshare, {static_cast<int64_t>(bit)});
    rets_b.push_back(_and(ctx, x_bshare_shift, k1));
  }

  std::vector<spu::Value> rets_a;
  vmap(rets_b.begin(), rets_b.end(), std::back_inserter(rets_a),
       [&](const Value &x) { return _prefer_a(ctx, x); });
  return rets_a;
}

// Generate vector of bit decomposition of sorting keys
std::vector<spu::Value> _gen_bv_vector(SPUContext *ctx,
                                       absl::Span<spu::Value const> keys,
                                       SortDirection direction,
                                       int64_t valid_bits) {
  std::vector<spu::Value> ret;
  const auto perm_field = internal::_get_field_from_n(ctx, keys[0].numel());
  const auto k1 = _constant(ctx, 1U, keys[0].shape(), perm_field);

  // keys[0] is the most significant key
  for (size_t i = keys.size(); i > 0; --i) {
    const auto t = _bit_decompose(ctx, keys[i - 1], valid_bits);

    SPU_ENFORCE(t.size() > 0);
    for (size_t j = 0; j < t.size() - 1; j++) {
      // Radix sort is a stable sorting algorithm for the ascending order, if
      // we flip the bit, then we can get the descending order for stable sort
      if (direction == SortDirection::Descending) {
        ret.emplace_back(_sub(ctx, k1, t[j]));
      } else {
        ret.emplace_back(t[j]);
      }
    }
    // The sign bit is opposite
    if (direction == SortDirection::Descending) {
      ret.emplace_back(t.back());
    } else {
      ret.emplace_back(_sub(ctx, k1, t.back()));
    }
  }
  return ret;
}

// Generate shared inverse permutation by key
spu::Value _gen_inv_perm_s(SPUContext *ctx, absl::Span<spu::Value const> keys,
                           SortDirection direction, int64_t valid_bits) {
  const auto perm_field = internal::_get_field_from_n(ctx, keys[0].numel());

  // 1. generate bit decomposition vector of keys
  std::vector<spu::Value> bv = _gen_bv_vector(ctx, keys, direction, valid_bits);
  size_t bv_size = bv.size();
  SPU_ENFORCE_GT(bv_size, 0U);

  // quick path for one or two valid bits
  if (bv_size == 1) {
    return _gen_inv_perm_by_bv(ctx, bv[0]);
  }
  if (bv_size == 2) {
    return _gen_inv_perm_by_bv(ctx, bv[0], bv[1]);
  }

  // 2. generate the first two-bits permutation for initialization
  auto shared_perm = _gen_inv_perm_by_bv(ctx, bv[0], bv[1]);

  // 3. generate shared inverse permutation by bit vector and process
  size_t bv_idx = 2;
  for (; bv_idx < bv_size - 1; bv_idx += 2) {
    // generate random permutation for shuffle
    auto random_perm = hal::_rand_perm_s(ctx, keys[0].shape(), perm_field);
    auto [shuffled_bv, shuffled_perm] = _opt_apply_inv_perm_ss(
        ctx, std::vector<spu::Value>{bv[bv_idx], bv[bv_idx + 1]}, shared_perm,
        random_perm);
    auto perm = _gen_inv_perm_by_bv(ctx, shuffled_bv[0], shuffled_bv[1]);
    shared_perm = _opt_apply_perm_ss(ctx, perm, shuffled_perm, random_perm);
  }

  if (bv_idx == bv_size - 1) {
    // generate random permutation for shuffle
    auto random_perm = hal::_rand_perm_s(ctx, keys[0].shape(), perm_field);
    auto [shuffled_bv, shuffled_perm] = _opt_apply_inv_perm_ss(
        ctx, std::vector<spu::Value>{bv[bv_idx]}, shared_perm, random_perm);
    auto perm = _gen_inv_perm_by_bv(ctx, shuffled_bv[0]);
    shared_perm = _opt_apply_perm_ss(ctx, perm, shuffled_perm, random_perm);
  }

  return shared_perm;
}

spu::Value _gen_inv_perm_s(SPUContext *ctx, const spu::Value &key,
                           bool is_ascending, int64_t valid_bits) {
  std::vector<spu::Value> keys{key};
  auto direction =
      is_ascending ? SortDirection::Ascending : SortDirection::Descending;
  auto ret = _gen_inv_perm_s(ctx, keys, direction, valid_bits);
  return ret;
}

// Apply inverse permutation on each tensor of x by a shared inverse
// permutation <perm>
std::vector<spu::Value> _apply_inv_perm_ss(SPUContext *ctx,
                                           absl::Span<spu::Value const> x,
                                           const spu::Value &perm) {
  const auto perm_field = internal::_get_field_from_n(ctx, perm.numel());
  // 1. <SP> = secure shuffle <perm>
  auto shuffle_perm = hal::_rand_perm_s(ctx, x[0].shape(), perm_field);
  auto sp = hal::_perm_ss(ctx, hal::_ring_cast_down(ctx, perm, perm_field),
                          shuffle_perm);

  // 2. <SX> = secure shuffle <x>
  std::vector<spu::Value> sx;
  for (size_t i = 0; i < x.size(); ++i) {
    sx.emplace_back(hal::_perm_ss(ctx, x[i], shuffle_perm));
  }

  // 3. M = reveal(<SP>)
  auto m = _s2p(ctx, sp);
  SPU_ENFORCE_EQ(m.shape().ndim(), 1U, "perm should be 1-d tensor");

  // 4. <T> = SP(<SX>)
  std::vector<spu::Value> v;
  for (size_t i = 0; i < sx.size(); ++i) {
    auto t = hal::_inv_perm_sp(ctx, sx[i], m);
    v.emplace_back(std::move(t));
  }

  return v;
}

spu::Value _apply_inv_perm_ss(SPUContext *ctx, const spu::Value &x,
                              const spu::Value &perm) {
  std::vector<spu::Value> inputs{x};
  auto ret = _apply_inv_perm_ss(ctx, inputs, perm);
  return std::move(ret[0]);
}

// Ref: https://eprint.iacr.org/2019/695.pdf
// Algorithm 5: Composition of two share-vector permutations
//
// Compose is actually a special case of apply_perm where both inputs are
// permutations. So to be more general, we use the name _apply_perm_ss
// rather than _compose_ss here
std::vector<spu::Value> _apply_perm_ss(SPUContext *ctx,
                                       absl::Span<spu::Value const> x,
                                       const Value &perm) {
  const auto perm_field = internal::_get_field_from_n(ctx, perm.numel());
  // 1. <SP> = secure shuffle <perm>
  auto shuffle_perm = hal::_rand_perm_s(ctx, x[0].shape(), perm_field);
  auto sp = hal::_perm_ss(ctx, hal::_ring_cast_down(ctx, perm, perm_field),
                          shuffle_perm);

  // 2. M = reveal(<SP>)
  auto m = _s2p(ctx, sp);
  SPU_ENFORCE_EQ(m.shape().ndim(), 1U, "perm should be 1-d tensor");

  // 3. sx = apply_perm(x,m)
  std::vector<spu::Value> sx;
  sx.reserve(x.size());
  for (const auto &item : x) {
    sx.emplace_back(hal::_perm_sp(ctx, item, m));
  }

  // 4. ret = unshuffle(<sx>)
  std::vector<spu::Value> ret;
  ret.reserve(x.size());
  for (const auto &item : sx) {
    ret.emplace_back(hal::_inv_perm_ss(ctx, item, shuffle_perm));
  }

  return ret;
}

spu::Value _apply_perm_ss(SPUContext *ctx, const Value &x, const Value &perm) {
  std::vector<spu::Value> inputs{x};
  auto ret = _apply_perm_ss(ctx, inputs, perm);
  return std::move(ret[0]);
}

// Find mergeable keys from keys. Consecutive public/private(belong to one
// owner) keys can be merged. Assume there are six keys, i.e., public_key0,
// bob_key0, bob_key1, alice_key0, alice_key1, secret_key0. We can merge the
// six keys into bob_new_key, alice_new_key, secret_key0 for the following
// sorting. This function will return a vector of indices [3,5,6] which means
// key[0,3), key[3,5), and key[5,6) can be merged.
std::vector<size_t> _find_mergeable_keys(SPUContext *ctx,
                                         absl::Span<spu::Value const> keys) {
  std::vector<size_t> split_indices;
  split_indices.push_back(keys.size());
  auto idx = keys.size() - 1;
  int64_t pre_owner = keys[idx].isPrivate() ? _get_owner(keys[idx]) : -1;

  while (idx > 0) {
    idx--;
    const auto &pre_key = keys[idx + 1];
    const auto &cur_key = keys[idx];
    // secret key cannot be merged
    if (pre_key.isSecret()) {
      split_indices.push_back(idx + 1);
    } else {
      // if current key are not belong to different owners of previous
      // keys, they cannot be merged
      if (cur_key.isPublic()) {
        continue;
      } else if (cur_key.isPrivate()) {
        if (pre_owner == -1 || _get_owner(cur_key) == pre_owner) {
          pre_owner = _get_owner(cur_key);
          continue;
        } else {
          split_indices.push_back(idx + 1);
        }
      } else {
        split_indices.push_back(idx + 1);
      }
      pre_owner = cur_key.isPrivate() ? _get_owner(cur_key) : -1;
    }
  }
  std::reverse(split_indices.begin(), split_indices.end());
  return split_indices;
}

// Given a 1-d array input, generate its inverse permutation
spu::Value _gen_inv_perm(SPUContext *ctx, const Value &in, bool is_ascending,
                         int64_t valid_bits = -1) {
  SPU_TRACE_HAL_DISP(ctx, in, is_ascending, valid_bits);
  if (in.isPublic()) {
    return _gen_inv_perm_p(ctx, in, is_ascending);
  } else if (in.isSecret()) {
    return _gen_inv_perm_s(ctx, in, is_ascending, valid_bits);
  } else if (in.isPrivate()) {
    return _gen_inv_perm_v(ctx, in, is_ascending);
  } else {
    SPU_THROW("should not be here");
  }
}

spu::Value _apply_inv_perm_sv(SPUContext *ctx, const Value &in,
                              const Value &perm) {
  const auto perm_field = internal::_get_field_from_n(ctx, perm.numel());
  if (ctx->hasKernel("inv_perm_av")) {
    return hal::_inv_perm_sv(ctx, in,
                             hal::_ring_cast_down(ctx, perm, perm_field));
  } else {
    return _apply_inv_perm_ss(
        ctx, in, _v2s(ctx, hal::_ring_cast_down(ctx, perm, perm_field)));
  }
}

std::vector<Value> _apply_inv_perm_sv(SPUContext *ctx,
                                      absl::Span<Value const> inputs,
                                      const Value &perm) {
  if (ctx->hasKernel("inv_perm_av")) {
    std::vector<spu::Value> ret;
    ret.reserve(inputs.size());
    for (const auto &input : inputs) {
      ret.emplace_back(
          _apply_inv_perm_sv(ctx, input, perm).setDtype(input.dtype()));
    }
    return ret;
  } else {
    return _apply_inv_perm_ss(ctx, inputs, _2s(ctx, perm));
  }
}

#define MAP_APPLY_PERM_OP(NAME)                                             \
  spu::Value _apply##NAME(SPUContext *ctx, const Value &in,                 \
                          const Value &perm) {                              \
    const auto perm_field = internal::_get_field_from_n(ctx, perm.numel()); \
    return hal::NAME(ctx, in, hal::_ring_cast_down(ctx, perm, perm_field)); \
  }                                                                         \
                                                                            \
  std::vector<Value> _apply##NAME(                                          \
      SPUContext *ctx, absl::Span<Value const> inputs, const Value &perm) { \
    std::vector<Value> ret;                                                 \
    ret.reserve(inputs.size());                                             \
    for (const auto &input : inputs) {                                      \
      ret.emplace_back(                                                     \
          _apply##NAME(ctx, input, perm).setDtype(input.dtype()));          \
    }                                                                       \
    return ret;                                                             \
  }

MAP_APPLY_PERM_OP(_perm_pp);
MAP_APPLY_PERM_OP(_perm_vv);
MAP_APPLY_PERM_OP(_perm_sp);
MAP_APPLY_PERM_OP(_inv_perm_pp);
MAP_APPLY_PERM_OP(_inv_perm_vv);
MAP_APPLY_PERM_OP(_inv_perm_sp);

#define MAP_VEC_CONVERT_OP(NAME)                                             \
  std::vector<Value> NAME(SPUContext *ctx, absl::Span<Value const> inputs) { \
    std::vector<Value> ret;                                                  \
    ret.reserve(inputs.size());                                              \
    for (const auto &input : inputs) {                                       \
      ret.emplace_back(hal::NAME(ctx, input).setDtype(input.dtype()));       \
    }                                                                        \
    return ret;                                                              \
  }

MAP_VEC_CONVERT_OP(_p2s);
MAP_VEC_CONVERT_OP(_v2s);

#undef MAP_VEC_CONVERT_OP

std::vector<Value> _p2v(SPUContext *ctx, absl::Span<Value const> inputs,
                        int owner) {
  std::vector<Value> ret;
  ret.reserve(inputs.size());
  for (const auto &input : inputs) {
    ret.emplace_back(hal::_p2v(ctx, input, owner).setDtype(input.dtype()));
  }
  return ret;
}

// Given a permutation, apply (inverse) permutation on a 1-d array input
#define MAP_PERM_OP(NAME)                                                   \
  std::vector<Value> NAME(SPUContext *ctx, absl::Span<Value const> in,      \
                          const Value &perm) {                              \
    SPU_ENFORCE(!in.empty(), "Inputs should not be empty");                 \
    SPU_ENFORCE(std::all_of(in.begin(), in.end(),                           \
                            [&in](const spu::Value &v) {                    \
                              return v.vtype() == in[0].vtype();            \
                            }),                                             \
                "Inputs visibility mismatched");                            \
    if (in[0].isPrivate()) {                                                \
      SPU_ENFORCE(std::all_of(in.begin(), in.end(),                         \
                              [&in](const spu::Value &v) {                  \
                                return internal::_has_same_owner(v, in[0]); \
                              }),                                           \
                  "Inputs owner mismatched");                               \
    }                                                                       \
    SPU_TRACE_HAL_DISP(ctx, in[0], perm);                                   \
    if (in[0].isPublic() && perm.isPublic()) { /*PP*/                       \
      return NAME##_pp(ctx, in, perm);                                      \
    } else if (in[0].isPublic() && perm.isSecret()) { /*PS*/                \
      return NAME##_ss(ctx, _p2s(ctx, in), perm);                           \
    } else if (in[0].isPublic() && perm.isPrivate()) { /*PV*/               \
      return NAME##_vv(ctx, _p2v(ctx, in, _get_owner(perm)), perm);         \
    } else if (in[0].isPrivate() && perm.isPrivate()) { /*VV*/              \
      if (_has_same_owner(in[0], perm)) {                                   \
        return NAME##_vv(ctx, in, perm);                                    \
      } else {                                                              \
        return NAME##_sv(ctx, _v2s(ctx, in), perm);                         \
      }                                                                     \
    } else if (in[0].isPrivate() && perm.isPublic()) { /*VP*/               \
      return NAME##_vv(ctx, in, hal::_p2v(ctx, perm, _get_owner(in[0])));   \
    } else if (in[0].isPrivate() && perm.isSecret()) { /*VS*/               \
      return NAME##_ss(ctx, _v2s(ctx, in), perm);                           \
    } else if (in[0].isSecret() && perm.isSecret()) { /*SS*/                \
      return NAME##_ss(ctx, in, perm);                                      \
    } else if (in[0].isSecret() && perm.isPublic()) { /*SP*/                \
      return NAME##_sp(ctx, in, perm);                                      \
    } else if (in[0].isSecret() && perm.isPrivate()) { /*SV*/               \
      return NAME##_sv(ctx, in, perm);                                      \
    } else {                                                                \
      SPU_THROW("should not be here");                                      \
    }                                                                       \
  }

// Inverse permute 1-D array x with a permutation perm
// ret[perm[i]] = x[i]
MAP_PERM_OP(_apply_inv_perm)

spu::Value _apply_inv_perm(SPUContext *ctx, const spu::Value &x,
                           const spu::Value &perm) {
  std::vector<spu::Value> inputs{x};
  auto ret = _apply_inv_perm(ctx, inputs, perm);
  return std::move(ret[0]);
}

namespace {
DataType _get_dtype_from_n(int64_t n) {
  if (n <= (1LL << 8)) {
    return spu::DT_I8;
  } else if (n <= (1LL << 16)) {
    return spu::DT_I16;
  } else if (n <= (1LL << 32)) {
    return spu::DT_I32;
  } else {
    return spu::DT_I64;
  }
}
}  // namespace

// Given a permutation, generate its inverse permutation
// ret[perm[i]] = i
spu::Value _inverse(SPUContext *ctx, const Value &perm) {
  const auto perm_field = internal::_get_field_from_n(ctx, perm.numel());
  const auto running_field = perm.storage_type().as<Ring2k>()->field();
  const auto dt = _get_dtype_from_n(perm.numel());

  auto used_perm = perm;
  if (perm_field != running_field && perm_field != FT_INVALID) {
    SPU_ENFORCE(SizeOf(perm_field) <= SizeOf(running_field),
                "cannot convert field {} to {}", running_field, perm_field);
    used_perm = _ring_cast_down(ctx, perm, perm_field);
  }

  auto iota_perm = iota(ctx, dt, perm.numel(), perm_field);
  return _apply_inv_perm(ctx, iota_perm, used_perm);
}

spu::Value _apply_perm_sv(SPUContext *ctx, const Value &in, const Value &perm) {
  if (ctx->hasKernel("inv_perm_av")) {
    return hal::_inv_perm_sv(ctx, in, _inverse(ctx, perm));
  } else {
    return _apply_inv_perm_ss(ctx, in, hal::_v2s(ctx, _inverse(ctx, perm)));
  }
}

std::vector<Value> _apply_perm_sv(SPUContext *ctx,
                                  absl::Span<Value const> inputs,
                                  const Value &perm) {
  return _apply_inv_perm_sv(ctx, inputs, _inverse(ctx, perm));
}

// Permute 1-D array x with a permutation perm
// ret[i] = x[perm[i]]
MAP_PERM_OP(_apply_perm)

spu::Value _apply_perm(SPUContext *ctx, const spu::Value &x,
                       const spu::Value &perm) {
  std::vector<spu::Value> inputs{x};
  auto ret = _apply_perm(ctx, inputs, perm);
  return std::move(ret[0]);
}

// Compose two permutations into one permutation
// If we have two permutations x and y, we want to get a permutation z from x
// and y that apply_inv_perm(in, z) = apply_inv_perm(apply_inv_perm(in, x), y)
spu::Value _compose_perm(SPUContext *ctx, const Value &x, const Value &y) {
  return _apply_perm(ctx, y, x);
}

#undef MAP_PERM_OP

spu::Value _merge_keys(SPUContext *ctx, absl::Span<Value const> inputs,
                       bool is_ascending) {
  if (inputs[0].isPublic()) {
    SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                            [](const spu::Value &v) { return v.isPublic(); }),
                "keys should be all public");
    return _merge_keys_p(ctx, inputs, is_ascending);
  } else if (inputs[0].isPrivate()) {
    SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                            [&inputs](const spu::Value &v) {
                              return v.isPrivate() &&
                                     _has_same_owner(v, inputs[0]);
                            }),
                "keys should have a same owner");
    return _merge_keys_v(ctx, inputs, is_ascending);
  } else if (inputs[0].isSecret()) {
    SPU_THROW("merge secret permutation is currently not supported");
  } else {
    SPU_THROW("should not be here");
  }
}

spu::Value _merge_pub_pri_keys(SPUContext *ctx,
                               absl::Span<spu::Value const> keys,
                               bool is_ascending) {
  SPU_ENFORCE(std::all_of(keys.begin(), keys.end(),
                          [](const spu::Value &v) { return !v.isSecret(); }),
              "secret keys should not be here");
  SPU_ENFORCE_GE(keys.size(), 1U, "there are at least 1 key to merge");
  const auto &pre_key = keys.back();

  auto inv_perm = _gen_inv_perm(ctx, pre_key, is_ascending);

  for (int64_t i = keys.size() - 2; i >= 0; --i) {
    const auto &cur_key = keys[i];
    auto cur_key_hat = _apply_inv_perm(ctx, cur_key, inv_perm);
    auto cur_inv_perm = _gen_inv_perm(ctx, cur_key_hat, is_ascending);
    inv_perm = _compose_perm(ctx, inv_perm, cur_inv_perm);
  }
  auto dt = ctx->config().field == FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
  std::vector<spu::Value> permed_keys;
  for (const auto &key : keys) {
    permed_keys.emplace_back(_apply_inv_perm(ctx, key, inv_perm));
  }
  auto merged_key = _merge_keys(ctx, permed_keys, is_ascending).setDtype(dt);
  return _apply_perm(ctx, merged_key, inv_perm);
}

// Merge consecutive private/public keys
std::vector<spu::Value> _merge_sorting_keys(SPUContext *ctx,
                                            absl::Span<spu::Value const> keys,
                                            bool is_ascending) {
  auto merge_pos = _find_mergeable_keys(ctx, keys);
  SPU_ENFORCE_GT(merge_pos.size(), 0U, "there is at least 1 key after merging");
  std::vector<spu::Value> new_keys;
  size_t beg_idx = 0;
  for (size_t end_idx : merge_pos) {
    // for a single private/public, merge the key can use valid_bits
    // optimization
    if (end_idx - beg_idx == 1 && keys[beg_idx].isSecret()) {
      new_keys.push_back(keys[beg_idx]);
    } else {
      auto merged_key = _merge_pub_pri_keys(
          ctx, keys.subspan(beg_idx, end_idx - beg_idx), is_ascending);
      new_keys.push_back(std::move(merged_key));
    }
    beg_idx = end_idx;
  }
  return new_keys;
}

// Generate an inverse permutation vector according to sorting keys. The
// permutation vector should be secret or private (if enabled) but cannot be
// public as we have already process sorting with public keys outside of
// radix sort.
spu::Value gen_inv_perm(SPUContext *ctx, absl::Span<spu::Value const> inputs,
                        SortDirection direction, int64_t num_keys,
                        int64_t valid_bits) {
  // merge consecutive private/public keys
  auto keys = inputs.subspan(0, num_keys);
  if (std::all_of(keys.begin(), keys.end(),
                  [](const spu::Value &v) { return v.isSecret(); })) {
    auto perm = _gen_inv_perm_s(ctx, keys, direction, valid_bits);
    return perm;
  }
  bool is_ascending = direction == SortDirection::Ascending;
  auto merged_keys = _merge_sorting_keys(ctx, keys, is_ascending);

  // generate inverse permutation
  const auto &pre_key = merged_keys.back();
  auto inv_perm = _gen_inv_perm(ctx, pre_key, is_ascending, valid_bits);
  for (int64_t i = merged_keys.size() - 2; i >= 0; --i) {
    const auto &cur_key = merged_keys[i];
    auto cur_key_hat = _apply_inv_perm(ctx, cur_key, inv_perm);
    // TODO: +2 to +1 if unsigned. (Log2Ceil may be better? -1 more?)
    auto real_valid_bits =
        cur_key.isSecret() ? valid_bits : Log2Floor(cur_key.numel()) + 2;
    auto cur_inv_perm =
        _gen_inv_perm(ctx, cur_key_hat, is_ascending, real_valid_bits);
    inv_perm = _compose_perm(ctx, inv_perm, cur_inv_perm);
  }

  return inv_perm;
}

std::vector<spu::Value> apply_inv_perm(SPUContext *ctx,
                                       absl::Span<spu::Value const> inputs,
                                       const spu::Value &perm) {
  if (perm.isSecret()) {
    std::vector<spu::Value> inputs_s;
    for (const auto &input : inputs) {
      inputs_s.emplace_back(_2s(ctx, input).setDtype(input.dtype()));
    }
    return _apply_inv_perm_ss(ctx, inputs_s, perm);
  } else if (perm.isPrivate()) {
    if (ctx->hasKernel("inv_perm_av")) {
      std::vector<spu::Value> ret;
      for (const auto &input : inputs) {
        ret.emplace_back(
            _apply_inv_perm(ctx, input, perm).setDtype(input.dtype()));
      }
      return ret;
    } else {
      std::vector<spu::Value> inputs_s;
      for (const auto &input : inputs) {
        inputs_s.emplace_back(_2s(ctx, input).setDtype(input.dtype()));
      }
      return _apply_inv_perm_ss(ctx, inputs_s, _2s(ctx, perm));
    }
  } else {
    SPU_THROW("Should not be here");
  }
}

// Secure Radix Sort
// Ref:
//  https://eprint.iacr.org/2019/695.pdf
//
// Each input is a 1-d tensor, inputs[0, num_keys) are the keys, and sort
// inputs according to keys
std::vector<spu::Value> radix_sort(SPUContext *ctx,
                                   absl::Span<spu::Value const> inputs,
                                   SortDirection direction, int64_t num_keys,
                                   int64_t valid_bits) {
  auto perm = gen_inv_perm(ctx, inputs, direction, num_keys, valid_bits);
  auto res = apply_inv_perm(ctx, inputs, perm);
  return res;
}

}  // namespace internal

std::vector<spu::Value> merge1d(SPUContext *ctx,
                                absl::Span<spu::Value const> inputs,
                                const CompFn &cmp,
                                Visibility comparator_ret_vis, bool is_stable) {
  // sanity check.
  SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
  SPU_ENFORCE(inputs[0].shape().ndim() == 1,
              "Inputs should be 1-d but actually have {} dimensions",
              inputs[0].shape().ndim());
  SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                          [&inputs](const spu::Value &v) {
                            return v.shape() == inputs[0].shape();
                          }),
              "Inputs shape mismatched");

  std::vector<spu::Value> ret;
  if (comparator_ret_vis == VIS_SECRET) {
    SPU_ENFORCE(!is_stable,
                "Stable sort is unsupported if comparator return is secret.");

    ret = internal::odd_even_merge(ctx, cmp, inputs);
  } else {
    SPU_THROW("Should not reach here");
  }

  return ret;
}

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
                          [&inputs](const spu::Value &v) {
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
      for (const auto &input : inputs) {
        values.push_back(hal::slice(ctx, input, {a}, {a + 1}));
        values.push_back(hal::slice(ctx, input, {b}, {b + 1}));
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
    for (const auto &input : inputs) {
      ret.push_back(internal::_permute_1d(ctx, input, indices_to_sort));
    }
  } else if (comparator_ret_vis == VIS_SECRET) {
    SPU_ENFORCE(!is_stable,
                "Stable sort is unsupported if comparator return is secret.");

    ret = internal::odd_even_merge_sort(ctx, cmp, inputs);
  } else {
    SPU_THROW("Should not reach here");
  }

  return ret;
}

std::vector<spu::Value> simple_sort1d(SPUContext *ctx,
                                      absl::Span<spu::Value const> inputs,
                                      SortDirection direction, int64_t num_keys,
                                      int64_t valid_bits) {
  // sanity check.
  SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
  SPU_ENFORCE(inputs[0].shape().ndim() == 1,
              "Inputs should be 1-d but actually have {} dimensions",
              inputs[0].shape().ndim());
  SPU_ENFORCE(std::all_of(inputs.begin(), inputs.end(),
                          [&inputs](const spu::Value &v) {
                            return v.shape() == inputs[0].shape();
                          }),
              "Inputs shape mismatched");
  SPU_ENFORCE(num_keys > 0 && num_keys <= static_cast<int64_t>(inputs.size()),
              "num_keys {} is not valid", num_keys);

  std::vector<spu::Value> ret;
  const auto sort_method = ctx->config().sort_method;

  // There are multiple sort methods supported by SPU, we will try to seek the
  // best method in the following order if the user does not specify the method
  // manually.
  //   1. If all keys are Public, then fallback to the plaintext sort.
  //   2. Else, sequentially check if it supports radix sort or quick sort. If a
  //   match is found, execute the corresponding algorithm; otherwise, the
  //   default sorting network algorithm will be executed.
  //
  // Some takeaways about the above algorithm:
  //   1. Radix sort is currently the only STABLE sorting algorithm, so we
  //   choose it as the highest priority algorithm (as long as it is supported
  //   by the underlying MPC protocol).
  //   2. It's worth to know that quick sort is indeed faster than radix
  //   sort when the field is FM64 or FM128 (When in FM32, radix sort is always
  //   faster).
  //   3. However, radix sort can be significantly accelerated if you set
  //   the valid_bits when you know exactly the ranges of the keys.
  //   4. Radix sort and quick sort are more friendly to multiple payloads but
  //   not to multiple keys. Increasing one payload only adds one secret
  //   shuffle; however, for n additional keys, the communication/time can be
  //   roughly considered to multiply by n.
  //   5. Quick sort is more adaptable to the expansion of the ring. When the
  //   ring size doubles, the communication volume of quick sort nearly doubles,
  //   and the number of rounds increases (poly) logarithmically. In contrast,
  //   when the ring size doubles in radix sort, the communication （roughly）
  //   quadruples and the number of rounds doubles.
  //   6. The above conclusions regarding performance apply only to
  //   the cases of SECRET input and SECRET permutation. In reality, only radix
  //   sort has implemented a complete mechanism for selecting the best
  //   implementation based on visibility. The other implementations will use
  //   local computation only when all keys are public; in other cases, they
  //   will revert to the scenarios of SECRET input and SECRET permutation.
  //

  // if all keys are public, fallback to plaintext sort.
  if (std::all_of(inputs.begin(), inputs.begin() + num_keys,
                  [](const spu::Value &v) { return v.isPublic(); })) {
    return internal::fallback_sort1d(ctx, inputs, num_keys, direction);
  }

  // if use default sort method, trying to find the most best method
  // currently, radix sort (has efficient `shuffle`) -> quick sort -> sorting
  // network
  if (sort_method == RuntimeConfig::SORT_DEFAULT) {
    if (internal::_check_method_require(ctx, RuntimeConfig::SORT_RADIX) &&
        internal::_has_efficient_shuffle(ctx)) {
      ret = internal::radix_sort(ctx, inputs, direction, num_keys, valid_bits);
    } else if (internal::_check_method_require(ctx,
                                               RuntimeConfig::SORT_QUICK)) {
      ret = internal::quick_sort(ctx, inputs, num_keys, direction);
    } else if (internal::_check_method_require(
                   ctx,
                   RuntimeConfig::SORT_NETWORK)) {  // always true now.
      ret = internal::fallback_sort1d(ctx, inputs, num_keys, direction);
    } else {
      SPU_THROW("should not reach here");
    }
  } else {
    auto selected_method = internal::select_sort_method(ctx, sort_method);
    if (selected_method != sort_method) {
      SPDLOG_WARN(
          "Manually set method: {}, which is not supported, falling back to "
          "{}.",
          sort_method, selected_method);
    }

    switch (selected_method) {
      case RuntimeConfig::SORT_RADIX:
        ret =
            internal::radix_sort(ctx, inputs, direction, num_keys, valid_bits);
        break;
      case RuntimeConfig::SORT_QUICK:
        ret = internal::quick_sort(ctx, inputs, num_keys, direction);
        break;
      case RuntimeConfig::SORT_NETWORK:
        ret = internal::fallback_sort1d(ctx, inputs, num_keys, direction);
        break;
      default:
        SPU_THROW("should not reach here");
    }
  }

  return ret;
}

std::vector<spu::Value> permute(SPUContext *ctx,
                                absl::Span<const spu::Value> inputs,
                                int64_t permute_dim,
                                const Permute1dFn &permute_fn) {
  // sanity check.
  SPU_ENFORCE(!inputs.empty(), "Inputs should not be empty");
  // put the to_permute dimension to the last dimension.
  const Shape shape = inputs[0].shape();

  // let
  // - M is the number of inputs.
  // - N is the number of vector to permute
  // - W is the vector length.
  const int64_t M = inputs.size();
  const int64_t W = shape.dim(permute_dim);
  if (W == 0) {
    return std::vector<spu::Value>(inputs.begin(), inputs.end());
  }
  const int64_t N = shape.numel() / W;
  Axes perm(shape.ndim());
  Axes unperm;
  {
    // 2 ==> {0, 1, 4, 3, 2}
    SPU_ENFORCE(permute_dim < shape.ndim());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[permute_dim], perm.back());

    auto q = internal::_inverse_index(Index(perm.begin(), perm.end()));
    unperm = Axes(q.begin(), q.end());
  }

  Shape perm_shape(shape.begin(), shape.end());
  std::swap(perm_shape[permute_dim], perm_shape.back());

  // Do permute in 2-dimensions.
  // First, reshape the input to (N, W)
  std::vector<spu::Value> inputs2d;
  for (auto const &input : inputs) {
    auto transposed = hal::transpose(ctx, input, perm);
    auto reshaped = hal::reshape(ctx, transposed, {N, W});
    inputs2d.push_back(std::move(reshaped));
  }

  // Call permute1d for each dim to permute.
  // results (N,M,W), each element is a vector with length W.
  std::vector<std::vector<spu::Value>> permuted1d;
  for (int64_t ni = 0; ni < N; ni++) {
    // TODO: all these small permutations could be done in parallel.
    std::vector<spu::Value> input_i;
    input_i.reserve(inputs2d.size());
    for (auto const &input : inputs2d) {
      // we need 1-d tensor here
      input_i.push_back(
          hal::reshape(ctx, hal::slice(ctx, input, {ni, 0}, {ni + 1, W}), {W}));
    }

    permuted1d.push_back(permute_fn(input_i));
  }

  // result is (M,shape)
  std::vector<spu::Value> results(M);
  for (int64_t mi = 0; mi < M; mi++) {
    std::vector<spu::Value> output2d;
    for (int64_t ni = 0; ni < N; ni++) {
      output2d.push_back(hal::unsqueeze(ctx, permuted1d[ni][mi]));
    }
    auto result = hal::concatenate(ctx, output2d, 0);
    // Permute it back, final result is (M, shape)
    result = hal::reshape(ctx, result, perm_shape);
    results[mi] = hal::transpose(ctx, result, unperm);
  }

  return results;
}

std::vector<Value> topk_1d(SPUContext *ctx, const spu::Value &input,
                           const SimpleCompFn &scalar_cmp,
                           const TopKConfig &config) {
  SPU_ENFORCE(input.shape().ndim() == 1,
              "Inputs should be 1-d but actually have {} dimensions",
              input.shape().ndim());
  SPU_ENFORCE(input.numel() >= config.k_hi,
              "k={} is larger than the last dimension={}", config.k_hi,
              input.numel());
  SPU_ENFORCE(config.k_lo <= config.k_hi);

  if (input.isPublic()) {
    Index indices_to_sort(input.numel());
    std::iota(indices_to_sort.begin(), indices_to_sort.end(), 0);
    auto comparator = [&scalar_cmp, &input, &ctx](int64_t a, int64_t b) {
      auto lhs_value = slice_scalar_at(ctx, input, {a});
      auto rhs_value = slice_scalar_at(ctx, input, {b});
      spu::Value cmp_ret = scalar_cmp(ctx, lhs_value, rhs_value);
      return getBooleanValue(ctx, cmp_ret);
    };

    std::nth_element(indices_to_sort.begin(),
                     indices_to_sort.begin() + config.k_lo - 1,
                     indices_to_sort.end(), comparator);
    if (config.k_lo < config.k_hi) {
      std::nth_element(indices_to_sort.begin() + config.k_lo,
                       indices_to_sort.begin() + (config.k_hi - 1),
                       indices_to_sort.end(), comparator);
    }

    std::vector<spu::Value> ret;

    auto topk_indices =
        Index(indices_to_sort.begin(), indices_to_sort.begin() + config.k_hi);
    ret.push_back(internal::_permute_1d(ctx, input, topk_indices));
    if (!config.value_only) {
      auto dt =
          ctx->config().field == FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
      ret.push_back(constant(ctx, topk_indices, dt,
                             {static_cast<int64_t>(topk_indices.size())}));
    }

    return ret;
  }

  if (ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am")) {
    auto inp = internal::PrepareInput(ctx, input, config);

    hal::CompFn comp_fn =
        [ctx, &scalar_cmp](absl::Span<const spu::Value> values) -> spu::Value {
      auto cmp = scalar_cmp(ctx, values[0], values[1]);
      if (values.size() == 2) {
        return cmp;
      }
      // equal has better performance for aby3
      // cmp+andbb has better performance for semi2k
      auto eq = hal::equal(ctx, values[0], values[1]);

      // comparision of random value
      auto result = scalar_cmp(ctx, values[2], values[3]);
      result = hal::bitwise_and(ctx, eq, result);
      result = hal::bitwise_or(ctx, cmp, result);
      return result;
    };

    return internal::QuickSelectTopk(ctx, comp_fn, absl::MakeSpan(inp), config);

  } else {
    // fall back to general sort
    SPDLOG_WARN(
        "Fallback to generic topk (using sort) because permutation-related "
        "kernels are not supported");

    auto dt =
        ctx->config().field == FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
    std::vector<spu::Value> inp;

    inp.push_back(input);
    if (!config.value_only) {
      inp.push_back(_p2s(ctx, hal::iota(ctx, dt, input.numel())).setDtype(dt));
    }

    hal::CompFn comp_fn =
        [ctx, &scalar_cmp](absl::Span<const spu::Value> values) -> spu::Value {
      // single key with extra payload
      return scalar_cmp(ctx, values[0], values[1]);
    };
    auto sorted =
        hal::sort1d(ctx, absl::MakeSpan(inp), comp_fn, VIS_SECRET, false);

    for (auto &item : sorted) {
      item = slice(ctx, item, {0}, {config.k_hi});
    }

    return sorted;
  }
}

std::vector<spu::Value> apply_inv_permute_1d(
    SPUContext *ctx, absl::Span<const spu::Value> inputs,
    const spu::Value &perm) {
  // Note: the kernel `inv_perm_am` in MPC layer is exactly the `unshuffle`
  // semantics, and we implement `apply_inv_perm_ss` in HAL layer. So we wrap
  // the `apply_inv_perm` to deal with the all inv_perm stuffs.
  return internal::_apply_inv_perm(ctx, inputs, perm);
}

std::vector<spu::Value> apply_permute_1d(SPUContext *ctx,
                                         absl::Span<const spu::Value> inputs,
                                         const spu::Value &perm) {
  // Note: the kernel `perm_am` in MPC layer is exactly the `shuffle`
  // semantics, and we implement `apply_perm_ss` in HAL layer. So we wrap the
  // `apply_perm` to deal with the all inv_perm stuffs.
  return internal::_apply_perm(ctx, inputs, perm);
}

spu::Value apply_general_permute_1d(SPUContext *ctx, const spu::Value &input,
                                    const spu::Value &perm) {
  SPU_TRACE_HAL_DISP(ctx, input, perm);
  SPU_ENFORCE(!perm.isSecret(), "Secret permutation is not supported");

  if (input.isPublic() && perm.isPublic()) { /*PP*/
    return hal::_perm_pp(ctx, input, perm);
  } else if (input.isPublic() && perm.isPrivate()) { /*PV*/
    return hal::_perm_vv(ctx, hal::_p2v(ctx, input, internal::_get_owner(perm)),
                         perm);
  } else if (input.isPrivate() && perm.isPrivate()) { /*VV*/
    if (internal::_has_same_owner(input, perm)) {
      return hal::_perm_vv(ctx, input, perm);
    } else {
      return hal::_perm2_sv(ctx, hal::_v2s(ctx, input), perm);
    }
  } else if (input.isPrivate() && perm.isPublic()) { /*VP*/
    return hal::_perm_vv(ctx, input,
                         hal::_p2v(ctx, perm, internal::_get_owner(input)));
  } else if (input.isSecret() && perm.isPublic()) { /*SP*/
    return hal::_perm2_sp(ctx, input, perm);
  } else if (input.isSecret() && perm.isPrivate()) { /*SV*/
    return hal::_perm2_sv(ctx, input, perm);
  } else {
    SPU_THROW("should not be here");
  }
}

}  // namespace spu::kernel::hal
