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
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

#include "libspu/spu.pb.h"

namespace spu::kernel::hal {

namespace {

// generate inverse permutation
Index GenInvPerm(const Index &p) {
  Index q(p.size());
  const auto n = static_cast<int64_t>(p.size());
  for (int64_t i = 0; i < n; ++i) {
    q[p[i]] = i;
  }
  return q;
}

Value Permute1D(SPUContext *, const Value &x, const Index &indices) {
  SPU_ENFORCE(x.shape().size() == 1);
  return Value(x.data().linear_gather(indices), x.dtype());
}

// FIXME: move to mpc layer
// Vectorized Prefix Sum
// Ref: https://en.algorithmica.org/hpc/algorithms/prefix/
Value PrefixSum(SPUContext *ctx, const Value &x) {
  SPU_ENFORCE(x.shape().ndim() == 2U && x.shape()[0] == 1,
              "x should be 1-row matrix");

  auto padding0 = _constant(ctx, 0U, {1, 1});
  auto x_t = x;
  for (int64_t shift = 1; shift < x.numel(); shift *= 2) {
    auto x_slice = slice(ctx, x_t, {0, 0}, {1, x.numel() - shift}, {});
    auto x_rshift = pad(ctx, x_slice, padding0, {0, shift}, {0, 0}, {0, 0});
    x_t = _add(ctx, x_t, x_rshift);
  }

  return x_t;
}

void CmpSwap(SPUContext *ctx, const CompFn &comparator_body,
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

inline int64_t Exp2(int64_t n) {
  SPU_ENFORCE_GE(n, 0, "input should be greater than or equal zero");
  int64_t ret = 1;
  return ret << n;
}

// Bitonic sort with all comparator sorts in the same direction
// Ref:
// https://en.wikipedia.org/wiki/Bitonic_sorter#Alternative_representation
std::vector<spu::Value> BitonicSort(SPUContext *ctx,
                                    const CompFn &comparator_body,
                                    absl::Span<spu::Value const> inputs) {
  // make a copy for inplace sort
  std::vector<spu::Value> ret;
  for (auto const &input : inputs) {
    if (input.isPublic()) {
      // we can not linear_scatter a secret value to a public operand
      // FIXME: clone() is need here as ref2k p2s is in-place operation, fixed
      // later
      ret.emplace_back(_p2s(ctx, input).setDtype(input.dtype()).clone());
    } else {
      ret.emplace_back(input.clone());
    }
  }

  // sort by per network layer for memory optimizations, sorting N elements
  // needs log2(N) stages, and the i_th stage has i layers
  const auto numel = inputs.front().numel();
  const auto n_stages = Log2Ceil(numel);
  for (int64_t stage = 1; stage <= n_stages; ++stage) {
    for (int64_t layer = 1; layer <= stage; ++layer) {
      // find index pairs that needs to be compared
      Index lhs_indices;
      Index rhs_indices;
      auto step = Exp2(stage - layer);
      for (int64_t idx = 0; idx + step < numel; idx += 2 * step) {
        for (int64_t offset = 0; offset < step; ++offset) {
          int64_t lhs_idx, rhs_idx;
          if (layer == 1) {
            lhs_idx = idx + step - offset - 1;
            rhs_idx = idx + step + offset;
          } else {
            lhs_idx = idx + offset;
            rhs_idx = idx + offset + step;
          }
          if (lhs_idx >= numel || rhs_idx >= numel) break;
          lhs_indices.emplace_back(lhs_idx);
          rhs_indices.emplace_back(rhs_idx);
        }
      }

      CmpSwap(ctx, comparator_body, absl::MakeSpan(ret), lhs_indices,
              rhs_indices);
    }
  }
  return ret;
}

// Secure shuffle a shared permutation <perm> and use it to permute shared bit
// vectors of x.
// x is a list of shared bit vectors, <perm> is a shared permutation,
// random_perm is a permutation for shuffling <perm>, and m is the
// revealed permutation of shuffled <perm>.
//
// The steps are as follows:
//   1) secure shuffle <perm> as <sp>
//   2) secure shuffle <x> as <sx>
//   3) reveal securely shuffled <sp> as m
//   4) inverse permute <sx> by m and return
std::pair<std::vector<spu::Value>, spu::Value> ShufflePerm(
    SPUContext *ctx, absl::Span<spu::Value const> x, spu::Value perm,
    spu::Value random_perm) {
  // 1. <SP> = secure shuffle <perm>
  auto sp = _perm_ss(ctx, perm, random_perm);

  // 2. <SX> = secure shuffle <x>
  std::vector<spu::Value> sx;
  for (size_t i = 0; i < x.size(); ++i) {
    sx.emplace_back(_perm_ss(ctx, x[i], random_perm));
  }

  // 3. M = reveal(<SP>)
  auto m = _s2p(ctx, sp);
  SPU_ENFORCE_EQ(m.shape().ndim(), 1U, "perm should be 1-d tensor");

  // 4. <T> = SP(<SX>)
  std::vector<spu::Value> v;

  for (size_t i = 0; i < sx.size(); ++i) {
    auto t = _inv_perm_sp(ctx, sx[i], m);
    v.emplace_back(std::move(t));
  }

  return {v, m};
}

// Process two bit vectors in one loop
// Reference: https://eprint.iacr.org/2019/695.pdf (5.2 Optimizations)
//
// perm = GenInvPermByTwoBitVectors(x, y)
//   input: bit vector x, bit vector y
//          bit vector y is more significant than x
//   output: shared inverse permutation
//
// We can generate inverse permutation by two bit vectors in one loop.
// It needs one extra mul op and 2 times memory to store intermediate data
// than GenInvPermByBitVector. But the number of invocations of
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
spu::Value GenInvPermByTwoBitVectors(SPUContext *ctx, const spu::Value &x,
                                     const spu::Value &y) {
  SPU_ENFORCE(x.shape() == y.shape(), "x and y should has the same shape");
  SPU_ENFORCE(x.shape().ndim() == 1, "x and y should be 1-d");

  const auto k1 = _constant(ctx, 1U, x.shape());
  auto rev_x = _sub(ctx, k1, x);
  auto rev_y = _sub(ctx, k1, y);
  auto f0 = _mul(ctx, rev_x, rev_y);
  auto f1 = _sub(ctx, rev_y, f0);
  auto f2 = _sub(ctx, rev_x, f0);
  auto f3 = _sub(ctx, y, f2);

  const auto numel = x.numel();
  Shape new_shape = {1, numel};
  auto f =
      concatenate(ctx,
                  {reshape(ctx, f0, new_shape), reshape(ctx, f1, new_shape),
                   reshape(ctx, f2, new_shape), reshape(ctx, f3, new_shape)},
                  1);

  // calculate prefix sum
  auto ps = PrefixSum(ctx, f);

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
spu::Value GenInvPermByBitVector(SPUContext *ctx, const spu::Value &x) {
  SPU_ENFORCE(x.shape().ndim() == 1, "x should be 1-d");

  const auto k1 = _constant(ctx, 1U, x.shape());
  auto rev_x = _sub(ctx, k1, x);

  const auto numel = x.numel();
  Shape new_shape = {1, numel};
  auto f = concatenate(
      ctx, {reshape(ctx, rev_x, new_shape), reshape(ctx, x, new_shape)}, 1);

  // calculate prefix sum
  auto ps = PrefixSum(ctx, f);

  // mul f and s
  auto fs = _mul(ctx, f, ps);

  auto fs0 = slice(ctx, fs, {0, 0}, {1, numel}, {});
  auto fs1 = slice(ctx, fs, {0, numel}, {1, 2 * numel}, {});

  // calculate result
  auto r = _add(ctx, fs0, fs1);
  auto res = _sub(ctx, reshape(ctx, r, x.shape()), k1);
  return res;
}

// This is the inverse of ShufflePerm.
// The input is a shared inverse permutation <perm>, a public permutation
// shuffled_perm generated by ShufflePerm, and a secret permutation
// random_perm for secure unshuffle.
//
// The steps are as follows:
//   1) permute <perm> by shuffled_perm as <sm>
//   2) secure unshuffle <sm> and return results
//
// By doing ShufflePerm and UnshufflePerm, we get the shared inverse
// permutation of initial shared bit vectors.
spu::Value UnshufflePerm(SPUContext *ctx, const spu::Value &perm,
                         const spu::Value &shuffled_perm,
                         const spu::Value &random_perm) {
  auto sm = _perm_sp(ctx, perm, shuffled_perm);
  auto res = _inv_perm_ss(ctx, sm, random_perm);
  return res;
}

std::vector<spu::Value> BitDecompose(SPUContext *ctx, const spu::Value &x,
                                     int64_t valid_bits) {
  auto x_bshare = _prefer_b(ctx, x);
  const auto k1 = _constant(ctx, 1U, x.shape());
  std::vector<spu::Value> rets;
  size_t nbits = valid_bits != -1
                     ? static_cast<size_t>(valid_bits)
                     : x_bshare.storage_type().as<BShare>()->nbits();
  rets.reserve(nbits);

  for (size_t bit = 0; bit < nbits; ++bit) {
    auto x_bshare_shift = right_shift_logical(ctx, x_bshare, bit);
    auto lowest_bit = _and(ctx, x_bshare_shift, k1);
    rets.emplace_back(_prefer_a(ctx, lowest_bit));
  }

  return rets;
}

// Generate vector of bit decomposition of sorting keys
std::vector<spu::Value> GenBvVector(SPUContext *ctx,
                                    absl::Span<spu::Value const> inputs,
                                    SortDirection direction, int64_t num_keys,
                                    int64_t valid_bits) {
  std::vector<spu::Value> ret;
  const auto k1 = _constant(ctx, 1U, inputs[0].shape());
  // inputs[0] is the most significant key
  for (int64_t i = num_keys - 1; i >= 0; --i) {
    const auto &t = BitDecompose(ctx, inputs[i], valid_bits);

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
spu::Value GenInvPerm(SPUContext *ctx, absl::Span<spu::Value const> inputs,
                      SortDirection direction, int64_t num_keys,
                      int64_t valid_bits) {
  // 1. generate bit decomposition vector of keys
  std::vector<spu::Value> bv =
      GenBvVector(ctx, inputs, direction, num_keys, valid_bits);
  SPU_ENFORCE_GT(bv.size(), 0U);

  // 2. generate natural permutation for initialization
  auto dt =
      ctx->config().field() == FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
  auto init_perm = iota(ctx, dt, inputs[0].numel());
  auto shared_perm = _p2s(ctx, init_perm);

  // 3. generate shared inverse permutation by bit vector and process
  size_t bv_size = bv.size();
  size_t bv_idx = 0;
  for (; bv_idx < bv_size - 1; bv_idx += 2) {
    auto random_perm = _rand_perm_s(ctx, inputs[0].shape());
    auto [shuffled_bv, shuffled_perm] =
        ShufflePerm(ctx, std::vector<spu::Value>{bv[bv_idx], bv[bv_idx + 1]},
                    shared_perm, random_perm);
    auto perm = GenInvPermByTwoBitVectors(ctx, shuffled_bv[0], shuffled_bv[1]);
    shared_perm = UnshufflePerm(ctx, perm, shuffled_perm, random_perm);
  }

  if (bv_idx == bv_size - 1) {
    auto random_perm = _rand_perm_s(ctx, inputs[0].shape());
    auto [shuffled_bv, shuffled_perm] = ShufflePerm(
        ctx, std::vector<spu::Value>{bv[bv_idx]}, shared_perm, random_perm);
    auto perm = GenInvPermByBitVector(ctx, shuffled_bv[0]);
    shared_perm = UnshufflePerm(ctx, perm, shuffled_perm, random_perm);
  }

  return shared_perm;
}

// Apply inverse permutation on each tensor of x by a shared inverse
// permutation <perm>
std::vector<spu::Value> ApplyInvPerm(SPUContext *ctx,
                                     absl::Span<spu::Value const> x,
                                     const spu::Value &perm) {
  // sanity check.
  SPU_ENFORCE(!x.empty(), "inputs should not be empty");
  SPU_ENFORCE(x[0].shape().ndim() == 1,
              "inputs should be 1-d but actually have {} dimensions",
              x[0].shape().ndim());
  SPU_ENFORCE(std::all_of(x.begin(), x.end(),
                          [&x](const spu::Value &input) {
                            return input.shape() == x[0].shape();
                          }),
              "inputs shape mismatched");

  // 1. <SP> = secure shuffle <perm>
  auto shuffle_perm = _rand_perm_s(ctx, x[0].shape());
  auto sp = _perm_ss(ctx, perm, shuffle_perm);

  // 2. <SX> = secure shuffle <x>
  std::vector<spu::Value> sx;
  for (size_t i = 0; i < x.size(); ++i) {
    sx.emplace_back(_perm_ss(ctx, x[i], shuffle_perm));
  }

  // 3. M = reveal(<SP>)
  auto m = _s2p(ctx, sp);
  SPU_ENFORCE_EQ(m.shape().ndim(), 1U, "perm should be 1-d tensor");

  // 4. <T> = SP(<SX>)
  std::vector<spu::Value> v;
  for (size_t i = 0; i < sx.size(); ++i) {
    auto t = _inv_perm_sp(ctx, sx[i], m);
    v.emplace_back(std::move(t));
  }

  return v;
}

// Secure Radix Sort
// Ref:
//  https://eprint.iacr.org/2019/695.pdf
//
// Each input is a 1-d tensor, inputs[0, num_keys) are the keys, and sort
// inputs according to keys
std::vector<spu::Value> RadixSort(SPUContext *ctx,
                                  absl::Span<spu::Value const> inputs,
                                  SortDirection direction, int64_t num_keys,
                                  int64_t valid_bits) {
  auto perm = GenInvPerm(ctx, inputs, direction, num_keys, valid_bits);
  auto res = ApplyInvPerm(ctx, inputs, perm);
  return res;
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
      ret.push_back(Permute1D(ctx, input, indices_to_sort));
    }
  } else if (comparator_ret_vis == VIS_SECRET) {
    SPU_ENFORCE(!is_stable,
                "Stable sort is unsupported if comparator return is secret.");

    ret = BitonicSort(ctx, cmp, inputs);
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

  bool fallback = false;
  // If all keys are secret values and the protocol supports secret shuffle and
  // unshuffle, we can use radix sort for fast 1-D sort. Otherwise, we fallback
  // to generic sort1d, and use the inputs[0] as the sorting key
  if (!std::all_of(inputs.begin(), inputs.begin() + num_keys,
                   [](const spu::Value &v) { return v.isSecret(); })) {
    fallback = true;
    SPDLOG_WARN("Fallback to generic sort1d because not all keys are secret");
  }

  if (!fallback &&
      !(ctx->hasKernel("rand_perm_s") && ctx->hasKernel("perm_as") &&
        ctx->hasKernel("perm_ap") && ctx->hasKernel("inv_perm_as") &&
        ctx->hasKernel("inv_perm_ap"))) {
    fallback = true;
    SPDLOG_WARN(
        "Fallback to generic sort1d because permutation-related kernels are "
        "not supported");
  }
  if (!fallback) {
    auto ret = RadixSort(ctx, inputs, direction, num_keys, valid_bits);
    return ret;
  } else {
    auto scalar_cmp = [direction](spu::SPUContext *ctx, const spu::Value &lhs,
                                  const spu::Value &rhs) {
      if (direction == SortDirection::Ascending) {
        return hal::less(ctx, lhs, rhs);
      }
      return hal::greater(ctx, lhs, rhs);
    };

    hal::CompFn comp_fn =
        [ctx, num_keys,
         &scalar_cmp](absl::Span<const spu::Value> values) -> spu::Value {
      spu::Value pre_equal = hal::constant(ctx, true, DT_I1, values[0].shape());
      spu::Value result = scalar_cmp(ctx, values[0], values[1]);
      // the idea here is that if the two values of the last key is equal, than
      // we compare the two values of the current key, and iteratively to update
      // the result which indicates whether to swap values
      for (int64_t idx = 2; idx < num_keys * 2; idx += 2) {
        pre_equal = hal::bitwise_and(
            ctx, pre_equal, hal::equal(ctx, values[idx - 2], values[idx - 1]));
        auto current = scalar_cmp(ctx, values[idx], values[idx + 1]);
        current = hal::bitwise_and(ctx, pre_equal, current);
        result = hal::bitwise_or(ctx, result, current);
      }
      return result;
    };

    Visibility vis =
        std::all_of(inputs.begin(), inputs.begin() + num_keys,
                    [](const spu::Value &v) { return v.isPublic(); })
            ? VIS_PUBLIC
            : VIS_SECRET;
    auto ret = sort1d(ctx, inputs, comp_fn, vis, false);
    return ret;
  }
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

    auto q = GenInvPerm(Index(perm.begin(), perm.end()));
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
    inputs2d.push_back(reshaped);
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
      output2d.push_back(hal::reshape(ctx, permuted1d[ni][mi], {1, W}));
    }
    auto result = hal::concatenate(ctx, output2d, 0);
    // Permute it back, final result is (M, shape)
    result = hal::reshape(ctx, result, perm_shape);
    results[mi] = hal::transpose(ctx, result, unperm);
  }

  return results;
}

}  // namespace spu::kernel::hal
