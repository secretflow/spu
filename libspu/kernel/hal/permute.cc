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
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/utils.h"

#include "libspu/spu.pb.h"

namespace spu::kernel::hal {

namespace internal {

inline int64_t _get_owner(const Value &x) {
  return x.storage_type().as<Private>()->owner();
}

inline bool _has_same_owner(const Value &x, const Value &y) {
  return _get_owner(x) == _get_owner(y);
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
    if (!input.isSecret()) {
      // we can not linear_scatter a secret value to a public operand
      ret.emplace_back(_2s(ctx, input.clone()).setDtype(input.dtype()));
    } else {
      ret.emplace_back(input.clone());
    }
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

// Ref: https://eprint.iacr.org/2019/695.pdf
// Algorithm 13 Optimized inverse application of a permutation
//
// The steps are as follows:
//   1) secure shuffle <perm> as <sp>
//   2) secure shuffle <x> as <sx>
//   3) reveal securely shuffled <sp> as m
//   4) inverse permute <sx> by m and return
std::pair<std::vector<spu::Value>, spu::Value> _opt_apply_inv_perm_ss(
    SPUContext *ctx, absl::Span<spu::Value const> x, spu::Value perm,
    spu::Value random_perm) {
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

  const auto k1 = _constant(ctx, 1U, x.shape());
  auto rev_x = _sub(ctx, k1, x);

  const auto numel = x.numel();
  Shape new_shape = {1, numel};
  auto f = concatenate(
      ctx, {reshape(ctx, rev_x, new_shape), reshape(ctx, x, new_shape)}, 1);

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
std::vector<spu::Value> _gen_bv_vector(SPUContext *ctx,
                                       absl::Span<spu::Value const> keys,
                                       SortDirection direction,
                                       int64_t valid_bits) {
  std::vector<spu::Value> ret;
  const auto k1 = _constant(ctx, 1U, keys[0].shape());
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
  // 1. generate bit decomposition vector of keys
  std::vector<spu::Value> bv = _gen_bv_vector(ctx, keys, direction, valid_bits);
  SPU_ENFORCE_GT(bv.size(), 0U);

  // 2. generate natural permutation for initialization
  auto dt =
      ctx->config().field() == FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
  auto init_perm = iota(ctx, dt, keys[0].numel());
  auto shared_perm = _p2s(ctx, init_perm);

  // 3. generate shared inverse permutation by bit vector and process
  size_t bv_size = bv.size();
  size_t bv_idx = 0;
  for (; bv_idx < bv_size - 1; bv_idx += 2) {
    // generate random permutation for shuffle
    auto random_perm = hal::_rand_perm_s(ctx, keys[0].shape());
    auto [shuffled_bv, shuffled_perm] = _opt_apply_inv_perm_ss(
        ctx, std::vector<spu::Value>{bv[bv_idx], bv[bv_idx + 1]}, shared_perm,
        random_perm);
    auto perm = _gen_inv_perm_by_bv(ctx, shuffled_bv[0], shuffled_bv[1]);
    shared_perm = _opt_apply_perm_ss(ctx, perm, shuffled_perm, random_perm);
  }

  if (bv_idx == bv_size - 1) {
    // generate random permutation for shuffle
    auto random_perm = hal::_rand_perm_s(ctx, keys[0].shape());
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
  // 1. <SP> = secure shuffle <perm>
  auto shuffle_perm = hal::_rand_perm_s(ctx, x[0].shape());
  auto sp = hal::_perm_ss(ctx, perm, shuffle_perm);

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
spu::Value _apply_perm_ss(SPUContext *ctx, const Value &x, const Value &perm) {
  // 1. <SP> = secure shuffle <perm>
  auto shuffle_perm = hal::_rand_perm_s(ctx, x.shape());
  auto sp = hal::_perm_ss(ctx, perm, shuffle_perm);

  // 2. M = reveal(<SP>)
  auto m = _s2p(ctx, sp);
  SPU_ENFORCE_EQ(m.shape().ndim(), 1U, "perm should be 1-d tensor");

  // 3. sx = apply_perm(x,m)
  auto sx = hal::_perm_sp(ctx, x, m);

  // 4. ret = unshuffle(<sx>)
  auto ret = hal::_inv_perm_ss(ctx, sx, shuffle_perm);

  return ret;
}

// Find mergeable keys from keys. Consecutive public/private(belong to one
// owner) keys can be merged. Assume there are six keys, i.e., public_key0,
// bob_key0, bob_key1, alice_key0, alice_key1, secret_key0. We can merge the six
// keys into bob_new_key, alice_new_key, secret_key0 for the following sorting.
// This function will return a vector of indices [3,5,6] which means key[0,3),
// key[3,5), and key[5,6) can be merged.
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
  if (ctx->hasKernel("inv_perm_av")) {
    return hal::_inv_perm_sv(ctx, in, perm);
  } else {
    return _apply_inv_perm_ss(ctx, in, _v2s(ctx, perm));
  }
}

#define MAP_APPLY_PERM_OP(NAME)                             \
  spu::Value _apply##NAME(SPUContext *ctx, const Value &in, \
                          const Value &perm) {              \
    return hal::NAME(ctx, in, perm);                        \
  }

MAP_APPLY_PERM_OP(_perm_pp);
MAP_APPLY_PERM_OP(_perm_vv);
MAP_APPLY_PERM_OP(_perm_sp);
MAP_APPLY_PERM_OP(_inv_perm_pp);
MAP_APPLY_PERM_OP(_inv_perm_vv);
MAP_APPLY_PERM_OP(_inv_perm_sp);

// Given a permutation, apply (inverse) permutation on a 1-d array input
#define MAP_PERM_OP(NAME)                                                \
  spu::Value NAME(SPUContext *ctx, const Value &in, const Value &perm) { \
    SPU_TRACE_HAL_DISP(ctx, in, perm);                                   \
    if (in.isPublic() && perm.isPublic()) { /*PP*/                       \
      return NAME##_pp(ctx, in, perm);                                   \
    } else if (in.isPublic() && perm.isSecret()) { /*PS*/                \
      return NAME##_ss(ctx, _p2s(ctx, in), perm);                        \
    } else if (in.isPublic() && perm.isPrivate()) { /*PV*/               \
      return NAME##_vv(ctx, _p2v(ctx, in, _get_owner(perm)), perm);      \
    } else if (in.isPrivate() && perm.isPrivate()) { /*VV*/              \
      if (_has_same_owner(in, perm)) {                                   \
        return NAME##_vv(ctx, in, perm);                                 \
      } else {                                                           \
        return NAME##_sv(ctx, _v2s(ctx, in), perm);                      \
      }                                                                  \
    } else if (in.isPrivate() && perm.isPublic()) { /*VP*/               \
      return NAME##_vv(ctx, in, _p2v(ctx, perm, _get_owner(in)));        \
    } else if (in.isPrivate() && perm.isSecret()) { /*VS*/               \
      return NAME##_ss(ctx, _v2s(ctx, in), perm);                        \
    } else if (in.isSecret() && perm.isSecret()) { /*SS*/                \
      return NAME##_ss(ctx, in, perm);                                   \
    } else if (in.isSecret() && perm.isPublic()) { /*SP*/                \
      return NAME##_sp(ctx, in, perm);                                   \
    } else if (in.isSecret() && perm.isPrivate()) { /*SV*/               \
      return NAME##_sv(ctx, in, perm);                                   \
    } else {                                                             \
      SPU_THROW("should not be here");                                   \
    }                                                                    \
  }

// Inverse permute 1-D array x with a permutation perm
// ret[perm[i]] = x[i]
MAP_PERM_OP(_apply_inv_perm)

// Given a permutation, generate its inverse permutation
// ret[perm[i]] = i
spu::Value _inverse(SPUContext *ctx, const Value &perm) {
  auto dt =
      ctx->config().field() == FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
  auto iota_perm = iota(ctx, dt, perm.numel());
  return _apply_inv_perm(ctx, iota_perm, perm);
}

spu::Value _apply_perm_sv(SPUContext *ctx, const Value &in, const Value &perm) {
  if (ctx->hasKernel("inv_perm_av")) {
    return hal::_inv_perm_sv(ctx, in, _inverse(ctx, perm));
  } else {
    return _apply_inv_perm_ss(ctx, in, _v2s(ctx, _inverse(ctx, perm)));
  }
}

// Permute 1-D array x with a permutation perm
// ret[i] = x[perm[i]]
MAP_PERM_OP(_apply_perm)

// Compose two permutations into one permutation
// If we have two permutations x and y, we want to get a permutation z from x
// and y that apply_inv_perm(in, z) = apply_inv_perm(apply_inv_perm(in, x), y)
spu::Value _compose_perm(SPUContext *ctx, const Value &x, const Value &y) {
  return _apply_perm(ctx, y, x);
}

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
  auto dt =
      ctx->config().field() == FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
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

  bool fallback = false;
  // if all keys are public, fallback to public sort
  if (std::all_of(inputs.begin(), inputs.begin() + num_keys,
                  [](const spu::Value &v) { return v.isPublic(); })) {
    fallback = true;
  }
  // If the protocol supports secret shuffle and unshuffle, we can use radix
  // sort for fast 1-D sort. Otherwise, we fallback to generic sort1d
  if (!fallback &&
      !(ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am") &&
        ctx->hasKernel("perm_ap") && ctx->hasKernel("inv_perm_am") &&
        ctx->hasKernel("inv_perm_ap"))) {
    fallback = true;
  }
  if (!fallback) {
    auto ret =
        internal::radix_sort(ctx, inputs, direction, num_keys, valid_bits);
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
