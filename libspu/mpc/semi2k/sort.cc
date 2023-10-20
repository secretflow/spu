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

#include "libspu/mpc/semi2k/sort.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

namespace {

NdArrayRef wrap_a2b(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(a2b(ctx, WrapValue(x)));
}

NdArrayRef wrap_a2v(SPUContext* ctx, const NdArrayRef& x, size_t rank) {
  return UnwrapValue(a2v(ctx, WrapValue(x), rank));
}

NdArrayRef wrap_a2p(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(a2p(ctx, WrapValue(x)));
}

NdArrayRef wrap_mul_aa(SPUContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_add_aa(SPUContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  return UnwrapValue(add_aa(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_sub_pa(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  SPU_ENFORCE(x.numel() == y.numel());

  auto* comm = ctx->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_sub(x, y).as(y.eltype());
  } else {
    return ring_neg(y).as(y.eltype());
  }
}

NdArrayRef wrap_sub_ap(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  SPU_ENFORCE(x.numel() == y.numel());

  auto* comm = ctx->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_sub(x, y).as(x.eltype());
  } else {
    return x;
  }
}

NdArrayRef wrap_sub_aa(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  SPU_ENFORCE(x.numel() == y.numel());
  SPU_ENFORCE(x.eltype() == y.eltype());

  return ring_sub(x, y).as(x.eltype());
}

// Reference:
//  III.D @ https://eprint.iacr.org/2019/599.pdf (SPDZ-2K primitives)
//
// Analysis:
//  Online Latency: 1 (x_xor_r reveal)
//  Communication: one element bits for one element
//  Vectorization: yes
//
// HighLevel Intuition:
//  Since: X = sum: Xi * 2^i
// If we have <Xi>A, then we can construct <X>A = sum: <Xi>A * 2^i.
//
// The problem is that we only have <Xi>B in hand. Details for how to
// construct <Xi>A from <Xi>B:
// - trusted third party choose a random bit r, where r == 0 or r == 1.
// - trusted third party send <r>A to parties
// - parties compute <r>B from <r>A
// - parties xor_open c = Xi ^ r = open(<Xi>B ^ <r>B), Xi is still safe due
// to protection from r.
// - parties compute: <x> = c + (1-2c)*<r>
//    <Xi>A = 1 - <r>A if c == 1, i.e. Xi != r
//    <Xi>A = <r>A if c == 0, i.e. Xi == r
//    i.e. <Xi>A = c + (1-2c) * <r>A
//
//  Online Communication:
//    = 1 (xor open)

// Unassemble BShr to AShr bit-by-bit
//  Input: BShr
//  Return: a vector of k AShr, k is the valid bits of BShr
std::vector<NdArrayRef> B2AUnassemble(KernelEvalContext* ctx,
                                      const NdArrayRef& x) {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  const int64_t nbits = x.eltype().as<BShare>()->nbits();
  SPU_ENFORCE((size_t)nbits > 0 && (size_t)nbits <= SizeOf(field) * 8,
              "invalid nbits={}", nbits);

  auto numel = x.numel();

  auto randbits = beaver->RandBit(field, {numel * static_cast<int64_t>(nbits)});

  std::vector<NdArrayRef> res;
  for (int64_t idx = 0; idx < nbits; ++idx) {
    res.emplace_back(makeType<AShrTy>(field), x.shape());
  }

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    NdArrayView<U> _randbits(randbits);
    NdArrayView<U> _x(x);

    std::vector<U> x_xor_r(numel);

    pforeach(0, numel, [&](int64_t idx) {
      // use _r[i*nbits, (i+1)*nbits) to construct rb[i]
      U mask = 0;
      for (int64_t bit = 0; bit < nbits; ++bit) {
        mask += (_randbits[idx * nbits + bit] & 0x1) << bit;
      }
      x_xor_r[idx] = _x[idx] ^ mask;
    });

    // open c = x ^ r
    x_xor_r = comm->allReduce<U, std::bit_xor>(x_xor_r, "open(x^r)");

    pforeach(0, numel, [&](int64_t idx) {
      pforeach(0, nbits, [&](int64_t bit) {
        NdArrayView<U> _res(res[bit]);
        auto c_i = (x_xor_r[idx] >> bit) & 0x1;
        if (comm->getRank() == 0) {
          _res[idx] = (c_i + (1 - c_i * 2) * _randbits[idx * nbits + bit]);
        } else {
          _res[idx] = ((1 - c_i * 2) * _randbits[idx * nbits + bit]);
        }
      });
    });
  });

  return res;
}

// Input: AShare of x
// Output: a vector of AShare of each bit of x
std::vector<NdArrayRef> BitDecompose(KernelEvalContext* ctx,
                                     const NdArrayRef& x) {
  auto b = wrap_a2b(ctx->sctx(), x);
  return B2AUnassemble(ctx, b);
}

// TODO(jimi): maybe support multiple keys in future
// Generate vector of bit decomposition
std::vector<NdArrayRef> GenBvVector(KernelEvalContext* ctx,
                                    const NdArrayRef& key) {
  std::vector<NdArrayRef> ret;
  const auto& t = BitDecompose(ctx, key);
  SPU_ENFORCE(t.size() > 0);
  ret.insert(ret.end(), t.begin(), t.end() - 1);
  const auto field = key.eltype().as<AShrTy>()->field();
  ret.emplace_back(wrap_sub_pa(ctx, ring_ones(field, key.shape()), t.back()));

  return ret;
}

// Secure inverse permutation of x by perm_rank's permutation pv
// The idea here is:
// Input permutation pv, beaver generates perm pair {<A>, <B>} that
// InversePermute(A, pv) = B. So we can get <y> = InversePermute(open(<x> -
// <A>), pv) + <B> that y = InversePermute(x, pv).
NdArrayRef SecureInvPerm(KernelEvalContext* ctx, const NdArrayRef& x,
                         size_t perm_rank, absl::Span<const int64_t> pv) {
  const auto lctx = ctx->lctx();
  const auto field = x.eltype().as<AShrTy>()->field();
  auto* beaver = ctx->getState<Semi2kState>()->beaver();

  auto perm_pair = beaver->PermPair(field, x.shape(), perm_rank, pv);

  auto t = wrap_a2v(ctx->sctx(), ring_sub(x, perm_pair.first).as(x.eltype()),
                    perm_rank);

  if (lctx->Rank() == perm_rank) {
    SPU_ENFORCE(pv.size());
    ring_add_(perm_pair.second, applyInvPerm(t, pv));
    return perm_pair.second.as(x.eltype());
  } else {
    return perm_pair.second.as(x.eltype());
  }
}

// Secure inverse permutation of a vector x by perm_rank's permutation pv
std::vector<NdArrayRef> SecureInvPerm(KernelEvalContext* ctx,
                                      absl::Span<const NdArrayRef> x,
                                      size_t perm_rank,
                                      absl::Span<const int64_t> pv) {
  std::vector<NdArrayRef> v;
  for (size_t i = 0; i < x.size(); ++i) {
    auto t = SecureInvPerm(ctx, x[i], perm_rank, pv);
    v.emplace_back(std::move(t));
  }
  return v;
}

// Secure shuffle a vector x by each party's local permutation pv.
// The shuffle involves multiple rounds of share inverse permutation. Each round
// the permutation is a local permutation pv generated by a perm_rank.
std::vector<NdArrayRef> Shuffle(KernelEvalContext* ctx,
                                absl::Span<const NdArrayRef> x,
                                absl::Span<const int64_t> pv) {
  std::vector<NdArrayRef> v;
  const auto lctx = ctx->lctx();
  SPU_ENFORCE(!x.empty(), "inputs should not be empty");

  for (size_t i = 0; i < lctx->WorldSize(); ++i) {
    if (i == 0) {
      v = SecureInvPerm(ctx, x, i, pv);
    } else {
      v = SecureInvPerm(ctx, v, i, pv);
    }
  }

  return v;
}

// Secure shuffle x by each party's local permutation pv.
NdArrayRef Shuffle(KernelEvalContext* ctx, const NdArrayRef& x,
                   absl::Span<const int64_t> pv) {
  auto vec = Shuffle(ctx, std::vector<NdArrayRef>{x}, pv);
  SPU_ENFORCE(vec.size() > 0);
  return vec[0];
}

// Inverse a securely shuffled perm on shuffled x.
// x is a list of shared bit vectors, <perm> is a shared permutation, pv is
// a local generated random permutation for secure shuffle, and random_perm is
// revealed permutation of shuffled <perm>.
//
// The steps are as follows:
//   1) secure shuffle <perm> as <sp>
//   2) secure shuffle <x> as <sx>
//   3) reveal securely shuffled <sp> as random_perm
//   4) inverse permute <sx> by random_perm and return
std::vector<NdArrayRef> InvShuffledPerm(KernelEvalContext* ctx,
                                        absl::Span<const NdArrayRef> x,
                                        const NdArrayRef& perm,
                                        PermVector* random_perm,
                                        absl::Span<const int64_t> pv) {
  // 1. <SP> = secure shuffle <perm>
  auto sp = Shuffle(ctx, perm, pv);

  // 2. <SX> = secure shuffle <x>
  auto sx = Shuffle(ctx, x, pv);

  // 3. M = reveal(<SP>)
  auto m = wrap_a2p(ctx->sctx(), sp);
  SPU_ENFORCE_EQ(m.shape().ndim(), 1U, "perm should be 1-d tensor");
  auto size = m.shape()[0];
  const auto field = m.eltype().as<Ring2k>()->field();
  PermVector perm_vector(size);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _m(m);
    pforeach(0, size,
             [&](int64_t idx) { perm_vector[idx] = (int64_t)_m[idx]; });
  });
  SPU_ENFORCE(random_perm != nullptr);
  *random_perm = perm_vector;

  // 4. <T> = SP(<SX>)
  std::vector<NdArrayRef> v;
  for (size_t i = 0; i < sx.size(); ++i) {
    auto t = applyInvPerm(sx[i], perm_vector);
    v.emplace_back(std::move(t));
  }

  return v;
}

// Inverse a securely shuffled perm on shuffled x.
NdArrayRef InvShuffledPerm(KernelEvalContext* ctx, const NdArrayRef& x,
                           const NdArrayRef& perm, PermVector* random_perm,
                           absl::Span<const int64_t> pv) {
  std::vector<NdArrayRef> v{x};
  auto vec = InvShuffledPerm(ctx, v, perm, random_perm, pv);
  SPU_ENFORCE(vec.size() > 0);
  return vec[0];
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
// It needs one extra mul op and 2 times memory to store intermediate data than
// GenInvPermByBitVector. But the number of invocations of permutation-related
// protocols such as SecureInvPerm or Compose will be reduced to half.
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
NdArrayRef GenInvPermByTwoBitVectors(KernelEvalContext* ctx,
                                     const NdArrayRef& x, const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape(), "x and y should has the same shape");
  SPU_ENFORCE(x.shape().ndim() == 1, "x and y should be 1-d");

  const auto field = x.eltype().as<AShrTy>()->field();
  const int64_t numel = x.numel();
  auto ones = ring_ones(field, x.shape());
  auto rev_x = wrap_sub_pa(ctx, ones, x);
  auto rev_y = wrap_sub_pa(ctx, ones, y);
  auto f0 = wrap_mul_aa(ctx->sctx(), rev_x, rev_y);
  auto f1 = wrap_sub_aa(ctx, rev_y, f0);
  auto f2 = wrap_sub_aa(ctx, rev_x, f0);
  auto f3 = wrap_sub_aa(ctx, y, f2);

  Shape new_shape = {1, numel};
  auto f = f0.reshape(new_shape).concatenate(
      {f1.reshape(new_shape), f2.reshape(new_shape), f3.reshape(new_shape)}, 1);
  auto s = f.clone();

  // calculate prefix sum
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _s(s);
    for (int64_t i = 1; i < s.numel(); ++i) {
      _s[i] += _s[i - 1];
    }
  });

  // mul f and s
  auto fs = wrap_mul_aa(ctx->sctx(), f, s);

  auto fs0 = fs.slice({0, 0}, {1, numel}, {});
  auto fs1 = fs.slice({0, numel}, {1, 2 * numel}, {});
  auto fs2 = fs.slice({0, 2 * numel}, {1, 3 * numel}, {});
  auto fs3 = fs.slice({0, 3 * numel}, {1, 4 * numel}, {});

  // calculate result
  auto s01 = wrap_add_aa(ctx->sctx(), fs0, fs1);
  auto s23 = wrap_add_aa(ctx->sctx(), fs2, fs3);
  auto r = wrap_add_aa(ctx->sctx(), s01, s23);
  auto res = wrap_sub_ap(ctx, r.reshape(x.shape()), ones);

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
NdArrayRef GenInvPermByBitVector(KernelEvalContext* ctx, const NdArrayRef& x) {
  SPU_ENFORCE(x.shape().ndim() == 1, "x should be 1-d");

  const auto field = x.eltype().as<AShrTy>()->field();
  const int64_t numel = x.numel();
  auto ones = ring_ones(field, x.shape());
  auto rev_x = wrap_sub_pa(ctx, ones, x);

  Shape new_shape = {1, numel};
  auto f = rev_x.reshape(new_shape).concatenate({x.reshape(new_shape)}, 1);
  auto s = f.clone();

  // calculate prefix sum
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _s(s);
    for (int64_t i = 1; i < s.numel(); ++i) {
      _s[i] += _s[i - 1];
    }
  });

  // mul f and s
  auto fs = wrap_mul_aa(ctx->sctx(), f, s);

  auto fs0 = fs.slice({0, 0}, {1, numel}, {});
  auto fs1 = fs.slice({0, numel}, {1, 2 * numel}, {});

  // calculate result
  auto r = wrap_add_aa(ctx->sctx(), fs0, fs1);
  auto res = wrap_sub_ap(ctx, r.reshape(x.shape()), ones);
  return res;
}

// The inverse of secure shuffle
NdArrayRef Unshuffle(KernelEvalContext* ctx, const NdArrayRef& x,
                     absl::Span<const int64_t> pv) {
  const auto lctx = ctx->lctx();
  NdArrayRef ret(x);

  auto inv_pv = genInversePerm(pv);
  for (int i = lctx->WorldSize() - 1; i >= 0; --i) {
    ret = SecureInvPerm(ctx, ret, i, inv_pv);
  }

  return ret;
}

// This is the inverse of InvShuffledPerm.
// The input is a shared inverse permutation <perm>, a permutation public_pv
// known to every parties, a locally generated permutation private_pv for secure
// unshuffle.
//
// The steps are as follows:
//   1) permute <perm> by public_pv as <sm>
//   2) secure unshuffle <sm> and  return results
//
// By doing InvShuffledPerm and UnshufflePerm, we get the shared inverse
// permutation of initial shared bit vectors.
NdArrayRef UnshufflePerm(KernelEvalContext* ctx, const NdArrayRef& perm,
                         absl::Span<const int64_t> public_pv,
                         absl::Span<const int64_t> private_pv) {
  auto sm = applyPerm(perm, public_pv);
  auto res = Unshuffle(ctx, sm, private_pv);
  return res;
}

// Generate shared inverse permutation by key
NdArrayRef GenInvPerm(KernelEvalContext* ctx, const NdArrayRef& key) {
  // key should be a 1-D tensor
  SPU_ENFORCE(key.shape().ndim() == 1, "key should be 1-d");
  const auto field = key.eltype().as<AShrTy>()->field();
  auto numel = key.numel();

  // 1. generate bit decomposition vector of key
  std::vector<NdArrayRef> v = GenBvVector(ctx, key);
  SPU_ENFORCE_GT(v.size(), 0U);

  // 2. generate natural permutation
  NdArrayRef s(key.eltype(), key.shape());
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _s(s);
    pforeach(0, numel, [&](int64_t idx) {
      _s[idx] = (ctx->lctx()->Rank() == 0) ? idx : 0;
    });
  });

  // 3. generate shared inverse permutation by bit vector and process
  PermVector random_perm;
  size_t v_size = v.size();
  size_t v_idx = 0;
  for (; v_idx < v_size - 1; v_idx += 2) {
    auto pv = genRandomPerm(s.shape()[0]);
    auto t =
        InvShuffledPerm(ctx, std::vector<NdArrayRef>{v[v_idx], v[v_idx + 1]}, s,
                        &random_perm, pv);
    auto perm = GenInvPermByTwoBitVectors(ctx, t[0], t[1]);
    s = UnshufflePerm(ctx, perm, random_perm, pv);
  }

  if (v_idx == v_size - 1) {
    auto pv = genRandomPerm(s.shape()[0]);
    auto t = InvShuffledPerm(ctx, v[v_idx], s, &random_perm, pv);
    auto perm = GenInvPermByBitVector(ctx, t);
    s = UnshufflePerm(ctx, perm, random_perm, pv);
  }

  return s;
}

// Apply inverse permutation on each tensor of x by a shared inverse permutation
// <perm>
std::vector<NdArrayRef> ApplyInvPerm(KernelEvalContext* ctx,
                                     absl::Span<NdArrayRef const> x,
                                     const NdArrayRef& perm) {
  // sanity check.
  SPU_ENFORCE(!x.empty(), "inputs should not be empty");
  SPU_ENFORCE(x[0].shape().ndim() == 1,
              "inputs should be 1-d but actually have {} dimensions",
              x[0].shape().ndim());
  SPU_ENFORCE(std::all_of(x.begin(), x.end(),
                          [&x](const NdArrayRef& input) {
                            return input.shape() == x[0].shape();
                          }),
              "inputs shape mismatched");

  // 1. <SP> = secure shuffle <perm>
  auto pv = genRandomPerm(x[0].shape()[0]);
  auto sp = Shuffle(ctx, perm, pv);

  // 2. <SX> = secure shuffle <x>
  auto sx = Shuffle(ctx, x, pv);

  // 3. M = reveal(<SP>)
  auto m = wrap_a2p(ctx->sctx(), sp);
  SPU_ENFORCE_EQ(m.shape().ndim(), 1U, "perm should be 1-d tensor");
  auto size = m.shape()[0];
  const auto field = m.eltype().as<Pub2kTy>()->field();
  PermVector perm_vector(size);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _m(m);
    pforeach(0, size,
             [&](int64_t idx) { perm_vector[idx] = (int64_t)_m[idx]; });
  });

  // 4. <T> = SP(<SX>)
  std::vector<NdArrayRef> v;

  for (size_t i = 0; i < sx.size(); ++i) {
    auto t = applyInvPerm(sx[i], perm_vector);
    v.emplace_back(std::move(t));
  }

  return v;
}

}  // namespace

// Radix sort
// Ref:
// https://eprint.iacr.org/2019/695.pdf
// in[0] is the key, each tensor of in is a 1-d tensor
std::vector<NdArrayRef> SimpleSortA::proc(
    KernelEvalContext* ctx, absl::Span<NdArrayRef const> in) const {
  SPU_ENFORCE(!in.empty(), "inputs should not be empty");

  auto perm = GenInvPerm(ctx, in[0]);
  auto res = ApplyInvPerm(ctx, in, perm);
  return res;
}

}  // namespace spu::mpc::semi2k