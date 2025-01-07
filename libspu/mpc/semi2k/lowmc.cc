// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/semi2k/lowmc.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/lowmc.h"
#include "libspu/mpc/utils/lowmc_utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

namespace {

NdArrayRef wrap_xor_bp(SPUContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(xor_bp(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_xor_bb(SPUContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(xor_bb(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_and_bb(SPUContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(and_bb(ctx, WrapValue(x), WrapValue(y)));
}

/// Some shape utils
NdArrayRef extract_bit_arr(const NdArrayRef& in, int64_t idx) {
  const auto field = in.eltype().as<BShrTy>()->field();
  SPU_ENFORCE((uint64_t)idx < SizeOf(field) * 8, "bit extract out of range.");
  const auto bty = makeType<BShrTy>(field, 1);

  NdArrayRef out(bty, in.shape());
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _in(in);
    NdArrayView<ring2k_t> _out(out);

    pforeach(0, in.numel(), [&](int64_t i) {  //
      _out[i] = (_in[i] >> idx) & 1;
    });
  });

  return out;
}

// offset=0 means c, offset=2 means a
NdArrayRef extract_packed_bit_arr(const NdArrayRef& state, int64_t n_boxes,
                                  int64_t offset) {
  const auto field = state.eltype().as<BShrTy>()->field();
  const auto bty = makeType<BShrTy>(field, 1);

  const auto& ori_shape = state.shape();
  const auto ori_numel = ori_shape.numel();
  Shape to_shape = ori_shape;
  to_shape[0] = ori_shape[0] * n_boxes;

  NdArrayRef ret(bty, to_shape);
  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _state(state);
    NdArrayView<ring2k_t> _ret(ret);

    for (int64_t i = 0; i < n_boxes; ++i) {
      const auto start_idx = 3 * i;
      pforeach(0, ori_shape.numel(), [&](int64_t idx) {  //
        _ret[idx + i * ori_numel] = (_state[idx] >> (start_idx + offset)) & 1;
      });
    }
  });

  return ret;
}

// do memory copying by hand, get packed (abc, bca)
std::tuple<NdArrayRef, NdArrayRef> construct_concat_arr(const NdArrayRef& state,
                                                        int64_t n_boxes) {
  const auto field = state.eltype().as<BShrTy>()->field();
  const auto bty = makeType<BShrTy>(field, 3);

  const auto& ori_shape = state.shape();
  const auto ori_numel = ori_shape.numel();
  Shape to_shape = ori_shape;
  to_shape[0] = ori_shape[0] * n_boxes;

  NdArrayRef abc(bty, to_shape);
  NdArrayRef bca(bty, to_shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _state(state);
    NdArrayView<ring2k_t> _abc(abc);
    NdArrayView<ring2k_t> _bca(bca);

    for (int64_t i = 0; i < n_boxes; ++i) {
      const auto start_idx = 3 * i;
      pforeach(0, ori_shape.numel(), [&](int64_t idx) {
        // xxxx xabc => 0000 0abc
        _abc[idx + i * ori_numel] = (_state[idx] >> start_idx) & 7;
        // xxxx xabc => 0000 0bca
        _bca[idx + i * ori_numel] = (((_state[idx] >> start_idx) & 3) << 1) |
                                    ((_state[idx] >> (start_idx + 2)) & 1);
      });
    }
  });

  return std::make_tuple(abc, bca);
}

// for shape (k * n0, n1, ...),
// get array with shape = (n0, n1, ...)
NdArrayRef slice_arr(const NdArrayRef& x, int64_t idx, const Shape& ori_shape) {
  const auto& whole_shape = x.shape();
  SPU_ENFORCE(ori_shape.ndim() == whole_shape.ndim(), "axis mismatch.");
  SPU_ENFORCE(std::equal(whole_shape.begin() + 1, whole_shape.end(),
                         ori_shape.begin() + 1),
              "mismatch of shape.");

  // compute slice indices
  Index start_ind(ori_shape.ndim(), 0);
  start_ind[0] = idx * ori_shape[0];
  Index end_ind(ori_shape.begin(), ori_shape.end());
  end_ind[0] = start_ind[0] + ori_shape[0];

  return x.slice(start_ind, end_ind, {});
}

/// Some core operations for LowMC layer
NdArrayRef Sbox(KernelEvalContext* ctx, const NdArrayRef& state,
                int64_t n_boxes, size_t n_bits) {
  // for SboxLayer, the initial definition is a look-up table, we use some
  // logical operations to replace it.
  // i.e. Sbox(a, b, c) = (a + b * c, a + b + a * c, a + b + c + a * b),
  // where `+` is XOR, `*` is AND
  // TODO: Lots of memory copying here to save rounds, use FM8 for temporay
  // a,b,c to save memory
  NdArrayRef abc_arr;
  NdArrayRef bca_arr;
  // the origin data: ... a2b2c2 a1b1c1 a0b0c0
  // we concat all abc to get [a2b2c2; a1b1c1; a0b0c0]
  // we concat all bca to get [b2c2a2; b1c1a1; b0c0a0]
  std::tie(abc_arr, bca_arr) = construct_concat_arr(state, n_boxes);

  // doing all expensive secret and op simultaneously
  auto abc_and_bca_arr = wrap_and_bb(ctx->sctx(), abc_arr, bca_arr);
  auto abc_xor_bca_arr = wrap_xor_bb(ctx->sctx(), abc_arr, bca_arr);

  // extract all ab, bc, ac
  auto ab_arr = extract_bit_arr(abc_and_bca_arr, 2);
  auto bc_arr = extract_bit_arr(abc_and_bca_arr, 1);
  auto ac_arr = extract_bit_arr(abc_and_bca_arr, 0);

  // extract a+b, b+c
  auto a_b_arr = extract_bit_arr(abc_xor_bca_arr, 2);
  auto b_c_arr = extract_bit_arr(abc_xor_bca_arr, 1);

  // extract a
  auto a_arr = extract_packed_bit_arr(state, n_boxes, 2);

  // a + b * c
  auto new_a = wrap_xor_bb(ctx->sctx(), a_arr, bc_arr);
  // a + b + a * c
  auto new_b = wrap_xor_bb(ctx->sctx(), a_b_arr, ac_arr);
  // a + b + c + a * b
  auto a_b_c_arr = wrap_xor_bb(ctx->sctx(), b_c_arr, a_arr);
  auto new_c = wrap_xor_bb(ctx->sctx(), a_b_c_arr, ab_arr);

  std::vector<NdArrayRef> bits_arr;
  bits_arr.reserve(n_bits);
  const auto& ori_shape = state.shape();
  // collect first 3*n_boxes bits
  for (int64_t i = 0; i < n_boxes; ++i) {
    bits_arr.push_back(slice_arr(new_c, i, ori_shape));
    bits_arr.push_back(slice_arr(new_b, i, ori_shape));
    bits_arr.push_back(slice_arr(new_a, i, ori_shape));
  }

  // concat all bits
  const auto field = state.eltype().as<BShrTy>()->field();
  auto ret = ring_zeros(field, state.shape()).as(state.eltype());

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> _ret(ret);
    NdArrayView<ring2k_t> _state(state);

    for (int64_t i = 0; i < 3 * n_boxes; ++i) {
      NdArrayView<ring2k_t> _tmp(bits_arr[i]);

      pforeach(0, ret.numel(), [&](int64_t idx) {  //
        _ret[idx] = _ret[idx] | ((_tmp[idx] & 1) << i);
      });
    }

    // The rest higher bits stay unchanged in SBoxLayer, so we copy them
    pforeach(0, ret.numel(), [&](int64_t idx) {  //
      _ret[idx] = _ret[idx] | ((_state[idx] >> (3 * n_boxes)) << (3 * n_boxes));
    });
  });

  return ret;
}

NdArrayRef Affine(KernelEvalContext* ctx, const LowMC& cipher,
                  const NdArrayRef& state, int64_t rounds) {
  const auto field = state.eltype().as<BShrTy>()->field();

  const auto L_matrix = cipher.Lmat()[rounds];
  return dot_product_gf2(L_matrix, state, field);
}

}  // namespace

NdArrayRef LowMcB::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* prg_state = ctx->getState<PrgState>();

  // generate the shared key, key0 ^ key1 = key
  uint128_t key;
  prg_state->fillPriv(absl::MakeSpan(&key, 1));

  // generate public seed
  uint128_t seed;
  prg_state->fillPubl(absl::MakeSpan(&seed, 1));

  return encrypt(ctx, in, key, seed);
}

NdArrayRef LowMcB::encrypt(KernelEvalContext* ctx, const NdArrayRef& in,
                           uint128_t key, uint128_t seed) const {
  const auto field = in.eltype().as<BShrTy>()->field();
  const auto numel = in.numel();
  const auto k = SizeOf(field) * 8;
  const auto shape = in.shape();
  const auto pub_ty = makeType<Pub2kTy>(field);

  NdArrayRef out;
  DISPATCH_ALL_FIELDS(field, [&]() {
    auto d = get_data_complexity(numel);
    auto cipher = LowMC(field, seed, d);
    SPU_ENFORCE(static_cast<int64_t>(k) == cipher.data_block_size(),
                "block size must be equal now.");

    // generate round keys
    auto round_keys =
        generate_round_keys(cipher.Kmat(), key, cipher.rounds(), field);

    // Following the same steps as in plaintext, with MPC primitives for bit
    // operations.
    //
    // 1. key whiten: state = in ^ roundKeys[0]
    auto round_key0 = round_keys[0].broadcast_to(shape, {}).as(pub_ty);
    out = wrap_xor_bb(ctx->sctx(), in, round_key0);

    // 2. round loop: for i = 1 to r
    // state = SboxLayer(state)
    // state = GF2Dot(Lmatrix[i-1], state)
    // state = state ^ RoundConstants[i-1]
    // state = state ^ RoundKeys[i]
    const auto n_boxes = cipher.number_of_boxes();
    SPU_ENFORCE((int64_t)k >= 3 * n_boxes, "invalid parameters setting.");

    for (int64_t r = 1; r <= cipher.rounds(); ++r) {
      // The only Non Linear Layer in LowMC
      out = Sbox(ctx, out, n_boxes, k);

      out = Affine(ctx, cipher, out, /*round idx*/ r - 1).as(in.eltype());

      auto round_constant =
          cipher.RoundConstants()[r - 1].broadcast_to(shape, {}).as(pub_ty);
      out = wrap_xor_bp(ctx->sctx(), out, round_constant);

      auto round_key = round_keys[r].broadcast_to(shape, {}).as(pub_ty);
      out = wrap_xor_bb(ctx->sctx(), out, round_key);
    }
  });

  return out;
}

namespace {
NdArrayRef wrap_lowmcb(KernelEvalContext* ctx, const NdArrayRef& in) {
  return LowMcB().proc(ctx, in);
}

FieldType get_dst_field(const int64_t k) {
  if (k <= 32) {
    return FM32;
  } else if (k <= 64) {
    return FM64;
  } else {
    // no matther how large k is, we always use FM128.
    return FM128;
  }
}

NdArrayRef concate_bits(const std::vector<NdArrayRef>& inputs,
                        const FieldType dst_field) {
  const auto field = inputs[0].eltype().as<Ring2k>()->field();
  const auto k = SizeOf(field) * 8;

  SPU_ENFORCE(k * inputs.size() <= SizeOf(dst_field) * 8,
              "too much inputs to concat!");

  auto ret = ring_zeros(dst_field, inputs[0].shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using src_el_t = ring2k_t;

    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dst_el_t = ring2k_t;
      NdArrayView<dst_el_t> _ret(ret);

      for (uint64_t i = 0; i < inputs.size(); ++i) {
        NdArrayView<src_el_t> _inp(inputs[i]);
        const auto shift_bits = k * i;

        pforeach(0, ret.numel(), [&](int64_t idx) {  //
          _ret[idx] |= (static_cast<dst_el_t>(_inp[idx]) << shift_bits);
        });
      }
    });
  });

  return ret;
}

}  // namespace

NdArrayRef MultiKeyLowMcB::proc(KernelEvalContext* ctx,
                                const std::vector<NdArrayRef>& inputs) const {
  SPU_ENFORCE(!inputs.empty());
  const auto field = inputs[0].eltype().as<Ring2k>()->field();
  SPU_ENFORCE(std::all_of(inputs.begin() + 1, inputs.end(),
                          [&field](const NdArrayRef& v) {
                            return v.eltype().as<Ring2k>()->field() == field;
                          }),
              "all inputs must have the same field");

  if (inputs.size() == 1) {
    return wrap_lowmcb(ctx, inputs[0]);
  }

  // SPU can now only native support FM128.
  static constexpr int64_t kMaxBits = 128;
  static constexpr FieldType kMaxField = FM128;

  const int64_t k = SizeOf(field) * 8;
  const auto total_bits = k * inputs.size();

  if (total_bits <= kMaxBits) {
    // just concat all bits if SPU can handle it.
    const auto dst_field = get_dst_field(total_bits);
    auto concat_inp =
        concate_bits(inputs, dst_field).as(makeType<BShrTy>(dst_field));
    return wrap_lowmcb(ctx, concat_inp);
  } else {
    // re-mapping to FM128
    auto* prg_state = ctx->getState<PrgState>();
    const Shape rand_mat_shape = {kMaxBits};
    auto remapping_inp = ring_zeros(kMaxField, inputs[0].shape());
    // e.g. inputs = [x0, x1, x2, x3], each xi is 64 bits, we want to remap
    // these to 128 bits.
    // Conceptually, we generate a public random binary matrix M (shape = (128,
    // 64*4)), compute gf2dot(M, inputs), which is 128 bits output.
    for (const auto& item : inputs) {
      // logically, (128, k) binary matrix
      const auto rand_mat = prg_state->genPubl(field, rand_mat_shape);
      // split the large gf2dot into several small gf2dot and use xor to combine
      // them.
      auto part_dot = dot_product_gf2(rand_mat, item, kMaxField);
      ring_xor_(remapping_inp, part_dot);
    }
    return wrap_lowmcb(ctx, remapping_inp.as(makeType<BShrTy>(kMaxField)));
  }
}

}  // namespace spu::mpc::semi2k
