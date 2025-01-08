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

#include "libspu/mpc/shamir/boolean.h"

#include <algorithm>

#include "libspu/core/bit_utils.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/common/pv_gfmp.h"
#include "libspu/mpc/shamir/type.h"
#include "libspu/mpc/shamir/value.h"
#include "libspu/mpc/utils/gfmp.h"
#include "libspu/mpc/utils/gfmp_ops.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::shamir {

namespace {

size_t getNumBits(const NdArrayRef& in) {
  if (in.eltype().isa<PubGfmpTy>()) {
    const auto field = in.eltype().as<PubGfmpTy>()->field();
    return DISPATCH_ALL_FIELDS(field,
                               [&]() { return maxBitWidth<ring2k_t>(in); });
  } else if (in.eltype().isa<BShrTy>()) {
    return in.eltype().as<BShrTy>()->nbits();
  } else {
    SPU_THROW("should not be here, {}", in.eltype());
  }
}

NdArrayRef wrap_mul_aa(SPUContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) {
  return UnwrapValue(mul_aa(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_p2a(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(p2a(ctx, WrapValue(x)));
}

NdArrayRef wrap_a2p(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(a2p(ctx, WrapValue(x)));
}

NdArrayRef wrap_a2v(SPUContext* ctx, const NdArrayRef& x, size_t rank) {
  return UnwrapValue(a2v(ctx, WrapValue(x), rank));
}

NdArrayRef wrap_b2a(SPUContext* ctx, const NdArrayRef& x) {
  return UnwrapValue(b2a(ctx, WrapValue(x)));
}

}  // namespace

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  const size_t lhs_nbits = lhs.as<BShrTy>()->nbits();
  const size_t rhs_nbits = rhs.as<BShrTy>()->nbits();

  const size_t out_nbits = std::max(lhs_nbits, rhs_nbits);
  const auto field = lhs.as<BShrTy>()->field();
  ctx->pushOutput(makeType<BShrTy>(field, out_nbits));
}

NdArrayRef CastTypeB::proc(KernelEvalContext*, const NdArrayRef& in,
                           const Type& to_type) const {
  NdArrayRef out(to_type, in.shape());
  const size_t in_nbits = in.eltype().as<BShrTy>()->nbits();
  const size_t out_nbits = to_type.as<BShrTy>()->nbits();
  SPU_ENFORCE_GE(out_nbits, in_nbits);
  memset(out.data(), 0, out.numel() * out.elsize());
  // FIXME: optimize me, all the following things are memory copies
  for (int64_t idx = 0; idx < static_cast<int64_t>(in_nbits); ++idx) {
    auto in_i = getBitShare(in, idx);
    auto out_i = getBitShare(out, idx);
    ring_assign(out_i, in_i);
  }
  return out;
}

NdArrayRef B2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return wrap_a2p(ctx->sctx(), wrap_b2a(ctx->sctx(), in));
}

NdArrayRef P2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<PubGfmpTy>();
  const auto field = in_ty->field();

  return DISPATCH_ALL_FIELDS(field, [&]() {
    const size_t nbits = maxBitWidth<ring2k_t>(in);
    const auto out_ty = makeType<BShrTy>(field, nbits);
    NdArrayRef out(out_ty, in.shape());
    NdArrayView<ring2k_t> _in(in);

    std::vector<NdArrayRef> bits;
    for (size_t i = 0; i < nbits; ++i) {
      NdArrayRef bit_i_p(makeType<AShrTy>(field), in.shape());
      NdArrayView<ring2k_t> _bit_i_p(bit_i_p);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _bit_i_p[idx] = static_cast<ring2k_t>(_in[idx] >> i) & 1U;
      });
      bits.push_back(std::move(bit_i_p));
    }
    std::vector<NdArrayRef> bits_a;
    vmap(bits.begin(), bits.end(), std::back_inserter(bits_a),
         [ctx](const NdArrayRef& a) { return wrap_p2a(ctx->sctx(), a); });
    for (size_t i = 0; i < nbits; ++i) {
      NdArrayRef bit_i_b = getBitShare(out, i);
      ring_assign(bit_i_b, bits_a[i]);
    }
    return out;
  });
}

NdArrayRef B2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  return wrap_a2v(ctx->sctx(), wrap_b2a(ctx->sctx(), in), rank);
}

NdArrayRef AndBP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<PubGfmpTy>();
  auto b_field = lhs_ty->field();
  const size_t out_nbits = std::min(lhs_ty->nbits(), getNumBits(rhs));
  auto out_ty = makeType<BShrTy>(b_field, out_nbits);
  NdArrayRef out(out_ty, lhs.shape());
  DISPATCH_ALL_FIELDS(b_field, [&]() {
    using LT = ring2k_t;
    DISPATCH_ALL_FIELDS(rhs_ty->field(), [&]() {
      using RT = ring2k_t;
      for (size_t i = 0; i < out_nbits; ++i) {
        auto lhs_i = getBitShare(lhs, i);
        auto out_i = getBitShare(out, i);
        NdArrayView<LT> _lhs_i(lhs_i);
        NdArrayView<LT> _out_i(out_i);
        NdArrayView<RT> _rhs(rhs);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          _out_i[idx] =
              mul_mod(static_cast<LT>((_rhs[idx] >> i) & 1U), _lhs_i[idx]);
        });
      }
    });
  });
  return out;
}

NdArrayRef AndBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();
  SPU_ENFORCE_EQ(lhs_ty->field(), rhs_ty->field());
  auto b_field = lhs_ty->field();
  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_ty->nbits());
  auto out_ty = makeType<BShrTy>(b_field, out_nbits);
  NdArrayRef out(out_ty, lhs.shape());

  auto and_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
    return wrap_mul_aa(ctx->sctx(), a, b);
  };

  std::vector<NdArrayRef> lhs_bits;
  std::vector<NdArrayRef> rhs_bits;
  for (size_t i = 0; i < out_nbits; ++i) {
    lhs_bits.push_back(getBitShare(lhs, i));
    rhs_bits.push_back(getBitShare(rhs, i));
  }
  std::vector<NdArrayRef> tmp_out;
  vmap(lhs_bits.begin(), lhs_bits.end(), rhs_bits.begin(), rhs_bits.end(),
       std::back_inserter(tmp_out), and_lambda);
  for (size_t i = 0; i < out_nbits; ++i) {
    auto out_i = getBitShare(out, i);
    ring_assign(out_i, tmp_out[i]);
  }
  return out;
}

NdArrayRef XorBP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<PubGfmpTy>();
  auto b_field = lhs_ty->field();
  const size_t l_nbits = lhs_ty->nbits();
  const size_t r_nbits = getNumBits(rhs);
  const size_t out_nbits = std::max(l_nbits, r_nbits);
  const size_t common_nbits = std::min(l_nbits, r_nbits);
  auto out_ty = makeType<BShrTy>(b_field, out_nbits);
  NdArrayRef out(out_ty, lhs.shape());
  DISPATCH_ALL_FIELDS(b_field, [&]() {
    using LT = ring2k_t;
    DISPATCH_ALL_FIELDS(rhs_ty->field(), [&]() {
      using RT = ring2k_t;
      // calculate common bits
      for (size_t i = 0; i < common_nbits; ++i) {
        auto lhs_i = getBitShare(lhs, i);
        auto out_i = getBitShare(out, i);
        NdArrayView<LT> _lhs_i(lhs_i);
        NdArrayView<LT> _out_i(out_i);
        NdArrayView<RT> _rhs(rhs);
        // x ^ y = (x + y) - 2 * (x * y) for x,y in [0,1]
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          LT _x = _lhs_i[idx];
          LT _y = static_cast<LT>((_rhs[idx] >> i) & 1U);
          _out_i[idx] =
              add_mod(add_mod(_x, _y),
                      add_inv(mul_mod(static_cast<LT>(2), mul_mod(_x, _y))));
        });
      }
      // calculate the rest bits
      for (size_t i = common_nbits; i < out_nbits; ++i) {
        auto out_i = getBitShare(out, i);
        if (l_nbits > r_nbits) {
          auto more_bit_share = getBitShare(lhs, i);
          ring_assign(out_i, more_bit_share);
        } else {
          NdArrayView<LT> _out_i(out_i);
          NdArrayView<RT> _rhs(rhs);
          pforeach(0, lhs.numel(), [&](int64_t idx) {
            _out_i[idx] = static_cast<LT>((_rhs[idx] >> i) & 1U);
          });
        }
      }
    });
  });
  return out;
}

NdArrayRef XorBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  SPU_ENFORCE_EQ(lhs_ty->field(), rhs_ty->field());
  auto b_field = lhs_ty->field();
  const size_t l_nbits = lhs_ty->nbits();
  const size_t r_nbits = rhs_ty->nbits();
  const size_t out_nbits = std::max(l_nbits, r_nbits);
  const size_t common_nbits = std::min(l_nbits, r_nbits);
  auto out_ty = makeType<BShrTy>(b_field, out_nbits);
  NdArrayRef out(out_ty, lhs.shape());
  DISPATCH_ALL_FIELDS(b_field, [&]() {
    // calculate common bits
    auto xor_lambda = [ctx](const NdArrayRef& a, const NdArrayRef& b) {
      auto tmp = wrap_mul_aa(ctx->sctx(), a, b);
      NdArrayRef xor_ret(a.eltype(), a.shape());
      NdArrayView<ring2k_t> _xor_ret(xor_ret);
      NdArrayView<ring2k_t> _a(a);
      NdArrayView<ring2k_t> _b(b);
      NdArrayView<ring2k_t> _tmp(tmp);
      // x ^ y = (x + y) - 2 * (x * y) for x,y in [0,1]
      pforeach(0, a.numel(), [&](int64_t idx) {
        _xor_ret[idx] =
            add_mod(add_mod(_a[idx], _b[idx]),
                    add_inv(mul_mod(static_cast<ring2k_t>(2), _tmp[idx])));
      });
      return xor_ret;
    };

    std::vector<NdArrayRef> lhs_bits;
    std::vector<NdArrayRef> rhs_bits;
    for (size_t i = 0; i < common_nbits; ++i) {
      lhs_bits.push_back(getBitShare(lhs, i));
      rhs_bits.push_back(getBitShare(rhs, i));
    }
    std::vector<NdArrayRef> tmp_out;
    vmap(lhs_bits.begin(), lhs_bits.end(), rhs_bits.begin(), rhs_bits.end(),
         std::back_inserter(tmp_out), xor_lambda);
    for (size_t i = 0; i < common_nbits; ++i) {
      auto out_i = getBitShare(out, i);
      ring_assign(out_i, tmp_out[i]);
    }
    // calculate the rest bits
    for (size_t i = common_nbits; i < out_nbits; ++i) {
      auto more_bit_share =
          l_nbits > r_nbits ? getBitShare(lhs, i) : getBitShare(rhs, i);
      auto out_i = getBitShare(out, i);
      ring_assign(out_i, more_bit_share);
    }
  });
  return out;
}

NdArrayRef LShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         const Sizes& bits) const {
  // Fixme
  SPU_ENFORCE(bits.size() == 1);
  if (bits[0] == 0) {
    return in;
  }
  const auto* in_ty = in.eltype().as<BShrTy>();
  const auto b_field = in_ty->field();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::min<size_t>(
      in_ty->nbits() + *std::max_element(bits.begin(), bits.end()),
      GetMersennePrimeExp(field));

  auto out_ty = makeType<BShrTy>(b_field, out_nbits);
  NdArrayRef out(out_ty, in.shape());
  memset(out.data(), 0, out.numel() * out.elsize());
  auto shift = bits[0];
  // FIXME: optimize me, all the following things are memory copies
  for (int64_t idx = 0; idx < static_cast<int64_t>(out_nbits) - shift; ++idx) {
    auto in_i = getBitShare(in, idx);
    auto out_i = getBitShare(out, idx + shift);
    ring_assign(out_i, in_i);
  }
  return out;
}

NdArrayRef RShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                         const Sizes& bits) const {
  // Fixme
  SPU_ENFORCE(bits.size() == 1);
  if (bits[0] == 0) {
    return in;
  }
  const auto* in_ty = in.eltype().as<BShrTy>();
  int64_t out_nbits = in_ty->nbits();
  out_nbits -= std::min(out_nbits, *std::min_element(bits.begin(), bits.end()));
  const auto b_field = in_ty->field();

  auto out_ty = makeType<BShrTy>(b_field, out_nbits);
  NdArrayRef out(out_ty, in.shape());
  auto shift = bits[0];
  // FIXME: optimize me, all the following things are memory copies
  for (int64_t idx = 0; idx < static_cast<int64_t>(out_nbits); ++idx) {
    auto in_i = getBitShare(in, idx + shift);
    auto out_i = getBitShare(out, idx);
    ring_assign(out_i, in_i);
  }
  return out;
}

NdArrayRef ARShiftB::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                          const Sizes& bits) const {
  // Fixme
  SPU_ENFORCE(bits.size() == 1);
  if (bits[0] == 0) {
    return in;
  }

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();

  // assume the nbits should not be greater than field bits
  SPU_ENFORCE(in_ty->nbits() <= GetMersennePrimeExp(field),
              "in.type={}, field={}", in.eltype(), field);
  const int64_t out_nbits = in_ty->nbits();
  const auto b_field = in_ty->field();

  auto out_ty = makeType<BShrTy>(b_field, out_nbits);
  NdArrayRef out(out_ty, in.shape());
  auto shift = bits[0];
  // FIXME: optimize me, all the following things are memory copies
  for (int64_t idx = 0; idx < static_cast<int64_t>(out_nbits) - shift; ++idx) {
    auto in_i = getBitShare(in, idx + shift);
    auto out_i = getBitShare(out, idx);
    ring_assign(out_i, in_i);
  }
  // fill the rest as msb
  NdArrayRef msb = getBitShare(in, in_ty->nbits() - 1);
  if (in_ty->nbits() < GetMersennePrimeExp(field)) {
    msb = ring_zeros(b_field, in.shape());
  }
  for (int64_t idx = static_cast<int64_t>(out_nbits) - shift; idx < out_nbits;
       ++idx) {
    auto out_i = getBitShare(out, idx);
    ring_assign(out_i, msb);
  }
  return out;
}

NdArrayRef BitrevB::proc(KernelEvalContext*, const NdArrayRef& in, size_t start,
                         size_t end) const {
  SPU_ENFORCE(start <= end && end <= 128);

  const auto* in_ty = in.eltype().as<BShrTy>();
  if (start > in_ty->nbits() || start == end) {
    return in;
  }
  const size_t out_nbits = std::max(in_ty->nbits(), end);
  const auto b_field = in_ty->field();
  const auto out_ty = makeType<BShrTy>(b_field, out_nbits);
  NdArrayRef out(out_ty, in.shape());
  memset(out.data(), 0, out.numel() * out.elsize());
  for (size_t i = 0; i < start; ++i) {
    auto src = getBitShare(in, i);
    auto dst = getBitShare(out, i);
    ring_assign(dst, src);
  }
  for (size_t i = end; i < in_ty->nbits(); ++i) {
    auto src = getBitShare(in, i);
    auto dst = getBitShare(out, i);
    ring_assign(dst, src);
  }
  for (size_t i = 0; i + start < end; ++i) {
    auto src = getBitShare(in, i + start);
    auto dst = getBitShare(out, end - 1 - i);
    ring_assign(dst, src);
  }
  return out;
}

NdArrayRef BitIntlB::proc(KernelEvalContext*, const NdArrayRef& in,
                          size_t stride) const {
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t nbits = in_ty->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  int64_t offset = 1 << stride;
  int64_t half_bits = nbits / 2;
  int64_t idx = 0;
  for (int64_t i = 0; i < half_bits; i += offset) {
    for (int j = 0; j < offset; ++j) {
      auto src = getBitShare(in, i + j);
      auto dst = getBitShare(out, idx++);
      ring_assign(dst, src);
    }
    for (int j = 0; j < offset; ++j) {
      auto src = getBitShare(in, i + j + half_bits);
      auto dst = getBitShare(out, idx++);
      ring_assign(dst, src);
    }
  }
  SPU_ENFORCE_EQ(idx, static_cast<int64_t>(nbits));
  return out;
}

NdArrayRef BitDeintlB::proc(KernelEvalContext*, const NdArrayRef& in,
                            size_t stride) const {
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t nbits = in_ty->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  int64_t offset = 1 << stride;
  int64_t half_bits = nbits / 2;
  int64_t idx = 0;
  while (idx < half_bits) {
    for (int j = 0; j < offset; ++j) {
      auto src = getBitShare(in, idx + j);
      auto dst = getBitShare(out, idx);
      ring_assign(dst, src);
    }
    for (int j = 0; j < offset; ++j) {
      auto src = getBitShare(in, idx + j);
      auto dst = getBitShare(out, idx + half_bits);
      ring_assign(dst, src);
    }
    idx += offset;
  }
  SPU_ENFORCE_EQ(idx, static_cast<int64_t>(nbits));
  return out;
}

}  // namespace spu::mpc::shamir
