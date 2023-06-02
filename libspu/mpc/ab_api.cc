// Copyright 2021 Ant Group Co., Ltd.
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

#include "libspu/mpc/ab_api.h"

#include "libspu/core/bit_utils.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/utils/tiling_util.h"

namespace spu::mpc {

#define FORCE_DISPATCH(CTX, ...)                      \
  {                                                   \
    SPU_TRACE_MPC_LEAF(CTX, __VA_ARGS__);             \
    return dynDispatch((CTX), __func__, __VA_ARGS__); \
  }

#define TRY_NAMED_DISPATCH(CTX, FNAME, ...)        \
  if ((CTX)->hasKernel(__func__)) {                \
    SPU_TRACE_MPC_LEAF(CTX, __VA_ARGS__);          \
    return dynDispatch((CTX), FNAME, __VA_ARGS__); \
  }

#define TRY_DISPATCH(CTX, ...) TRY_NAMED_DISPATCH(CTX, __func__, __VA_ARGS__)

template <typename... Args>
Value tiledDynDispatch(const std::string& fn_name, SPUContext* ctx,
                       Args&&... args) {
  auto impl = [fn_name](SPUContext* sh_ctx, Args&&... sh_args) {
    return dynDispatch(sh_ctx, fn_name, std::forward<Args>(sh_args)...);
  };

  return tiled(impl, ctx, std::forward<Args>(args)...);
}

#define TILED_DISPATCH(CTX, ...)                           \
  {                                                        \
    SPU_TRACE_MPC_LEAF(ctx, __VA_ARGS__);                  \
    return tiledDynDispatch(__func__, (CTX), __VA_ARGS__); \
  }

// TODO: now we handcode mark some of the functions as tiled dispatch according
// to experience.
// Note: now tracing is done just before dynamic dispatch, so the time record is
// OK no matter if tiling is enabled.

Value a2p(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value p2a(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value a2v(SPUContext* ctx, const Value& x, size_t owner) {
  // Note: Private is not mandatory for now, it's the protocol author's
  // responsibility to decide if private should be supported. He/she can:
  // 1. Do not support private from IO interface, then all computation should be
  //    in Secret/Public domain, Private related kernel will never be
  //    dispatched.
  // 2. Support private in IO interface, then the private computation/conversion
  //    interface should also be supported.
  FORCE_DISPATCH(ctx, x, owner);
}

Value v2a(SPUContext* ctx, const Value& x) {
  // Note: it's the protocol author's responsibility to ensure private is
  // supported
  FORCE_DISPATCH(ctx, x);
}

Value msb_a2b(SPUContext* ctx, const Value& x) { TILED_DISPATCH(ctx, x); }

Value rand_a(SPUContext* ctx, const Shape& shape) {
  FORCE_DISPATCH(ctx, shape);
}

Value rand_b(SPUContext* ctx, const Shape& shape) {
  FORCE_DISPATCH(ctx, shape);
}

Value not_a(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value add_ap(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value equal_aa(SPUContext* ctx, const Value& x, const Value& y) {
  TILED_DISPATCH(ctx, x, y);
}

Value equal_ap(SPUContext* ctx, const Value& x, const Value& y) {
  TILED_DISPATCH(ctx, x, y);
}

Value add_aa(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

OptionalAPI<Value> add_av(SPUContext* ctx, const Value& x, const Value& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

Value mul_ap(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value mul_aa(SPUContext* ctx, const Value& x, const Value& y) {
  TILED_DISPATCH(ctx, x, y);
}

OptionalAPI<Value> mul_av(SPUContext* ctx, const Value& x, const Value& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

Value mul_a1b(SPUContext* ctx, const Value& x, const Value& y) {
  TILED_DISPATCH(ctx, x, y);
}

Value lshift_a(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value trunc_a(SPUContext* ctx, const Value& x, size_t nbits) {
  TILED_DISPATCH(ctx, x, nbits);
}

Value mmul_ap(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k) {
  FORCE_DISPATCH(ctx, x, y, m, n, k);
}

Value mmul_aa(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k) {
  FORCE_DISPATCH(ctx, x, y, m, n, k);
}

OptionalAPI<Value> mmul_av(SPUContext* ctx, const Value& x, const Value& y,
                           size_t m, size_t n, size_t k) {
  TRY_DISPATCH(ctx, x, y, m, n, k);
  return NotAvailable;
}

Type common_type_b(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_MPC_LEAF(ctx, a, b);
  return dynDispatch<Type>(ctx, __func__, a, b);
}

Value cast_type_b(SPUContext* ctx, const Value& a, const Type& to_type) {
  FORCE_DISPATCH(ctx, a, to_type);
}

Value b2p(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value p2b(SPUContext* ctx, const Value& x) { FORCE_DISPATCH(ctx, x); }

Value b2v(SPUContext* ctx, const Value& x, size_t owner) {
  FORCE_DISPATCH(ctx, x, owner);
}

Value a2b(SPUContext* ctx, const Value& x) { TILED_DISPATCH(ctx, x); }

Value b2a(SPUContext* ctx, const Value& x) { TILED_DISPATCH(ctx, x); }

Value and_bp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value and_bb(SPUContext* ctx, const Value& x, const Value& y) {
  TILED_DISPATCH(ctx, x, y);
}

OptionalAPI<Value> and_bv(SPUContext* ctx, const Value& x, const Value& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

Value xor_bp(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

Value xor_bb(SPUContext* ctx, const Value& x, const Value& y) {
  FORCE_DISPATCH(ctx, x, y);
}

OptionalAPI<Value> xor_bv(SPUContext* ctx, const Value& x, const Value& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

Value lshift_b(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value rshift_b(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value arshift_b(SPUContext* ctx, const Value& x, size_t nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

Value bitrev_b(SPUContext* ctx, const Value& x, size_t start, size_t end) {
  FORCE_DISPATCH(ctx, x, start, end);
}

static inline size_t numBits(const Value& in) {
  return in.storage_type().as<BShare>()->nbits();
}

static inline Value setNumBits(const Value& in, size_t nbits) {
  Value out = in;
  out.storage_type().as<BShare>()->setNbits(nbits);
  return out;
}

// TODO: we can not ref api.h, circular reference
static Value hack_make_p(SPUContext* ctx, uint128_t init,
                         const std::vector<int64_t>& shape) {
  return dynDispatch(ctx, "make_p", init, Shape(shape));
}

Value bitintl_b(SPUContext* ctx, const Value& x, size_t stride) {
  TRY_DISPATCH(ctx, x, stride);

  // default implementation.
  // algorithm:
  //      0000000011111111
  // swap     ^^^^^^^^
  //      0000111100001111
  // swap   ^^^^    ^^^^
  //      0011001100110011
  // swap  ^^  ^^  ^^  ^^
  //      0101010101010101
  const size_t nbits = x.storage_type().as<BShare>()->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  Value out;
  for (int64_t idx = Log2Ceil(nbits) - 2; idx >= static_cast<int64_t>(stride);
       idx--) {
    auto K = hack_make_p(ctx, spu::detail::kBitIntlKeepMasks[idx], x.shape());
    auto M = hack_make_p(ctx, spu::detail::kBitIntlSwapMasks[idx], x.shape());
    int64_t S = 1 << idx;
    // out = (out & K) ^ ((out >> S) & M) ^ ((out & M) << S);
    out = xor_bb(
        ctx,
        xor_bb(ctx, and_bp(ctx, out, K), and_bp(ctx, rshift_b(ctx, out, S), M)),
        lshift_b(ctx, and_bp(ctx, out, M), S));
  }
  out = setNumBits(out, numBits(x));
  return out;
}

Value bitdeintl_b(SPUContext* ctx, const Value& x, size_t stride) {
  TRY_DISPATCH(ctx, x, stride);

  // algorithm:
  //      0101010101010101
  // swap  ^^  ^^  ^^  ^^
  //      0011001100110011
  // swap   ^^^^    ^^^^
  //      0000111100001111
  // swap     ^^^^^^^^
  //      0000000011111111
  const size_t nbits = x.storage_type().as<BShare>()->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  Value out;
  for (int64_t idx = stride; idx + 1 < Log2Ceil(nbits); idx++) {
    auto K = hack_make_p(ctx, spu::detail::kBitIntlKeepMasks[idx], x.shape());
    auto M = hack_make_p(ctx, spu::detail::kBitIntlSwapMasks[idx], x.shape());
    int64_t S = 1 << idx;
    // out = (out & K) ^ ((out >> S) & M) ^ ((out & M) << S);
    out = xor_bb(
        ctx,
        xor_bb(ctx, and_bp(ctx, out, K), and_bp(ctx, rshift_b(ctx, out, S), M)),
        lshift_b(ctx, and_bp(ctx, out, M), S));
  }
  out = setNumBits(out, numBits(x));
  return out;
}

namespace {

// TODO: move this to RuntimeConfig
enum class CircuitType {
  KoggeStone,
  Sklansky,
  Count,
};

// The kogge-stone adder.
//
// P stands for propagate, G stands for generate, where:
//  (G0, P0) = (g0, p0)
//  (Gi, Pi) = (gi, pi) o (Gi-1, Pi-1)
//
// The `o` here is:
//  (G0, P0) o (G1, P1) = (G0 ^ (P0 & G1), P0 & P1)
//
// Latency log(k) + 1
Value ppa_kogge_stone(SPUContext* ctx, const Value& lhs, const Value& rhs,
                      size_t nbits) {
  // Generate p & g.
  auto P = xor_bb(ctx, lhs, rhs);
  auto G = and_bb(ctx, lhs, rhs);

  for (int idx = 0; idx < Log2Ceil(nbits); ++idx) {
    const size_t offset = 1UL << idx;
    auto G1 = lshift_b(ctx, G, offset);
    auto P1 = lshift_b(ctx, P, offset);

    // P1 = P & P1
    // G1 = G ^ (P & G1)
    std::vector<Value> res = spu::vectorize(
        {P, P}, {P1, G1},
        [&](const Value& xx, const Value& yy) { return and_bb(ctx, xx, yy); });
    P = std::move(res[0]);
    G = xor_bb(ctx, G, res[1]);
  }

  // out = (G << 1) ^ p0
  auto C = lshift_b(ctx, G, 1);
  return xor_bb(ctx, xor_bb(ctx, lhs, rhs), C);
}

std::pair<Value, Value> bit_scatter(SPUContext* ctx, const Value& in,
                                    size_t stride) {
  // TODO: use faster bit scatter implementation for ABY3
  const size_t nbits = numBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits), "unsupported {}", nbits);
  auto out = bitdeintl_b(ctx, in, stride);

  auto hi = rshift_b(ctx, out, nbits / 2);
  auto mask = hack_make_p(ctx, (static_cast<uint128_t>(1) << (nbits / 2)) - 1,
                          in.shape());
  auto lo = and_bp(ctx, out, mask);
  return std::make_pair(hi, lo);
}

Value bit_gather(SPUContext* ctx, const Value& hi, const Value& lo,
                 size_t stride) {
  const size_t nbits = numBits(hi);
  SPU_ENFORCE(absl::has_single_bit(nbits), "unsupported {}", nbits);
  SPU_ENFORCE(nbits == numBits(lo), "nbits mismatch {}, {}", nbits,
              numBits(lo));

  auto out = xor_bb(ctx, lshift_b(ctx, hi, nbits), lo);
  return bitintl_b(ctx, out, stride);
}

// The sklansky adder.
Value ppa_sklansky(SPUContext* ctx, Value const& lhs, Value const& rhs,
                   size_t nbits) {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  constexpr std::array<uint128_t, 7> kSelMask = {{
      yacl::MakeUint128(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),  // invalid
      yacl::MakeUint128(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA),  // 10101010
      yacl::MakeUint128(0x8888888888888888, 0x8888888888888888),  // 10001000
      yacl::MakeUint128(0x8080808080808080, 0x8080808080808080),  // 10000000
      yacl::MakeUint128(0x8000800080008000, 0x8000800080008000),  // ...
      yacl::MakeUint128(0x8000000080000000, 0x8000000080000000),  // ...
      yacl::MakeUint128(0x8000000000000000, 0x8000000000000000),  // ...
  }};

  // Generate P & G.
  auto P = xor_bb(ctx, lhs, rhs);
  auto G = and_bb(ctx, lhs, rhs);

  const size_t bit_width = numBits(lhs);
  SPU_ENFORCE(bit_width == numBits(rhs), "nbits mismatch {}, {}", bit_width,
              numBits(rhs));
  for (int idx = 0; idx < Log2Ceil(nbits); ++idx) {
    auto [Ph, Pl] = bit_scatter(ctx, P, idx);
    auto [Gh, Gl] = bit_scatter(ctx, G, idx);
    // SPU_ENFORCE(numBits(Ph) == bit_width / 2);
    // SPU_ENFORCE(numBits(Pl) == bit_width / 2);
    // SPU_ENFORCE(numBits(Gh) == bit_width / 2);
    // SPU_ENFORCE(numBits(Gl) == bit_width / 2);

    const auto s_mask = hack_make_p(ctx, kSelMask[idx], lhs.shape());
    auto Gs = and_bp(ctx, Gl, s_mask);
    auto Ps = and_bp(ctx, Pl, s_mask);
    for (int j = 0; j < idx; j++) {
      Gs = xor_bb(ctx, Gs, rshift_b(ctx, Gs, 1 << j));
      Ps = xor_bb(ctx, Ps, rshift_b(ctx, Ps, 1 << j));
    }
    // SPU_ENFORCE(numBits(Ps) == bit_width / 2);
    // SPU_ENFORCE(numBits(Gs) == bit_width / 2);

    // Ph = Ph & Ps
    // Gh = Gh ^ (Ph & Gs)
    std::vector<Value> PG = spu::vectorize(
        {Ph, Ph}, {Ps, Gs},
        [&](const Value& xx, const Value& yy) { return and_bb(ctx, xx, yy); });
    Ph = std::move(PG[0]);
    Gh = xor_bb(ctx, Gh, PG[1]);
    // SPU_ENFORCE(numBits(Gh) == numBits(G) / 2);
    // SPU_ENFORCE(numBits(Ph) == numBits(P) / 2);

    P = bit_gather(ctx, Ph, Pl, idx);
    G = bit_gather(ctx, Gh, Gl, idx);
  }

  // out = (G0 << 1) ^ p0
  auto C = lshift_b(ctx, G, 1);
  return xor_bb(ctx, xor_bb(ctx, lhs, rhs), C);
}

}  // namespace

Value add_bb(SPUContext* ctx, const Value& x, const Value& y) {
  // TRY_DISPATCH
  if (ctx->hasKernel(__func__)) {
    SPU_TRACE_MPC_LEAF(ctx, x, y);
    return tiledDynDispatch(__func__, ctx, x, y);
  }

  // default implementation
  SPU_TRACE_MPC_LEAF(ctx, x, y);

  const size_t nbits = numBits(x);
  SPU_ENFORCE(nbits == numBits(y), "nbits mismatch {}!={}", nbits, numBits(y));

  auto type = CircuitType::KoggeStone;

  switch (type) {
    case CircuitType::KoggeStone:
      return ppa_kogge_stone(ctx, x, y, nbits);
    case CircuitType::Sklansky:
      return ppa_sklansky(ctx, x, y, nbits);
    default:
      SPU_THROW("unknown circuit type {}", static_cast<uint32_t>(type));
  }
}

Value carry_a2b(SPUContext* ctx, const Value& x, const Value& y, size_t k) {
  // init P & G
  auto P = xor_bb(ctx, x, y);
  auto G = and_bb(ctx, x, y);

  // Use kogge stone layout.
  while (k > 1) {
    if (k % 2 != 0) {
      k += 1;
      P = lshift_b(ctx, P, 1);
      G = lshift_b(ctx, G, 1);
    }
    auto [P1, P0] = bit_scatter(ctx, P, 0);
    auto [G1, G0] = bit_scatter(ctx, G, 0);

    // Calculate next-level of P, G
    //   P = P1 & P0
    //   G = G1 | (P1 & G0)
    //     = G1 ^ (P1 & G0)
    std::vector<Value> v = vectorize(
        {P0, G0}, {P1, P1},
        [&](const Value& xx, const Value& yy) { return and_bb(ctx, xx, yy); });
    P = std::move(v[0]);
    G = xor_bb(ctx, G1, v[1]);
    k >>= 1;
  }

  return G;
}

}  // namespace spu::mpc
