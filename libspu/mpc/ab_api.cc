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
MemRef tiledDynDispatch(const std::string& fn_name, SPUContext* ctx,
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

MemRef a2p(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

MemRef p2a(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

MemRef a2v(SPUContext* ctx, const MemRef& x, size_t owner) {
  // Note: Private is not mandatory for now, it's the protocol author's
  // responsibility to decide if private should be supported. He/she can:
  // 1. Do not support private from IO interface, then all computation should be
  //    in Secret/Public domain, Private related kernel will never be
  //    dispatched.
  // 2. Support private in IO interface, then the private computation/conversion
  //    interface should also be supported.
  FORCE_DISPATCH(ctx, x, owner);
}

MemRef v2a(SPUContext* ctx, const MemRef& x) {
  // Note: it's the protocol author's responsibility to ensure private is
  // supported
  FORCE_DISPATCH(ctx, x);
}

MemRef msb_a2b(SPUContext* ctx, const MemRef& x) { TILED_DISPATCH(ctx, x); }

MemRef rand_a(SPUContext* ctx, SemanticType type, const Shape& shape) {
  FORCE_DISPATCH(ctx, type, shape);
}

MemRef rand_b(SPUContext* ctx, const Shape& shape) {
  FORCE_DISPATCH(ctx, shape);
}

MemRef negate_a(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

MemRef add_ap(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef equal_aa(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TILED_DISPATCH(ctx, x, y);
}

MemRef equal_ap(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TILED_DISPATCH(ctx, x, y);
}

MemRef add_aa(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

OptionalAPI<MemRef> add_av(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

MemRef mul_ap(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef mul_aa(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TILED_DISPATCH(ctx, x, y);
}

MemRef square_a(SPUContext* ctx, const MemRef& x) { TILED_DISPATCH(ctx, x); }

OptionalAPI<MemRef> mul_av(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

MemRef mul_a1b(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TILED_DISPATCH(ctx, x, y);
}

OptionalAPI<MemRef> mul_a1bv(SPUContext* ctx, const MemRef& x,
                             const MemRef& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

MemRef lshift_a(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  FORCE_DISPATCH(ctx, x, nbits);
}

MemRef trunc_a(SPUContext* ctx, const MemRef& x, size_t nbits, SignType sign) {
  TILED_DISPATCH(ctx, x, nbits, sign);
}

MemRef mmul_ap(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef mmul_aa(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

OptionalAPI<MemRef> mmul_av(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

Type common_type_a(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_MPC_LEAF(ctx, a, b);
  return dynDispatch<Type>(ctx, __func__, a, b);
}

Type common_type_b(SPUContext* ctx, const Type& a, const Type& b) {
  SPU_TRACE_MPC_LEAF(ctx, a, b);
  return dynDispatch<Type>(ctx, __func__, a, b);
}

MemRef cast_type_b(SPUContext* ctx, const MemRef& a, const Type& to_type) {
  FORCE_DISPATCH(ctx, a, to_type);
}

MemRef cast_type_a(SPUContext* ctx, const MemRef& a, const Type& to_type) {
  FORCE_DISPATCH(ctx, a, to_type);
}

MemRef b2p(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

MemRef p2b(SPUContext* ctx, const MemRef& x) { FORCE_DISPATCH(ctx, x); }

MemRef b2v(SPUContext* ctx, const MemRef& x, size_t owner) {
  FORCE_DISPATCH(ctx, x, owner);
}

MemRef a2b(SPUContext* ctx, const MemRef& x) { TILED_DISPATCH(ctx, x); }

MemRef b2a(SPUContext* ctx, const MemRef& x) { TILED_DISPATCH(ctx, x); }

MemRef and_bp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef and_bb(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TILED_DISPATCH(ctx, x, y);
}

OptionalAPI<MemRef> and_bv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

MemRef xor_bp(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

MemRef xor_bb(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  FORCE_DISPATCH(ctx, x, y);
}

OptionalAPI<MemRef> xor_bv(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  TRY_DISPATCH(ctx, x, y);
  return NotAvailable;
}

MemRef lshift_b(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  return dynDispatch(ctx, "lshift", x, nbits);
}

MemRef rshift_b(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  return dynDispatch(ctx, "rshift", x, nbits);
}

MemRef arshift_b(SPUContext* ctx, const MemRef& x, const Sizes& nbits) {
  return dynDispatch(ctx, "arshift", x, nbits);
}

MemRef bitrev_b(SPUContext* ctx, const MemRef& x, size_t start, size_t end) {
  return dynDispatch(ctx, "bitrev", x, start, end);
}

static inline size_t numBits(const MemRef& in) {
  return in.eltype().as<BaseRingType>()->valid_bits();
}

static inline MemRef setNumBits(const MemRef& in, size_t nbits) {
  MemRef out = in;
  out.eltype().as<BaseRingType>()->set_valid_bits(nbits);
  return out;
}

// TODO: we can not ref api.h, circular reference
static MemRef hack_make_p(SPUContext* ctx, uint128_t init, SemanticType st,
                          const Shape& shape) {
  return dynDispatch(ctx, "make_p", init, st, shape);
}

MemRef bitintl_b(SPUContext* ctx, const MemRef& x, size_t stride) {
  TRY_NAMED_DISPATCH(ctx, "bitintl", x, stride);

  // default implementation.
  // algorithm:
  //      0000000011111111
  // swap     ^^^^^^^^
  //      0000111100001111
  // swap   ^^^^    ^^^^
  //      0011001100110011
  // swap  ^^  ^^  ^^  ^^
  //      0101010101010101
  const size_t nbits = x.eltype().as<BaseRingType>()->valid_bits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  MemRef out = x;
  for (int64_t idx = Log2Ceil(nbits) - 2; idx >= static_cast<int64_t>(stride);
       idx--) {
    auto K = hack_make_p(ctx, spu::detail::kBitIntlKeepMasks[idx], SE_I128,
                         x.shape());
    auto M = hack_make_p(ctx, spu::detail::kBitIntlSwapMasks[idx], SE_I128,
                         x.shape());
    int64_t S = static_cast<uint64_t>(1) << idx;
    // out = (out & K) ^ ((out >> S) & M) ^ ((out & M) << S);
    out = xor_bb(ctx,
                 xor_bb(ctx, and_bp(ctx, out, K),
                        and_bp(ctx, rshift_b(ctx, out, {S}), M)),
                 lshift_b(ctx, and_bp(ctx, out, M), {S}));
  }
  out = setNumBits(out, numBits(x));
  return out;
}

MemRef bitdeintl_b(SPUContext* ctx, const MemRef& x, size_t stride) {
  TRY_NAMED_DISPATCH(ctx, "bitdeintl", x, stride);

  // algorithm:
  //      0101010101010101
  // swap  ^^  ^^  ^^  ^^
  //      0011001100110011
  // swap   ^^^^    ^^^^
  //      0000111100001111
  // swap     ^^^^^^^^
  //      0000000011111111
  const size_t nbits = x.eltype().as<BaseRingType>()->valid_bits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  MemRef out = x;
  for (int64_t idx = stride; idx + 1 < Log2Ceil(nbits); idx++) {
    auto K = hack_make_p(ctx, spu::detail::kBitIntlKeepMasks[idx], SE_I128,
                         x.shape());
    auto M = hack_make_p(ctx, spu::detail::kBitIntlSwapMasks[idx], SE_I128,
                         x.shape());
    int64_t S = static_cast<uint64_t>(1) << idx;
    // out = (out & K) ^ ((out >> S) & M) ^ ((out & M) << S);
    out = xor_bb(ctx,
                 xor_bb(ctx, and_bp(ctx, out, K),
                        and_bp(ctx, rshift_b(ctx, out, {S}), M)),
                 lshift_b(ctx, and_bp(ctx, out, M), {S}));
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
MemRef ppa_kogge_stone(SPUContext* ctx, const MemRef& lhs, const MemRef& rhs,
                       size_t nbits) {
  // Generate p & g.
  auto P = xor_bb(ctx, lhs, rhs);
  auto G = and_bb(ctx, lhs, rhs);

  for (int idx = 0; idx < Log2Ceil(nbits); ++idx) {
    const int64_t offset = static_cast<uint64_t>(1) << idx;
    auto G1 = lshift_b(ctx, G, {offset});
    auto P1 = lshift_b(ctx, P, {offset});

    // P1 = P & P1
    // G1 = G ^ (P & G1)
    std::vector<MemRef> res =
        spu::vmap({P, P}, {P1, G1}, [&](const MemRef& xx, const MemRef& yy) {
          return and_bb(ctx, xx, yy);
        });
    P = std::move(res[0]);
    G = xor_bb(ctx, G, res[1]);
  }

  // out = (G << 1) ^ p0
  auto C = lshift_b(ctx, G, {1});
  return xor_bb(ctx, xor_bb(ctx, lhs, rhs), C);
}

std::pair<MemRef, MemRef> bit_scatter(SPUContext* ctx, const MemRef& in,
                                      size_t stride) {
  // TODO: use faster bit scatter implementation for ABY3
  const size_t nbits = numBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits), "unsupported {}", nbits);
  auto out = bitdeintl_b(ctx, in, stride);

  auto hi = rshift_b(ctx, out, {static_cast<int64_t>(nbits / 2)});
  auto mask = hack_make_p(ctx, (static_cast<uint128_t>(1) << (nbits / 2)) - 1,
                          SE_I128, in.shape());
  auto lo = and_bp(ctx, out, mask);

  return std::make_pair(hi, lo);
}

MemRef bit_gather(SPUContext* ctx, const MemRef& hi, const MemRef& lo,
                  size_t stride) {
  const size_t nbits = numBits(hi);
  SPU_ENFORCE(absl::has_single_bit(nbits), "unsupported {}", nbits);
  SPU_ENFORCE(nbits == numBits(lo), "nbits mismatch {}, {}", nbits,
              numBits(lo));

  auto out = xor_bb(ctx, lshift_b(ctx, hi, {static_cast<int64_t>(nbits)}), lo);
  return bitintl_b(ctx, out, stride);
}

// The sklansky adder.
MemRef ppa_sklansky(SPUContext* ctx, MemRef const& lhs, MemRef const& rhs,
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

    const auto s_mask = hack_make_p(ctx, kSelMask[idx], SE_I128, lhs.shape());
    auto Gs = and_bp(ctx, Gl, s_mask);
    auto Ps = and_bp(ctx, Pl, s_mask);
    for (int j = 0; j < idx; j++) {
      Gs = xor_bb(ctx, Gs, rshift_b(ctx, Gs, {1 << j}));
      Ps = xor_bb(ctx, Ps, rshift_b(ctx, Ps, {1 << j}));
    }
    // SPU_ENFORCE(numBits(Ps) == bit_width / 2);
    // SPU_ENFORCE(numBits(Gs) == bit_width / 2);

    // Ph = Ph & Ps
    // Gh = Gh ^ (Ph & Gs)
    std::vector<MemRef> PG =
        spu::vmap({Ph, Ph}, {Ps, Gs}, [&](const MemRef& xx, const MemRef& yy) {
          return and_bb(ctx, xx, yy);
        });
    Ph = std::move(PG[0]);
    Gh = xor_bb(ctx, Gh, PG[1]);
    // SPU_ENFORCE(numBits(Gh) == numBits(G) / 2);
    // SPU_ENFORCE(numBits(Ph) == numBits(P) / 2);

    P = bit_gather(ctx, Ph, Pl, idx);
    G = bit_gather(ctx, Gh, Gl, idx);
  }

  // out = (G0 << 1) ^ p0
  auto C = lshift_b(ctx, G, {1});
  return xor_bb(ctx, xor_bb(ctx, lhs, rhs), C);
}

}  // namespace

MemRef add_bb(SPUContext* ctx, const MemRef& x, const MemRef& y) {
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

MemRef carry_a2b(SPUContext* ctx, const MemRef& x, const MemRef& y, size_t k) {
  // init P & G
  auto P = xor_bb(ctx, x, y);

  // k bits
  auto G = and_bb(ctx, x, y);

  // Use kogge stone layout.
  //    Theoretically: k + k/2 + k/4 + ... + 1 = 2k
  //    Actually: K + k/2 + k/4 + ... + 8 (8) + 8 (4) + 8 (2) + 8 (1) = 2k + 16
  while (k > 1) {
    if (k % 2 != 0) {
      k += 1;
      P = lshift_b(ctx, P, {1});
      G = lshift_b(ctx, G, {1});
    }
    auto [P1, P0] = bit_scatter(ctx, P, 0);
    auto [G1, G0] = bit_scatter(ctx, G, 0);

    // Calculate next-level of P, G
    //   P = P1 & P0
    //   G = G1 | (P1 & G0)
    //     = G1 ^ (P1 & G0)
    std::vector<MemRef> v =
        vmap({P0, G0}, {P1, P1}, [&](const MemRef& xx, const MemRef& yy) {
          return and_bb(ctx, xx, yy);
        });
    P = std::move(v[0]);
    G = xor_bb(ctx, G1, v[1]);
    k >>= 1;
  }

  return G;
}

std::vector<MemRef> bit_decompose_b(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_MPC_LEAF(ctx, x);
  return dynDispatch<std::vector<MemRef>>(ctx, "bit_decompose_b", x);
}

MemRef bit_compose_b(SPUContext* ctx, const std::vector<MemRef>& x) {
  SPU_TRACE_MPC_LEAF(ctx, x.size(), x[0].eltype());
  return dynDispatch(ctx, "bit_compose_b", x);
}

}  // namespace spu::mpc
