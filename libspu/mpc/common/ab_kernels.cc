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

#include "libspu/mpc/common/ab_kernels.h"

#include "yacl/base/int128.h"

#include "libspu/core/bit_utils.h"
#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/ab_api.h"

namespace spu::mpc::common {
namespace {

constexpr std::array<uint128_t, 6> kBitIntlSwapMasks = {{
    yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
    yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
    yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
    yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
    yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
    yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
}};

constexpr std::array<uint128_t, 6> kBitIntlKeepMasks = {{
    yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
    yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
    yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
    yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
    yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
    yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
}};

inline size_t numBits(const ArrayRef& in) {
  return in.eltype().as<BShare>()->nbits();
}

inline ArrayRef setNumBits(const ArrayRef& in, size_t nbits) {
  ArrayRef out = in;
  out.eltype().as<BShare>()->setNbits(nbits);
  return out;
}

}  // namespace

// implement interleave by xor_bb/and_bp/shift_b, assume all these
// instructions are communication free.
void BitIntlB::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<ArrayRef>(0);
  const size_t stride = ctx->getParam<size_t>(1);
  auto* obj = ctx->caller();

  SPU_TRACE_MPC_LEAF(ctx, in, stride);

  // algorithm:
  //      0000000011111111
  // swap     ^^^^^^^^
  //      0000111100001111
  // swap   ^^^^    ^^^^
  //      0011001100110011
  // swap  ^^  ^^  ^^  ^^
  //      0101010101010101
  const size_t nbits = in.eltype().as<BShare>()->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));

  const size_t size = in.numel();
  ArrayRef out = in.clone();
  for (int64_t idx = Log2Ceil(nbits) - 2; idx >= static_cast<int64_t>(stride);
       idx--) {
    auto K = make_p(obj, kBitIntlKeepMasks[idx], size);
    auto M = make_p(obj, kBitIntlSwapMasks[idx], size);
    int64_t S = 1 << idx;
    // out = (out & K) ^ ((out >> S) & M) ^ ((out & M) << S);
    out = xor_bb(
        obj,
        xor_bb(obj, and_bp(obj, out, K), and_bp(obj, rshift_b(obj, out, S), M)),
        lshift_b(obj, and_bp(obj, out, M), S));
  }
  out = setNumBits(out, numBits(in));
  ctx->setOutput(out);
}

// implement interleave by xor_bb/and_bp/shift_b, assume all these
// instructions are communication free.
void BitDeintlB::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<ArrayRef>(0);
  const size_t stride = ctx->getParam<size_t>(1);
  auto* obj = ctx->caller();

  SPU_TRACE_MPC_LEAF(ctx, in, stride);

  // algorithm:
  //      0101010101010101
  // swap  ^^  ^^  ^^  ^^
  //      0011001100110011
  // swap   ^^^^    ^^^^
  //      0000111100001111
  // swap     ^^^^^^^^
  //      0000000011111111
  const size_t nbits = in.eltype().as<BShare>()->nbits();
  SPU_ENFORCE(absl::has_single_bit(nbits));
  const size_t size = in.numel();

  ArrayRef out = in.clone();
  for (int64_t idx = stride; idx + 1 < Log2Ceil(nbits); idx++) {
    auto K = make_p(obj, kBitIntlKeepMasks[idx], size);
    auto M = make_p(obj, kBitIntlSwapMasks[idx], size);
    int64_t S = 1 << idx;
    // out = (out & K) ^ ((out >> S) & M) ^ ((out & M) << S);
    out = xor_bb(
        obj,
        xor_bb(obj, and_bp(obj, out, K), and_bp(obj, rshift_b(obj, out, S), M)),
        lshift_b(obj, and_bp(obj, out, M), S));
  }
  out = setNumBits(out, numBits(in));
  ctx->setOutput(out);
}

namespace {

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
ArrayRef ppa_kogge_stone(Object* ctx, const ArrayRef& lhs, const ArrayRef& rhs,
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
    std::vector<ArrayRef> res = spu::vectorize(
        {P, P}, {P1, G1}, [&](const ArrayRef& xx, const ArrayRef& yy) {
          return and_bb(ctx, xx, yy);
        });
    P = std::move(res[0]);
    G = xor_bb(ctx, G, res[1]);
  }

  // out = (G << 1) ^ p0
  auto C = lshift_b(ctx, G, 1);
  return xor_bb(ctx, xor_bb(ctx, lhs, rhs), C);
}

std::pair<ArrayRef, ArrayRef> bit_scatter(Object* ctx, const ArrayRef& in,
                                          size_t stride) {
  const size_t nbits = numBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits), "unsupported {}", nbits);
  auto out = bitdeintl_b(ctx, in, stride);

  auto hi = rshift_b(ctx, out, nbits / 2);
  auto mask =
      make_p(ctx, (static_cast<uint128_t>(1) << (nbits / 2)) - 1, in.numel());
  auto lo = and_bp(ctx, out, mask);
  return std::make_pair(hi, lo);
}

ArrayRef bit_gather(Object* ctx, const ArrayRef& hi, const ArrayRef& lo,
                    size_t stride) {
  const size_t nbits = numBits(hi);
  SPU_ENFORCE(absl::has_single_bit(nbits), "unsupported {}", nbits);
  SPU_ENFORCE(nbits == numBits(lo), "nbits mismatch {}, {}", nbits,
              numBits(lo));

  auto out = xor_bb(ctx, lshift_b(ctx, hi, nbits), lo);
  return bitintl_b(ctx, out, stride);
}

// The sklansky adder.
ArrayRef ppa_sklansky(Object* ctx, ArrayRef const& lhs, ArrayRef const& rhs,
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

    const auto s_mask = make_p(ctx, kSelMask[idx], lhs.numel());
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
    std::vector<ArrayRef> PG = spu::vectorize(
        {Ph, Ph}, {Ps, Gs}, [&](const ArrayRef& xx, const ArrayRef& yy) {
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
  auto C = lshift_b(ctx, G, 1);
  return xor_bb(ctx, xor_bb(ctx, lhs, rhs), C);
}

}  // namespace

ArrayRef AddBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const size_t nbits = numBits(lhs);
  SPU_ENFORCE(nbits == numBits(rhs), "nbits mismatch {}!={}", nbits,
              numBits(rhs));

  switch (type_) {
    case CircuitType::KoggeStone:
      return ppa_kogge_stone(ctx->caller(), lhs, rhs, nbits);
    case CircuitType::Sklansky:
      return ppa_sklansky(ctx->caller(), lhs, rhs, nbits);
    default:
      SPU_THROW("unknown circuit type {}", static_cast<uint32_t>(type_));
  }
}

// compute the k'th bit of x + y
ArrayRef carry_out(Object* ctx, const ArrayRef& x, const ArrayRef& y,
                   size_t k) {
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
    std::vector<ArrayRef> v = vectorize(
        {P0, G0}, {P1, P1}, [&](const ArrayRef& xx, const ArrayRef& yy) {
          return and_bb(ctx, xx, yy);
        });
    P = std::move(v[0]);
    G = xor_bb(ctx, G1, v[1]);
    k >>= 1;
  }

  return G;
}

}  // namespace spu::mpc::common
