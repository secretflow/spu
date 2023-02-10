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

#pragma once

#include <array>
#include <functional>
#include <iostream>

#include "absl/numeric/bits.h"
#include "yacl/base/int128.h"

#include "libspu/core/bit_utils.h"
#include "libspu/core/vectorize.h"

namespace spu::mpc {

template <typename T>
struct CircuitBasicBlock {
  // multi-bit xor. i.e. 0010 xor 1010 -> 1000
  using Xor = std::function<T(T const&, T const&)>;

  // multi-bit and. i.e. 0010 xor 1010 -> 0010
  using And = std::function<T(T const&, T const&)>;

  // (logical) left shift
  using LShift = std::function<T(T const&, size_t)>;

  // (logical) right shift
  using RShift = std::function<T(T const&, size_t)>;

  // Init a constant.
  using InitLike = std::function<T(T const&, uint128_t)>;

  // Set number of bits.
  using SetNBits = std::function<void(T&, size_t)>;

  Xor _xor = nullptr;
  And _and = nullptr;
  LShift lshift = nullptr;
  RShift rshift = nullptr;
  InitLike init_like = nullptr;
  SetNBits set_nbits = nullptr;
};

// Parallel Prefix Graph: Kogge Stone.
//
// P stands for propogate, G stands for generate, where:
//  (G0, P0) = (g0, p0)
//  (Gi, Pi) = (gi, pi) o (Gi-1, Pi-1)
//
// The `o` here is:
//  (G0, P0) o (G1, P1) = (G0 ^ (P0 & G1), P0 & P1)
//
// Latency log(k) + 1
template <typename T>
T kogge_stone(const CircuitBasicBlock<T>& ctx, T const& lhs, T const& rhs,
              size_t nbits) {
  // Generate p & g.
  auto P = ctx._xor(lhs, rhs);
  auto G = ctx._and(lhs, rhs);

  for (int idx = 0; idx < Log2Ceil(nbits); ++idx) {
    const size_t offset = 1UL << idx;
    auto G1 = ctx.lshift(G, offset);
    auto P1 = ctx.lshift(P, offset);

    // P1 = P & P1
    // G1 = G ^ (P & G1)
    if constexpr (HasSimdTrait<T>::value) {
      std::vector<T> res = vectorize({P, P}, {P1, G1}, ctx._and);
      P = std::move(res[0]);
      G = ctx._xor(G, std::move(res[1]));
    } else {
      auto tmp = ctx._and(P, G1);
      P = ctx._and(P, P1);
      G = ctx._xor(G, tmp);
    }
  }

  // out = (G << 1) ^ p0
  auto C = ctx.lshift(G, 1);
  return ctx._xor(ctx._xor(lhs, rhs), C);
}

template <typename T>
T sklansky(const CircuitBasicBlock<T>& ctx, T const& lhs, T const& rhs,
           size_t nbits) {
  constexpr std::array<uint128_t, 7> kKeepMasks = {{
      yacl::MakeUint128(0x5555555555555555, 0x5555555555555555),
      yacl::MakeUint128(0x3333333333333333, 0x3333333333333333),
      yacl::MakeUint128(0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F),
      yacl::MakeUint128(0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF),
      yacl::MakeUint128(0x0000FFFF0000FFFF, 0x0000FFFF0000FFFF),
      yacl::MakeUint128(0x00000000FFFFFFFF, 0x00000000FFFFFFFF),
      yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFFFFFFFFFF),
  }};

  constexpr std::array<uint128_t, 7> kSelMask = {{
      yacl::MakeUint128(0x5555555555555555, 0x5555555555555555),
      yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),
      yacl::MakeUint128(0x0808080808080808, 0x0808080808080808),
      yacl::MakeUint128(0x0080008000800080, 0x0080008000800080),
      yacl::MakeUint128(0x0000800000008000, 0x0000800000008000),
      yacl::MakeUint128(0x0000000080000000, 0x0000000080000000),
      yacl::MakeUint128(0x0000000000000000, 0x8000000000000000),
  }};

  // Generate p & g.
  auto P = ctx._xor(lhs, rhs);
  auto G = ctx._and(lhs, rhs);
  for (int idx = 0; idx < Log2Ceil(nbits); ++idx) {
    const auto s_mask = ctx.init_like(G, kSelMask[idx]);
    auto G1 = ctx.lshift(ctx._and(G, s_mask), 1);
    auto P1 = ctx.lshift(ctx._and(P, s_mask), 1);

    for (int j = 0; j < idx; j++) {
      G1 = ctx._xor(G1, ctx.lshift(G1, 1 << j));
      P1 = ctx._xor(P1, ctx.lshift(P1, 1 << j));
    }

    const auto k_mask = ctx.init_like(G, kKeepMasks[idx]);
    P1 = ctx._xor(P1, k_mask);

    // P = P & P1
    // G = G ^ (P & G1)
    if constexpr (HasSimdTrait<T>::value) {
      std::vector<T> res = vectorize({P, P}, {P1, G1}, ctx._and);
      P = std::move(res[0]);
      G = ctx._xor(G, std::move(res[1]));
    } else {
      auto tmp = ctx._and(P, G1);
      P = ctx._and(P, P1);
      G = ctx._xor(G, tmp);
    }
  }

  // out = (G0 << 1) ^ p0
  auto C = ctx.lshift(G, 1);
  return ctx._xor(ctx._xor(lhs, rhs), C);
}

template <typename T>
T odd_even_split(const CircuitBasicBlock<T>& ctx, const T& v, size_t nbits) {
  // algorithm:
  //
  //      0101010101010101
  // swap  ^^  ^^  ^^  ^^
  //      0011001100110011
  // swap   ^^^^    ^^^^
  //      0000111100001111
  // swap     ^^^^^^^^
  //      0000000011111111

  constexpr std::array<uint128_t, 6> kSwapMasks = {{
      yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
      yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
      yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
      yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
      yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
      yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
  }};
  constexpr std::array<uint128_t, 6> kKeepMasks = {{
      yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
      yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
      yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
      yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
      yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
      yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
  }};

  // let r = v
  T r = ctx.lshift(v, 0);
  for (int idx = 0; idx + 1 < Log2Ceil(nbits); ++idx) {
    // r = (r & keep) ^ ((r >> i) & move) ^ ((r & move) << i)
    const auto keep = ctx.init_like(r, kKeepMasks[idx]);
    const auto move = ctx.init_like(r, kSwapMasks[idx]);

    r = ctx._xor(ctx._and(r, keep),
                 ctx._xor(ctx._and(ctx.rshift(r, 1 << idx), move),
                          ctx.lshift(ctx._and(r, move), 1 << idx)));
  }

  if (!absl::has_single_bit(nbits)) {
    // handle non 2^k bits case.
    T mask = ctx.init_like(r, (1ULL << (nbits / 2)) - 1);
    r = ctx._xor(ctx.lshift(ctx.rshift(r, 1 << Log2Floor(nbits)), nbits / 2),
                 ctx._and(r, mask));
  }

  return r;
}

//    7  6  5  4  3  2  1  0
//    |_/   |_/   |_/   |_/
//    |____/      |____/
//    |__________/
//
//    6  5  5  4  3  2  1
//    |_/   |_/   |_/   |
//    |____/      |____/
//    |__________/
//
// # Reference
// [1](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.9499&rep=rep1&type=pdf)
// CarryOutL
template <typename T>
T carry_out(const CircuitBasicBlock<T>& ctx, const T& x, const T& y,
            size_t nbits) {
  SPU_ENFORCE(nbits != 0, "carry out with 0 is meaningless");
  // split even and odd bits. e.g.
  //   xAyBzCwD -> [xyzw, ABCD]
  auto bit_split = [&](T const& in, size_t kk) -> std::tuple<T, T> {
    SPU_ENFORCE(kk % 2 == 0 && kk <= 128);
    const size_t hk = kk / 2;

    auto perm = odd_even_split(ctx, in, kk);
    T mask = ctx.init_like(perm, (static_cast<uint128_t>(1) << hk) - 1);
    T t0 = ctx._and(perm, mask);
    T t1 = ctx._and(ctx.rshift(perm, hk), mask);
    ctx.set_nbits(t0, hk);
    ctx.set_nbits(t1, hk);
    return std::make_tuple(t0, t1);
  };

  // init P & G
  auto P = ctx._xor(x, y);
  auto G = ctx._and(x, y);

  if (nbits == 1) {
    return ctx._and(G, ctx.init_like(G, 1));
  }

  // Use kogge stone layout.
  size_t k = nbits;
  while (k > 1) {
    if (k % 2 != 0) {
      k += 1;
      P = ctx.lshift(P, 1);
      G = ctx.lshift(G, 1);
    }
    auto [P0, P1] = bit_split(P, k);
    auto [G0, G1] = bit_split(G, k);

    // Calculate next-level of P, G
    //   P = P1 & P0
    //   G = G1 | (P1 & G0)
    //     = G1 ^ (P1 & G0)
    if constexpr (HasSimdTrait<T>::value) {
      std::vector<T> v = vectorize({P0, G0}, {P1, P1}, ctx._and);
      P = std::move(v[0]);
      G = ctx._xor(G1, std::move(v[1]));
    } else {
      P = ctx._and(P1, P0);
      G = ctx._xor(G1, ctx._and(P1, G0));
    }
    k >>= 1;
  }

  return G;
}

}  // namespace spu::mpc
