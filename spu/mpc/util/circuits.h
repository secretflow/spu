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

#include <functional>
#include <iostream>

#include "absl/numeric/bits.h"

#include "spu/core/vectorize.h"

namespace spu::mpc {
namespace details {

// TODO(jint): general utility
template <class T>
struct dependent_false : std::false_type {};

}  // namespace details

// multi-bit circuit basic block
//
// with LShift/RShift, we can build any constant input, plus xor/and, we
// can build any complicate circuits.
template <typename T>
struct CircuitBasicBlock {
  // multi-bit xor. i.e. 0010 xor 1010 -> 1000
  using Xor = std::function<T(T const&, T const&)>;

  // multi-bit and. i.e. 0010 xor 1010 -> 0010
  using And = std::function<T(T const&, T const&)>;

  // (logic) left shift
  using LShift = std::function<T(T const&, size_t)>;

  // (logic) right shift
  using RShift = std::function<T(T const&, size_t)>;

  // TODO(jint) uint64_t is not good idea?
  using InitLike = std::function<T(T const& x, uint64_t init)>;

  size_t num_bits = 0;
  Xor _xor = nullptr;
  And _and = nullptr;
  LShift lshift = nullptr;
  RShift rshift = nullptr;
  InitLike init_like = nullptr;
};

template <typename T>
CircuitBasicBlock<T> DefaultCircuitBasicBlock() {
  if constexpr (std::is_integral_v<T>) {
    CircuitBasicBlock<T> cbb;
    cbb.num_bits = sizeof(T) * 8;
    cbb.init_like = [](T const&, uint64_t x) -> T { return static_cast<T>(x); };
    cbb._xor = [](T const& lhs, T const& rhs) -> T { return lhs ^ rhs; };
    cbb._and = [](T const& lhs, T const& rhs) -> T { return lhs & rhs; };
    cbb.lshift = [](T const& x, size_t bits) -> T { return x << bits; };
    cbb.rshift = [](T const& x, size_t bits) -> T { return x >> bits; };
    return cbb;
  } else {
    static_assert(details::dependent_false<T>::value,
                  "Not implemented for circuit basic block.");
  }
}

/// Reference:
///  PPA (Parallel Prefix Adder)
///  http://users.encs.concordia.ca/~asim/COEN_6501/Lecture_Notes/Parallel%20prefix%20adders%20presentation.pdf
///
/// Why KoggleStone:
///  - easy to implement.
///
/// Analysis:
///  AND Gates: 1 + log(k) (additional 1 for `g` generation)
template <typename T>
T KoggleStoneAdder(
    const T& lhs, const T& rhs,
    const CircuitBasicBlock<T>& bb = DefaultCircuitBasicBlock<T>()) {
  // Generate p & g.
  T p = bb._xor(lhs, rhs);
  T g = bb._and(lhs, rhs);

  // Parallel Prefix Graph: Koggle Stone.
  // We write prefix element as P, G, where:
  //  (G0, P0) = (g0, p0)
  //  (Gi, Pi) = (gi, pi) o (Gi-1, Pi-1)
  // The `o` here is:
  //  (G0, P0) o (G1, P1) = (G0 ^ (P0 & G1), P0 & P1)
  //
  // We can perform AND vectorization for above two AND:
  T G0 = g;
  T P0 = p;
  for (int idx = 0; idx < static_cast<int>(absl::bit_width(bb.num_bits)) - 1;
       ++idx) {
    const size_t offset = 1UL << idx;

    // G1 = G << offset
    // P1 = P << offset
    T G1 = bb.lshift(G0, offset);
    T P1 = bb.lshift(P0, offset);

    // In the Kogge-Stone graph, we need to keep the lowest |offset| P, G
    // unmodified.
    //
    //// P0 = P0 & P1
    //// G0 = G0 ^ (P0 & G1)
    if constexpr (hasSimdTrait<T>::value) {
      std::vector<T> res = vectorize({P0, P0}, {P1, G1}, bb._and);
      P0 = std::move(res[0]);
      G1 = std::move(res[1]);
    } else {
      G1 = bb._and(P0, G1);
      P0 = bb._and(P0, P1);
    }

    G0 = bb._xor(G1, G0);
  }

  // Carry = G0
  // C = Carry << 1;
  // out = C ^ P
  T C = bb.lshift(G0, 1);
  return bb._xor(p, C);
}

// Calculate the carry-out bit of the sum of two elements.
// Output m elements, with each element n bit number, where:
//   c = carry(x + y)
//
// The basic idea is to apply the CarryOutL circuit described in [1] on a m*n
// bit vector, with the following assumptions.
// - CPU could always do n-bit SIMD.
// - m element may not be continuous (it's strided).
//
//    b7 b6 b5 b4 b3 b2 b1 b0
//    |_/   |_/   |_/   |_/
//    |____/      |____/
//    |__________/
//
// # Reference
// [1](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.9499&rep=rep1&type=pdf)
// CarryOutL
// [2](https://users.encs.concordia.ca/~asim/COEN_6501/Lecture_Notes/Parallel%20prefix%20adders%20presentation.pdf)
// J. Sklansky – conditional adder
template <typename T>
T CarryOut(const T& lhs, const T& rhs,
           const CircuitBasicBlock<T>& bb = DefaultCircuitBasicBlock<T>()) {
  auto compress = [&](const T& in, size_t nbits) -> T {
    // out[i] = in[i*2]
    T out = bb.init_like(in, 0);
    const T kOne = bb.init_like(in, 1);
    // TODO: use log(nbits) method.
    for (size_t i = 0; i < nbits / 2; i++) {
      // (in >> (2*i)) << i
      T tmp = bb.lshift(bb._and(bb.rshift(in, 2 * i), kOne), i);
      out = bb._xor(out, tmp);
    }
    return out;
  };

  // init P & G
  T P0 = bb._xor(lhs, rhs);
  T G0 = bb._and(lhs, rhs);

  for (size_t idx = 0; idx < absl::bit_width(bb.num_bits) - 1; ++idx) {
    // TODO: we should split odd/even bits instead of shift.
    // In current implementation, communication is doubled.
    T P1 = bb.rshift(P0, 1);
    T G1 = bb.rshift(G0, 1);

    // Calculate next-level of P, G
    //   P = P1 & P0
    //   G = G1 | (P1 & G0)
    //     = G1 ^ (P1 & G0)
    if constexpr (hasSimdTrait<T>::value) {
      std::vector<T> v = vectorize({P0, G0}, {P1, P1}, bb._and);
      P0 = std::move(v[0]);
      G0 = bb._xor(G1, v[1]);
    } else {
      P0 = bb._and(P1, P0);
      G0 = bb._xor(G1, bb._and(P1, G0));
    }

    // Compress it.
    //    a7 a6 a5 a4 a3 a2 a1 a0
    //     \_|   \_|   \_|   \_|
    //       a6    a4    a2    a0
    //
    //                a6 a4 a2 a0
    G0 = compress(G0, bb.num_bits >> idx);
    P0 = compress(P0, bb.num_bits >> idx);

    // TODO: set the new bitwidth.
  }

  return G0;
}

}  // namespace spu::mpc
