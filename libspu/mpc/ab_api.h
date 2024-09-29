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

#include "libspu/core/context.h"
#include "libspu/core/memref.h"

namespace spu::mpc {

MemRef a2p(SPUContext* ctx, const MemRef& x);
MemRef p2a(SPUContext* ctx, const MemRef& x);
MemRef a2v(SPUContext* ctx, const MemRef& x, size_t owner);
MemRef v2a(SPUContext* ctx, const MemRef& x);

MemRef msb_a2b(SPUContext* ctx, const MemRef& x);

MemRef rand_a(SPUContext* ctx, SemanticType type, const Shape& shape);
MemRef rand_b(SPUContext* ctx, const Shape& shape);

MemRef negate_a(SPUContext* ctx, const MemRef& x);

MemRef equal_ap(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef equal_aa(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef add_ap(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef add_aa(SPUContext* ctx, const MemRef& x, const MemRef&);
OptionalAPI<MemRef> add_av(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef mul_ap(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef mul_aa(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef square_a(SPUContext* ctx, const MemRef& x);
OptionalAPI<MemRef> mul_av(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef mul_a1b(SPUContext* ctx, const MemRef& x, const MemRef& y);
OptionalAPI<MemRef> mul_a1bv(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef lshift_a(SPUContext* ctx, const MemRef& x, const Sizes& nbits);
MemRef trunc_a(SPUContext* ctx, const MemRef& x, size_t nbits, SignType sign);

MemRef mmul_ap(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef mmul_aa(SPUContext* ctx, const MemRef& x, const MemRef& y);
OptionalAPI<MemRef> mmul_av(SPUContext* ctx, const MemRef& x, const MemRef& y);

Type common_type_a(SPUContext* ctx, const Type& a, const Type& b);
Type common_type_b(SPUContext* ctx, const Type& a, const Type& b);
MemRef cast_type_b(SPUContext* ctx, const MemRef& a, const Type& to_type);
MemRef cast_type_a(SPUContext* ctx, const MemRef& a, const Type& to_type);

MemRef b2p(SPUContext* ctx, const MemRef& x);
MemRef p2b(SPUContext* ctx, const MemRef& x);
MemRef b2v(SPUContext* ctx, const MemRef& x, size_t owner);

MemRef a2b(SPUContext* ctx, const MemRef& x);
MemRef b2a(SPUContext* ctx, const MemRef& x);

MemRef and_bp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef and_bb(SPUContext* ctx, const MemRef& x, const MemRef& y);
OptionalAPI<MemRef> and_bv(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef xor_bp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef xor_bb(SPUContext* ctx, const MemRef& x, const MemRef& y);
OptionalAPI<MemRef> xor_bv(SPUContext* ctx, const MemRef& x,
                           const MemRef& y);  // TODO

MemRef lshift_b(SPUContext* ctx, const MemRef& x, const Sizes& nbits);
MemRef rshift_b(SPUContext* ctx, const MemRef& x, const Sizes& nbits);
MemRef arshift_b(SPUContext* ctx, const MemRef& x, const Sizes& nbits);

// Bit reverse for binary share.
MemRef bitrev_b(SPUContext* ctx, const MemRef& x, size_t start, size_t end);

// TODO: maybe we should add more general PDEP/PEXT instruction.

/// Bit interleave for binary share.
//
// Interleave bits of input, so the upper bits of input are in the even
// positions and lower bits in the odd. Also called Morton Number.
//
//   abcdXYZW -> aXbYcZdW
//
// stride represent the log shift of the interleaved distance.
//
//   00001111 -> 01010101     stride = 0
//   00001111 -> 00110011     stride = 1
//   00001111 -> 00001111     stride = 2
//
MemRef bitintl_b(SPUContext* ctx, const MemRef& x, size_t stride);

// Bit de-interleave for binary share.
//
// The reverse bit interleave method, put the even bits at lower half, and odd
// bits at upper half.
//
//   aXbYcZdW -> abcdXYZW
//
// stride represent the log shift of the interleaved distance.
//
//   01010101 -> 00001111     stride = 0
//   00110011 -> 00001111     stride = 1
//   00001111 -> 00001111     stride = 2
//
MemRef bitdeintl_b(SPUContext* ctx, const MemRef& x, size_t stride);

MemRef add_bb(SPUContext* ctx, const MemRef& x, const MemRef& y);

// compute the k'th bit of x + y
MemRef carry_a2b(SPUContext* ctx, const MemRef& x, const MemRef& y, size_t k);

std::vector<MemRef> bit_decompose_b(SPUContext* ctx, const MemRef& x);
MemRef bit_compose_b(SPUContext* ctx, const std::vector<MemRef>& x);

}  // namespace spu::mpc
