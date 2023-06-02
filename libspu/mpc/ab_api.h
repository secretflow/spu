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
#include "libspu/core/value.h"

namespace spu::mpc {

Value a2p(SPUContext* ctx, const Value& x);
Value p2a(SPUContext* ctx, const Value& x);
Value a2v(SPUContext* ctx, const Value& x, size_t owner);
Value v2a(SPUContext* ctx, const Value& x);

Value msb_a2b(SPUContext* ctx, const Value& x);

Value rand_a(SPUContext* ctx, const Shape& shape);
Value rand_b(SPUContext* ctx, const Shape& shape);

Value not_a(SPUContext* ctx, const Value& x);

Value equal_ap(SPUContext* ctx, const Value& x, const Value& y);
Value equal_aa(SPUContext* ctx, const Value& x, const Value& y);

Value add_ap(SPUContext* ctx, const Value& x, const Value& y);
Value add_aa(SPUContext* ctx, const Value& x, const Value&);
OptionalAPI<Value> add_av(SPUContext* ctx, const Value& x, const Value& y);

Value mul_ap(SPUContext* ctx, const Value& x, const Value& y);
Value mul_aa(SPUContext* ctx, const Value& x, const Value& y);
OptionalAPI<Value> mul_av(SPUContext* ctx, const Value& x, const Value& y);

Value mul_a1b(SPUContext* ctx, const Value& x, const Value& y);

Value lshift_a(SPUContext* ctx, const Value& x, size_t nbits);
Value trunc_a(SPUContext* ctx, const Value& x, size_t nbits);

Value mmul_ap(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k);
Value mmul_aa(SPUContext* ctx, const Value& x, const Value& y, size_t m,
              size_t n, size_t k);
OptionalAPI<Value> mmul_av(SPUContext* ctx, const Value& x, const Value& y,
                           size_t m, size_t n, size_t k);

Type common_type_b(SPUContext* ctx, const Type& a, const Type& b);
Value cast_type_b(SPUContext* ctx, const Value& a, const Type& to_type);

Value b2p(SPUContext* ctx, const Value& x);
Value p2b(SPUContext* ctx, const Value& x);
Value b2v(SPUContext* ctx, const Value& x, size_t owner);

Value a2b(SPUContext* ctx, const Value& x);
Value b2a(SPUContext* ctx, const Value& x);

Value and_bp(SPUContext* ctx, const Value& x, const Value& y);
Value and_bb(SPUContext* ctx, const Value& x, const Value& y);
OptionalAPI<Value> and_bv(SPUContext* ctx, const Value& x, const Value& y);

Value xor_bp(SPUContext* ctx, const Value& x, const Value& y);
Value xor_bb(SPUContext* ctx, const Value& x, const Value& y);
OptionalAPI<Value> xor_bv(SPUContext* ctx, const Value& x,
                          const Value& y);  // TODO

Value lshift_b(SPUContext* ctx, const Value& x, size_t nbits);
Value rshift_b(SPUContext* ctx, const Value& x, size_t nbits);
Value arshift_b(SPUContext* ctx, const Value& x, size_t nbits);

// Bit reverse for binary share.
Value bitrev_b(SPUContext* ctx, const Value& x, size_t start, size_t end);

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
Value bitintl_b(SPUContext* ctx, const Value& x, size_t stride);

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
Value bitdeintl_b(SPUContext* ctx, const Value& x, size_t stride);

Value add_bb(SPUContext* ctx, const Value& x, const Value& y);

// compute the k'th bit of x + y
Value carry_a2b(SPUContext* ctx, const Value& x, const Value& y, size_t k);

}  // namespace spu::mpc
