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

#include "libspu/mpc/api.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/object.h"

namespace spu::mpc {

ArrayRef a2p(Object* ctx, const ArrayRef&);
ArrayRef p2a(Object* ctx, const ArrayRef&);
ArrayRef msb_a2b(Object* ctx, const ArrayRef&);

ArrayRef zero_a(Object* ctx, size_t);
ArrayRef rand_a(Object* ctx, size_t);
ArrayRef rand_b(Object* ctx, size_t);

ArrayRef not_a(Object* ctx, const ArrayRef&);

ArrayRef equal_pp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef equal_sp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef equal_ss(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef add_ap(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef add_aa(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef mul_ap(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef mul_aa(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef mul_a1b(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef lshift_a(Object* ctx, const ArrayRef&, size_t);
ArrayRef trunc_a(Object* ctx, const ArrayRef&, size_t);

ArrayRef mmul_ap(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);
ArrayRef mmul_aa(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);

ArrayRef zero_b(Object* ctx, size_t);

Type common_type_b(Object* ctx, const Type& a, const Type& b);
ArrayRef cast_type_b(Object* ctx, const ArrayRef& a, const Type& to_type);

ArrayRef b2p(Object* ctx, const ArrayRef&);
ArrayRef p2b(Object* ctx, const ArrayRef&);

ArrayRef a2b(Object* ctx, const ArrayRef&);
ArrayRef b2a(Object* ctx, const ArrayRef&);

ArrayRef and_bp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef and_bb(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef xor_bp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef xor_bb(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef lshift_b(Object* ctx, const ArrayRef&, size_t);
ArrayRef rshift_b(Object* ctx, const ArrayRef&, size_t);
ArrayRef arshift_b(Object* ctx, const ArrayRef&, size_t);

// Bit reverse for binary share.
ArrayRef bitrev_b(Object* ctx, const ArrayRef&, size_t, size_t);

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
ArrayRef bitintl_b(Object* ctx, const ArrayRef& in, size_t stride);

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
ArrayRef bitdeintl_b(Object* ctx, const ArrayRef& in, size_t stride);

ArrayRef add_bb(Object* ctx, const ArrayRef&, const ArrayRef&);

void regABKernels(Object* obj);

}  // namespace spu::mpc
