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

#include "spu/core/array_ref.h"
#include "spu/mpc/object.h"

namespace spu::mpc {

// Convert a public to a secret.
//
// In most of cases, you should not do this, becasue:
// 1. This only convert the 'type' to secret, but partipants still knows its
//    value at the moment.
// 2. Nearly all ops has public parameter overload, we should use it directly.
ArrayRef p2s(Object* ctx, const ArrayRef&);

// Convert a secret to a public, aka, reveal.
//
// Note: this API indicates information leak.
ArrayRef s2p(Object* ctx, const ArrayRef&);

// Compute bitwise_not(invert) of a value in ring 2k space.
ArrayRef not_s(Object* ctx, const ArrayRef&);
ArrayRef not_p(Object* ctx, const ArrayRef&);

ArrayRef msb_p(Object* ctx, const ArrayRef&);
ArrayRef msb_s(Object* ctx, const ArrayRef&);

ArrayRef eqz_p(Object* ctx, const ArrayRef&);
ArrayRef eqz_s(Object* ctx, const ArrayRef&);

ArrayRef lshift_p(Object* ctx, const ArrayRef&, size_t);
ArrayRef lshift_s(Object* ctx, const ArrayRef&, size_t);

ArrayRef rshift_p(Object* ctx, const ArrayRef&, size_t);
ArrayRef rshift_s(Object* ctx, const ArrayRef&, size_t);

ArrayRef arshift_p(Object* ctx, const ArrayRef&, size_t);
ArrayRef arshift_s(Object* ctx, const ArrayRef&, size_t);
ArrayRef truncpr_s(Object* ctx, const ArrayRef&, size_t);

ArrayRef bitrev_s(Object* ctx, const ArrayRef&, size_t, size_t);
ArrayRef bitrev_p(Object* ctx, const ArrayRef&, size_t, size_t);

ArrayRef add_pp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef add_sp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef add_ss(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef mul_pp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef mul_sp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef mul_ss(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef and_pp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef and_sp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef and_ss(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef xor_pp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef xor_sp(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef xor_ss(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef mmul_pp(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);
ArrayRef mmul_sp(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);
ArrayRef mmul_ss(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);

// parties random a public together.
ArrayRef rand_p(Object* ctx, FieldType, size_t);
ArrayRef rand_s(Object* ctx, FieldType, size_t);

/// Arithmetic & Boolean ops

ArrayRef a2p(Object* ctx, const ArrayRef&);
ArrayRef p2a(Object* ctx, const ArrayRef&);

ArrayRef zero_a(Object* ctx, FieldType, size_t);

ArrayRef not_a(Object* ctx, const ArrayRef&);
ArrayRef msb_a(Object* ctx, const ArrayRef&);

ArrayRef add_ap(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef add_aa(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef mul_ap(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef mul_aa(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef lshift_a(Object* ctx, const ArrayRef&, size_t);
ArrayRef truncpr_a(Object* ctx, const ArrayRef&, size_t);

ArrayRef mmul_ap(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);
ArrayRef mmul_aa(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);

ArrayRef zero_b(Object* ctx, FieldType, size_t);

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

// Reverse bit, abbreviation inspired from MISP BITREV instruction, and linux
// bitrev library.
ArrayRef bitrev_b(Object* ctx, const ArrayRef&, size_t, size_t);

ArrayRef add_bb(Object* ctx, const ArrayRef&, const ArrayRef&);

}  // namespace spu::mpc
