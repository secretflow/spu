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

// Get the common type of secrets.
//
// Unlike public types, which has only one form, secrets has multiple storage
// formats, like AShare/BShare, which make them not concatable.
//
// This api calculate the common type.
Type common_type_s(Object* ctx, const Type& a, const Type& b);
ArrayRef cast_type_s(Object* ctx, const ArrayRef& a, const Type& to_type);

// Make a public variable with given plaintext input.
//
// All parties knowns the value.
ArrayRef make_p(Object* ctx, const ArrayRef& plaintext);

// parties random a public together.
// TODO: move field into ctx
ArrayRef rand_p(Object* ctx, FieldType, size_t);
ArrayRef rand_s(Object* ctx, FieldType, size_t);

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

// Reverse bit, like MISP BITREV instruction, and linux bitrev library.
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

}  // namespace spu::mpc

#define SPU_MPC_DEF_UNARY_OP(NAME)                 \
  ArrayRef NAME(Object* ctx, const ArrayRef& in) { \
    return ctx->call(#NAME, in);                   \
  }

#define SPU_MPC_DEF_UNARY_OP_WITH_SIZE(NAME)                  \
  ArrayRef NAME(Object* ctx, const ArrayRef& in, size_t sz) { \
    return ctx->call(#NAME, in, sz);                          \
  }

#define SPU_MPC_DEF_UNARY_OP_WITH_2SIZE(NAME)                              \
  ArrayRef NAME(Object* ctx, const ArrayRef& in, size_t sz1, size_t sz2) { \
    return ctx->call(#NAME, in, sz1, sz2);                                 \
  }

#define SPU_MPC_DEF_BINARY_OP(NAME)                                  \
  ArrayRef NAME(Object* ctx, const ArrayRef& x, const ArrayRef& y) { \
    return ctx->call(#NAME, x, y);                                   \
  }

#define SPU_MPC_DEF_MMUL(NAME)                                                 \
  ArrayRef NAME(Object* ctx, const ArrayRef& x, const ArrayRef& y, size_t sz1, \
                size_t sz2, size_t sz3) {                                      \
    return ctx->call(#NAME, x, y, sz1, sz2, sz3);                              \
  }
