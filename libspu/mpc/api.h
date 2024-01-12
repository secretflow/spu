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

// TODO: add to naming conventions.
// - use x,y,z for value
// - use a,b,c for type
// - follow current module style.

// Convert a public to a secret.
//
// In most of cases, you should not do this, because:
// 1. This only convert the 'type' to secret, but participants still know its
//    value at the moment.
// 2. Nearly all ops has public parameter overload, we should use it directly.
//
// These ops are useful for shape related ops, like pad/concat.
Value p2s(SPUContext* ctx, const Value& x);
// Convert a public to a private.
Value p2v(SPUContext* ctx, const Value& x, size_t owner);
// Convert a private to a secret.
Value v2s(SPUContext* ctx, const Value& x);

// Convert a private to a public, same as reveal.
// Note: this API indicates information leak.
Value v2p(SPUContext* ctx, const Value& x);

// Convert a secret to a private. aka, reveal_to.
// Note: this API indicates information leak.
Value s2v(SPUContext* ctx, const Value& x, size_t owner);

// Convert a secret to a public, aka, reveal.
// Note: this API indicates information leak.
Value s2p(SPUContext* ctx, const Value& x);

// Import will be called on all parameters at the beginning program.
//
// The import stage can be used:
// - for malicious protocols, adding party privately generated mac.
// - sharing conversion, import shares generated by other protocols.
//
// @param ctx, the evaluation context.
// @param x, the type may not be of current protocol's type, but
//            it should be a Secret type.
Value import_s(SPUContext* ctx, const Value& x);

// Export a secret value as a given type.
//
// The export stage can be used:
// - strip party private information.
// - sharing conversion, export shares for other protocols.
//
// @param ctx, the evaluation context.
// @param x, the input should be one of current protocol's type.
// @param as_type, the target type, it should be a Secret type.
Value export_s(SPUContext* ctx, const Value& x, const Type& t);

// Get the common type of secrets.
//
// Unlike public types, which has only one form, secrets has multiple storage
// formats, like AShare/BShare, which make them not concatable.
//
// This api calculate the common type.
Type common_type_s(SPUContext* ctx, const Type& a, const Type& b);
Type common_type_v(SPUContext* ctx, const Type& a, const Type& b);
Value cast_type_s(SPUContext* ctx, const Value& frm, const Type& to_type);

// Make a public variable with given plaintext input.
//
// All parties knowns the value.
Value make_p(SPUContext* ctx, uint128_t init, const Shape& shape);

// parties random a public together.
Value rand_p(SPUContext* ctx, const Shape& shape);
Value rand_s(SPUContext* ctx, const Shape& shape);

// Compute bitwise_not(invert) of a value in ring 2k space.
Value not_p(SPUContext* ctx, const Value& x);
Value not_s(SPUContext* ctx, const Value& x);
Value not_v(SPUContext* ctx, const Value& x);

Value msb_p(SPUContext* ctx, const Value& x);
Value msb_s(SPUContext* ctx, const Value& x);
Value msb_v(SPUContext* ctx, const Value& x);

Value equal_pp(SPUContext* ctx, const Value& x, const Value& y);
OptionalAPI<Value> equal_sp(SPUContext* ctx, const Value& x, const Value& y);
OptionalAPI<Value> equal_ss(SPUContext* ctx, const Value& x, const Value& y);

Value add_ss(SPUContext* ctx, const Value& x, const Value& y);
Value add_sv(SPUContext* ctx, const Value& x, const Value& y);
Value add_sp(SPUContext* ctx, const Value& x, const Value& y);
// Note: add_vv may result in secret or private.
Value add_vv(SPUContext* ctx, const Value& x, const Value& y);
Value add_vp(SPUContext* ctx, const Value& x, const Value& y);
Value add_pp(SPUContext* ctx, const Value& x, const Value& y);

Value mul_ss(SPUContext* ctx, const Value& x, const Value& y);
Value mul_sv(SPUContext* ctx, const Value& x, const Value& y);
Value mul_sp(SPUContext* ctx, const Value& x, const Value& y);
Value mul_vv(SPUContext* ctx, const Value& x, const Value& y);
Value mul_vp(SPUContext* ctx, const Value& x, const Value& y);
Value mul_pp(SPUContext* ctx, const Value& x, const Value& y);

Value mmul_ss(SPUContext* ctx, const Value& x, const Value& y);
Value mmul_sv(SPUContext* ctx, const Value& x, const Value& y);
Value mmul_sp(SPUContext* ctx, const Value& x, const Value& y);
Value mmul_vv(SPUContext* ctx, const Value& x, const Value& y);
Value mmul_vp(SPUContext* ctx, const Value& x, const Value& y);
Value mmul_pp(SPUContext* ctx, const Value& x, const Value& y);

Value and_ss(SPUContext* ctx, const Value& x, const Value& y);
Value and_sv(SPUContext* ctx, const Value& x, const Value& y);
Value and_sp(SPUContext* ctx, const Value& x, const Value& y);
Value and_vv(SPUContext* ctx, const Value& x, const Value& y);
Value and_vp(SPUContext* ctx, const Value& x, const Value& y);
Value and_pp(SPUContext* ctx, const Value& x, const Value& y);

Value xor_ss(SPUContext* ctx, const Value& x, const Value& y);
Value xor_sv(SPUContext* ctx, const Value& x, const Value& y);
Value xor_sp(SPUContext* ctx, const Value& x, const Value& y);
Value xor_vv(SPUContext* ctx, const Value& x, const Value& y);
Value xor_vp(SPUContext* ctx, const Value& x, const Value& y);
Value xor_pp(SPUContext* ctx, const Value& x, const Value& y);

Value lshift_s(SPUContext* ctx, const Value& x, size_t nbits);
Value lshift_v(SPUContext* ctx, const Value& x, size_t nbits);
Value lshift_p(SPUContext* ctx, const Value& x, size_t nbits);

Value rshift_s(SPUContext* ctx, const Value& x, size_t nbits);
Value rshift_v(SPUContext* ctx, const Value& x, size_t nbits);
Value rshift_p(SPUContext* ctx, const Value& x, size_t nbits);

Value arshift_s(SPUContext* ctx, const Value& x, size_t nbits);
Value arshift_v(SPUContext* ctx, const Value& x, size_t nbits);
Value arshift_p(SPUContext* ctx, const Value& x, size_t nbits);

Value trunc_s(SPUContext* ctx, const Value& x, size_t nbits, SignType sign);
Value trunc_v(SPUContext* ctx, const Value& x, size_t nbits, SignType sign);
Value trunc_p(SPUContext* ctx, const Value& x, size_t nbits, SignType sign);

// Reverse bit, like MIPS BITREV instruction, and linux bitrev library.
Value bitrev_s(SPUContext* ctx, const Value& x, size_t start, size_t end);
Value bitrev_v(SPUContext* ctx, const Value& x, size_t start, size_t end);
Value bitrev_p(SPUContext* ctx, const Value& x, size_t start, size_t end);

//////////////////////////////////////////////////////////////////////////////
// TODO: Formalize these permutation APIs
//////////////////////////////////////////////////////////////////////////////
// Generate a 1-D random secret permutation. Here secret means the permutation
// is composed of a series of individual permutations hold by each party.
// Specifically, if Perm = Perm1(Perm0), then party0 holds Perm0 and party1
// holds Perm1
OptionalAPI<Value> rand_perm_s(SPUContext* ctx, const Shape& shape);

// Permute 1-D x with permutation perm
// ret[i] = x[perm[i]]
OptionalAPI<Value> perm_sp(SPUContext* ctx, const Value& x, const Value& perm);
OptionalAPI<Value> perm_ss(SPUContext* ctx, const Value& x, const Value& perm);
Value perm_pp(SPUContext* ctx, const Value& x, const Value& perm);
Value perm_vv(SPUContext* ctx, const Value& x, const Value& perm);

// Inverse permute 1-D x with permutation perm
// ret[perm[i]] = x[i]
OptionalAPI<Value> inv_perm_sp(SPUContext* ctx, const Value& x,
                               const Value& perm);
OptionalAPI<Value> inv_perm_ss(SPUContext* ctx, const Value& x,
                               const Value& perm);
OptionalAPI<Value> inv_perm_sv(SPUContext* ctx, const Value& x,
                               const Value& perm);
Value inv_perm_pp(SPUContext* ctx, const Value& x, const Value& perm);
Value inv_perm_vv(SPUContext* ctx, const Value& x, const Value& perm);

/*---------------------------- Value APIs ----------------------------------*/
// Broadcast a Value
Value broadcast(SPUContext* ctx, const Value& in, const Shape& to_shape,
                const Axes& in_dims);

// Resahpe a Value
Value reshape(SPUContext* ctx, const Value& in, const Shape& to_shape);

// Extract a slice from a Value
Value extract_slice(SPUContext* ctx, const Value& in,
                    const Index& start_indices, const Index& end_indices,
                    const Strides& strides);

// Update a Value at index with given value
Value update_slice(SPUContext* ctx, const Value& in, const Value& update,
                   const Index& start_indices);

// Transpose a Value
Value transpose(SPUContext* ctx, const Value& in, const Axes& permutation);

// Reverse a Value at dimensions
Value reverse(SPUContext* ctx, const Value& in, const Axes& dimensions);

// Fill a Value with input value
Value fill(SPUContext* ctx, const Value& in, const Shape& to_shape);

// Pad a Value
Value pad(SPUContext* ctx, const Value& in, const Value& padding_value,
          const Sizes& edge_padding_low, const Sizes& edge_padding_high,
          const Sizes& interior_padding);

// Concate Values at an axis
Value concatenate(SPUContext* ctx, const std::vector<Value>& values,
                  int64_t axis);
}  // namespace spu::mpc
