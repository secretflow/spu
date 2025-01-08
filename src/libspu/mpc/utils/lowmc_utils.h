// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc {

// recap: For n*k binary matrix, we regard it as a (n,) NdArrayRef, each
// element is a row (k bits).
// For n*k binary matrix A, k-bits binary vector B, n-bits C = dot(A, B):
// C[r] = bit_parity(A[r] & B) for r in [0, n)
NdArrayRef dot_product_gf2(const NdArrayRef& x, const NdArrayRef& y,
                           FieldType to_field);

// Key is strongly dependent on the sharing semantics, so we leave the key
// setting procedure in kernel layer.
// Here we implement the plaintext scheme, which can also be used in n-n
// xor sharing semantics (e.g. SEMI2K, CHEETAH, etc.).
// For ABY3, can call this function twice to get two sharing of the round
// keys.
std::vector<NdArrayRef> generate_round_keys(
    const std::vector<NdArrayRef>& key_matrices, uint128_t key, uint64_t rounds,
    FieldType to_field);

// we only support three choices for data complexity now.
// n <= 2^20 (about 1 million); n <= 2^30 (about 1 billion); n <= 2^40
// (about 1 trillion)
int64_t get_data_complexity(int64_t n);

}  // namespace spu::mpc
