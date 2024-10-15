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

NdArrayRef gfmp_rand(FieldType field, const Shape& shape);
NdArrayRef gfmp_rand(FieldType field, const Shape& shape, uint128_t prg_seed,
                     uint64_t* prg_counter);

NdArrayRef gfmp_zeros(FieldType field, const Shape& shape);

NdArrayRef gfmp_mod(const NdArrayRef& x);
void gfmp_mod_(NdArrayRef& x);

NdArrayRef gfmp_mul_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_mul_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_div_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_div_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_add_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_add_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_sub_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_sub_mod_(NdArrayRef& x, const NdArrayRef& y);

NdArrayRef gfmp_exp_mod(const NdArrayRef& x, const NdArrayRef& y);
void gfmp_exp_mod_(NdArrayRef& x, const NdArrayRef& y);

}  // namespace spu::mpc
