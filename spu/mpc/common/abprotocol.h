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

#include "spu/mpc/api.h"
#include "spu/mpc/object.h"
#include "spu/mpc/util/circuits.h"

namespace spu::mpc {

ArrayRef a2p(Object* ctx, const ArrayRef&);
ArrayRef p2a(Object* ctx, const ArrayRef&);

// FIXME(jint) drop FieldType.
ArrayRef zero_a(Object* ctx, FieldType, size_t);
ArrayRef rand_a(Object* ctx, FieldType, size_t);
ArrayRef rand_b(Object* ctx, FieldType, size_t);

ArrayRef not_a(Object* ctx, const ArrayRef&);
ArrayRef msb_a(Object* ctx, const ArrayRef&);

ArrayRef add_ap(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef add_aa(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef mul_ap(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef mul_aa(Object* ctx, const ArrayRef&, const ArrayRef&);
ArrayRef mul_a1b(Object* ctx, const ArrayRef&, const ArrayRef&);

ArrayRef lshift_a(Object* ctx, const ArrayRef&, size_t);
ArrayRef truncpr_a(Object* ctx, const ArrayRef&, size_t);

ArrayRef mmul_ap(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);
ArrayRef mmul_aa(Object* ctx, const ArrayRef&, const ArrayRef&, size_t, size_t,
                 size_t);

ArrayRef zero_b(Object* ctx, FieldType, size_t);

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

ArrayRef bitrev_b(Object* ctx, const ArrayRef&, size_t, size_t);

ArrayRef add_bb(Object* ctx, const ArrayRef&, const ArrayRef&);

void regABKernels(Object* obj);

CircuitBasicBlock<ArrayRef> makeABProtBasicBlock(Object* ctx);

}  // namespace spu::mpc
