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

#include "spu/mpc/interfaces.h"

namespace spu::mpc {

#define IMPL_UNARY_OP(NAME)                        \
  ArrayRef NAME(Object* ctx, const ArrayRef& in) { \
    return ctx->call(#NAME, in);                   \
  }

#define IMPL_UNARY_OP_WITH_SIZE(NAME)                         \
  ArrayRef NAME(Object* ctx, const ArrayRef& in, size_t sz) { \
    return ctx->call(#NAME, in, sz);                          \
  }

#define IMPL_UNARY_OP_WITH_2SIZE(NAME)                                     \
  ArrayRef NAME(Object* ctx, const ArrayRef& in, size_t sz1, size_t sz2) { \
    return ctx->call(#NAME, in, sz1, sz2);                                 \
  }

#define IMPL_BINARY_OP(NAME)                                         \
  ArrayRef NAME(Object* ctx, const ArrayRef& x, const ArrayRef& y) { \
    return ctx->call(#NAME, x, y);                                   \
  }

#define IMPL_BINARY_OP_WITH_3SIZE(NAME)                                        \
  ArrayRef NAME(Object* ctx, const ArrayRef& x, const ArrayRef& y, size_t sz1, \
                size_t sz2, size_t sz3) {                                      \
    return ctx->call(#NAME, x, y, sz1, sz2, sz3);                              \
  }

IMPL_UNARY_OP(p2s)
IMPL_UNARY_OP(s2p)
IMPL_UNARY_OP(not_s)
IMPL_UNARY_OP(not_p)
IMPL_UNARY_OP(msb_s)
IMPL_UNARY_OP(msb_p)
IMPL_UNARY_OP(eqz_s)
IMPL_UNARY_OP(eqz_p)
IMPL_UNARY_OP_WITH_SIZE(lshift_p)
IMPL_UNARY_OP_WITH_SIZE(lshift_s)
IMPL_UNARY_OP_WITH_SIZE(rshift_p)
IMPL_UNARY_OP_WITH_SIZE(rshift_s)
IMPL_UNARY_OP_WITH_SIZE(arshift_p)
IMPL_UNARY_OP_WITH_SIZE(arshift_s)
IMPL_UNARY_OP_WITH_SIZE(truncpr_s)
IMPL_UNARY_OP_WITH_2SIZE(bitrev_s)
IMPL_UNARY_OP_WITH_2SIZE(bitrev_p)
IMPL_BINARY_OP(add_pp)
IMPL_BINARY_OP(add_sp)
IMPL_BINARY_OP(add_ss)
IMPL_BINARY_OP(mul_pp)
IMPL_BINARY_OP(mul_sp)
IMPL_BINARY_OP(mul_ss)
IMPL_BINARY_OP(and_pp)
IMPL_BINARY_OP(and_sp)
IMPL_BINARY_OP(and_ss)
IMPL_BINARY_OP(xor_pp)
IMPL_BINARY_OP(xor_sp)
IMPL_BINARY_OP(xor_ss)
IMPL_BINARY_OP_WITH_3SIZE(mmul_pp)
IMPL_BINARY_OP_WITH_3SIZE(mmul_sp)
IMPL_BINARY_OP_WITH_3SIZE(mmul_ss)
// TODO: move field into ctx
ArrayRef rand_p(Object* ctx, FieldType field, size_t sz) {
  return ctx->call("rand_p", field, sz);
}
ArrayRef rand_s(Object* ctx, FieldType field, size_t sz) {
  return ctx->call("rand_s", field, sz);
}

// Arithmetic & Boolean ops
// TODO: zero_a
ArrayRef zero_a(Object* ctx, FieldType field, size_t sz) {
  return ctx->call("zero_a", field, sz);
}
IMPL_UNARY_OP(a2p)
IMPL_UNARY_OP(p2a)
IMPL_UNARY_OP(not_a)
IMPL_UNARY_OP(msb_a)
IMPL_BINARY_OP(add_ap)
IMPL_BINARY_OP(add_aa)
IMPL_BINARY_OP(mul_ap)
IMPL_BINARY_OP(mul_aa)
IMPL_UNARY_OP_WITH_SIZE(lshift_a)
IMPL_UNARY_OP_WITH_SIZE(truncpr_a)
IMPL_BINARY_OP_WITH_3SIZE(mmul_ap)
IMPL_BINARY_OP_WITH_3SIZE(mmul_aa)

// TODO: zero_b
ArrayRef zero_b(Object* ctx, FieldType field, size_t sz) {
  return ctx->call("zero_b", field, sz);
}
IMPL_UNARY_OP(b2p)
IMPL_UNARY_OP(p2b)
IMPL_UNARY_OP(a2b)
IMPL_UNARY_OP(b2a)
IMPL_BINARY_OP(and_bp)
IMPL_BINARY_OP(and_bb)
IMPL_BINARY_OP(xor_bp)
IMPL_BINARY_OP(xor_bb)
IMPL_UNARY_OP_WITH_SIZE(lshift_b)
IMPL_UNARY_OP_WITH_SIZE(rshift_b)
IMPL_UNARY_OP_WITH_SIZE(arshift_b)
IMPL_UNARY_OP_WITH_2SIZE(bitrev_b)
IMPL_BINARY_OP(add_bb)

//
}  // namespace spu::mpc
