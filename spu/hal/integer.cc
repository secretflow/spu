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

#include "spu/hal/integer.h"

#include "spu/hal/prot_wrapper.h"
#include "spu/hal/ring.h"

namespace spu::hal {

#define ENSURE_INT_AND_DTYPE_MATCH(X, Y)                                   \
  YASL_ENFORCE(X.isInt(), "expect lhs int, got {]", X.dtype());            \
  YASL_ENFORCE(Y.isInt(), "expect rhs int, got {]", X.dtype());            \
  YASL_ENFORCE(X.dtype() == Y.dtype(), "dtype mismatch {}, {}", X.dtype(), \
               Y.dtype());

#define DEF_UNARY_OP(Name, Fn2K)                              \
  Value Name(HalContext* ctx, const Value& x) {               \
    SPU_TRACE_HAL(ctx, x);                                    \
    SPU_PROFILE_OP(ctx, x);                                   \
                                                              \
    YASL_ENFORCE(x.isInt(), "expect Int, got {]", x.dtype()); \
    return Fn2K(ctx, x).setDtype(x.dtype());                  \
  }

/*           name,     op_2k */
DEF_UNARY_OP(i_negate, _negate)

#undef DEF_UNARY_OP

#define DEF_BINARY_OP(Name, Fn2K)                               \
  Value Name(HalContext* ctx, const Value& x, const Value& y) { \
    SPU_TRACE_HAL(ctx, x, y);                                   \
    SPU_PROFILE_OP(ctx, x, y);                                  \
    ENSURE_INT_AND_DTYPE_MATCH(x, y);                           \
    return Fn2K(ctx, x, y).setDtype(x.dtype());                 \
  }

DEF_BINARY_OP(i_add, _add)
DEF_BINARY_OP(i_mul, _mul)
DEF_BINARY_OP(i_mmul, _mmul)

Value i_less(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL(ctx, x, y);
  SPU_PROFILE_OP(ctx, x, y);
  ENSURE_INT_AND_DTYPE_MATCH(x, y);

  return _less(ctx, x, y).setDtype(DT_I1);
}

#undef DEF_BINARY_OP

Value i_abs(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL(ctx, x);
  SPU_PROFILE_OP(ctx, x);

  YASL_ENFORCE(x.isInt());

  // abs(x) = _sign(x) * x
  return _mul(ctx, _sign(ctx, x), x).setDtype(x.dtype());
}

Value i_equal(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL(ctx, x, y);
  SPU_PROFILE_OP(ctx, x, y);

  YASL_ENFORCE(x.isInt());
  YASL_ENFORCE(y.isInt());

  return _eqz(ctx, i_sub(ctx, x, y)).setDtype(DT_I1);
}

Value i_sub(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL(ctx, x, y);
  SPU_PROFILE_OP(ctx, x, y);

  YASL_ENFORCE(x.isInt());
  YASL_ENFORCE(y.isInt());
  return i_add(ctx, x, i_negate(ctx, y));
}

}  // namespace spu::hal
