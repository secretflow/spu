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

#include "libspu/kernel/hal/integer.h"

#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hal {

// FIXME(jint) handle cross (int) dtype binary operators
#define ENSURE_INT_AND_DTYPE_MATCH(X, Y)                           \
  SPU_ENFORCE((X).isInt(), "expect lhs int, got {]", (X).dtype()); \
  SPU_ENFORCE((Y).isInt(), "expect rhs int, got {]", (X).dtype());

#define DEF_UNARY_OP(Name, Fn2K)                             \
  Value Name(SPUContext* ctx, const Value& x) {              \
    SPU_TRACE_HAL_LEAF(ctx, x);                              \
                                                             \
    SPU_ENFORCE(x.isInt(), "expect Int, got {]", x.dtype()); \
    return Fn2K(ctx, x).setDtype(x.dtype());                 \
  }

/*           name,     op_2k */
DEF_UNARY_OP(i_negate, _negate)

#undef DEF_UNARY_OP

#define DEF_BINARY_OP(Name, Fn2K)                               \
  Value Name(SPUContext* ctx, const Value& x, const Value& y) { \
    SPU_TRACE_HAL_LEAF(ctx, x, y);                              \
    ENSURE_INT_AND_DTYPE_MATCH(x, y);                           \
    return Fn2K(ctx, x, y).setDtype(x.dtype());                 \
  }

DEF_BINARY_OP(i_add, _add)
DEF_BINARY_OP(i_mul, _mul)
DEF_BINARY_OP(i_mmul, _mmul)

Value i_less(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  ENSURE_INT_AND_DTYPE_MATCH(x, y);

  return _less(ctx, x, y).setDtype(DT_I1);
}

#undef DEF_BINARY_OP

Value i_abs(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isInt());

  // abs(x) = _sign(x) * x
  return _mul(ctx, _sign(ctx, x), x).setDtype(x.dtype());
}

Value i_equal(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isInt());
  SPU_ENFORCE(y.isInt());

  return _equal(ctx, x, y).setDtype(DT_I1);
}

Value i_sub(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isInt());
  SPU_ENFORCE(y.isInt());
  return i_add(ctx, x, i_negate(ctx, y));
}

Value i_conv2d(SPUContext* ctx, const Value& x, const Value& y,
               absl::Span<const int64_t> window_strides,
               absl::Span<const int64_t> result_shape) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  ENSURE_INT_AND_DTYPE_MATCH(x, y);
  return _conv2d(ctx, x, y, window_strides, result_shape).setDtype(x.dtype());
}

}  // namespace spu::kernel::hal
