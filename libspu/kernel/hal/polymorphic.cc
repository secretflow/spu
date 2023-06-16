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

#include "libspu/kernel/hal/polymorphic.h"

#include "fmt/format.h"
#include "fmt/ostream.h"

#include "libspu/core/context.h"
#include "libspu/core/encoding.h"  // for bitcast
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp.h"
#include "libspu/kernel/hal/integer.h"
#include "libspu/kernel/hal/ring.h"  // for fast fxp x int
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"

// TODO: handle dtype promotion inside integer dtypes.
namespace spu::kernel::hal {
namespace {

using UnaryOp = Value(SPUContext*, const Value&);
using BinaryOp = Value(SPUContext*, const Value&, const Value&);

DataType common_dtype(DataType lhs, DataType rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  return std::max(lhs, rhs);  // Always results to higher rank type
}

template <BinaryOp* FnFxp, BinaryOp* FnInt>
Value dtypeBinaryDispatch(std::string_view op_name, SPUContext* ctx,
                          const Value& x, const Value& y) {
  // Promote int to fxp if mismatch.
  if (x.isInt() && y.isInt()) {
    auto common_type = common_dtype(x.dtype(), y.dtype());
    return FnInt(ctx, dtype_cast(ctx, x, common_type),
                 dtype_cast(ctx, y, common_type));
  } else if (x.isInt() && y.isFxp()) {
    return FnFxp(ctx, dtype_cast(ctx, x, y.dtype()), y);
  } else if (x.isFxp() && y.isInt()) {
    return FnFxp(ctx, x, dtype_cast(ctx, y, x.dtype()));
  } else if (x.isFxp() && y.isFxp()) {
    return FnFxp(ctx, x, y);
  } else {
    SPU_THROW("unsupported op {} for x={}, y={}", op_name, x, y);
  }
}

template <UnaryOp* FnFxp, UnaryOp* FnInt>
Value dtypeUnaryDispatch(std::string_view op_name, SPUContext* ctx,
                         const Value& x) {
  // Promote int to fxp if mismatch.
  if (x.isInt()) {
    return FnInt(ctx, x);
  } else if (x.isFxp()) {
    return FnFxp(ctx, x);
  } else {
    SPU_THROW("unsupported op {} for x={}", op_name, x);
  }
}

bool isCrossIntFxp(const Value& x, const Value& y) {
  return (x.isFxp() && y.isInt()) || (x.isInt() && y.isFxp());
}

}  // namespace

Value add(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  return dtypeBinaryDispatch<f_add, i_add>("add", ctx, x, y);
}

Value sub(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  return dtypeBinaryDispatch<f_sub, i_sub>("sub", ctx, x, y);
}

Value mixed_mul(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  auto new_dtype = x.isFxp() ? x.dtype() : y.dtype();
  return _mul(ctx, x, y).setDtype(new_dtype);
}

Value mixed_mmul(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  auto new_dtype = x.isFxp() ? x.dtype() : y.dtype();
  return _mmul(ctx, x, y).setDtype(new_dtype);
}

Value mul(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  // fast dispatch, avoid truncation cost
  if (isCrossIntFxp(x, y)) {
    return mixed_mul(ctx, x, y);
  }

  return dtypeBinaryDispatch<f_mul, i_mul>("mul", ctx, x, y);
}

Value matmul(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  // fast dispatch, avoid truncation cost
  if (isCrossIntFxp(x, y)) {
    return mixed_mmul(ctx, x, y);
  }

  return dtypeBinaryDispatch<f_mmul, i_mmul>("mmul", ctx, x, y);
}

Value logical_not(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_LEAF(ctx, in);

  auto _k1 = _constant(ctx, 1, in.shape());

  // TODO: we should NOT dispatch according to AShr/BShr trait here.
  if (in.storage_type().isa<BShare>()) {
    return _xor(ctx, in, _k1).setDtype(in.dtype());
  } else {
    return _sub(ctx, _k1, in).setDtype(in.dtype());
  }
}

Value equal(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape(), "x = {}, y = {}", x, y);

  return dtypeBinaryDispatch<f_equal, i_equal>("equal", ctx, x, y);
}

Value not_equal(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return logical_not(ctx, equal(ctx, x, y));
}

Value less(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return dtypeBinaryDispatch<f_less, i_less>("less", ctx, x, y);
}

Value less_equal(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  // not (x > y)
  return logical_not(ctx, greater(ctx, x, y));
}

Value greater(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return less(ctx, y, x);
}

Value greater_equal(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  // not (x < y)
  return logical_not(ctx, less(ctx, x, y));
}

Value negate(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return dtypeUnaryDispatch<f_negate, i_negate>("negate", ctx, x);
}

Value abs(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return dtypeUnaryDispatch<f_abs, i_abs>("abs", ctx, x);
}

Value exp(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_exp(ctx, in);
}

Value select(SPUContext* ctx, const Value& pred, const Value& a,
             const Value& b) {
  SPU_TRACE_HAL_DISP(ctx, pred, a, b);

  SPU_ENFORCE(pred.isInt());
  SPU_ENFORCE(a.shape() == b.shape());
  SPU_ENFORCE(a.dtype() == b.dtype());

  // To ensure pred is {0, 1} on integer range, we have to promote pred to an
  // actual integer here. Otherwise, when we use pred to do computation the
  // result will be wrong
  return _mux(ctx, pred, a, b).setDtype(a.dtype());
}

Value bitwise_and(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  SPU_ENFORCE(x.isInt() && y.isInt());
  SPU_ENFORCE(x.shape() == y.shape());

  return _and(ctx, x, y).setDtype(x.dtype());
}

Value bitwise_xor(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.isInt() && y.isInt());
  SPU_ENFORCE(x.shape() == y.shape());

  return _xor(ctx, x, y).setDtype(x.dtype());
}

Value bitwise_or(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.isInt() && y.isInt());
  SPU_ENFORCE(x.shape() == y.shape());

  return _or(ctx, x, y).setDtype(x.dtype());
}

Value bitwise_not(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  return _not(ctx, in).setDtype(in.dtype());
}

Value logistic(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_sigmoid(ctx, in);
}

Value log(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_log(ctx, in);
}

Value log1p(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_log1p(ctx, in);
}

Value reciprocal(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  SPU_ENFORCE(in.isFxp());

  return f_reciprocal(ctx, in);
}

Value floor(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_floor(ctx, in);
}

Value ceil(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_ceil(ctx, in);
}

Value max(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  SPU_ENFORCE(x.dtype() == y.dtype());

  return select(ctx, greater(ctx, x, y), x, y);
}

Value min(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  SPU_ENFORCE(x.dtype() == y.dtype());

  return select(ctx, less(ctx, x, y), x, y);
}

Value power(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  if (x.isInt() && y.isInt()) {
    auto x_f = dtype_cast(ctx, x, DT_F32);
    auto y_f = dtype_cast(ctx, y, DT_F32);
    auto ret = power(ctx, x_f, y_f);
    return dtype_cast(ctx, ret, x.dtype());
  }

  // x^y = e^(y*ln(x))
  return exp(ctx, mul(ctx, y, log(ctx, x)));
}

Value idiv(SPUContext* ctx, const Value& x, const Value& y) {
  auto sign_x = sign(ctx, x);
  auto sign_y = sign(ctx, y);

  auto abs_x = mul(ctx, x, sign_x);
  auto abs_y = mul(ctx, y, sign_y);

  Value q;
  {
    const auto x_f = dtype_cast(ctx, abs_x, DT_F32);
    const auto y_f = dtype_cast(ctx, abs_y, DT_F32);

    auto approx_q = div(ctx, x_f, y_f);

    // Due to truncation error and limited precision of fxp, the approximate
    // quotient should be corrected
    approx_q = dtype_cast(ctx, approx_q, x.dtype());

    auto approx_x = mul(ctx, abs_y, approx_q);

    // if (approx_q + 1) * y <= x, then ++approx_q;
    auto v1 = less_equal(ctx, add(ctx, approx_x, abs_y), abs_x);
    // if approx_q * y > x, then --approx_q;
    auto v2 = greater(ctx, approx_x, abs_x);

    q = sub(ctx, add(ctx, approx_q, v1), v2);
  }

  return mul(ctx, q, mul(ctx, sign_x, sign_y));
}

Value div(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  if (x.isInt() && y.isInt()) {
    return idiv(ctx, x, y);
  }

  // Kind of a hack to compute max dtype
  auto dtype = std::max(x.dtype(), y.dtype());

  const auto x_f = dtype_cast(ctx, x, dtype);
  const auto y_f = dtype_cast(ctx, y, dtype);

#define F_DIV_WITH_DIRECT_GOLDSCHMIDT_METHOD
#ifdef F_DIV_WITH_DIRECT_GOLDSCHMIDT_METHOD
  auto res_f = f_div(ctx, x_f, y_f);
#else
  auto res_f = mul(ctx, x, reciprocal(ctx, y));
#endif

  return res_f;
}

Value clamp(SPUContext* ctx, const Value& x, const Value& minv,
            const Value& maxv) {
  SPU_TRACE_HAL_DISP(ctx, x, minv, maxv);

  // TODO(jint) are these type contraint required?
  SPU_ENFORCE(minv.dtype() == maxv.dtype());
  SPU_ENFORCE(minv.dtype() == x.dtype());

  return min(ctx, max(ctx, minv, x), maxv);
}

Value bitcast(SPUContext* ctx, const Value& x, DataType dtype) {
  SPU_TRACE_HAL_DISP(ctx, x, dtype);

  // FIXME(jint) should we directly use fixed point binary expr for bitcast?
  return Value(x.data().clone(), dtype);
}

Value left_shift(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_DISP(ctx, x, bits);

  return _lshift(ctx, x, bits).setDtype(x.dtype());
}

Value right_shift_logical(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_DISP(ctx, x, bits);

  return _rshift(ctx, x, bits).setDtype(x.dtype());
}

Value right_shift_arithmetic(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_DISP(ctx, x, bits);

  return _arshift(ctx, x, bits).setDtype(x.dtype());
}

Value log2(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_log2(ctx, in);
}

Value exp2(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_exp2(ctx, x);
}

Value tanh(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_tanh(ctx, x);
}

Value rsqrt(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_rsqrt(ctx, x);
}

Value sqrt(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_sqrt(ctx, x);
}

Value sign(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return _sign(ctx, x).setDtype(DT_I8);
}

Value conv2d(SPUContext* ctx, const Value& x, const Value& y,
             absl::Span<const int64_t> window_strides,
             absl::Span<const int64_t> result_shape) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  if (x.isFxp() && y.isFxp()) {
    return f_conv2d(ctx, x, y, window_strides, result_shape);
  }

  if (x.isInt() && y.isInt()) {
    auto common_type = common_dtype(x.dtype(), y.dtype());
    return i_conv2d(ctx, dtype_cast(ctx, x, common_type),
                    dtype_cast(ctx, y, common_type), window_strides,
                    result_shape);
  }
  SPU_THROW("unsupported op {} for x={}, y={}", "conv2d", x, y);
}

}  // namespace spu::kernel::hal
