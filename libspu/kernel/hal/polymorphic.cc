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

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/fxp_cleartext.h"
#include "libspu/kernel/hal/integer.h"
#include "libspu/kernel/hal/ring.h"  // for fast fxp x int
#include "libspu/kernel/hal/type_cast.h"

// TODO: handle dtype promotion inside integer dtypes.
namespace spu::kernel::hal {
namespace {

DataType common_dtype(DataType lhs, DataType rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  return std::max(lhs, rhs);  // Always results to higher rank type
}

template <typename FnFxp, typename FnInt, typename... Args>
Value dtypeBinaryDispatch(std::string_view op_name, FnFxp&& fn_fxp,
                          FnInt&& fn_int, SPUContext* ctx, const Value& x,
                          const Value& y, Args&&... args) {
  // Promote int to fxp if mismatch.
  if (x.isInt() && y.isInt()) {
    auto common_type = common_dtype(x.dtype(), y.dtype());
    auto xx = dtype_cast(ctx, x, common_type);
    auto yy = dtype_cast(ctx, y, common_type);
    return fn_int(ctx, xx, yy, std::forward<Args>(args)...);
  } else if (x.isInt() && y.isFxp()) {
    auto xx = dtype_cast(ctx, x, y.dtype());
    return fn_fxp(ctx, xx, y, std::forward<Args>(args)...);
  } else if (x.isFxp() && y.isInt()) {
    auto yy = dtype_cast(ctx, y, x.dtype());
    return fn_fxp(ctx, x, yy, std::forward<Args>(args)...);
  } else if (x.isFxp() && y.isFxp()) {
    auto common_type = common_dtype(x.dtype(), y.dtype());
    auto xx = dtype_cast(ctx, x, common_type);
    auto yy = dtype_cast(ctx, y, common_type);
    return fn_fxp(ctx, xx, yy, std::forward<Args>(args)...);
  } else {
    SPU_THROW("unsupported op {} for x={}, y={}", op_name, x, y);
  }
}

template <typename FnFxp, typename FnInt, typename... Args>
Value dtypeUnaryDispatch(std::string_view op_name, FnFxp&& fn_fxp,
                         FnInt&& fn_int, SPUContext* ctx, const Value& x,
                         Args&&... args) {
  // Promote int to fxp if mismatch.
  if (x.isInt()) {
    return fn_int(ctx, x, std::forward<Args>(args)...);
  } else if (x.isFxp()) {
    return fn_fxp(ctx, x, std::forward<Args>(args)...);
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
  return dtypeBinaryDispatch("add", f_add, i_add, ctx, x, y);
}

Value sub(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  return dtypeBinaryDispatch("sub", f_sub, i_sub, ctx, x, y);
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

static Value f_mul_impl(SPUContext* ctx, const Value& x, const Value& y) {
  return f_mul(ctx, x, y, SignType::Unknown);
}

Value mul(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  // fast dispatch, avoid truncation cost
  if (isCrossIntFxp(x, y)) {
    return mixed_mul(ctx, x, y);
  }

  return dtypeBinaryDispatch("mul", f_mul_impl, i_mul, ctx, x, y);
}

Value square(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return dtypeUnaryDispatch("square", f_square, i_square, ctx, x);
}

Value matmul(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  // fast dispatch, avoid truncation cost
  if (isCrossIntFxp(x, y)) {
    return mixed_mmul(ctx, x, y);
  }

  return dtypeBinaryDispatch("mmul", f_mmul, i_mmul, ctx, x, y);
}

Value tensordot(SPUContext* ctx, const Value& x, const Value& y,
                const Index& ix, const Index& iy) {
  SPU_TRACE_HAL_DISP(ctx, x, y, ix, iy);
  return dtypeBinaryDispatch("tensordot", f_tensordot, i_tensordot, ctx, x, y,
                             ix, iy);
}

Value conv2d(SPUContext* ctx, const Value& x, const Value& y,
             const Strides& window_strides) {
  SPU_TRACE_HAL_DISP(ctx, x, y, window_strides);

  return dtypeBinaryDispatch("conv2d", f_conv2d, i_conv2d, ctx, x, y,
                             window_strides);
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

  return dtypeBinaryDispatch("equal", f_equal, i_equal, ctx, x, y);
}

Value not_equal(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return logical_not(ctx, equal(ctx, x, y));
}

Value less(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return dtypeBinaryDispatch("less", f_less, i_less, ctx, x, y);
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

  return dtypeUnaryDispatch("negate", f_negate, i_negate, ctx, x);
}

Value abs(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return dtypeUnaryDispatch("abs", f_abs, i_abs, ctx, x);
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

  if (x.isInt()) {
    // ref:
    // https://github.com/openxla/stablehlo/blob/main/stablehlo/reference/Element.cpp#L912
    // Although there are some "strange" semantics in stablehlo, we still follow
    // them yet:
    //   1. when x is int, then the return value must be int type.
    //   2. if x is int, then y must be int
    //   3. if x is int and y<0, then
    //      a. when |x|!=1, then always return 0;
    //      b. when |x|=1, then y=|y|;
    //
    // However, for jax.numpy.power, it behaves differently:
    //   1. if any x or y is float, then both x and y will be upcast to float.
    //   2. if both x and y are int, then y must be non-negative.
    SPU_ENFORCE(y.isInt(), "when base is int, then y must be int.");
    auto k0 = _constant(ctx, 0, x.shape());
    auto k1 = _constant(ctx, 1, x.shape());
    const auto bit_width = SizeOf(ctx->getField()) * 8;

    auto y_b = _prefer_b(ctx, y);
    auto msb_y = _rshift(ctx, y_b, {static_cast<int64_t>(bit_width - 1)});
    auto x_abs1 = _equal(ctx, abs(ctx, x), k1);

    auto ret = _constant(ctx, 1, x.shape());
    // To compute ret = x^y,
    // although y has `bit_width` bits, we only consider `y_bits` bits here.
    // The reason are two folds (recall that both x and y are int):
    //   1. if |x|>1, then `ret` will OVERFLOW/UNDERFLOW if y>63 (e.g. FM64),
    //   which means the valid bits of y can't exceed `log(bit_width - 1)` .
    //   2. if |x|=1:
    //      a). x=1, then we always get `ret`=1;
    //      b). x=-1, then the sign of `ret` is decided on the LSB of y;
    // So we can "truncate" y to `y_bits` bits safely.
    const size_t y_bits = Log2Ceil(bit_width - 1);

    auto base = x;
    // TODO: do this in parallel
    // To compute x^y, it is necessary to compute all x^(2^idx), we use base
    // (init as `x`) to store it, update base to base*base till last
    // iteration, and multiply all these numbers according to y_{idx}.
    // e.g. y=0101, then ret = (x) * (1) * (x^(2^2)) * (1) = x^5
    for (size_t idx = 0; idx < y_bits; idx++) {
      // x^(2^idx) * y_{idx}
      auto cur_pow = _mux(
          ctx, _and(ctx, _rshift(ctx, y_b, {static_cast<int64_t>(idx)}), k1),
          base, k1);
      ret = _mul(ctx, cur_pow, ret);
      if (idx < y_bits - 1) {
        base = _mul(ctx, base, base);
      }
    }

    // when x=-1 and y<0, we can still get a correct result
    return _mux(ctx, _and(ctx, msb_y, _not(ctx, x_abs1)), k0, ret)
        .setDtype(x.dtype());
  }
  if (x.isPublic() && y.isPublic()) {
    return f_pow_p(ctx, x, y);
  }

  auto msb = _msb(ctx, x);
  auto msb_a = _prefer_a(ctx, msb);
  auto x_abs = _mux(ctx, msb_a, _negate(ctx, x), x).setDtype(x.dtype());

  // if x=0 is public, then log(x) get -inf, the wrong output will be got after
  // multiplying y. So we force x to be secret, then computing log(x) leads to
  // a small negative numbers, so exp(y*log(x))=0.
  auto x_s = x.isPublic() ? hal::seal(ctx, x_abs) : x_abs;
  // x^y = e^(y*ln(x))
  // the precision is highly dependent on the precision of exp and log, so we
  // choose the most precise methods here.
  auto val = detail::exp_pade(ctx, mul(ctx, y, detail::log_minmax(ctx, x_s)));

  // the final sign is decided on both sign of x and the parity of y
  // when x<0 and y is odd, e.g. (-2)^3 = -8
  auto odd =
      _and(ctx, _rshift(ctx, y, {static_cast<int64_t>(ctx->getFxpBits())}),
           _constant(ctx, 1, y.shape()));
  auto sign = _and(ctx, msb, odd);

  return _mux(ctx, sign, _negate(ctx, val), val).setDtype(x.dtype());
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

Value left_shift(SPUContext* ctx, const Value& x, const Sizes& bits) {
  SPU_TRACE_HAL_DISP(ctx, x, bits);

  return _lshift(ctx, x, bits).setDtype(x.dtype());
}

Value right_shift_logical(SPUContext* ctx, const Value& x, const Sizes& bits) {
  SPU_TRACE_HAL_DISP(ctx, x, bits);

  return _rshift(ctx, x, bits).setDtype(x.dtype());
}

Value right_shift_arithmetic(SPUContext* ctx, const Value& x,
                             const Sizes& bits) {
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

Value sine(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_sine(ctx, x);
}

Value cosine(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_cosine(ctx, x);
}

Value atan2(SPUContext* ctx, const Value& y, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, y, x);

  SPU_ENFORCE(x.isFxp() && y.isFxp());

  return f_atan2(ctx, y, x);
}

Value acos(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_acos(ctx, x);
}

Value asin(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  SPU_ENFORCE(x.isFxp());

  return f_asin(ctx, x);
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

std::optional<Value> oramonehot(SPUContext* ctx, const Value& x,
                                int64_t db_size, bool db_is_secret) {
  SPU_ENFORCE(x.isInt(), "onehot_point should be int");

  auto ret = _oramonehot(ctx, x, db_size, db_is_secret);
  if (!ret.has_value()) {
    return std::nullopt;
  }
  ret->setDtype(x.dtype());
  return ret;
}

Value oramread(SPUContext* ctx, const Value& x, const Value& y,
               int64_t offset) {
  SPU_ENFORCE(x.isInt(), "onehot_point should be int");

  Value ret = _oramread(ctx, x, y, offset).setDtype(y.dtype());
  return ret;
}
}  // namespace spu::kernel::hal
