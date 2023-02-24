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

#include "libspu/core/encoding.h"  // for bitcast
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp.h"
#include "libspu/kernel/hal/integer.h"
#include "libspu/kernel/hal/ring.h"  // for fast fxp x int
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"

// TODO: handle dtype promotion inside integer dtypes.
namespace spu::kernel::hal {
namespace {

using UnaryOp = Value(HalContext*, const Value&);
using BinaryOp = Value(HalContext*, const Value&, const Value&);

DataType common_dtype(DataType lhs, DataType rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  return std::max(lhs, rhs);  // Always results to higher rank type
}

template <BinaryOp* FnFxp, BinaryOp* FnInt>
Value dtypeBinaryDispatch(std::string_view op_name, HalContext* ctx,
                          const Value& x, const Value& y) {
  // Promote int to fxp if mismatch.
  if (x.isInt() && y.isInt()) {
    auto common_type = common_dtype(x.dtype(), y.dtype());
    return FnInt(ctx, dtype_cast(ctx, x, common_type),
                 dtype_cast(ctx, y, common_type));
  } else if (x.isInt() && y.isFxp()) {
    return FnFxp(ctx, dtype_cast(ctx, x, DT_FXP), y);
  } else if (x.isFxp() && y.isInt()) {
    return FnFxp(ctx, x, dtype_cast(ctx, y, DT_FXP));
  } else if (x.isFxp() && y.isFxp()) {
    return FnFxp(ctx, x, y);
  } else {
    SPU_THROW("unsupported op {} for x={}, y={}", op_name, x, y);
  }
}

template <UnaryOp* FnFxp, UnaryOp* FnInt>
Value dtypeUnaryDispatch(std::string_view op_name, HalContext* ctx,
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

Value logisticMM1(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  // SigmoidMM1: f(x) = 0.5 + 0.125 * x
  const auto c1 = constant(ctx, 0.5F, x.shape());
  const auto c2 = constant(ctx, 0.125F, x.shape());
  return add(ctx, c1, mul(ctx, c2, x));
}

Value logisticReal(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  // f(x) = 1/(1+exp(-x))
  const auto c1 = constant(ctx, 1.0F, x.shape());
  return reciprocal(ctx, add(ctx, c1, exp(ctx, negate(ctx, x))));
}

Value logisticSEG3(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  // f(x) = 0.5 + 0.125x if -4 <= x <= 4
  //        1            if       x > 4
  //        0            if  -4 > x
  // Rounds = Gt + Mux*2 = 4 + Log(K)
  auto upper = constant(ctx, 1.0F, x.shape());
  auto lower = constant(ctx, 0.0F, x.shape());
  auto middle = logisticMM1(ctx, x);

  auto upper_bound = constant(ctx, 4.0F, x.shape());
  auto lower_bound = constant(ctx, -4.0F, x.shape());

  auto ret = select(ctx, greater(ctx, x, upper_bound), upper, middle);
  return select(ctx, less(ctx, x, lower_bound), lower, ret);
}

}  // namespace

Value identity(HalContext* ctx, const Value& x) {
  // FIXME: constant should be the same dtype as input.
  auto zeros = constant(ctx, 0, x.shape());
  return add(ctx, x, zeros);
}

Value add(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  return dtypeBinaryDispatch<f_add, i_add>("add", ctx, x, y);
}

Value sub(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  return dtypeBinaryDispatch<f_sub, i_sub>("sub", ctx, x, y);
}

Value mixed_mul(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  return _mul(ctx, x, y).asFxp();
}

Value mixed_mmul(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  return _mmul(ctx, x, y).asFxp();
}

Value mul(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  // fast dispatch, avoid truncation cost
  if (isCrossIntFxp(x, y)) {
    return mixed_mul(ctx, x, y);
  }

  return dtypeBinaryDispatch<f_mul, i_mul>("mul", ctx, x, y);
}

Value matmul(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  // fast dispatch, avoid truncation cost
  if (isCrossIntFxp(x, y)) {
    return mixed_mmul(ctx, x, y);
  }

  return dtypeBinaryDispatch<f_mmul, i_mmul>("mmul", ctx, x, y);
}

Value logical_not(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_LEAF(ctx, in);

  auto _k1 = constant(ctx, true, in.shape());

  // TODO: we should NOT dispatch according to AShr/BShr trait here.
  if (in.storage_type().isa<BShare>()) {
    return _xor(ctx, in, _k1).setDtype(in.dtype());
  } else {
    return _sub(ctx, _k1, in).setDtype(in.dtype());
  }
}

Value equal(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape(), "x = {}, y = {}", x, y);

  if (ctx->rt_config().protocol() == ProtocolKind::CHEETAH &&
      (x.isSecret() || y.isSecret())) {
    // For 2PC, equal can be done with the same cost of half MSB.
    //      x0 + x1 = y0 + y1 mod 2^k
    // <=>  x0 - y0 = y1 - x1 mod 2^k
    // <=>  [1{x = y}]_B <- EQ(x0 - y0, y1 - x1) where EQ is a 2PC protocol.
    return dtypeBinaryDispatch<f_equal, i_equal>("equal", ctx, x, y);
  }

  // Note: following method does work, but slower ...
  // With optimized msb kernel, A2B+PreOr is slower than 2*MSB
  // return _eqz(ctx, sub(ctx, x, y)).setDtype(DT_I1);
  return bitwise_and(ctx, logical_not(ctx, less(ctx, x, y)),
                     logical_not(ctx, less(ctx, y, x)));
}

Value not_equal(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return logical_not(ctx, equal(ctx, x, y));
}

Value less(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return dtypeBinaryDispatch<f_less, i_less>("less", ctx, x, y);
}

Value less_equal(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  // not (x > y)
  return logical_not(ctx, greater(ctx, x, y));
}

Value greater(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  return less(ctx, y, x);
}

Value greater_equal(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.shape() == y.shape());

  // not (x < y)
  return logical_not(ctx, less(ctx, x, y));
}

Value negate(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return dtypeUnaryDispatch<f_negate, i_negate>("negate", ctx, x);
}

Value abs(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return dtypeUnaryDispatch<f_abs, i_abs>("abs", ctx, x);
}

Value exp(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  switch (ctx->rt_config().fxp_exp_mode()) {
    case RuntimeConfig::EXP_DEFAULT:
    case RuntimeConfig::EXP_TAYLOR:
      return f_exp(ctx, dtype_cast(ctx, in, DT_FXP));
    case RuntimeConfig::EXP_PADE: {
      // The valid input for exp_pade_approx is [-exp_input_limit,
      // exp_input_limit].
      // TODO(junfeng): We should merge clamp into exp_pade_approx to save msb
      // ops.
      const float exp_input_limit = 32 / std::log2(std::exp(1));
      const auto x = clamp(ctx, constant(ctx, -exp_input_limit, in.shape()),
                           dtype_cast(ctx, in, DT_FXP),
                           constant(ctx, exp_input_limit, in.shape()));
      return f_exp(ctx, x);
    }
    default:
      SPU_THROW("unexpected exp approximation method {}",
                ctx->rt_config().fxp_exp_mode());
  }
}

Value select(HalContext* ctx, const Value& pred, const Value& a,
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

Value bitwise_and(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  SPU_ENFORCE(x.isInt() && y.isInt());
  SPU_ENFORCE(x.shape() == y.shape());

  return _and(ctx, x, y).setDtype(x.dtype());
}

Value bitwise_xor(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.isInt() && y.isInt());
  SPU_ENFORCE(x.shape() == y.shape());

  return _xor(ctx, x, y).setDtype(x.dtype());
}

Value bitwise_or(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.isInt() && y.isInt());
  SPU_ENFORCE(x.shape() == y.shape());

  return _or(ctx, x, y).setDtype(x.dtype());
}

Value bitwise_not(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  return _not(ctx, in).setDtype(in.dtype());
}

Value logistic(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  switch (ctx->rt_config().sigmoid_mode()) {
    case RuntimeConfig::SIGMOID_DEFAULT:
    case RuntimeConfig::SIGMOID_MM1: {
      return logisticMM1(ctx, in);
    }
    case RuntimeConfig::SIGMOID_SEG3: {
      return logisticSEG3(ctx, in);
    }
    case RuntimeConfig::SIGMOID_REAL: {
      return logisticReal(ctx, in);
    }
    default: {
      SPU_THROW("Should not hit");
    }
  }
}

Value log(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  return f_log(ctx, dtype_cast(ctx, in, DT_FXP));
}

Value log1p(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  return f_log1p(ctx, dtype_cast(ctx, in, DT_FXP));
}

Value reciprocal(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  SPU_ENFORCE(in.isFxp());

  return f_reciprocal(ctx, in);
}

Value floor(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_floor(ctx, in);
}

Value ceil(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  SPU_ENFORCE(in.isFxp());

  return f_ceil(ctx, in);
}

Value max(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  SPU_ENFORCE(x.dtype() == y.dtype());

  return select(ctx, greater(ctx, x, y), x, y);
}

Value min(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  SPU_ENFORCE(x.dtype() == y.dtype());

  return select(ctx, less(ctx, x, y), x, y);
}

Value power(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  // x^y = e^(y*ln(x))
  return exp(ctx, mul(ctx, y, log(ctx, x)));
}

Value idiv(HalContext* ctx, const Value& x, const Value& y) {
  auto sign_x = sign(ctx, x);
  auto sign_y = sign(ctx, y);

  auto abs_x = mul(ctx, x, sign_x);
  auto abs_y = mul(ctx, y, sign_y);

  Value q;
  {
    const auto x_f = dtype_cast(ctx, abs_x, DT_FXP);
    const auto y_f = dtype_cast(ctx, abs_y, DT_FXP);

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

Value div(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);

  if (x.isInt() && y.isInt()) {
    return idiv(ctx, x, y);
  }

  const auto x_f = dtype_cast(ctx, x, DT_FXP);
  const auto y_f = dtype_cast(ctx, y, DT_FXP);

#define F_DIV_WITH_DIRECT_GOLDSCHMIDT_METHOD
#ifdef F_DIV_WITH_DIRECT_GOLDSCHMIDT_METHOD
  auto res_f = f_div(ctx, x_f, y_f);
#else
  auto res_f = mul(ctx, x, reciprocal(ctx, y));
#endif

  return res_f;
}

Value clamp(HalContext* ctx, const Value& minv, const Value& x,
            const Value& maxv) {
  SPU_TRACE_HAL_DISP(ctx, minv, x, maxv);

  SPU_ENFORCE(minv.dtype() == maxv.dtype());
  SPU_ENFORCE(minv.dtype() == x.dtype());

  return min(ctx, max(ctx, minv, x), maxv);
}

Value bitcast(HalContext* ctx, const Value& x, DataType dtype) {
  SPU_TRACE_HAL_DISP(ctx, x, dtype);

  // FIXME(jint) should we directly use fixed point binary expr for bitcast?
  return Value(x.data().clone(), dtype);
}

Value left_shift(HalContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_DISP(ctx, x, bits);

  return _lshift(ctx, x, bits).setDtype(x.dtype());
}

Value right_shift_logical(HalContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_DISP(ctx, x, bits);

  return _rshift(ctx, x, bits).setDtype(x.dtype());
}

Value right_shift_arithmetic(HalContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_DISP(ctx, x, bits);

  return _arshift(ctx, x, bits).setDtype(x.dtype());
}

Value log2(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  return f_log2(ctx, dtype_cast(ctx, in, DT_FXP));
}

Value exp2(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return f_exp2(ctx, dtype_cast(ctx, x, DT_FXP));
}

Value tanh(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return f_tanh(ctx, dtype_cast(ctx, x, DT_FXP));
}

Value rsqrt(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return f_rsqrt(ctx, dtype_cast(ctx, x, DT_FXP));
}

Value sqrt(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return f_sqrt(ctx, dtype_cast(ctx, x, DT_FXP));
}

Value sign(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  return _sign(ctx, x).setDtype(DT_I8);
}

Value conv2d(HalContext* ctx, const Value& x, const Value& y,
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
