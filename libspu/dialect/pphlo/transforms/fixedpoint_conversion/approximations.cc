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

#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/approximations.h"

#include "libspu/core/prelude.h"

namespace mlir::spu::pphlo::fixedpoint {

namespace impl {

Value highestOneBit(builder::FxpBuilder &builder, Value x) {
  auto int_type = builder.getIntTypeWithSameWidth(x.getType(), true);
  auto _x = builder.bitcast(x, int_type);
  auto y = builder.prefix_or(_x);
  auto y1 = builder.rshift(y, 1);
  return builder.xor_(y, y1);
}

Value polynomial(builder::FxpBuilder &builder, Value x,
                 llvm::ArrayRef<double> coeffs, SignType sign_x,
                 SignType sign_ret) {
  SPU_ENFORCE(coeffs.size() > 1);

  auto x_pow = builder.fxp_constant(1.0F);
  auto res = builder.fxp_constant(coeffs[0]);

  auto trunc_bits = builder.getCurrentFxpBits();

  for (size_t i = 1; i < coeffs.size(); ++i) {
    if ((i & 1) == 0U) {
      // x^{even order} is always positive
      x_pow = builder.mul(x_pow, x, SignType::Positive);
    } else {
      if (i > 1) {
        x_pow = builder.mul(x_pow, x, sign_x);
      } else {
        // i=1, then save a _trunc
        x_pow = x;
      }
    }
    res = builder.add(
        res, builder.mul_no_trunc(x_pow, builder.fxp_constant(coeffs[i])));
  }

  return builder.truncation(res, trunc_bits, sign_ret);
}

}  // namespace impl

Value cosine_approx(builder::FxpBuilder &builder, Value input) {
  return impl::cosine_chebyshev(builder, input);
}

Value sine_approx(builder::FxpBuilder &builder, Value input) {
  return impl::sine_chebyshev(builder, input);
}

Value tanh_approx(builder::FxpBuilder &builder, Value input) {
  return impl::tanh_chebyshev(builder, input);
}

Value exponential_approx(builder::FxpBuilder &builder, Value input) {
  switch (builder.getConfig().exp_mode) {
    case ::spu::EXP_TAYLOR: {
      return impl::exponential_taylor(builder, input);
    }
    case ::spu::EXP_PADE: {
      return impl::exponential_pade(builder, input);
    }
    default:
      llvm_unreachable("Should not reach");
  }
}

Value log_approx(builder::FxpBuilder &builder, Value input) {
  switch (builder.getConfig().log_mode) {
    case ::spu::LOG_PADE: {
      return impl::log_pade(builder, input);
    }
    case ::spu::LOG_NEWTON: {
      return impl::log_newton(builder, input);
    }
    case ::spu::LOG_MINMAX: {
      return impl::log_minmax(builder, input);
    }
    default:
      llvm_unreachable("Should not reach");
  }
}

Value reciprocal_approx(builder::FxpBuilder &builder, Value input) {
  return impl::reciprocal_goldschmidt(builder, input);
}

Value logistic_approx(builder::FxpBuilder &builder, Value input) {
  switch (builder.getConfig().sig_mode) {
    case ::spu::SIGMOID_MM1: {
      return impl::logistic_mm1(builder, input);
    }
    case ::spu::SIGMOID_SEG3: {
      return impl::logistic_seg3(builder, input);
    }
    case ::spu::SIGMOID_REAL: {
      return impl::logistic_real(builder, input);
    }
    default:
      llvm_unreachable("Should not reach");
  }
}

Value rsqrt_approx(builder::FxpBuilder &builder, Value input) {
  return impl::rsqrt_expand(builder, input);
}

Value sqrt_approx(builder::FxpBuilder &builder, Value input) {
  return impl::sqrt_with_rsqrt(builder, input);
}

Value div_approx(builder::FxpBuilder &builder, Value lhs, Value rhs) {
  return impl::div_goldschmidt_general(builder, lhs, rhs);
}

Value erf_approx(builder::FxpBuilder &builder, Value input) {
  return impl::erf_poly(builder, input);
}

Value power_approx(builder::FxpBuilder &builder, Value lhs, Value rhs) {
  // abs_lhs
  auto msb_lhs = builder.less(lhs, builder.fxp_constant(0));
  auto abs_lhs =
      builder.select(msb_lhs, builder.negate(lhs), lhs, lhs.getType());

  // if x=0 is public, then log(x) get -inf, the wrong output will be got after
  // multiplying y. So we force x to be secret, then computing log(x) leads to
  // a small negative numbers, so exp(y*log(x))=0.
  // the precision is highly dependent on the precision of exp and log, so we
  // choose the most precise methods here.
  auto val = impl::log_minmax(builder, abs_lhs);

  val = builder.mul(rhs, val);
  val = impl::exponential_pade(builder, val);

  // the final sign is decided on both sign of x and the parity of y
  // when x<0 and y is odd, e.g. (-2)^3 = -8
  Value i_y;
  i_y = builder.bitcast(rhs, builder.getIntTypeWithSameWidth(rhs.getType()));
  i_y = builder.rshift(i_y, builder.getCurrentFxpBits());

  auto odd = builder.and_(i_y, builder.int_constant(1));
  auto sign = builder.and_(msb_lhs, builder.convert(odd, msb_lhs.getType()));

  return builder.select(sign, builder.negate(val), val, val.getType());
}

Value atan2_approx(builder::FxpBuilder &builder, Value lhs, Value rhs) {
  return impl::atan2_minmax(builder, lhs, rhs);
}

}  // namespace mlir::spu::pphlo::fixedpoint
