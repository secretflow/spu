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

#include "libspu/kernel/hal/fxp_base.h"

#include <cmath>

#include "libspu/core/prelude.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp_cleartext.h"
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hal {
namespace detail {

// Calc:
//   y = x*c0 + x^2*c1 + x^3*c2 + ... + x^n*c[n-1]
//
// Coefficients should be ordered from the order 1 (linear) term first, ending
// with the highest order term. (Constant is not included).
Value f_polynomial(SPUContext* ctx, const Value& x,
                   const std::vector<Value>& coeffs) {
  SPU_TRACE_HAL_DISP(ctx, x);
  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(!coeffs.empty());

  Value x_pow = x;
  Value res = _mul(ctx, x_pow, coeffs[0]);

  for (size_t i = 1; i < coeffs.size(); i++) {
    if (i & 1) {
      // x^{even order} is always positive
      x_pow =
          _trunc_with_sign(ctx, _mul(ctx, x_pow, x), ctx->getFxpBits(), true);
    } else {
      x_pow = _trunc(ctx, _mul(ctx, x_pow, x));
    }
    res = _add(ctx, res, _mul(ctx, x_pow, coeffs[i]));
  }

  return _trunc(ctx, res).setDtype(x.dtype());
}

Value highestOneBit(SPUContext* ctx, const Value& x) {
  auto y = _prefix_or(ctx, x);
  auto y1 = _rshift(ctx, y, 1);
  return _xor(ctx, y, y1);
}

// FIXME:
// Use range propatation instead of directly set.
// or expose bit_decompose as mpc level api.
void hintNumberOfBits(const Value& a, size_t nbits) {
  if (a.storage_type().isa<BShare>()) {
    const_cast<Type&>(a.storage_type()).as<BShare>()->setNbits(nbits);
  }
}

// Reference:
//   Charpter 3.4 Division @ Secure Computation With Fixed Point Number
//   http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.1305&rep=rep1&type=pdf
//
// Goldschmidt main idea:
// Target:
//   calculate a/b
//
// Symbols:
//   f: number of fractional bits in fixed point.
//   m: the highest position of bit with value == 1.
//
// Initial guess:
//   let b = c*2^{m} where c = normalize(x), c \in [0.5, 1)
//   let w = 1/c ≈ (2.9142 - 2*c) as the initial guess.
//
// Iteration (reduce error):
//   let r = w, denotes result
//   let e = 1-c*w, denotes error
//   for _ in iters:
//     r = r(1 + e)
//     e = e * e
//
//   return r * a * 2^{-m}
//
// Precision is decided by magic number, i.e 2.9142 and f.
Value div_goldschmidt(SPUContext* ctx, const Value& a, const Value& b) {
  SPU_TRACE_HAL_DISP(ctx, a, b);

  // We prefer  b_abs = b < 0 ? -b : b over b_abs = sign(b) * b
  // because MulA1B is a better choice than MulAA for CHEETAH.
  // For ABY3, these two computations give the same cost though.
  auto is_negative = _msb(ctx, b);
  // insert ``prefer_a'' because the msb bit are used twice.
  is_negative = _prefer_a(ctx, is_negative);
  auto b_abs = _mux(ctx, is_negative, _negate(ctx, b), b).setDtype(b.dtype());

  auto b_msb = detail::highestOneBit(ctx, b_abs);

  // factor = 2^{2f-m} = 2^{f-m} * 2^f, the fixed point repr of 2^{f-m}
  const size_t num_fxp_bits = ctx->getFxpBits();
  auto factor = _bitrev(ctx, b_msb, 0, 2 * num_fxp_bits).setDtype(b.dtype());
  detail::hintNumberOfBits(factor, 2 * num_fxp_bits);

  // compute normalize x_abs, [0.5, 1)
  auto c = f_mul_with_sign(ctx, b_abs, factor, SignType::POSITIVE);

  // initial guess:
  //   w = 1/c ≈ 2.9142 - 2c when c >= 0.5 and c < 1
  const auto k2 = _constant(ctx, 2, c.shape());
  const auto k2_9142 = constant(ctx, 2.9142F, b.dtype(), c.shape());
  auto w = f_sub(ctx, k2_9142, _mul(ctx, k2, c).setDtype(b.dtype()));

  // init r=w, e=1-c*w
  const auto& k1_ = constant(ctx, 1.0F, b.dtype(), c.shape());
  auto r = w;
  auto e = f_sub(ctx, k1_, f_mul_with_sign(ctx, c, w, SignType::POSITIVE));

  const size_t num_iters = ctx->config().fxp_div_goldschmidt_iters();
  SPU_ENFORCE(num_iters != 0, "fxp_div_goldschmidt_iters should not be {}",
              num_iters);

  // iterate, r=r(1+e), e=e*e
  for (size_t itr = 0; itr < num_iters; itr++) {
    r = f_mul_with_sign(ctx, r, f_add(ctx, e, k1_), SignType::POSITIVE);
    if (itr + 1 < num_iters) {
      e = f_square(ctx, e);
    }
  }

  // NOTE(juhou): I hope to perform r*factor first which can use truncate_msb=0
  // However, it might overflow when the input x is too small.
  r = f_mul(ctx, r, a);
  r = f_mul(ctx, r, factor);
  return _mux(ctx, is_negative, _negate(ctx, r), r).setDtype(a.dtype());
}

Value reciprocal_goldschmidt_positive(SPUContext* ctx, const Value& b_abs) {
  auto b_msb = detail::highestOneBit(ctx, b_abs);

  // factor = 2^{2f-m} = 2^{f-m} * 2^f, the fixed point repr of 2^{f-m}
  const size_t num_fxp_bits = ctx->getFxpBits();
  auto factor =
      _bitrev(ctx, b_msb, 0, 2 * num_fxp_bits).setDtype(b_abs.dtype());
  detail::hintNumberOfBits(factor, 2 * num_fxp_bits);

  // compute normalize x_abs, [0.5, 1)
  auto c = f_mul_with_sign(ctx, b_abs, factor, SignType::POSITIVE);

  // initial guess:
  //   w = 1/b = 2.9142 - 2c when c >= 0.5 and c < 1
  const auto k2 = _constant(ctx, 2, c.shape());
  const auto k2_9142 = constant(ctx, 2.9142F, b_abs.dtype(), c.shape());
  auto w =
      f_mul(ctx, f_sub(ctx, k2_9142, _mul(ctx, k2, c).setDtype(b_abs.dtype())),
            factor);

  // init r=a*w, e=1-b*w
  const auto& k1_ = constant(ctx, 1.0F, b_abs.dtype(), c.shape());
  auto r = w;
  auto e = f_sub(ctx, k1_, f_mul_with_sign(ctx, b_abs, w, SignType::POSITIVE));

  const size_t num_iters = ctx->config().fxp_div_goldschmidt_iters();
  SPU_ENFORCE(num_iters != 0, "fxp_div_goldschmidt_iters should not be {}",
              num_iters);

  // iterate, r=r(1+e), e=e*e
  for (size_t itr = 0; itr < num_iters; itr++) {
    r = f_mul_with_sign(ctx, r, f_add(ctx, e, k1_), SignType::POSITIVE);
    if (itr + 1 < num_iters) {
      e = f_square(ctx, e);
    }
  }

  return r;
}

// NOTE(junfeng): we have a seperate reciprocal_goldschmidt is to avoid
// unnecessary f_mul for y initiation in div_goldschmidt.
Value reciprocal_goldschmidt(SPUContext* ctx, const Value& b) {
  SPU_TRACE_HAL_DISP(ctx, b);

  auto is_negative = _msb(ctx, b);
  is_negative = _prefer_a(ctx, is_negative);

  auto b_abs = _mux(ctx, is_negative, _negate(ctx, b), b).setDtype(b.dtype());
  auto r = reciprocal_goldschmidt_positive(ctx, b_abs);
  return _mux(ctx, is_negative, _negate(ctx, r), r).setDtype(b.dtype());
}

}  // namespace detail

Value f_negate(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());
  return _negate(ctx, x).setDtype(x.dtype());
}

Value f_abs(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());
  const Value sign = _sign(ctx, x);

  return _mul(ctx, sign, x).setDtype(x.dtype());
}

Value f_reciprocal(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());
  if (x.isPublic()) {
    return f_reciprocal_p(ctx, x);
  }

  return detail::reciprocal_goldschmidt(ctx, x);
}

Value f_add(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  return _add(ctx, x, y).setDtype(x.dtype());
}

Value f_sub(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  return f_add(ctx, x, f_negate(ctx, y));
}

Value f_mul(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  return _trunc(ctx, _mul(ctx, x, y)).setDtype(x.dtype());
}

Value f_mul_with_sign(SPUContext* ctx, const Value& x, const Value& y,
                      SignType sign) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  switch (sign) {
    case SignType::POSITIVE:
      return _trunc_with_sign(ctx, _mul(ctx, x, y), ctx->getFxpBits(),
                              /*positive*/ true)
          .setDtype(x.dtype());
      break;
    case SignType::NEGATIVE:
      return _trunc_with_sign(ctx, _mul(ctx, x, y), ctx->getFxpBits(),
                              /*positive*/ false)
          .setDtype(x.dtype());
      break;
    case SignType::UNKNOWN:
    default:
      return f_mul(ctx, x, y);
      break;
  }
}

Value f_mmul(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  return _trunc(ctx, _mmul(ctx, x, y)).setDtype(x.dtype());
}

Value f_conv2d(SPUContext* ctx, const Value& x, const Value& y,
               absl::Span<const int64_t> window_strides,
               absl::Span<const int64_t> result_shape) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  return _trunc(ctx, _conv2d(ctx, x, y, window_strides, result_shape))
      .setDtype(x.dtype());
}

Value f_div(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  if (x.isPublic() && y.isPublic()) {
    return f_div_p(ctx, x, y);
  }

  return detail::div_goldschmidt(ctx, x, y);
}

Value f_equal(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  return _equal(ctx, x, y).setDtype(DT_I1);
}

Value f_less(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp() && y.isFxp() && x.dtype() == y.dtype());

  return _less(ctx, x, y).setDtype(DT_I1);
}

Value f_square(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());
  // TODO(jint) optimize me.
  // TODO(juhou) can use truncate with msb=0

  return _trunc_with_sign(ctx, _mul(ctx, x, x), ctx->getFxpBits(), true)
      .setDtype(x.dtype());
}

Value f_floor(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());

  const size_t fbits = ctx->getFxpBits();
  return _lshift(ctx, _arshift(ctx, x, fbits), fbits).setDtype(x.dtype());
}

Value f_ceil(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());

  // ceil(x) = floor(x + 1.0 - epsilon)
  const auto k1 = constant(ctx, 1.0F, x.dtype(), x.shape());
  return f_floor(
      ctx, f_add(ctx, x, f_sub(ctx, k1, epsilon(ctx, x.dtype(), x.shape()))));
}

}  // namespace spu::kernel::hal
