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
Value f_polynomial(HalContext* ctx, const Value& x,
                   const std::vector<Value>& coeffs) {
  SPU_TRACE_HAL_DISP(ctx, x);
  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(!coeffs.empty());

  Value x_pow = x;
  Value res = _mul(ctx, x_pow, coeffs[0]);

  for (size_t i = 1; i < coeffs.size(); i++) {
    x_pow = _trunc(ctx, _mul(ctx, x_pow, x));
    res = _add(ctx, res, _mul(ctx, x_pow, coeffs[i]));
  }

  return _trunc(ctx, res).asFxp();
}

Value highestOneBit(HalContext* ctx, const Value& x) {
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
Value div_goldschmidt(HalContext* ctx, const Value& a, const Value& b) {
  SPU_TRACE_HAL_DISP(ctx, a, b);

  auto b_sign = _sign(ctx, b);
  auto b_abs = _mul(ctx, b_sign, b).asFxp();

  auto b_msb = detail::highestOneBit(ctx, b_abs);

  // factor = 2^{2f-m} = 2^{f-m} * 2^f, the fixed point repr of 2^{f-m}
  const size_t num_fxp_bits = ctx->getFxpBits();
  auto factor = _bitrev(ctx, b_msb, 0, 2 * num_fxp_bits).asFxp();
  detail::hintNumberOfBits(factor, 2 * num_fxp_bits);

  // compute normalize x_abs, [0.5, 1)
  auto c = f_mul(ctx, b_abs, factor);

  // initial guess:
  //   w = 1/c ≈ 2.9142 - 2c when c >= 0.5 and c < 1
  const auto k2 = constant(ctx, 2, c.shape());
  const auto k2_9142 = constant(ctx, 2.9142F, c.shape());
  auto w = f_sub(ctx, k2_9142, _mul(ctx, k2, c).asFxp());

  // init r=w, e=1-c*w
  const auto& k1_ = constant(ctx, 1.0F, c.shape());
  auto r = w;
  auto e = f_sub(ctx, k1_, f_mul(ctx, c, w));

  const size_t config_num_iters = ctx->rt_config().fxp_div_goldschmidt_iters();
  const size_t num_iters = config_num_iters == 0 ? 2 : config_num_iters;

  // iterate, r=r(1+e), e=e*e
  for (size_t itr = 0; itr < num_iters; itr++) {
    r = f_mul(ctx, r, f_add(ctx, e, k1_));
    e = f_square(ctx, e);
  }

  r = f_mul(ctx, r, a);
  r = f_mul(ctx, r, factor);
  return _mul(ctx, r, b_sign).asFxp();
}

Value reciprocal_goldschmidt_positive(HalContext* ctx, const Value& b_abs) {
  auto b_msb = detail::highestOneBit(ctx, b_abs);

  // factor = 2^{2f-m} = 2^{f-m} * 2^f, the fixed point repr of 2^{f-m}
  const size_t num_fxp_bits = ctx->getFxpBits();
  auto factor = _bitrev(ctx, b_msb, 0, 2 * num_fxp_bits).asFxp();
  detail::hintNumberOfBits(factor, 2 * num_fxp_bits);

  // compute normalize x_abs, [0.5, 1)
  auto c = f_mul(ctx, b_abs, factor);

  // initial guess:
  //   w = 1/b = 2.9142 - 2c when c >= 0.5 and c < 1
  const auto k2 = constant(ctx, 2, c.shape());
  const auto k2_9142 = constant(ctx, 2.9142F, c.shape());
  auto w = f_mul(ctx, f_sub(ctx, k2_9142, _mul(ctx, k2, c).asFxp()), factor);

  // init r=a*w, e=1-b*w
  const auto& k1_ = constant(ctx, 1.0F, c.shape());
  auto r = w;
  auto e = f_sub(ctx, k1_, f_mul(ctx, b_abs, w));

  const size_t config_num_iters = ctx->rt_config().fxp_div_goldschmidt_iters();
  const size_t num_iters = config_num_iters == 0 ? 2 : config_num_iters;

  // iterate, r=r(1+e), e=e*e
  for (size_t itr = 0; itr < num_iters; itr++) {
    r = f_mul(ctx, r, f_add(ctx, e, k1_));
    e = f_square(ctx, e);
  }

  return r;
}

// NOTE(junfeng): we have a seperate reciprocal_goldschmidt is to avoid
// unnecessary f_mul for y initiation in div_goldschmidt.
Value reciprocal_goldschmidt(HalContext* ctx, const Value& b) {
  SPU_TRACE_HAL_DISP(ctx, b);

  auto b_sign = _sign(ctx, b);
  auto b_abs = _mul(ctx, b_sign, b).asFxp();

  return _mul(ctx, reciprocal_goldschmidt_positive(ctx, b_abs), b_sign).asFxp();
}

}  // namespace detail

Value f_negate(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());
  return _negate(ctx, x).asFxp();
}

Value f_abs(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());
  const Value sign = _sign(ctx, x);

  return _mul(ctx, sign, x).asFxp();
}

Value f_reciprocal(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());
  if (x.isPublic()) {
    return f_reciprocal_p(ctx, x);
  }

  return detail::reciprocal_goldschmidt(ctx, x);
}

Value f_add(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(y.isFxp());

  return _add(ctx, x, y).asFxp();
}

Value f_sub(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(y.isFxp());
  return f_add(ctx, x, f_negate(ctx, y));
}

Value f_mul(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(y.isFxp());

  return _trunc(ctx, _mul(ctx, x, y)).asFxp();
}

Value f_mmul(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(y.isFxp());

  return _trunc(ctx, _mmul(ctx, x, y)).asFxp();
}

Value f_div(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(y.isFxp());

  if (x.isPublic() && y.isPublic()) {
    return f_div_p(ctx, x, y);
  }

  return detail::div_goldschmidt(ctx, x, y);
}

Value f_equal(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(y.isFxp());

  return _eqz(ctx, f_sub(ctx, x, y)).setDtype(DT_I1);
}

Value f_less(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(x.isFxp());
  SPU_ENFORCE(y.isFxp());

  return _less(ctx, x, y).setDtype(DT_I1);
}

Value f_square(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());
  // TODO(jint) optimize me.

  return f_mul(ctx, x, x);
}

Value f_floor(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());

  const size_t fbits = ctx->getFxpBits();
  return _lshift(ctx, _arshift(ctx, x, fbits), fbits).asFxp();
}

Value f_ceil(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(x.isFxp());

  // ceil(x) = floor(x + 1.0 - epsilon)
  const auto& k1 = constant(ctx, 1.0F, x.shape());
  return f_floor(ctx, f_add(ctx, x, f_sub(ctx, k1, epsilon(ctx, x.shape()))));
}

}  // namespace spu::kernel::hal
