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

#include "spu/kernel/hal/fxp.h"

#include <algorithm>
#include <cmath>

#include "absl/numeric/bits.h"

#include "spu/kernel/hal/constants.h"
#include "spu/kernel/hal/integer.h"
#include "spu/kernel/hal/public_intrinsic.h"
#include "spu/kernel/hal/ring.h"

namespace spu::kernel::hal {
namespace {

// Calc:
//   y = x*c0 + x^2*c1 + x^3*c2 + ... + x^n*c[n-1]
//
// Coefficients should be ordered from the order 1 (linear) term first, ending
// with the highest order term. (Constant is not included).
Value f_polynomial(HalContext* ctx, const Value& x,
                   const std::vector<Value>& coeffs) {
  SPU_TRACE_HAL_DISP(ctx, x);
  YACL_ENFORCE(x.isFxp());
  YACL_ENFORCE(!coeffs.empty());

  Value x_pow = x;
  Value res = _mul(ctx, x_pow, coeffs[0]);

  for (size_t i = 1; i < coeffs.size(); i++) {
    x_pow = _trunc(ctx, _mul(ctx, x_pow, x));
    res = _add(ctx, res, _mul(ctx, x_pow, coeffs[i]));
  }

  return _trunc(ctx, res).asFxp();
}

// Extract the most significant bit. see
// https://docs.oracle.com/javase/7/docs/api/java/lang/Integer.html#highestOneBit(int)
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

}  // namespace

namespace detail {

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

  auto b_msb = highestOneBit(ctx, b_abs);

  // factor = 2^{2f-m} = 2^{f-m} * 2^f, the fixed point repr of 2^{f-m}
  const size_t num_fxp_bits = ctx->getFxpBits();
  auto factor = _bitrev(ctx, b_msb, 0, 2 * num_fxp_bits).asFxp();
  hintNumberOfBits(factor, 2 * num_fxp_bits);

  // compute normalize x_abs, [0.5, 1)
  auto c = f_mul(ctx, b_abs, factor);

  // initial guess:
  //   w = 1/c ≈ 2.9142 - 2c when c >= 0.5 and c < 1
  const auto k2 = constant(ctx, 2, c.shape());
  const auto k2_9142 = constant(ctx, 2.9142f, c.shape());
  auto w = f_sub(ctx, k2_9142, _mul(ctx, k2, c).asFxp());

  // init r=w, e=1-c*w
  const auto& k1_ = constant(ctx, 1.0f, c.shape());
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
  auto b_msb = highestOneBit(ctx, b_abs);

  // factor = 2^{2f-m} = 2^{f-m} * 2^f, the fixed point repr of 2^{f-m}
  const size_t num_fxp_bits = ctx->getFxpBits();
  auto factor = _bitrev(ctx, b_msb, 0, 2 * num_fxp_bits).asFxp();
  hintNumberOfBits(factor, 2 * num_fxp_bits);

  // compute normalize x_abs, [0.5, 1)
  auto c = f_mul(ctx, b_abs, factor);

  // initial guess:
  //   w = 1/b = 2.9142 - 2c when c >= 0.5 and c < 1
  const auto k2 = constant(ctx, 2, c.shape());
  const auto k2_9142 = constant(ctx, 2.9142f, c.shape());
  auto w = f_mul(ctx, f_sub(ctx, k2_9142, _mul(ctx, k2, c).asFxp()), factor);

  // init r=a*w, e=1-b*w
  const auto& k1_ = constant(ctx, 1.0f, c.shape());
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

// Pade approximation fo x belongs to [0.5, 1]:
//
// p2524(x) = -0.205466671951 * 10
//          + x * -0.88626599391 * 10
//          + x^2 * 0.610585199015 * 10
//          + x^3 * 0.481147460989 * 10
// q2524(x) = 0.353553425277
//          + x * 0.454517087629 * 10
//          + x^2 * 0.642784209029 * 10
//          + x^3 * 0.1 *10
// log2(x) = p2524(x) / q2524(x)
//
Value log2_pade_approx_for_normalized(HalContext* ctx, const Value& x) {
  const auto x2 = f_square(ctx, x);
  const auto x3 = f_mul(ctx, x2, x);

  const auto p0 = constant(ctx, -0.205466671951 * 10, x.shape());
  const auto p1 = constant(ctx, -0.88626599391 * 10, x.shape());
  const auto p2 = constant(ctx, 0.610585199015 * 10, x.shape());
  const auto p3 = constant(ctx, 0.481147460989 * 10, x.shape());

  const auto q0 = constant(ctx, 0.353553425277, x.shape());
  const auto q1 = constant(ctx, 0.454517087629 * 10, x.shape());
  const auto q2 = constant(ctx, 0.642784209029 * 10, x.shape());
  const auto q3 = constant(ctx, 0.1 * 10, x.shape());

  auto p2524 = _mul(ctx, x, p1);
  p2524 = _add(ctx, p2524, _mul(ctx, x2, p2));
  p2524 = _add(ctx, p2524, _mul(ctx, x3, p3));
  p2524 = _add(ctx, _trunc(ctx, p2524), p0).asFxp();

  auto q2524 = _mul(ctx, x, q1);
  q2524 = _add(ctx, q2524, _mul(ctx, x2, q2));
  q2524 = _add(ctx, q2524, _mul(ctx, x3, q3));
  q2524 = _add(ctx, _trunc(ctx, q2524), q0).asFxp();

  return div_goldschmidt(ctx, p2524, q2524);
}

// Refer to
// Chapter 5 Exponentiation and Logarithms
// Benchmarking Privacy Preserving Scientific Operations
// https://www.esat.kuleuven.be/cosic/publications/article-3013.pdf
Value log2_pade_approx(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_DISP(ctx, x);

  auto k = _popcount(ctx, _prefix_or(ctx, x));

  const size_t num_fxp_bits = ctx->getFxpBits();

  // let x = x_norm * factor, where x in [0.5, 1.0)
  auto msb = highestOneBit(ctx, x);
  auto factor = _bitrev(ctx, msb, 0, 2 * num_fxp_bits).asFxp();
  hintNumberOfBits(factor, 2 * num_fxp_bits);
  auto norm = f_mul(ctx, x, factor);

  // log2(x) = log2(x_norm * factor)
  //         = log2(x_norm) + log2(factor)
  //         = log2(x_norm) + (k-fxp_bits)
  return _add(ctx, log2_pade_approx_for_normalized(ctx, norm),
              _lshift(ctx,
                      _sub(ctx, k,
                           constant(ctx, static_cast<uint64_t>(num_fxp_bits),
                                    x.shape())),
                      num_fxp_bits))
      .asFxp();
}

// See P11, A.2.4 Logarithm and Exponent,
// https://lvdmaaten.github.io/publications/papers/crypten.pdf
// https://github.com/facebookresearch/CrypTen/blob/master/crypten/common/functions/approximations.py#L55-L104
// Approximates the natural logarithm using 8th order modified
// Householder iterations. This approximation is accurate within 2% relative
// error on [0.0001, 250].
Value log_householder_approx(HalContext* ctx, const Value& x) {
  Value term_1 = f_div(ctx, x, constant(ctx, 120.0f, x.shape()));
  Value term_2 = f_mul(
      ctx,
      f_exp(ctx,
            f_negate(ctx,
                     f_add(ctx, f_mul(ctx, x, constant(ctx, 2.0f, x.shape())),
                           constant(ctx, 1.0f, x.shape())))),
      constant(ctx, 20.0f, x.shape()));
  Value y =
      f_add(ctx, f_sub(ctx, term_1, term_2), constant(ctx, 3.0f, x.shape()));

  std::vector<Value> coeffs;
  const size_t config_orders = ctx->rt_config().fxp_log_orders();
  const size_t num_order = config_orders == 0 ? 8 : config_orders;
  for (size_t i = 0; i < num_order; i++) {
    coeffs.emplace_back(constant(ctx, 1.0f / (1.0f + i), x.shape()));
  }

  const size_t config_iters = ctx->rt_config().fxp_log_iters();
  const size_t num_iters = config_iters == 0 ? 3 : config_iters;
  for (size_t i = 0; i < num_iters; i++) {
    Value h = f_sub(ctx, constant(ctx, 1.0f, x.shape()),
                    f_mul(ctx, x, f_exp(ctx, f_negate(ctx, y))));
    y = f_sub(ctx, y, f_polynomial(ctx, h, coeffs));
  }

  return y;
}

// see https://lvdmaaten.github.io/publications/papers/crypten.pdf
//   exp(x) = (1 + x / n) ^ n, when n is infinite large.
Value exp_taylor_series(HalContext* ctx, const Value& x) {
  const size_t config_iters = ctx->rt_config().fxp_exp_iters();
  const size_t num_iters = config_iters == 0 ? 8 : config_iters;

  Value res = f_add(ctx, _trunc(ctx, x, num_iters).asFxp(),
                    constant(ctx, 1.0f, x.shape()));

  for (size_t i = 0; i < num_iters; i++) {
    res = f_square(ctx, res);
  }

  return res;
}

// Pade approximation of exp2(x), x is in [0, 1].
// p1015(x) = 0.100000007744302 * 10
//             + x * 0.693147180426163
//             + x^2 * 0.240226510710170
//             + x^3 * 0.555040686204663 / 10
//             + x^4 * 0.961834122588046 / 100
//             + x^5 * 0.133273035928143 / 100
Value exp2_pade_approx_for_positive_pure_decimal(HalContext* ctx,
                                                 const Value& x) {
  auto x2 = f_mul(ctx, x, x);
  auto x3 = f_mul(ctx, x, x2);
  auto x4 = f_mul(ctx, x, x3);
  auto x5 = f_mul(ctx, x, x4);

  const auto p0 = constant(ctx, 0.100000007744302 * 10, x.shape());
  const auto p1 = constant(ctx, 0.693147180426163, x.shape());
  const auto p2 = constant(ctx, 0.240226510710170, x.shape());
  const auto p3 = constant(ctx, 0.555040686204663 / 10, x.shape());
  const auto p4 = constant(ctx, 0.961834122588046 / 100, x.shape());
  const auto p5 = constant(ctx, 0.133273035928143 / 100, x.shape());

  auto res = _mul(ctx, x, p1);
  res = _add(ctx, res, _mul(ctx, x2, p2));
  res = _add(ctx, res, _mul(ctx, x3, p3));
  res = _add(ctx, res, _mul(ctx, x4, p4));
  res = _add(ctx, res, _mul(ctx, x5, p5));

  return _add(ctx, _trunc(ctx, res), p0).asFxp();
}

// Refer to
// Chapter 5 Exponentiation and Logarithms
// Benchmarking Privacy Preserving Scientific Operations
// https://www.esat.kuleuven.be/cosic/publications/article-3013.pdf
// NOTE(junfeng): The valid integer bits of x is 5. Otherwise, the output is
// incorrect.
Value exp2_pade_approx(HalContext* ctx, const Value& x) {
  const size_t fbits = ctx->getFxpBits();
  const auto k1 = constant(ctx, 1U, x.shape());
  const auto k0 = constant(ctx, 0U, x.shape());
  const auto k2 = constant(ctx, 2U, x.shape());
  // TODO(junfeng): Make int_bits configurable.
  const size_t int_bits = 5;
  const size_t bit_width = SizeOf(ctx->getField()) * 8;

  const auto x_bshare = _or(ctx, x, k0);  // noop, to bshare
  const auto x_msb = _rshift(ctx, x_bshare, bit_width - 1);
  auto x_integer = _rshift(ctx, x_bshare, fbits);
  auto x_fraction = _sub(ctx, x, _lshift(ctx, x_integer, fbits)).asFxp();
  auto ret = exp2_pade_approx_for_positive_pure_decimal(ctx, x_fraction);

  for (size_t idx = 0; idx < int_bits; idx++) {
    auto a = _and(ctx, _rshift(ctx, x_integer, idx), k1);
    hintNumberOfBits(a, 1);
    a = _mul(ctx, k1, a);  // noop, to ashare
    const auto K = 1U << std::min(1UL << idx, bit_width - 2);
    ret = _mul(ctx, ret,
               _add(ctx, _mul(ctx, a, constant(ctx, K, x.shape())),
                    _sub(ctx, k1, a)))
              .asFxp();
  }

  // If we could ensure the integer bits of x is 5.
  // we have x, -x, -x_hat. x_hat is 2's complement of -x.
  // Then,
  //             x + (x_hat) = 32
  //            (x_hat) - 32 = -x
  //  exp2(x_hat) / exp2(32) = exp(-x)
  //  so exp(-x) = exp2(x_hat) / exp2(32)
  auto ret_reciprocal = _trunc(ctx, ret, std::pow(2, int_bits)).asFxp();

  // ret + msb * (reciprocal - ret)
  return f_add(ctx, ret,
               _mul(ctx, x_msb, f_sub(ctx, ret_reciprocal, ret)).asFxp());
}

Value exp_pade_approx(HalContext* ctx, const Value& x) {
  return f_exp2(
      ctx, f_mul(ctx, x, constant(ctx, std::log2(std::exp(1)), x.shape())));
}

// Refer to
// https://www.wolframalpha.com/input?i=Pade+approximation+tanh%28x%29+order+5%2C5.
// tanh(x) = (x + x^3 / 9.0 + x^5 /945.0) /
//           (1 + 4 * x^2 / 9.0 + x^4 / 63.0)
Value tanh_pade_approx(HalContext* ctx, const Value& x) {
  const auto x_2 = f_square(ctx, x);
  const auto x_3 = f_mul(ctx, x_2, x);
  const auto x_4 = f_square(ctx, x_2);
  const auto x_5 = f_mul(ctx, x_2, x_3);

  const auto dividend =
      f_add(ctx, x,
            f_add(ctx, f_div(ctx, x_3, constant(ctx, 9.0, x.shape())),
                  f_div(ctx, x_5, constant(ctx, 945.0, x.shape()))));

  const auto divisor =
      f_add(ctx, constant(ctx, 1.0, x.shape()),
            f_add(ctx, f_div(ctx, x_2, constant(ctx, 9.0 / 4.0, x.shape())),
                  f_div(ctx, x_4, constant(ctx, 63.0, x.shape()))));

  return f_div(ctx, dividend, divisor);
}

}  // namespace detail

Value f_square(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());
  // TODO(jint) optimize me.

  return f_mul(ctx, x, x);
}

Value f_exp(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());

  if (x.isPublic()) {
    return f_exp_p(ctx, x);
  }

  switch (ctx->rt_config().fxp_exp_mode()) {
    case RuntimeConfig::EXP_DEFAULT:
    case RuntimeConfig::EXP_TAYLOR:
      return detail::exp_taylor_series(ctx, x);
    case RuntimeConfig::EXP_PADE:
      return detail::exp_pade_approx(ctx, x);
    default:
      YACL_THROW("unexpected exp approxmation method {}",
                 ctx->rt_config().fxp_exp_mode());
  }
}

Value f_negate(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());
  return _negate(ctx, x).asFxp();
}

Value f_abs(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());
  const Value sign = _sign(ctx, x);

  return _mul(ctx, sign, x).asFxp();
}

Value f_reciprocal(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());
  if (x.isPublic()) {
    return f_reciprocal_p(ctx, x);
  }

  return detail::reciprocal_goldschmidt(ctx, x);
}

Value f_add(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  YACL_ENFORCE(x.isFxp());
  YACL_ENFORCE(y.isFxp());

  return _add(ctx, x, y).asFxp();
}

Value f_sub(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  YACL_ENFORCE(x.isFxp());
  YACL_ENFORCE(y.isFxp());
  return f_add(ctx, x, f_negate(ctx, y));
}

Value f_mul(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  YACL_ENFORCE(x.isFxp());
  YACL_ENFORCE(y.isFxp());

  return _trunc(ctx, _mul(ctx, x, y)).asFxp();
}

Value f_mmul(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  YACL_ENFORCE(x.isFxp());
  YACL_ENFORCE(y.isFxp());

  return _trunc(ctx, _mmul(ctx, x, y)).asFxp();
}

Value f_div(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  YACL_ENFORCE(x.isFxp());
  YACL_ENFORCE(y.isFxp());

  if (x.isPublic() && y.isPublic()) {
    return f_div_p(ctx, x, y);
  }

  return detail::div_goldschmidt(ctx, x, y);
}

Value f_equal(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  YACL_ENFORCE(x.isFxp());
  YACL_ENFORCE(y.isFxp());

  return _eqz(ctx, f_sub(ctx, x, y)).setDtype(DT_I1);
}

Value f_less(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  YACL_ENFORCE(x.isFxp());
  YACL_ENFORCE(y.isFxp());

  return _less(ctx, x, y).setDtype(DT_I1);
}

Value f_log(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());

  if (x.isPublic()) {
    return f_log_p(ctx, x);
  }

  switch (ctx->rt_config().fxp_log_mode()) {
    case RuntimeConfig::LOG_DEFAULT:
    case RuntimeConfig::LOG_PADE:
      return f_mul(ctx, constant(ctx, std::log(2.0f), x.shape()),
                   f_log2(ctx, x));
    case RuntimeConfig::LOG_NEWTON:
      return detail::log_householder_approx(ctx, x);
    default:
      YACL_THROW("unlogected log approxmation method {}",
                 ctx->rt_config().fxp_log_mode());
  }
}

Value f_log1p(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());

  return f_log(ctx, f_add(ctx, constant(ctx, 1.0f, x.shape()), x));
}

Value f_floor(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());

  const size_t fbits = ctx->getFxpBits();
  return _lshift(ctx, _arshift(ctx, x, fbits), fbits).asFxp();
}

Value f_ceil(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());

  // TODO: Add fxp::epsilon
  return f_floor(
      ctx,
      f_add(ctx, x,
            constant(ctx, 1.0 - (1.0 / (1 << ctx->getFxpBits())), x.shape())
                .asFxp()));
}

Value f_log2(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  YACL_ENFORCE(x.isFxp());

  return detail::log2_pade_approx(ctx, x).asFxp();
}

Value f_exp2(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  return detail::exp2_pade_approx(ctx, x);
}

Value f_tanh(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  return detail::tanh_pade_approx(ctx, x);
}

// Reference:
//  1. https://dl.acm.org/doi/10.1145/3411501.3419427
// Main idea:
//  1. convert x to u * 2^(e + 1) while u belongs to [0.25, 0.5).
//  2. get a nice approximation for u part.
//  3. get the compensation for 2^(e + 1) part.
//  4. multiple two parts and get the result.
Value f_rsqrt(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  const size_t k = SizeOf(ctx->getField()) * 8;
  const size_t f = ctx->getFxpBits();
  const auto k1 = constant(ctx, 1U, x.shape());

  // let e = NP2(x)
  //   , z = 2^(e+f)
  auto z = _lshift(ctx, highestOneBit(ctx, x), 1);

  // let u in [0.25, 0.5)
  auto z_rev = _bitrev(ctx, z, 0, 2 * f);
  hintNumberOfBits(z_rev, 2 * f);
  auto u = _trunc(ctx, _mul(ctx, x, z_rev)).asFxp();

  // let rsqrt(u) = 26.02942339 * u^4 - 49.86605845 * u^3 + 38.4714796 * u^2
  // - 15.47994394 * u + 4.14285016
  std::vector<Value> coeffs = {constant(ctx, -15.47994394, x.shape()),
                               constant(ctx, 38.4714796, x.shape()),
                               constant(ctx, -49.86605845, x.shape()),
                               constant(ctx, 26.02942339, x.shape())};
  auto r = f_add(ctx, f_polynomial(ctx, u, coeffs),
                 constant(ctx, 4.14285016, x.shape()));

  // let a = 2^((e+f)/2), that is a[i] = 1 for i = (e+f)/2 else 0
  // let b = lsb(e+f)
  auto a = constant(ctx, 0U, x.shape());
  auto b = constant(ctx, 0U, x.shape());
  for (size_t i = 0; i < k / 2; i++) {
    auto z_2i = _rshift(ctx, z, 2 * i);
    auto z_2i1 = _rshift(ctx, z, 2 * i + 1);
    // a[i] = z[2*i] ^ z[2*i+1]
    auto a_i = _and(ctx, _xor(ctx, z_2i, z_2i1), k1);
    a = _xor(ctx, a, _lshift(ctx, a_i, i));

    // b ^= z[2*i]
    b = _xor(ctx, b, z_2i);
  }
  b = _and(ctx, b, k1);
  hintNumberOfBits(b, 1);
  auto a_rev = _bitrev(ctx, a, 0, (f / 2) * 2);
  hintNumberOfBits(a_rev, (f / 2) * 2);

  // do compensation
  // Note:
  //   https://arxiv.org/pdf/2107.00501.pdf
  // - the magic number c0 & c1 seems to be wrong.
  // - the LSB algorithm is correct and used in this implementation.
  //
  // The following constant is deduced exactly from:
  //   https://dl.acm.org/doi/10.1145/3411501.3419427
  Value c0;
  Value c1;
  if (f % 2 == 1) {
    c0 = constant(ctx, 1 << ((f + 3) / 2), x.shape());  // f+e even
    c1 = _trunc(ctx, constant(ctx, (1 << (f / 2 + 1)) * std::sqrt(2),
                              x.shape()));  // f+e odd
  } else {
    c0 = _trunc(ctx, constant(ctx, (1 << (f / 2)) * std::sqrt(2),
                              x.shape()));        // f+e even
    c1 = constant(ctx, 1 << (f / 2), x.shape());  // f+e odd
  }

  // let comp = 2^(-(e-1)/2) = mux(b, c1, c0) * a_rev
  auto comp = _mul(ctx, _mux(ctx, b, c0, c1), a_rev);

  return _trunc(ctx, _mul(ctx, r, comp)).asFxp();
}

// Referrence:
// 1. https://eprint.iacr.org/2012/405.pdf, section 6.1
// 2.
// https://github.com/tf-encrypted/tf-encrypted/blob/3b0f14d26e900caf12a92a9ea2284ccd4d58e683/tf_encrypted/protocol/aby3/fp.py#L35-L52
// Goldschmidt iteration, needs an initial approximation of sqrt_inv(x).
// In the end, g is an approximation of sqrt(x) while h is an approximation of
// 1 / (2 * sqrt(x)).
Value f_sqrt(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  const auto c0 = constant(ctx, 0.5, x.shape());
  const auto c1 = constant(ctx, 1.5, x.shape());

  Value y0 = f_rsqrt(ctx, x);
  Value g = f_mul(ctx, x, y0);
  Value h = f_mul(ctx, y0, c0);

  // iterations of 1 is enough.
  const int iterations = 1;

  for (int i = 0; i < iterations; i++) {
    const auto r = f_sub(ctx, c1, f_mul(ctx, g, h));
    g = f_mul(ctx, g, r);
    h = f_mul(ctx, h, r);
  }

  return g;
}

}  // namespace spu::kernel::hal
