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
#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/builder.h"

namespace mlir::spu::pphlo::fixedpoint::impl {

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
Value log2_pade_normalized(builder::FxpBuilder &builder, Value x) {
  auto x2 = builder.mul(x, x);
  auto x3 = builder.mul(x2, x);

  auto p0 = builder.fxp_constant(-0.205466671951F * 10);
  auto p1 = builder.fxp_constant(-0.88626599391F * 10);
  auto p2 = builder.fxp_constant(0.610585199015F * 10);
  auto p3 = builder.fxp_constant(0.481147460989F * 10);

  auto q0 = builder.fxp_constant(0.353553425277F);
  auto q1 = builder.fxp_constant(0.454517087629F * 10);
  auto q2 = builder.fxp_constant(0.642784209029F * 10);
  auto q3 = builder.fxp_constant(0.1F * 10);

  auto p2524 = builder.mul_no_trunc(x, p1);
  p2524 = builder.add(p2524, builder.mul_no_trunc(x2, p2));
  p2524 = builder.add(p2524, builder.mul_no_trunc(x3, p3));

  auto default_fxp_bits = builder.getCurrentFxpBits();
  auto p2524_frac_bits = builder.getTypeTools().getFxpBits(p2524.getType());

  p2524 = builder.add(
      builder.truncation(p2524, p2524_frac_bits - default_fxp_bits), p0);

  auto q2524 = builder.mul_no_trunc(x, q1);
  q2524 = builder.add(q2524, builder.mul_no_trunc(x2, q2));
  q2524 = builder.add(q2524, builder.mul_no_trunc(x3, q3));

  auto q2524_frac_bits = builder.getTypeTools().getFxpBits(q2524.getType());

  q2524 = builder.add(
      builder.truncation(q2524, q2524_frac_bits - default_fxp_bits), q0);

  return div_approx(builder, p2524, q2524);
}

// Refer to
// Chapter 5 Exponentiation and Logarithms
// Benchmarking Privacy Preserving Scientific Operations
// https://www.esat.kuleuven.be/cosic/publications/article-3013.pdf
Value log2_pade(builder::FxpBuilder &builder, Value input) {
  auto fxp_bits = builder.getCurrentFxpBits();
  auto uint_type = builder.getIntTypeWithSameWidth(input.getType(), true);

  auto x_int = builder.bitcast(input, uint_type);

  // k = popcnt(prefix_or(x))
  auto k = builder.popcnt(builder.prefix_or(x_int));
  k = builder.bitcast(k, builder.getIntTypeWithSameWidth(k.getType()));

  // let x = x_norm * factor, where x in [0.5, 1.0)
  auto msb = highestOneBit(builder, x_int);
  auto factor = builder.bitrev(msb, 0, 2 * fxp_bits);
  factor = builder.bitcast(factor, input.getType());

  auto norm = builder.mul(input, factor);

  // log2(x) = log2(x_norm * factor)
  //         = log2(x_norm) + log2(factor)
  //         = log2(x_norm) + (k-fxp_bits)
  auto log2_normed = log2_pade_normalized(builder, norm);
  auto k_sub_fxp_bits = builder.substract(k, builder.int_constant(fxp_bits));
  auto rhs = builder.convert(
      k_sub_fxp_bits, builder.getTypeTools().getType(log2_normed.getType(),
                                                     Visibility::PUBLIC));
  return builder.add(log2_normed, rhs);
}

Value log_pade(builder::FxpBuilder &builder, Value input) {
  // log(2)*log2(x)
  auto c1 = builder.fxp_constant(std::log(2.0F));
  auto log2_x = log2_pade(builder, input);

  return builder.mul(c1, log2_x);
}

// See P11, A.2.4 Logarithm and Exponent,
// https://lvdmaaten.github.io/publications/papers/crypten.pdf
// https://github.com/facebookresearch/CrypTen/blob/master/crypten/common/functions/approximations.py#L55-L104
// Approximates the natural logarithm using 8th order modified
// Householder iterations. This approximation is accurate within 2% relative
// error on [0.0001, 250].
Value log_newton(builder::FxpBuilder &builder, Value input) {
  // term1 = x/120
  auto c1 = builder.fxp_constant(1 / 120.0);
  auto term1 = builder.mul(input, c1);

  // term2 = 20 * exp(-(2x+1)) + 3
  auto twenty = builder.int_constant(20);
  auto two = builder.int_constant(2);
  auto three = builder.fxp_constant(3.0F);
  auto one = builder.fxp_constant(1.0F);
  auto two_x = builder.mul_no_trunc(two, input);
  auto two_x_p_1 = builder.add(two_x, one);
  auto exp = exponential_approx(builder, two_x_p_1);
  auto twenty_exp = builder.mul_no_trunc(twenty, exp);
  auto term2 = builder.add(twenty_exp, three);

  // y = term1 - term2 + 3.0
  auto s = builder.substract(term1, term2);
  auto y = builder.add(s, three);

  // coeffs
  auto order = builder.getConfig().log_order;
  llvm::SmallVector<double> coeffs{0.0};
  for (int64_t i = 0; i < order; ++i) {
    coeffs.emplace_back(1.0 / (1.0 + i));
  }

  auto iter = builder.getConfig().log_iter;

  for (int64_t i = 0; i < iter; ++i) {
    // h = 1-x*exp(-y)
    auto h = builder.negate(y);
    h = exponential_approx(builder, h);
    h = builder.mul(input, h);
    h = builder.substract(one, h);
    // y = y - poly(h, coeff)
    auto p = polynomial(builder, h, coeffs);
    y = builder.substract(y, p);
  }

  return y;
}

Value log_minmax_normalized(builder::FxpBuilder &builder, Value x) {
  llvm::SmallVector<double, 9> kLogCoefficient{
      0.0,          0.9999964239,  -0.4998741238, 0.3317990258, -0.2407338084,
      0.1676540711, -0.0953293897, 0.0360884937,  -0.0064535442};

  // we have approximation of log(1+x) when x is in [0, 1]
  const auto k1 = builder.fxp_constant(1.0F);
  auto xm1 = builder.substract(x, k1);

  return polynomial(builder, xm1, kLogCoefficient, SignType::Positive,
                    SignType::Positive);
}

// Ref:
// Handbook of Mathematical Functions: with Formulas, Graphs, and Mathematical
// Tables, equation 4.1.44
Value log_minmax(builder::FxpBuilder &builder, Value input) {
  auto num_fxp_bits = builder.getCurrentFxpBits();
  auto int_type = builder.getIntTypeWithSameWidth(input.getType(), true);

  auto x_int = builder.bitcast(input, int_type);

  auto pre_x = builder.prefix_or(x_int);

  // because of bitrev, we can only handle x between (0, 2**(fxp+1)),
  // so we can limit the size of bits
  auto k = builder.popcnt(pre_x, 2 * num_fxp_bits + 2);
  k = builder.bitcast(k, builder.getIntTypeWithSameWidth(k.getType()));

  // get most significant non-zero bit of x
  // we avoid direct using detail::highestOneBit for saving one _prefix_or
  auto pre_x1 = builder.rshift(pre_x, 1);
  auto msb = builder.xor_(pre_x, pre_x1);

  // let x = x_norm * factor, where x in [1.0, 2.0)
  auto factor = builder.bitrev(msb, 0, 2 * num_fxp_bits + 1);
  factor = builder.bitcast(factor, input.getType());
  auto norm = builder.mul(input, factor);

  // log(x) = log(x_norm * factor)
  //        = log(x_norm) + log(factor)
  //        = log(x_norm) + (k - fxp_bits - 1) * log(2)
  auto log_norm = log_minmax_normalized(builder, norm);
  auto log2_e = builder.bitcast(
      builder.lshift(
          builder.substract(k, builder.int_constant(num_fxp_bits + 1)),
          num_fxp_bits),
      input.getType());
  auto k_log2 = builder.fxp_constant(std::log(2));
  auto log_e = builder.mul(log2_e, k_log2);

  return builder.add(log_norm, log_e);
}

}  // namespace mlir::spu::pphlo::fixedpoint::impl