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

Value mask_number_of_bits(builder::FxpBuilder &builder, const Value &in,
                          int64_t nbits) {
  auto mask = builder.uint_constant((static_cast<uint128_t>(1) << nbits) - 1);
  auto out = builder.and_(in, mask);
  return out;
}

Value reciprocal_goldschmidt_normalized_approx(builder::FxpBuilder &builder,
                                               const Value &b_abs,
                                               const Value &factor) {
  // compute normalize x_abs, [0.5, 1)
  auto c = builder.mul(b_abs, factor, SignType::Positive);

  // initial guess:
  //   w = 1/b = 2.9142 - 2c when c >= 0.5 and c < 1
  auto k2 = builder.int_constant(2);
  auto k2_9142 = builder.fxp_constant(2.9142F);
  auto w = builder.substract(k2_9142, builder.mul_no_trunc(k2, c));

  // init r=w, e=1-c*w
  auto k1_ = builder.fxp_constant(1.0F);
  auto r = w;
  auto e = builder.substract(k1_, builder.mul(c, w, SignType::Positive));

  auto num_iters = builder.getConfig().div_iter;
  if (builder.getCurrentFxpBits() >= 30) {
    // default 2 iters of goldschmidt can only get precision about 14 bits.
    // so if fxp>=30, we use 3 iters by default, which get about 28 bits
    // precision.
    num_iters = std::max<int64_t>(num_iters, 3);
  }

  // iterate, r=r(1+e), e=e*e
  for (int64_t itr = 0; itr < num_iters; itr++) {
    r = builder.mul(r, builder.add(e, k1_), SignType::Positive);
    e = builder.mul(e, e, SignType::Positive);
  }
  return r;
}

// Reference:
//   Chapter 3.4 Division @ Secure Computation With Fixed Point Number
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
Value div_goldschmidt_general(builder::FxpBuilder &builder, Value a, Value b,
                              SignType a_sign, SignType b_sign) {
  Value b_abs;
  Value sign;
  switch (b_sign) {
    case SignType::Unknown: {
      // hack: get an int sign
      sign = builder.sign(
          builder.bitcast(b, builder.getIntTypeWithSameWidth(b.getType())));
      b_abs = builder.mul_no_trunc(sign, b);
      break;
    }
    case SignType::Positive: {
      b_abs = b;
      sign = builder.int_constant(1);
      break;
    }
    case SignType::Negative: {
      b_abs = builder.negate(b);
      sign = builder.int_constant(-1);
      break;
    }
  }

  auto b_msb = highestOneBit(builder, b_abs);

  // factor = 2^{f-m} = 2^{-m} * 2^f, the fixed point repr of 2^{-m}
  auto factor = builder.bitrev(b_msb, 0, 2 * builder.getCurrentFxpBits());
  factor =
      mask_number_of_bits(builder, factor, 2 * builder.getCurrentFxpBits());
  factor = builder.bitcast(factor, b.getType());

  auto r = reciprocal_goldschmidt_normalized_approx(builder, b_abs, factor);

  // r from goldschmidt iteration is always positive
  // so sign(r*a) = sign(a)
  r = builder.mul(r, a, a_sign);
  // also, sign(r*factor) = sign(r)
  r = builder.mul(r, factor, a_sign);

  return builder.mul_no_trunc(r, sign);
}

Value reciprocal_goldschmidt_positive(builder::FxpBuilder &builder,
                                      Value abs_in) {
  auto b_msb = highestOneBit(builder, abs_in);

  // factor = 2^{f-m} = 2^{-m} * 2^f, the fixed point repr of 2^{-m}
  auto factor = builder.bitrev(b_msb, 0, 2 * builder.getCurrentFxpBits());
  factor =
      mask_number_of_bits(builder, factor, 2 * builder.getCurrentFxpBits());
  factor = builder.bitcast(factor, abs_in.getType());

  // compute approximation of normalize b_abs
  auto r = reciprocal_goldschmidt_normalized_approx(builder, abs_in, factor);

  r = builder.mul(r, factor, SignType::Positive);

  return r;
}

Value reciprocal_goldschmidt(builder::FxpBuilder &builder, Value input) {
  // hack: get an int sign
  auto sign = builder.sign(
      builder.bitcast(input, builder.getIntTypeWithSameWidth(input.getType())));

  auto abs_in = builder.mul_no_trunc(sign, input);

  auto r = reciprocal_goldschmidt_positive(builder, abs_in);

  return builder.mul_no_trunc(sign, r);
}

Value ErfImpl(builder::FxpBuilder &builder, Value input) {
  llvm::SmallVector<double, 5> kErfCoefficient{1.0, 0.278393, 0.230389,
                                               0.000972, 0.078108};

  auto z = polynomial(builder, input, kErfCoefficient, SignType::Positive,
                      SignType::Positive);
  z = builder.square(z);
  z = builder.square(z);
  z = reciprocal_goldschmidt_positive(builder, z);

  auto one = builder.fxp_constant(1.0F);
  return builder.substract(one, z);
}

Value erf_poly(builder::FxpBuilder &builder, Value input) {
  auto sign = builder.sign(
      builder.bitcast(input, builder.getIntTypeWithSameWidth(input.getType())));

  auto abs_x = builder.mul_no_trunc(sign, input);

  auto three = builder.fxp_constant(3.0F);
  auto cond = builder.less(abs_x, three);

  auto erf = ErfImpl(builder, abs_x);

  // we do this truncation because:
  // 1. for large abs_x, reciprocal may overflow
  // 2. error is sufficiently small (< 2.2e-5)
  auto one = builder.fxp_constant(1.0F);
  erf = builder.select(cond, erf, one, erf.getType());

  return builder.mul_no_trunc(erf, sign);
}

}  // namespace mlir::spu::pphlo::fixedpoint::impl