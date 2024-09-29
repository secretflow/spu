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

#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/builder.h"

namespace mlir::spu::pphlo::fixedpoint::impl {

// Pade approximation of exp2(x), x is in [0, 1].
// p1015(x) = 0.100000007744302 * 10
//             + x * 0.693147180426163
//             + x^2 * 0.240226510710170
//             + x^3 * 0.555040686204663 / 10
//             + x^4 * 0.961834122588046 / 100
//             + x^5 * 0.133273035928143 / 100
Value exp2_pade_normalized(builder::FxpBuilder &builder, const Value &x) {
  auto x2 = builder.mul(x, x);
  auto x3 = builder.mul(x, x2);
  auto x4 = builder.mul(x, x3);
  auto x5 = builder.mul(x, x4);

  const auto p0 = builder.fxp_constant(0.100000007744302F * 10);
  const auto p1 = builder.fxp_constant(0.693147180426163F);
  const auto p2 = builder.fxp_constant(0.240226510710170F);
  const auto p3 = builder.fxp_constant(0.555040686204663F / 10);
  const auto p4 = builder.fxp_constant(0.961834122588046F / 100);
  const auto p5 = builder.fxp_constant(0.133273035928143F / 100);

  auto res = builder.mul_no_trunc(x, p1);
  res = builder.add(res, builder.mul_no_trunc(x2, p2));
  res = builder.add(res, builder.mul_no_trunc(x3, p3));
  res = builder.add(res, builder.mul_no_trunc(x4, p4));
  res = builder.add(res, builder.mul_no_trunc(x5, p5));

  auto p0_fxp_bits = builder.getCurrentFxpBits();
  return builder.add(builder.truncation(res, p0_fxp_bits), p0);
}

// Refer to
// Chapter 5 Exponentiation and Logarithms
// Benchmarking Privacy Preserving Scientific Operations
// https://www.esat.kuleuven.be/cosic/publications/article-3013.pdf
// NOTE(junfeng): The valid integer bits of x is 5. Otherwise, the output is
// incorrect.
Value exp2_pade(builder::FxpBuilder &builder, Value input) {
  auto int_type = builder.getIntTypeWithSameWidth(input.getType(), true);
  auto in_fxp_bits = builder.getCurrentFxpBits();

  auto k1 = builder.uint_constant(1U);
  auto bit_width = builder.getCurrentFxpWidth();

  auto x_msb = builder.rshift(builder.bitcast(input, int_type), bit_width - 1);
  auto x_int = builder.rshift(builder.bitcast(input, int_type), in_fxp_bits);
  auto x_int_f =
      builder.bitcast(builder.lshift(x_int, in_fxp_bits), input.getType());
  auto x_fraction = builder.substract(input, x_int_f);

  auto ret = exp2_pade_normalized(builder, x_fraction);

  // TODO(junfeng): Make int_bits configurable.
  const size_t int_bits = 5;
  for (size_t idx = 0; idx < int_bits; idx++) {
    auto a = builder.and_(builder.rshift(x_int, idx), k1);
    auto K = builder.uint_constant(
        1U << std::min<int64_t>(1UL << idx, bit_width - 2));
    ret = builder.mul_no_trunc(
        ret, builder.add(builder.mul_no_trunc(a, K), builder.substract(k1, a)));
  }

  // If we could ensure the integer bits of x is 5.
  // we have x, -x, -x_hat. x_hat is 2's complement of -x.
  // Then,
  //             x + (x_hat) = 32
  //            (x_hat) - 32 = -x
  //  exp2(x_hat) / exp2(32) = exp(-x)
  //  so exp(-x) = exp2(x_hat) / exp2(32)
  auto ret_i = builder.bitcast(ret, int_type);
  auto ret_reciprocal = builder.rshift(ret_i, std::pow(2, int_bits));
  ret_reciprocal = builder.bitcast(ret_reciprocal, ret.getType());

  // ret + msb * (reciprocal - ret)
  return builder.add(
      ret, builder.mul_no_trunc(x_msb, builder.substract(ret_reciprocal, ret)));
}

Value exponential_pade(builder::FxpBuilder &builder, Value input) {
  // c1 = log2(exp(1.0))
  auto c1 = builder.fxp_constant(std::log2(std::exp(1.0F)));
  auto normalized_input = builder.mul(input, c1);

  return exp2_pade(builder, normalized_input);
}

// see https://lvdmaaten.github.io/publications/papers/crypten.pdf
//   exp(x) = (1 + x / n) ^ n, when n is infinite large.
Value exponential_taylor(builder::FxpBuilder &builder, Value input) {
  auto exp_iter = builder.getConfig().exp_iter;

  // trunc x exp iter
  auto trunc = builder.truncation(input, exp_iter);
  auto casted = builder.bitcast(trunc, input.getType());

  // 1 + casted
  auto one = builder.fxp_constant(1.0F);
  auto added = builder.add(one, casted);

  // for i = 0...exp_iter { ret = mul(added, added) }
  Value accu = added;
  for (int64_t idx = 0; idx < exp_iter; ++idx) {
    accu = builder.mul(accu, accu);
  }

  return accu;
}

}  // namespace mlir::spu::pphlo::fixedpoint::impl
