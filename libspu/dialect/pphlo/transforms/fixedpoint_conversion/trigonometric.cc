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

// Reference:
// https://github.com/facebookresearch/CrypTen/blob/6ef151101668591bcfb2bbf7e7ebd39ab6db0413/crypten/common/functions/approximations.py#L365
Value compute_chebyshev_polynomials(builder::FxpBuilder &builder, Value x,
                                    int64_t terms) {
  // Ref:
  // https://en.wikipedia.org/wiki/Chebyshev_polynomials#Recurrence_definition
  // Chebyshev Polynomials of the first kind are defined as
  //.. math::
  //    P_0(x) = 1
  //    P_1(x) = x
  //    P_{n+1}(x) = 2xP_{n}(x) - P_{n-1}(x)
  llvm::SmallVector<Value> poly = {x};

  // y = 4*x^2 - 2
  auto four = builder.int_constant(4);
  auto two = builder.fxp_constant(2.0F);
  // x^2
  auto square_x = builder.mul(x, x);
  // 4*x^2
  auto mul = builder.mul_no_trunc(four, square_x);
  // y  = 4*x^2 - 2
  auto y = builder.substract(mul, two);
  // z = y - 1
  auto one = builder.fxp_constant(1.0F);
  auto z = builder.substract(y, one);

  // poly = x * z
  poly.emplace_back(builder.mul(x, z));

  for (int64_t idx = 2; idx < terms; ++idx) {
    // next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
    auto p1 = builder.mul(y, poly[idx - 1]);
    auto next = builder.substract(p1, poly[idx - 2]);
    poly.emplace_back(next);
  }

  return builder.concate(poly, 0);
}

Value tanh_chebyshev(builder::FxpBuilder &builder, Value input) {
  // Cheb coeff, deg = 17, domain = [-5,5]
  llvm::SmallVector<double, 9> kCoeffs = {
      1.2514045938932097,   -0.3655987797163166,   0.17253141478140663,
      -0.08943445792774211, 0.047703017901250824,  -0.025830290571688078,
      0.014338801903468182, -0.008541730970059077, 0.0061230685785789475};

  auto in_rt = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto coeff_type = RankedTensorType::get({1, 9}, in_rt.getElementType());
  auto coeff_value = builder.fxp_constant_with_type(coeff_type, kCoeffs);

  auto flatten_x =
      builder.reshape(input, {static_cast<int64_t>(1), in_rt.getNumElements()});

  // Use flattened_type for builder
  builder.replaceBaseFxpValue(flatten_x);

  // Clamp input to [-5.0, 5.0]
  auto neg_five = builder.fxp_constant(-5.0F);
  auto five = builder.fxp_constant(5.0F);
  auto clamped = builder.clamp(neg_five, flatten_x, five);

  // x = 0.2*x
  auto p_2 = builder.fxp_constant(0.2F);
  auto normalized_x = builder.mul(p_2, clamped);

  auto poly =
      compute_chebyshev_polynomials(builder, normalized_x, kCoeffs.size());

  auto mmul = builder.dot(coeff_value, poly);

  // Restore buidler type
  builder.replaceBaseFxpValue(input);

  return builder.reshape(mmul, in_rt.getShape());
}

Value sine_chebyshev(builder::FxpBuilder &builder, Value input) {
  llvm::SmallVector<double, 5> kCoeffs = {
      -0.07570787578233389, -0.8532364056408055, 0.2474789050491474,
      -0.02719844932262742, 0.0016750058127101841};

  auto in_rt = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto coeff_type = RankedTensorType::get({1, 5}, in_rt.getElementType());
  auto coeff_value = builder.fxp_constant_with_type(coeff_type, kCoeffs);

  // Normalize input to[-pi, pi]
  // theta - TWO_PI * Math.floor((theta + Math.PI) / TWO_PI)
  auto pi = builder.fxp_constant(M_PI);
  auto two_pi = builder.fxp_constant(2 * M_PI);
  auto two_pi_inv = builder.fxp_constant(1 / (2 * M_PI));

  // theta + pi
  Value normalized = builder.add(input, pi);
  // (theta + Math.PI) / TWO_PI
  normalized = builder.mul(normalized, two_pi_inv);
  // Math.floor((theta + Math.PI) / TWO_PI)
  normalized = builder.floor(normalized);
  // TWO_PI * Math.floor((theta + Math.PI) / TWO_PI)
  normalized = builder.mul(normalized, two_pi);
  // theta - TWO_PI * Math.floor((theta + Math.PI) / TWO_PI)
  normalized = builder.substract(input, normalized);

  normalized = builder.reshape(normalized, {1, in_rt.getNumElements()});

  // Flattened
  builder.replaceBaseFxpValue(normalized);

  // rescale the original x
  auto c1 = builder.fxp_constant(0.25464790894703254F);
  normalized = builder.mul(normalized, c1);

  auto poly =
      compute_chebyshev_polynomials(builder, normalized, kCoeffs.size());

  // Unflattened
  builder.replaceBaseFxpValue(input);

  auto ret = builder.dot(coeff_value, poly);

  return builder.reshape(
      ret, mlir::dyn_cast<ShapedType>(input.getType()).getShape());
}

Value cosine_chebyshev(builder::FxpBuilder &builder, Value input) {
  auto half_pi = builder.fxp_constant(M_PI / 2);
  // cos(x) = sin(pi/2 - x)
  // pi/2 - x
  auto shifted = builder.substract(half_pi, input);
  return sine_approx(builder, shifted);
}

}  // namespace mlir::spu::pphlo::fixedpoint::impl
