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

Value AtanApproxLocal(builder::FxpBuilder &builder, const Value &x) {
  // 6-order minimax approximation with max error < 6.3893490851163973e-6
  static std::array<double, 7> kAtanCoefficientSmall{
      6.3893490851163976e-06, 0.99938232039482577, 0.0096717091887422429,
      -0.38851091678439126,   0.13850820695354954, 0.065822467870128534,
      -0.039488402923576769};

  // 10-order minimax approximation with max error < 1.4802815832055511e-9
  static std::array<double, 11> kAtanCoefficientLarge{
      7.3035884235708622e-09, 0.99999906394905635,   3.5324890092487464e-05,
      -0.33393042345794194,   0.0054765660426422556, 0.16982068444578205,
      0.10531189733914688,    -0.37905943050720364,  0.32946653597875702,
      -0.1337452245060563,    0.022023163399866309};

  if (builder.getCurrentFxpBits() <= 20) {
    return polynomial(builder, x, kAtanCoefficientSmall, SignType::Positive,
                      SignType::Positive);
  } else {
    return polynomial(builder, x, kAtanCoefficientLarge, SignType::Positive,
                      SignType::Positive);
  }
}

Value atan2_minmax(builder::FxpBuilder &builder, Value y, Value x) {
  auto sign_x = builder.sign(
      builder.bitcast(x, builder.getIntTypeWithSameWidth(x.getType())));

  auto sign_y = builder.sign(
      builder.bitcast(y, builder.getIntTypeWithSameWidth(y.getType())));

  auto abs_x = builder.mul_no_trunc(sign_x, x);
  auto abs_y = builder.mul_no_trunc(sign_y, y);

  auto cmp = builder.less(abs_x, abs_y);

  auto abs_xy = builder.add(abs_x, abs_y);

  auto bigger = builder.select(cmp, abs_y, abs_x, abs_xy.getType());
  auto smaller = builder.substract(abs_xy, bigger);

  auto tangent = div_goldschmidt_general(
      builder, smaller, bigger, SignType::Positive, SignType::Positive);

  // approximation of arctan(tangent) when tancleargent is in [0,1]
  auto theta = AtanApproxLocal(builder, tangent);

  // To do re-mapping:
  //   1. if abs_y > abs_x (indeed, we compute cot(\theta) before), then \theta
  //   = pi/2 - \theta
  //   2. if x < 0 (we compute tan(pi - \theta) before), then \theta = pi -
  //   \theta
  //   3. if y < 0 (we compute tan(-\theta) before), then \theta = -\theta
  auto m_pi_2 = builder.fxp_constant(M_PI_2);
  auto m_pi = builder.fxp_constant(M_PI);
  theta = builder.select(cmp, builder.substract(m_pi_2, theta), theta,
                         theta.getType());
  theta =
      builder.select(builder.equal(sign_x, builder.int_constant(-1)),
                     builder.substract(m_pi, theta), theta, theta.getType());

  return builder.mul_no_trunc(sign_y, theta);
}

}  // namespace mlir::spu::pphlo::fixedpoint::impl