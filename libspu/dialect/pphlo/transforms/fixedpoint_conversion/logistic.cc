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

// logisticMM1: f(x) = 0.5 + 0.125 * x
Value logistic_mm1(builder::FxpBuilder &builder, Value input) {
  auto c1 = builder.fxp_constant(0.5);
  auto c2 = builder.fxp_constant(0.125);

  // t1 = c2 * x
  auto mul = builder.mul(c2, input);

  // c1 + t1
  return builder.add(c1, mul);
}

// logistic SEG3
// f(x) = 0.5 + 0.125x if -4 <= x <= 4
//        1            if       x > 4
//        0            if  -4 > x
Value logistic_seg3(builder::FxpBuilder &builder, Value input) {
  // meddile section mm1
  auto mm1 = logistic_mm1(builder, input);

  // Seg constant
  auto left = builder.fxp_constant(-4.0);
  auto right = builder.fxp_constant(4.0);

  auto zero = builder.fxp_constant(0.0);
  auto one = builder.fxp_constant(1.0);

  // select(x > 4, 1, mm1)
  auto comp1 = builder.greater(input, right);
  auto right_select = builder.select(comp1, one, mm1, mm1.getType());
  // select(x < -4, 0, mm1)
  auto comp2 = builder.less(input, left);
  return builder.select(comp2, zero, right_select, mm1.getType());
}

// f(x) = 1/(1+exp(-x))
Value logistic_real(builder::FxpBuilder &builder, Value input) {
  // -x
  auto neg_x = builder.negate(input);
  // exp(-x)
  auto exp_n_x = exponential_approx(builder, neg_x);

  // 1 + exp(-x)
  auto one = builder.fxp_constant(1.0);
  auto add = builder.add(one, exp_n_x);

  return reciprocal_approx(builder, add);
}

}  // namespace mlir::spu::pphlo::fixedpoint::impl
