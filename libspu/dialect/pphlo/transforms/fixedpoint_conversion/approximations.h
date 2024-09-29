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

#pragma once

#include "mlir/IR/Value.h"

#include "libspu/dialect/pphlo/transforms/fixedpoint_conversion/builder.h"

namespace mlir::spu::pphlo::fixedpoint {

namespace impl {

// Common helper
Value highestOneBit(builder::FxpBuilder &builder, Value x);

// Calc:
//   y = c0 + x*c1 + x^2*c2 + x^3*c3 + ... + x^n*c[n]
Value polynomial(builder::FxpBuilder &builder, Value x,
                 llvm::ArrayRef<double> coeffs,
                 SignType sign_x = SignType::Unknown,
                 SignType sign_ret = SignType::Unknown);

// logistic
Value logistic_mm1(builder::FxpBuilder &builder, Value input);
Value logistic_seg3(builder::FxpBuilder &builder, Value input);
Value logistic_real(builder::FxpBuilder &builder, Value input);

// exponential
Value exponential_pade(builder::FxpBuilder &builder, Value input);
Value exponential_taylor(builder::FxpBuilder &builder, Value input);

// trigonometric
Value tanh_chebyshev(builder::FxpBuilder &builder, Value input);
Value sine_chebyshev(builder::FxpBuilder &builder, Value input);
Value cosine_chebyshev(builder::FxpBuilder &builder, Value input);

// logarithmic
Value log_newton(builder::FxpBuilder &builder, Value input);
Value log_pade(builder::FxpBuilder &builder, Value input);
Value log_minmax(builder::FxpBuilder &builder, Value input);

// division based
// we provide this general function to support some special cases (a or b has
// guarranteed sign) in fxp_approx for better both performance and accuracy.
Value div_goldschmidt_general(builder::FxpBuilder &builder, Value a, Value b,
                              SignType a_sign = SignType::Unknown,
                              SignType b_sign = SignType::Unknown);
Value reciprocal_goldschmidt(builder::FxpBuilder &builder, Value input);
Value erf_poly(builder::FxpBuilder &builder, Value input);

// rsqrt & sqrt
Value rsqrt_expand(builder::FxpBuilder &builder, Value input);
Value sqrt_with_rsqrt(builder::FxpBuilder &builder, Value input);

// atan2
Value atan2_minmax(builder::FxpBuilder &builder, Value lhs, Value rhs);

}  // namespace impl

Value sine_approx(builder::FxpBuilder &builder, Value input);

Value cosine_approx(builder::FxpBuilder &builder, Value input);

Value tanh_approx(builder::FxpBuilder &builder, Value input);

Value exponential_approx(builder::FxpBuilder &builder, Value input);

Value log_approx(builder::FxpBuilder &builder, Value input);

Value rsqrt_approx(builder::FxpBuilder &builder, Value input);

Value reciprocal_approx(builder::FxpBuilder &builder, Value input);

Value sqrt_approx(builder::FxpBuilder &builder, Value input);

Value div_approx(builder::FxpBuilder &builder, Value lhs, Value rhs);

Value logistic_approx(builder::FxpBuilder &builder, Value input);

Value erf_approx(builder::FxpBuilder &builder, Value input);

Value power_approx(builder::FxpBuilder &builder, Value lhs, Value rhs);

Value atan2_approx(builder::FxpBuilder &builder, Value lhs, Value rhs);

}  // namespace mlir::spu::pphlo::fixedpoint
