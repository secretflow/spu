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

#pragma once

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

/// the element-wise absolute value function
// @param in, the value to negate
Value abs(SPUContext* ctx, const Value& x);

/// general element-wise add operator
// @param x, the first parameter
// @param y, the second parameter
Value add(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise and operator
// @param x, the first parameter
// @param y, the second parameter
Value bitwise_and(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise xor operator
// @param x, the first parameter
// @param y, the second parameter
Value bitwise_xor(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise or operator
// @param x, the first parameter
// @param y, the second parameter
Value bitwise_or(SPUContext* ctx, const Value& x, const Value& y);

/// see numpy.bitwise_not(in)
// @param in, the input parameter
Value bitwise_not(SPUContext* ctx, const Value& in);

/// matrix production operator
// @param x, the first parameter
// @param y, the second parameter
Value matmul(SPUContext* ctx, const Value& x, const Value& y);

/// matrix production operator
// @param x, the first parameter
// @param y, the second parameter
// @param ix, contract dimensions of first parameter
// @param iy, contract dimensions of second parameter
Value tensordot(SPUContext* ctx, const Value& x, const Value& y,
                const Index& ix, const Index& iy);

/// 2-dimensional convolution operator
// @param x, the input tensor
// @param y, the kernel weight
// @param window_strides, sized-2 window strides
Value conv2d(SPUContext* ctx, const Value& x, const Value& y,
             const Strides& window_strides);

/// general element-wise bitwise equal operator
// @param x, the first parameter
// @param y, the second parameter
Value equal(SPUContext* ctx, const Value& x, const Value& y);

/// element-wise natural exponential x -> e^x
// @param in, the input value
Value exp(SPUContext* ctx, const Value& in);

/// element-wise floor
// @param in, the input value
Value floor(SPUContext* ctx, const Value& in);

/// element-wise ceil
// @param in, the input value
Value ceil(SPUContext* ctx, const Value& in);

/// general element-wise bitwise greater operator
// @param x, the first parameter
// @param y, the second parameter
Value greater(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise greater or equal operator
// @param x, the first parameter
// @param y, the second parameter
Value greater_equal(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise less operator
// @param x, the first parameter
// @param y, the second parameter
Value less(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise less or equal operator
// @param x, the first parameter
// @param y, the second parameter
Value less_equal(SPUContext* ctx, const Value& x, const Value& y);

/// the element-wise natural logarithm
// @param in, the param
Value log(SPUContext* ctx, const Value& in);

/// the element-wise natural logarithm of (1 + x)
// @param in, the param
Value log1p(SPUContext* ctx, const Value& in);

/// see numpy.logical_not(in)
// @param in, requires integer one or zero
Value logical_not(SPUContext* ctx, const Value& in);

/// the element-wise sigmoid function
// @param in, the param
Value logistic(SPUContext* ctx, const Value& in);

/// element-wise maximum
// @param x, first input value
// @param y, second input value
Value max(SPUContext* ctx, const Value& x, const Value& y);

/// element-wise minimum
// @param x, first input value
// @param y, second input value
Value min(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise multiply operator
// @param x, the first parameter
// @param y, the second parameter
Value mul(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise square operator
// @param x, the value to square
Value square(SPUContext* ctx, const Value& x);

Value div(SPUContext* ctx, const Value& x, const Value& y);

/// see numpy.negate(in)
// @param in, the value to negate
Value negate(SPUContext* ctx, const Value& x);

/// general element-wise bitwise equal operator
// @param x, the first parameter
// @param y, the second parameter
Value not_equal(SPUContext* ctx, const Value& x, const Value& y);

/// element-wise power x ^ y
// @param x, first input value, must be positive at this moment.
// @param y, second input value
// FIXME(junfeng): fix negative x.
Value power(SPUContext* ctx, const Value& x, const Value& y);

/// the element-wise reciprocal function
// @param in, the param
Value reciprocal(SPUContext* ctx, const Value& in);

/// see numpy.select
// @param pred, the predicate, requires integer zero or one
// @param a, the first param
// @param b, the second param
Value select(SPUContext* ctx, const Value& pred, const Value& a,
             const Value& b);

/// general element-wise subtract operator
// @param x, the first parameter
// @param y, the second parameter
Value sub(SPUContext* ctx, const Value& x, const Value& y);

/// general element-wise clamp operator
// @param x, the first parameter
// @param min, the second parameter
// @param max, the third parameter
Value clamp(SPUContext* ctx, const Value& x, const Value& min,
            const Value& max);

/// element-wise bitcast (reinterpret_cast)
// @param x, first input value
// @param dtype, second input value
Value bitcast(SPUContext* ctx, const Value& x, DataType dtype);

Value left_shift(SPUContext* ctx, const Value& x, const Sizes& bits);

Value right_shift_logical(SPUContext* ctx, const Value& x, const Sizes& bits);

Value right_shift_arithmetic(SPUContext* ctx, const Value& x,
                             const Sizes& bits);

/// the element-wise base-2 logarithm of x
// @param in, should be positive, or the result is implementation defined.
Value log2(SPUContext* ctx, const Value& in);

/// element-wise 2 to the power x, i.e. x -> 2^x
// @param in, the input value
Value exp2(SPUContext* ctx, const Value& x);

/// element-wise hyperbolic tangent, i.e. x -> tanh(x)
// @param in, the input value
Value tanh(SPUContext* ctx, const Value& x);

/// element-wise sine, i.e. x -> sin(x)
// @param in, the input value
Value sine(SPUContext* ctx, const Value& x);

/// element-wise cosine, i.e. x -> cos(x)
// @param in, the input value
Value cosine(SPUContext* ctx, const Value& x);

/// element-wise the angle between the positive x-axis and the ray from the
/// origin to the point (x, y), confined to (−π, π]
/// Note: -π will not be returned because of the missing of -0 and -inf in
/// fixed-point arithmetic.
// @param y, vertical coordinate
// @param x, horizontal coordinate
Value atan2(SPUContext* ctx, const Value& y, const Value& x);

/// element-wise arccosine, i.e. x -> acos(x)
// @param x, the input value
Value acos(SPUContext* ctx, const Value& x);

/// element-wise arcsine, i.e. x -> asin(x)
// @param x, the input value
Value asin(SPUContext* ctx, const Value& x);

/// element-wise reciprocal of square root operation, i.e. x - > 1.0 / sqrt(x)
// @param in, the input value
Value rsqrt(SPUContext* ctx, const Value& x);

/// element-wise square root operation.
// @param in, the input value
Value sqrt(SPUContext* ctx, const Value& x);

/// element-wise sign operation
// @param in, the input value
Value sign(SPUContext* ctx, const Value& x);

std::optional<Value> oramonehot(SPUContext* ctx, const Value& x,
                                int64_t db_size, bool db_is_secret);

Value oramread(SPUContext* ctx, const Value& x, const Value& y, int64_t offset);

}  // namespace spu::kernel::hal
