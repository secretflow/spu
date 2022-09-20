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

#include "spu/hal/context.h"
#include "spu/hal/value.h"

namespace spu::hal {

/// return the identity representation of in value.
// Useful to normalize stateful value (like lazy value).
Value identity(HalContext* ctx, const Value& in);

/// the element-wise absolute value function
// @param in, the value to negate
Value abs(HalContext* ctx, const Value& in);

/// general element-wise add operator
// @param x, the first parameter
// @param y, the second parameter
Value add(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise and operator
// @param x, the first parameter
// @param y, the second parameter
Value bitwise_and(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise xor operator
// @param x, the first parameter
// @param y, the second parameter
Value bitwise_xor(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise or operator
// @param x, the first parameter
// @param y, the second parameter
Value bitwise_or(HalContext* ctx, const Value& x, const Value& y);

/// see numpy.bitwise_not(in)
// @param in, the input parameter
Value bitwise_not(HalContext* ctx, const Value& in);

/// matrix production operator
// @param x, the first parameter
// @param y, the second parameter
Value matmul(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise equal operator
// @param x, the first parameter
// @param y, the second parameter
Value equal(HalContext* ctx, const Value& x, const Value& y);

/// element-wise natural exponential x -> e^x
// @param in, the input value
Value exp(HalContext* ctx, const Value& in);

/// element-wise floor
// @param in, the input value
Value floor(HalContext* ctx, const Value& in);

/// element-wise ceil
// @param in, the input value
Value ceil(HalContext* ctx, const Value& in);

/// general element-wise bitwise greater operator
// @param x, the first parameter
// @param y, the second parameter
Value greater(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise greater or equal operator
// @param x, the first parameter
// @param y, the second parameter
Value greater_equal(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise less operator
// @param x, the first parameter
// @param y, the second parameter
Value less(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise bitwise less or equal operator
// @param x, the first parameter
// @param y, the second parameter
Value less_equal(HalContext* ctx, const Value& x, const Value& y);

/// the element-wise natural logarithm
// @param in, the param
Value log(HalContext* ctx, const Value& in);

/// the element-wise natural logarithm of (1 + x)
// @param in, the param
Value log1p(HalContext* ctx, const Value& in);

/// see numpy.logical_not(in)
// @param in, requires integer one or zero
Value logical_not(HalContext* ctx, const Value& in);

/// the element-wise sigmoid function
// @param in, the param
Value logistic(HalContext* ctx, const Value& in);

/// element-wise maximum
// @param x, first input value
// @param y, second input value
Value max(HalContext* ctx, const Value& x, const Value& y);

/// element-wise minimum
// @param x, first input value
// @param y, second input value
Value min(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise multiply operator
// @param x, the first parameter
// @param y, the second parameter
Value mul(HalContext* ctx, const Value& x, const Value& y);

Value div(HalContext* ctx, const Value& x, const Value& y);

/// see numpy.negate(in)
// @param in, the value to negate
Value negate(HalContext* ctx, const Value& in);

/// general element-wise bitwise equal operator
// @param x, the first parameter
// @param y, the second parameter
Value not_equal(HalContext* ctx, const Value& x, const Value& y);

/// element-wise power x ^ y
// @param x, first input value, must be positive at this moment.
// @param y, second input value
// FIXME(junfeng): fix negative x.
Value power(HalContext* ctx, const Value& x, const Value& y);

/// the element-wise reciprocal function
// @param in, the param
Value reciprocal(HalContext* ctx, const Value& in);

/// see numpy.select
// @param pred, the predicate, requires integer zero or one
// @param a, the first param
// @param b, the second param
Value select(HalContext* ctx, const Value& pred, const Value& a,
             const Value& b);

/// general element-wise subtract operator
// @param x, the first parameter
// @param y, the second parameter
Value sub(HalContext* ctx, const Value& x, const Value& y);

/// general element-wise clamp operator
// @param min, the first parameter
// @param x, the second parameter
// @param max, the third parameter
Value clamp(HalContext* ctx, const Value& min, const Value& x,
            const Value& max);

/// element-wise bitcast (reinterpret_cast)
// @param x, first input value
// @param dtype, second input value
// @param elsize, size of element
Value bitcast(HalContext* ctx, const Value& x, DataType dtype, size_t elsize);

Value left_shift(HalContext* ctx, const Value& x, size_t bits);

Value right_shift_logical(HalContext* ctx, const Value& x, size_t bits);

Value right_shift_arithmetic(HalContext* ctx, const Value& x, size_t bits);

// TODO: why this function? It's not involoved in XLA dispatch path.
Value permute(HalContext* ctx, const Value& x, size_t dimension,
              const Value& permutations);

Value popcount(HalContext* ctx, const Value& x);

/// the element-wise base-2 logarithm of x
// @param in, should be positive, or the result is implementation defined.
Value log2(HalContext* ctx, const Value& in);

/// element-wise 2 to the power x, i.e. x -> 2^x
// @param in, the input value
Value exp2(HalContext* ctx, const Value& x);

/// element-wise hyperbolic tangent, i.e. x -> tanh(x)
// @param in, the input value
Value tanh(HalContext* ctx, const Value& x);

/// element-wise reciprocal of square root operation, i.e. x - > 1.0 / sqrt(x)
// @param in, the input value
Value sqrt_inv(HalContext* ctx, const Value& x);

}  // namespace spu::hal
