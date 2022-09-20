// Copyright 2022 Ant Group Co., Ltd.
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

#include "spu/device/pphlo/kernels/basic_binary.h"

#include "spu/hal/constants.h"
#include "spu/hal/polymorphic.h"
#include "spu/hal/type_cast.h"

namespace spu::device::pphlo::kernel {

#define SIMPLE_BINARY_KERNEL_DEFN(NAME, HalFcn)                                \
  hal::Value NAME(HalContext *ctx, const hal::Value &lhs,                      \
                  const hal::Value &rhs) {                                     \
    return HalFcn(ctx, lhs, rhs);                                              \
  }

SIMPLE_BINARY_KERNEL_DEFN(Add, hal::add)
SIMPLE_BINARY_KERNEL_DEFN(Equal, hal::equal);
SIMPLE_BINARY_KERNEL_DEFN(Sub, hal::sub)
SIMPLE_BINARY_KERNEL_DEFN(Less, hal::less)
SIMPLE_BINARY_KERNEL_DEFN(Greater, hal::greater)
SIMPLE_BINARY_KERNEL_DEFN(Mul, hal::mul)
SIMPLE_BINARY_KERNEL_DEFN(Power, hal::power)
SIMPLE_BINARY_KERNEL_DEFN(Max, hal::max)
SIMPLE_BINARY_KERNEL_DEFN(Min, hal::min)
SIMPLE_BINARY_KERNEL_DEFN(And, hal::bitwise_and)
SIMPLE_BINARY_KERNEL_DEFN(Or, hal::bitwise_or)
SIMPLE_BINARY_KERNEL_DEFN(Xor, hal::bitwise_xor)
SIMPLE_BINARY_KERNEL_DEFN(Div, hal::div)

#undef SIMPLE_BINARY_KERNEL_DEFN

hal::Value Remainder(HalContext *ctx, const hal::Value &lhs,
                     const hal::Value &rhs) {
  YASL_ENFORCE(lhs.dtype() == rhs.dtype(), "dtype mismatch {} != {}",
               lhs.dtype(), rhs.dtype());

  auto lhs_f = lhs;
  auto rhs_f = rhs;

  // 1st: find quotient by x/y
  if (lhs_f.isInt()) {
    lhs_f = hal::dtype_cast(ctx, lhs_f, DT_FXP);
    rhs_f = hal::dtype_cast(ctx, rhs_f, DT_FXP);
  }

  auto quotient = hal::div(ctx, lhs_f, rhs_f);
  // 2nd: round to nearst number through (x >= 0.0) ? floor(x) : ceil(x)...
  auto zero = hal::constant(ctx, 0.0F, quotient.shape());
  auto rquot = hal::select(ctx, hal::greater_equal(ctx, quotient, zero),
                           hal::floor(ctx, quotient), hal::ceil(ctx, quotient));
  // 3rd: rem = numer - rquot * denom
  auto ret = hal::sub(ctx, lhs_f, hal::mul(ctx, rquot, rhs_f));

  if (lhs.isInt()) {
    ret = hal::dtype_cast(ctx, ret, lhs.dtype());
  }
  return ret;
}

hal::Value Dot(HalContext *ctx, const hal::Value &lhs, const hal::Value &rhs) {
  YASL_ENFORCE(!lhs.shape().empty() && lhs.shape().size() <= 2);
  YASL_ENFORCE(!rhs.shape().empty() && rhs.shape().size() <= 2);

  return hal::matmul(ctx, lhs, rhs);
}

} // namespace spu::device::pphlo::kernel
