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

#include "libspu/kernel/hlo/basic_binary.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::kernel::hlo {

#define SIMPLE_BINARY_KERNEL_DEFN(NAME, HalFcn)           \
  spu::Value NAME(SPUContext *ctx, const spu::Value &lhs, \
                  const spu::Value &rhs) {                \
    return HalFcn(ctx, lhs, rhs);                         \
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
SIMPLE_BINARY_KERNEL_DEFN(NotEqual, hal::not_equal)
SIMPLE_BINARY_KERNEL_DEFN(LessEqual, hal::less_equal)
SIMPLE_BINARY_KERNEL_DEFN(GreaterEqual, hal::greater_equal)

#undef SIMPLE_BINARY_KERNEL_DEFN

spu::Value Remainder(SPUContext *ctx, const spu::Value &lhs,
                     const spu::Value &rhs) {
  SPU_ENFORCE(lhs.dtype() == rhs.dtype(), "dtype mismatch {} != {}",
              lhs.dtype(), rhs.dtype());

  // 1st: find quotient by x/y
  auto quotient = hal::div(ctx, lhs, rhs);

  if (lhs.isFxp() || rhs.isFxp()) {
    // 2nd: round to nearst number through (x >= 0.0) ? floor(x) : ceil(x)...
    auto zero = hal::zeros(ctx, quotient.dtype(), quotient.shape());
    quotient = hal::select(ctx, hal::greater_equal(ctx, quotient, zero),
                           hal::floor(ctx, quotient), hal::ceil(ctx, quotient));
  }

  // 3rd: rem = numer - rquot * denom
  return hal::sub(ctx, lhs, hal::mul(ctx, quotient, rhs));
}

spu::Value Dot(SPUContext *ctx, const spu::Value &lhs, const spu::Value &rhs) {
  SPU_ENFORCE(!lhs.shape().empty() && lhs.shape().size() <= 2);
  SPU_ENFORCE(!rhs.shape().empty() && rhs.shape().size() <= 2);

  return hal::matmul(ctx, lhs, rhs);
}

}  // namespace spu::kernel::hlo
