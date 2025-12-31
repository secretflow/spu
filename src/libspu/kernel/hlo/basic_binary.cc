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

#include "libspu/kernel/hal/complex.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/utils.h"

namespace spu::kernel::hlo {

#define SIMPLE_BINARY_KERNEL_DEFN(NAME, HalFcn)           \
  spu::Value NAME(SPUContext *ctx, const spu::Value &lhs, \
                  const spu::Value &rhs) {                \
    SPU_ENFORCE(!lhs.isComplex() && !rhs.isComplex());    \
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
SIMPLE_BINARY_KERNEL_DEFN(Complex, hal::complex)
SIMPLE_BINARY_KERNEL_DEFN(Atan2, hal::atan2)

#undef SIMPLE_BINARY_KERNEL_DEFN

spu::Value Remainder(SPUContext *ctx, const spu::Value &lhs,
                     const spu::Value &rhs) {
  SPU_ENFORCE(lhs.dtype() == rhs.dtype(), "dtype mismatch {} != {}",
              lhs.dtype(), rhs.dtype());
  SPU_ENFORCE(!lhs.isComplex() && !rhs.isComplex());

  // 1st: find quotient by x/y
  auto quotient = hal::div(ctx, lhs, rhs);

  if (lhs.isFxp() || rhs.isFxp()) {
    // 2nd: round to nearest number through (x >= 0.0) ? floor(x) : ceil(x)...
    auto zero = hal::zeros(ctx, quotient.dtype(), quotient.shape());
    quotient = hal::select(ctx, hal::greater_equal(ctx, quotient, zero),
                           hal::floor(ctx, quotient), hal::ceil(ctx, quotient));
  }

  // 3rd: rem = numer - rquot * denom
  return hal::sub(ctx, lhs, hal::mul(ctx, quotient, rhs));
}

spu::Value Dot(SPUContext *ctx, const spu::Value &lhs, const spu::Value &rhs) {
  SPU_ENFORCE(lhs.shape().isTensor() && lhs.shape().size() <= 2);
  SPU_ENFORCE(rhs.shape().isTensor() && rhs.shape().size() <= 2);
  SPU_ENFORCE(!lhs.isComplex() && !rhs.isComplex());

  return hal::matmul(ctx, lhs, rhs);
}

spu::Value DotGeneral(SPUContext *ctx, const spu::Value &lhs,
                      const spu::Value &rhs) {
  int64_t num_batch = lhs.shape()[0];
  if (ctx->config().experimental_enable_bmm) {
    return hal::batch_matmul(ctx, lhs, rhs);
  }

  std::vector<spu::Value> results(num_batch);
  Index lhs_slice_begin(3, 0);
  Index lhs_slice_end(lhs.shape().begin(), lhs.shape().end());
  Index rhs_slice_begin(3, 0);
  Index rhs_slice_end(rhs.shape().begin(), rhs.shape().end());
  Strides strides(lhs.shape().size(), 1);

  Shape lhs_slice_shape{lhs.shape()[1], lhs.shape()[2]};
  Shape rhs_slice_shape{rhs.shape()[1], rhs.shape()[2]};

  for (int64_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
    lhs_slice_begin[0] = batch_idx;
    lhs_slice_end[0] = batch_idx + 1;
    rhs_slice_begin[0] = batch_idx;
    rhs_slice_end[0] = batch_idx + 1;
    auto lhs_slice = kernel::hal::reshape(
        ctx,
        kernel::hal::slice(ctx, lhs, lhs_slice_begin, lhs_slice_end, strides),
        lhs_slice_shape);
    auto rhs_slice = kernel::hal::reshape(
        ctx,
        kernel::hal::slice(ctx, rhs, rhs_slice_begin, rhs_slice_end, strides),
        rhs_slice_shape);
    results[batch_idx] = kernel::hal::unsqueeze(
        ctx, kernel::hal::matmul(ctx, lhs_slice, rhs_slice));
  }

  return kernel::hal::concatenate(ctx, results, 0);
}

}  // namespace spu::kernel::hlo
