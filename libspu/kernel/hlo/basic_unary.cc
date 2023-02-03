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

#include "libspu/kernel/hlo/basic_unary.h"

#include "libspu/kernel/context.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/value.h"

namespace spu::kernel::hlo {

#define SIMPLE_UNARY_KERNEL_DEFN(NAME, HalFcn)             \
  spu::Value NAME(HalContext *ctx, const spu::Value &in) { \
    return HalFcn(ctx, in);                                \
  }

SIMPLE_UNARY_KERNEL_DEFN(Reciprocal, hal::reciprocal)
SIMPLE_UNARY_KERNEL_DEFN(Neg, hal::negate)
SIMPLE_UNARY_KERNEL_DEFN(Exp, hal::exp)
SIMPLE_UNARY_KERNEL_DEFN(Log, hal::log)
SIMPLE_UNARY_KERNEL_DEFN(Log1p, hal::log1p)
SIMPLE_UNARY_KERNEL_DEFN(Floor, hal::floor)
SIMPLE_UNARY_KERNEL_DEFN(Ceil, hal::ceil)
SIMPLE_UNARY_KERNEL_DEFN(Abs, hal::abs)
SIMPLE_UNARY_KERNEL_DEFN(Logistic, hal::logistic)
SIMPLE_UNARY_KERNEL_DEFN(Tanh, hal::tanh)
SIMPLE_UNARY_KERNEL_DEFN(Rsqrt, hal::rsqrt)
SIMPLE_UNARY_KERNEL_DEFN(Sqrt, hal::sqrt)

#undef SIMPLE_UNARY_KERNEL_DEFN

spu::Value Expm1(HalContext *ctx, const spu::Value &in) {
  // FIXME: By numpy spec, expm1 should have a higher numeric accuracy compare
  // with exp(x) - 1. SPU is not doing so right now, rethink about what we
  // should do here.
  return hal::sub(ctx, hal::exp(ctx, in), hal::constant(ctx, 1.0F, in.shape()));
}

spu::Value Not(HalContext *ctx, const spu::Value &in) {
  if (in.dtype() == DT_I1) {
    return hal::logical_not(ctx, in);
  } else {
    // By XLA semantics, NotOp for int other than boolean, it should be
    // bitwise not
    return hal::bitwise_not(ctx, in);
  }
}

spu::Value Sign(HalContext *ctx, const spu::Value &in) {
  auto s = hal::sign(ctx, in);
  auto zero =
      hal::dtype_cast(ctx, hal::constant(ctx, 0, in.shape()), s.dtype());
  s = hal::select(ctx, hal::equal(ctx, in, zero), zero, s);
  return hal::dtype_cast(ctx, s, in.dtype());
}

spu::Value Round_AFZ(HalContext *ctx, const spu::Value &in) {
  // select(x < 0, (int)(x-0.5), (int)(x+0.5))
  // -> (float)(int)(x + sign(x) * 0.5)
  YACL_ENFORCE(in.isFxp(), "Round only supports fxp");

  auto sign_in = hal::sign(ctx, in);
  auto p_half = hal::constant(ctx, 0.5, in.shape());
  p_half = hal::mul(ctx, sign_in, p_half);

  auto round = hal::add(ctx, in, p_half);

  return hal::dtype_cast(ctx, hal::dtype_cast(ctx, round, DT_I64), in.dtype());
}

}  // namespace spu::kernel::hlo
