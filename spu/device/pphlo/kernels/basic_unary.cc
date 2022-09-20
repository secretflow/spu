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

#include "spu/device/pphlo/kernels/basic_unary.h"

#include "spu/hal/polymorphic.h"

namespace spu::device::pphlo::kernel {

#define SIMPLE_UNARY_KERNEL_DEFN(NAME, HalFcn)                                 \
  hal::Value NAME(HalContext *ctx, const hal::Value &in) {                     \
    return HalFcn(ctx, in);                                                    \
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
SIMPLE_UNARY_KERNEL_DEFN(Not, hal::logical_not)
SIMPLE_UNARY_KERNEL_DEFN(Rsqrt, hal::sqrt_inv)

#undef SIMPLE_UNARY_KERNEL_DEFN

} // namespace spu::device::pphlo::kernel
