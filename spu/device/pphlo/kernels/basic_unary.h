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

#pragma once

#include "spu/hal/hal.h"

namespace spu::device::pphlo::kernel {

#define SIMPLE_UNARY_KERNEL_DECL(NAME)                                         \
  hal::Value NAME(HalContext *ctx, const hal::Value &in);

SIMPLE_UNARY_KERNEL_DECL(Reciprocal)
SIMPLE_UNARY_KERNEL_DECL(Neg)
SIMPLE_UNARY_KERNEL_DECL(Exp)
SIMPLE_UNARY_KERNEL_DECL(Log)
SIMPLE_UNARY_KERNEL_DECL(Log1p)
SIMPLE_UNARY_KERNEL_DECL(Floor)
SIMPLE_UNARY_KERNEL_DECL(Ceil)
SIMPLE_UNARY_KERNEL_DECL(Abs)
SIMPLE_UNARY_KERNEL_DECL(Logistic)
SIMPLE_UNARY_KERNEL_DECL(Tanh)
SIMPLE_UNARY_KERNEL_DECL(Not)
SIMPLE_UNARY_KERNEL_DECL(Rsqrt)

#undef SIMPLE_UNARY_KERNEL_DECL

} // namespace spu::device::pphlo::kernel
