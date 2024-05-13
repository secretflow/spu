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

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hlo {

#define SIMPLE_BINARY_KERNEL_DECL(NAME)                   \
  spu::Value NAME(SPUContext *ctx, const spu::Value &lhs, \
                  const spu::Value &rhs);

SIMPLE_BINARY_KERNEL_DECL(Add)
SIMPLE_BINARY_KERNEL_DECL(Equal);
SIMPLE_BINARY_KERNEL_DECL(NotEqual)
SIMPLE_BINARY_KERNEL_DECL(LessEqual)
SIMPLE_BINARY_KERNEL_DECL(GreaterEqual)
SIMPLE_BINARY_KERNEL_DECL(Sub)
SIMPLE_BINARY_KERNEL_DECL(Less)
SIMPLE_BINARY_KERNEL_DECL(Greater)
SIMPLE_BINARY_KERNEL_DECL(Mul)
SIMPLE_BINARY_KERNEL_DECL(Power)
SIMPLE_BINARY_KERNEL_DECL(Max)
SIMPLE_BINARY_KERNEL_DECL(Min)
SIMPLE_BINARY_KERNEL_DECL(And)
SIMPLE_BINARY_KERNEL_DECL(Or)
SIMPLE_BINARY_KERNEL_DECL(Xor)
SIMPLE_BINARY_KERNEL_DECL(Div)
SIMPLE_BINARY_KERNEL_DECL(Remainder)
SIMPLE_BINARY_KERNEL_DECL(Dot)
SIMPLE_BINARY_KERNEL_DECL(Complex)
SIMPLE_BINARY_KERNEL_DECL(DotGeneral)
SIMPLE_BINARY_KERNEL_DECL(Atan2)

#undef SIMPLE_BINARY_KERNEL_DECL

}  // namespace spu::kernel::hlo
