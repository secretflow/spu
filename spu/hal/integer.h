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

// !!please read [README.md] for api naming conventions.

// This module provide integral arithmetic and logical operations by `erase`
// security semantics, it dispatches functionality by value's security type to
// the underline mpc module.

Value i_negate(HalContext* ctx, const Value& x);

Value i_abs(HalContext* ctx, const Value& x);

Value i_add(HalContext* ctx, const Value& x, const Value& y);

Value i_sub(HalContext* ctx, const Value& x, const Value& y);

Value i_mul(HalContext* ctx, const Value& x, const Value& y);

Value i_mmul(HalContext* ctx, const Value& x, const Value& y);

Value i_equal(HalContext* ctx, const Value& x, const Value& y);

Value i_less(HalContext* ctx, const Value& x, const Value& y);

}  // namespace spu::hal
