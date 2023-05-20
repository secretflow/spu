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

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {

// !!please read [README.md] for api naming conventions.

// This module provide integral arithmetic and logical operations by `erase`
// security semantics, it dispatches functionality by value's security type to
// the underline mpc module.

Value i_negate(SPUContext* ctx, const Value& x);

Value i_abs(SPUContext* ctx, const Value& x);

Value i_add(SPUContext* ctx, const Value& x, const Value& y);

Value i_sub(SPUContext* ctx, const Value& x, const Value& y);

Value i_mul(SPUContext* ctx, const Value& x, const Value& y);

Value i_mmul(SPUContext* ctx, const Value& x, const Value& y);

Value i_conv2d(SPUContext* ctx, const Value& x, const Value& y,
               absl::Span<const int64_t> window_strides,
               absl::Span<const int64_t> result_shape);

Value i_equal(SPUContext* ctx, const Value& x, const Value& y);

Value i_less(SPUContext* ctx, const Value& x, const Value& y);

}  // namespace spu::kernel::hal
