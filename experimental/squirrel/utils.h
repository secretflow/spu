// Copyright 2024 Ant Group Co., Ltd.
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

namespace squirrel {

spu::Value ReduceSum(spu::SPUContext* ctx, const spu::Value& x, int axis,
                     bool keepdims = false);

// ArgMax
// We skip NaN check and simply return the 1st operand when two operands are
// identical.
spu::Value ArgMax(spu::SPUContext* ctx, const spu::Value& x, int axis,
                  bool keepdims = false);

spu::Value ArgMaxWithValue(spu::SPUContext* ctx, const spu::Value& x, int axis,
                           spu::Value& max);

// [x]_A * b where b \in {0, 1} is held by one party
spu::Value MulArithShareWithPrivateBoolean(spu::SPUContext* ctx,
                                           const spu::Value& ashr);

// [x]_A * b where b \in {0, 1} is held by one party
spu::Value MulArithShareWithPrivateBoolean(
    spu::SPUContext* ctx, const spu::Value& ashr,
    absl::Span<const uint8_t> prv_boolean);

// Compute (x0 + x1) * (b0 & b1), i.e., the Boolean is AND-style.
// Basically two COTs, (x0*b0)*b1 and (x1*b1)*b0
spu::Value MulArithShareWithANDBoolShare(spu::SPUContext* ctx,
                                         const spu::Value& ashr,
                                         absl::Span<const uint8_t> bshr);

// |bshr| = batch_size * |ashr|
// require 1D input
spu::Value BatchMulArithShareWithANDBoolShare(spu::SPUContext* ctx,
                                              const spu::Value& ashr,
                                              size_t batch_size,
                                              absl::Span<const uint8_t> bshr);

}  // namespace squirrel
