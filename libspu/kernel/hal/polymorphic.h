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

#include "libspu/core/memref.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

/// general element-wise bitwise greater operator
// @param x, the first parameter
// @param y, the second parameter
MemRef greater(SPUContext* ctx, const MemRef& x, const MemRef& y);

/// general element-wise bitwise greater or equal operator
// @param x, the first parameter
// @param y, the second parameter
MemRef greater_equal(SPUContext* ctx, const MemRef& x, const MemRef& y);

/// general element-wise bitwise less or equal operator
// @param x, the first parameter
// @param y, the second parameter
MemRef less_equal(SPUContext* ctx, const MemRef& x, const MemRef& y);

/// see numpy.logical_not(in)
// @param in, requires integer one or zero
MemRef logical_not(SPUContext* ctx, const MemRef& in);

/// general element-wise bitwise equal operator
// @param x, the first parameter
// @param y, the second parameter
MemRef not_equal(SPUContext* ctx, const MemRef& x, const MemRef& y);

/// general element-wise clamp operator
// @param x, the first parameter
// @param min, the second parameter
// @param max, the third parameter
MemRef clamp(SPUContext* ctx, const MemRef& x, const MemRef& min,
             const MemRef& max);

MemRef round_tne(SPUContext* ctx, const MemRef& in);

std::optional<MemRef> oramonehot(SPUContext* ctx, const MemRef& x,
                                 int64_t db_size, bool db_is_secret);

MemRef oramread(SPUContext* ctx, const MemRef& x, const MemRef& y,
                int64_t offset);

}  // namespace spu::kernel::hal
