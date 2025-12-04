#pragma once

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {
// 声明你的聚合函数
Value AggregateBrentKung(SPUContext* ctx, const Value& x_full,
                         const Value& g_full);
Value AggregateBrentKung_NonVectorized(SPUContext* ctx, const Value& x_full,
                                       const Value& g_full);

}  // namespace spu::kernel::hal