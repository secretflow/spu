#pragma once

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {
// AggregateBrentKung without valid bits
// Value AggregateBrentKung(SPUContext* ctx, const Value& x_full,
//                          const Value& g_full);


// AggregateBrentKung with valid bits
std::pair<Value, Value> AggregateBrentKung(SPUContext* ctx, 
    const Value& x_full,
    const Value& valid_full, 
    const Value& g_full);

   Value AggregateBrentKung_NonVectorized(SPUContext* ctx, const Value& x_full,
                                       const Value& g_full);

}  // namespace spu::kernel::hal