#pragma once

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hlo {
// AggregateBrentKung without valid bits
Value AggregateBrentKung_without_valids(SPUContext* ctx, const Value& x_full,
                                        const Value& g_full);

// AggregateBrentKung with valid bits
std::pair<Value, Value> AggregateBrentKung(SPUContext* ctx, const Value& x,
                                           const Value& valids,
                                           const Value& g_in);

Value AggregateBrentKung_NonVectorized(SPUContext* ctx, const Value& x_full,
                                       const Value& g_full);

std::pair<std::vector<spu::Value>, int64_t> extract_ordered(
    SPUContext* ctx, const spu::Value& arr, const spu::Value& condition);
}  // namespace spu::kernel::hlo