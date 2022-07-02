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

Value _p2s(HalContext* ctx, const Value& x);
Value _s2p(HalContext* ctx, const Value& x);

Value _not_p(HalContext* ctx, const Value& x);
Value _not_s(HalContext* ctx, const Value& x);

Value _msb_p(HalContext* ctx, const Value& x);
Value _msb_s(HalContext* ctx, const Value& x);

Value _eqz_p(HalContext* ctx, const Value& x);
Value _eqz_s(HalContext* ctx, const Value& x);

Value _lshift_p(HalContext* ctx, const Value& x, size_t bits);
Value _lshift_s(HalContext* ctx, const Value& x, size_t bits);

Value _rshift_p(HalContext* ctx, const Value& x, size_t bits);
Value _rshift_s(HalContext* ctx, const Value& x, size_t bits);

Value _arshift_p(HalContext* ctx, const Value& x, size_t bits);
Value _arshift_s(HalContext* ctx, const Value& x, size_t bits);
Value _truncpr_s(HalContext* ctx, const Value& x, size_t bits);

Value _add_pp(HalContext* ctx, const Value& x, const Value& y);
Value _add_sp(HalContext* ctx, const Value& x, const Value& y);
Value _add_ss(HalContext* ctx, const Value& x, const Value& y);

Value _mul_pp(HalContext* ctx, const Value& x, const Value& y);
Value _mul_sp(HalContext* ctx, const Value& x, const Value& y);
Value _mul_ss(HalContext* ctx, const Value& x, const Value& y);

Value _mmul_pp(HalContext* ctx, const Value& x, const Value& y);
Value _mmul_sp(HalContext* ctx, const Value& x, const Value& y);
Value _mmul_ss(HalContext* ctx, const Value& x, const Value& y);

Value _and_pp(HalContext* ctx, const Value& x, const Value& y);
Value _and_sp(HalContext* ctx, const Value& x, const Value& y);
Value _and_ss(HalContext* ctx, const Value& x, const Value& y);

Value _xor_pp(HalContext* ctx, const Value& x, const Value& y);
Value _xor_sp(HalContext* ctx, const Value& x, const Value& y);
Value _xor_ss(HalContext* ctx, const Value& x, const Value& y);

Value _bitrev_p(HalContext* ctx, const Value& in, size_t start, size_t end);
Value _bitrev_s(HalContext* ctx, const Value& in, size_t start, size_t end);

}  // namespace spu::hal
