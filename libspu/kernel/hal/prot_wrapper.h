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

#include <optional>

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace spu::kernel::hal {

// !!please read [README.md] for api naming conventions.
Type _common_type_s(SPUContext* ctx, const Type& a, const Type& b);
Value _cast_type_s(SPUContext* ctx, const Value& in, const Type& to);

Value _p2s(SPUContext* ctx, const Value& in);
Value _s2p(SPUContext* ctx, const Value& in);

Value _not_p(SPUContext* ctx, const Value& in);
Value _not_s(SPUContext* ctx, const Value& in);

Value _msb_p(SPUContext* ctx, const Value& in);
Value _msb_s(SPUContext* ctx, const Value& in);

Value _equal_pp(SPUContext* ctx, const Value& x, const Value& y);
std::optional<Value> _equal_sp(SPUContext* ctx, const Value& x, const Value& y);
std::optional<Value> _equal_ss(SPUContext* ctx, const Value& x, const Value& y);

Value _lshift_p(SPUContext* ctx, const Value& in, size_t bits);
Value _lshift_s(SPUContext* ctx, const Value& in, size_t bits);

Value _rshift_p(SPUContext* ctx, const Value& in, size_t bits);
Value _rshift_s(SPUContext* ctx, const Value& in, size_t bits);

Value _arshift_p(SPUContext* ctx, const Value& in, size_t bits);
Value _arshift_s(SPUContext* ctx, const Value& in, size_t bits);
Value _trunc_p(SPUContext* ctx, const Value& in, size_t bits);
Value _trunc_s(SPUContext* ctx, const Value& in, size_t bits);
Value _trunc_p_with_sign(SPUContext* ctx, const Value& in, size_t bits,
                         bool is_positive);
Value _trunc_s_with_sign(SPUContext* ctx, const Value& in, size_t bits,
                         bool is_positive);

Value _add_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _add_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _add_ss(SPUContext* ctx, const Value& x, const Value& y);

Value _mul_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _mul_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _mul_ss(SPUContext* ctx, const Value& x, const Value& y);

Value _mmul_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _mmul_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _mmul_ss(SPUContext* ctx, const Value& x, const Value& y);

Value _conv2d_ss(SPUContext* ctx, Value x, const Value& y,
                 absl::Span<const int64_t> window_strides,
                 absl::Span<const int64_t> result_shape);

Value _and_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _and_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _and_ss(SPUContext* ctx, const Value& x, const Value& y);

Value _xor_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _xor_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _xor_ss(SPUContext* ctx, const Value& x, const Value& y);

Value _bitrev_p(SPUContext* ctx, const Value& in, size_t start, size_t end);
Value _bitrev_s(SPUContext* ctx, const Value& in, size_t start, size_t end);

Value _make_p(SPUContext* ctx, uint128_t init, absl::Span<const int64_t> shape);

Value _rand_p(SPUContext* ctx, absl::Span<const int64_t> shape);
Value _rand_s(SPUContext* ctx, absl::Span<const int64_t> shape);

}  // namespace spu::kernel::hal
