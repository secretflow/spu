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

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

// NOLINTBEGIN(readability-identifier-naming)

// !!please read [README.md] for api naming conventions.
Type _common_type_s(SPUContext* ctx, const Type& a, const Type& b);
Type _common_type_v(SPUContext* ctx, const Type& a, const Type& b);
Value _cast_type_s(SPUContext* ctx, const Value& in, const Type& to);

Value _p2s(SPUContext* ctx, const Value& in);
Value _s2p(SPUContext* ctx, const Value& in);

Value _p2v(SPUContext* ctx, const Value& in, int owner);
Value _v2p(SPUContext* ctx, const Value& in);

Value _s2v(SPUContext* ctx, const Value& in, int owner);
Value _v2s(SPUContext* ctx, const Value& in);

Value _not_p(SPUContext* ctx, const Value& in);
Value _not_s(SPUContext* ctx, const Value& in);
Value _not_v(SPUContext* ctx, const Value& in);

Value _msb_p(SPUContext* ctx, const Value& in);
Value _msb_s(SPUContext* ctx, const Value& in);
Value _msb_v(SPUContext* ctx, const Value& in);

Value _equal_pp(SPUContext* ctx, const Value& x, const Value& y);
std::optional<Value> _equal_sp(SPUContext* ctx, const Value& x, const Value& y);
std::optional<Value> _equal_ss(SPUContext* ctx, const Value& x, const Value& y);

Value _lshift_p(SPUContext* ctx, const Value& in, size_t bits);
Value _lshift_s(SPUContext* ctx, const Value& in, size_t bits);
Value _lshift_v(SPUContext* ctx, const Value& in, size_t bits);

Value _rshift_p(SPUContext* ctx, const Value& in, size_t bits);
Value _rshift_s(SPUContext* ctx, const Value& in, size_t bits);
Value _rshift_v(SPUContext* ctx, const Value& in, size_t bits);

Value _arshift_p(SPUContext* ctx, const Value& in, size_t bits);
Value _arshift_s(SPUContext* ctx, const Value& in, size_t bits);
Value _arshift_v(SPUContext* ctx, const Value& in, size_t bits);

Value _trunc_p(SPUContext* ctx, const Value& in, size_t bits, SignType sign);
Value _trunc_s(SPUContext* ctx, const Value& in, size_t bits, SignType sign);
Value _trunc_v(SPUContext* ctx, const Value& in, size_t bits, SignType sign);

Value _add_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _add_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _add_ss(SPUContext* ctx, const Value& x, const Value& y);
Value _add_vv(SPUContext* ctx, const Value& x, const Value& y);
Value _add_vp(SPUContext* ctx, const Value& x, const Value& y);
Value _add_sv(SPUContext* ctx, const Value& x, const Value& y);

Value _mul_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _mul_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _mul_ss(SPUContext* ctx, const Value& x, const Value& y);
Value _mul_vv(SPUContext* ctx, const Value& x, const Value& y);
Value _mul_vp(SPUContext* ctx, const Value& x, const Value& y);
Value _mul_sv(SPUContext* ctx, const Value& x, const Value& y);

Value _rm_mmul_ss(SPUContext* ctx, const Value& x, const Value& y);

Value _mmul_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _mmul_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _mmul_ss(SPUContext* ctx, const Value& x, const Value& y);
Value _mmul_vv(SPUContext* ctx, const Value& x, const Value& y);
Value _mmul_vp(SPUContext* ctx, const Value& x, const Value& y);
Value _mmul_sv(SPUContext* ctx, const Value& x, const Value& y);

Value _conv2d_ss(SPUContext* ctx, const Value& input, const Value& kernel,
                 const Strides& strides);

Value _and_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _and_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _and_ss(SPUContext* ctx, const Value& x, const Value& y);
Value _and_vv(SPUContext* ctx, const Value& x, const Value& y);
Value _and_vp(SPUContext* ctx, const Value& x, const Value& y);
Value _and_sv(SPUContext* ctx, const Value& x, const Value& y);

Value _xor_pp(SPUContext* ctx, const Value& x, const Value& y);
Value _xor_sp(SPUContext* ctx, const Value& x, const Value& y);
Value _xor_ss(SPUContext* ctx, const Value& x, const Value& y);
Value _xor_vv(SPUContext* ctx, const Value& x, const Value& y);
Value _xor_vp(SPUContext* ctx, const Value& x, const Value& y);
Value _xor_sv(SPUContext* ctx, const Value& x, const Value& y);

Value _bitrev_p(SPUContext* ctx, const Value& in, size_t start, size_t end);
Value _bitrev_s(SPUContext* ctx, const Value& in, size_t start, size_t end);
Value _bitrev_v(SPUContext* ctx, const Value& in, size_t start, size_t end);

Value _make_p(SPUContext* ctx, uint128_t init, const Shape& shape);

Value _rand_p(SPUContext* ctx, const Shape& shape);
Value _rand_s(SPUContext* ctx, const Shape& shape);

// FIXME: temporary API
Value _rand_perm_s(SPUContext* ctx, const Shape& shape);
Value _perm_ss(SPUContext* ctx, const Value& x, const Value& perm);
Value _perm_sp(SPUContext* ctx, const Value& x, const Value& perm);
Value _inv_perm_ss(SPUContext* ctx, const Value& x, const Value& perm);
Value _inv_perm_sp(SPUContext* ctx, const Value& x, const Value& perm);

// NOLINTEND(readability-identifier-naming)

}  // namespace spu::kernel::hal
