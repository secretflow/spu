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

#include "libspu/core/value.h"

namespace spu {
class SPUContext;
}

// !!please read [README.md] for api naming conventions.
// this module implements ops x ring 2k space WITHOUT dtype check.
//
// for example, when multiply sfxp with sint, it first dispatch to `_mul`, which
// multiply the underline data x a typed unchecked way, then set the result
// dtype to fxp.

namespace spu::kernel::hal {

// NOLINTBEGIN(readability-identifier-naming)

Type _common_type(SPUContext* ctx, const Type& a, const Type& b);

Value _cast_type(SPUContext* ctx, const Value& x, const Type& to);

Value _negate(SPUContext* ctx, const Value& x);

// Return 1 when x >= 0 else -1.
Value _sign(SPUContext* ctx, const Value& x);

Value _add(SPUContext* ctx, const Value& x, const Value& y);

Value _sub(SPUContext* ctx, const Value& x, const Value& y);

Value _mul(SPUContext* ctx, const Value& x, const Value& y);

Value _square(SPUContext* ctx, const Value& x);

// Note: there is no div (multiplicative inverse) in ring 2^k.
// Value _div(SPUContext* ctx, const Value& x, const Value& y);

Value _mmul(SPUContext* ctx, const Value& x, const Value& y);

Value _conv2d(SPUContext* ctx, const Value& x, const Value& y,
              const Strides& strides);

Value _and(SPUContext* ctx, const Value& x, const Value& y);

Value _xor(SPUContext* ctx, const Value& x, const Value& y);

Value _or(SPUContext* ctx, const Value& x, const Value& y);

Value _not(SPUContext* ctx, const Value& in);

Value _msb(SPUContext* ctx, const Value& in);

// Return 1{x == y}
Value _equal(SPUContext* ctx, const Value& x, const Value& y);

Value _less(SPUContext* ctx, const Value& x, const Value& y);

Value _lshift(SPUContext* ctx, const Value& in, const Sizes& bits);

Value _rshift(SPUContext* ctx, const Value& in, const Sizes& bits);

Value _arshift(SPUContext* ctx, const Value& in, const Sizes& bits);

Value _trunc(SPUContext* ctx, const Value& x, size_t bits = 0,
             SignType sign = SignType::Unknown);

Value _bitrev(SPUContext* ctx, const Value&, size_t start_idx, size_t end_idx);

// Expect pred is either {0, 1}.
Value _mux(SPUContext* ctx, const Value& pred, const Value& a, const Value& b);

// TODO: test me
Value _clamp(SPUContext* ctx, const Value& x, const Value& minv,
             const Value& maxv);

Value _clamp_lower(SPUContext* ctx, const Value& x, const Value& minv);

Value _clamp_upper(SPUContext* ctx, const Value& x, const Value& maxv);

// Make a public value from uint128_t init value.
//
// If the current working field has less than 128bit, the lower sizeof(field)
// bits are used.
Value _constant(SPUContext* ctx, uint128_t init, const Shape& shape);

// Return the parity of bits, that is
// - 1 if there are odd number of 1s.
// - 0 if there are even number of 1s.
Value _bit_parity(SPUContext* ctx, const Value& x, size_t bits);

Value _popcount(SPUContext* ctx, const Value& x, size_t bits);

// out[i] = OR(in[0..i])
Value _prefix_or(SPUContext* ctx, const Value& x);

// separate even and odd bits. e.g.
//   xAyBzCwD -> xyzwABCD
Value _bitdeintl(SPUContext* ctx, const Value& in);

// Return value in arithmetic shared form if it's not.
//
// Note: theoretically, we should not leak `share` concept to hal layer.
Value _prefer_a(SPUContext* ctx, const Value& x);

// Return value in binary shared form if it's not.
// Note: theoretically, we should not leak `share` concept to hal layer.
Value _prefer_b(SPUContext* ctx, const Value& x);

// Tensor contraction x and y on index ix and iy.
// See awesome [tutorial](https://www.tensors.net/tutorial-1) for details.
Value _tensordot(SPUContext* ctx, const Value& x, const Value& y,
                 const Index& ix, const Index& iy);

std::optional<Value> _oramonehot(SPUContext* ctx, const Value& x,
                                 int64_t db_size, bool db_is_public);

Value _oramread(SPUContext* ctx, const Value& x, const Value& y,
                int64_t offset);

// NOLINTEND(readability-identifier-naming)

}  // namespace spu::kernel::hal
