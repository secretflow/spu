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
#include "libspu/core/pt_buffer_view.h"

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

MemRef _cast_type(SPUContext* ctx, const MemRef& x, const Type& to);

MemRef _negate(SPUContext* ctx, const MemRef& x);

// Return 1 when x >= 0 else -1.
MemRef _sign(SPUContext* ctx, const MemRef& x);

MemRef _add(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _sub(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _mul(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _square(SPUContext* ctx, const MemRef& x);

// Note: there is no div (multiplicative inverse) in ring 2^k.
// MemRef _div(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _mmul(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _conv2d(SPUContext* ctx, const MemRef& x, const MemRef& y,
               const Strides& strides);

MemRef _and(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _xor(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _or(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _not(SPUContext* ctx, const MemRef& in);

MemRef _msb(SPUContext* ctx, const MemRef& in);

// Return 1{x == y}
MemRef _equal(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _less(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _lshift(SPUContext* ctx, const MemRef& in, const Sizes& bits);

MemRef _rshift(SPUContext* ctx, const MemRef& in, const Sizes& bits);

MemRef _arshift(SPUContext* ctx, const MemRef& in, const Sizes& bits);

MemRef _trunc(SPUContext* ctx, const MemRef& x, size_t bits = 0,
              SignType sign = SignType::Unknown);

MemRef _bitrev(SPUContext* ctx, const MemRef&, size_t start_idx,
               size_t end_idx);

// Expect pred is either {0, 1}.
MemRef _mux(SPUContext* ctx, const MemRef& pred, const MemRef& a,
            const MemRef& b);

// TODO: test me
MemRef _clamp(SPUContext* ctx, const MemRef& x, const MemRef& minv,
              const MemRef& maxv);

MemRef _ring_cast(SPUContext* ctx, const MemRef& x, SemanticType to_type);

// Make a public value from uint128_t init value.
//
// If the current working field has less than 128bit, the lower sizeof(field)
// bits are used.
MemRef _constant(SPUContext* ctx, uint128_t init, SemanticType type,
                 const Shape& shape);

// Return the parity of bits, that is
// - 1 if there are odd number of 1s.
// - 0 if there are even number of 1s.
MemRef _bit_parity(SPUContext* ctx, const MemRef& x, size_t bits);

MemRef _popcount(SPUContext* ctx, const MemRef& x, size_t bits);

// out[i] = OR(in[0..i])
MemRef _prefix_or(SPUContext* ctx, const MemRef& x);

// separate even and odd bits. e.g.
//   xAyBzCwD -> xyzwABCD
MemRef _bitdeintl(SPUContext* ctx, const MemRef& in);

// Return value in arithmetic shared form if it's not.
//
// Note: theoretically, we should not leak `share` concept to hal layer.
MemRef _prefer_a(SPUContext* ctx, const MemRef& x);

// Return value in binary shared form if it's not.
// Note: theoretically, we should not leak `share` concept to hal layer.
MemRef _prefer_b(SPUContext* ctx, const MemRef& x);

// Encode integer to ring
MemRef _encode_int(SPUContext* ctx, PtBufferView bv, SemanticType type);

// Encode float to ring
MemRef _encode_fp(SPUContext* ctx, PtBufferView bv, int64_t fxp_bits,
                  SemanticType type);

MemRef _copy_fp(SPUContext* ctx, PtBufferView bv);

// Decode integer from ring
void _decode_int(SPUContext* ctx, const MemRef& encoded, PtBufferView* bv);

// Decode float to ring
void _decode_fp(SPUContext* ctx, const MemRef& encoded, PtBufferView* bv,
                int64_t fxp_bits);

// TODO(jimmy): move to somewhere else
// get an iota tensor on ring
MemRef _iota(SPUContext* ctx, PtType pt_type, int64_t numel,
             int64_t fxp_bits = 0);

std::optional<MemRef> _oramonehot(SPUContext* ctx, const MemRef& x,
                                  int64_t db_size, bool db_is_public);

MemRef _oramread(SPUContext* ctx, const MemRef& x, const MemRef& y,
                 int64_t offset);

// NOLINTEND(readability-identifier-naming)

}  // namespace spu::kernel::hal
