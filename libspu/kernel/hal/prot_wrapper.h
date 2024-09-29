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

#include "libspu/core/memref.h"

namespace spu {
class SPUContext;
}

namespace spu::kernel::hal {

// NOLINTBEGIN(readability-identifier-naming)

// !!please read [README.md] for api naming conventions.
Type _common_type_s(SPUContext* ctx, const Type& a, const Type& b);
Type _common_type_v(SPUContext* ctx, const Type& a, const Type& b);
MemRef _cast_type_s(SPUContext* ctx, const MemRef& in, const Type& to);

MemRef _p2s(SPUContext* ctx, const MemRef& in);
MemRef _s2p(SPUContext* ctx, const MemRef& in);

MemRef _p2v(SPUContext* ctx, const MemRef& in, int owner);
MemRef _v2p(SPUContext* ctx, const MemRef& in);

MemRef _ring_cast_p(SPUContext* ctx, const MemRef& in, PtType to_type);
MemRef _ring_cast_s(SPUContext* ctx, const MemRef& in, PtType to_type);

MemRef _s2v(SPUContext* ctx, const MemRef& in, int owner);
MemRef _v2s(SPUContext* ctx, const MemRef& in);

MemRef _not_p(SPUContext* ctx, const MemRef& in);
MemRef _not_s(SPUContext* ctx, const MemRef& in);
MemRef _not_v(SPUContext* ctx, const MemRef& in);

MemRef _negate_p(SPUContext* ctx, const MemRef& in);
MemRef _negate_s(SPUContext* ctx, const MemRef& in);
MemRef _negate_v(SPUContext* ctx, const MemRef& in);

MemRef _msb_p(SPUContext* ctx, const MemRef& in);
MemRef _msb_s(SPUContext* ctx, const MemRef& in);
MemRef _msb_v(SPUContext* ctx, const MemRef& in);

MemRef _equal_pp(SPUContext* ctx, const MemRef& x, const MemRef& y);
std::optional<MemRef> _equal_sp(SPUContext* ctx, const MemRef& x,
                                const MemRef& y);
std::optional<MemRef> _equal_ss(SPUContext* ctx, const MemRef& x,
                                const MemRef& y);

MemRef _lshift_p(SPUContext* ctx, const MemRef& in, const Sizes& bits);
MemRef _lshift_s(SPUContext* ctx, const MemRef& in, const Sizes& bits);
MemRef _lshift_v(SPUContext* ctx, const MemRef& in, const Sizes& bits);

MemRef _rshift_p(SPUContext* ctx, const MemRef& in, const Sizes& bits);
MemRef _rshift_s(SPUContext* ctx, const MemRef& in, const Sizes& bits);
MemRef _rshift_v(SPUContext* ctx, const MemRef& in, const Sizes& bits);

MemRef _arshift_p(SPUContext* ctx, const MemRef& in, const Sizes& bits);
MemRef _arshift_s(SPUContext* ctx, const MemRef& in, const Sizes& bits);
MemRef _arshift_v(SPUContext* ctx, const MemRef& in, const Sizes& bits);

MemRef _trunc_p(SPUContext* ctx, const MemRef& in, size_t bits, SignType sign);
MemRef _trunc_s(SPUContext* ctx, const MemRef& in, size_t bits, SignType sign);
MemRef _trunc_v(SPUContext* ctx, const MemRef& in, size_t bits, SignType sign);

MemRef _add_pp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _add_sp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _add_ss(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _add_vv(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _add_vp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _add_sv(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _mul_pp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mul_sp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mul_ss(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mul_vv(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mul_vp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mul_sv(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _square_p(SPUContext* ctx, const MemRef& x);
MemRef _square_s(SPUContext* ctx, const MemRef& x);
MemRef _square_v(SPUContext* ctx, const MemRef& x);

MemRef _mmul_pp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mmul_sp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mmul_ss(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mmul_vv(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mmul_vp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _mmul_sv(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _conv2d_ss(SPUContext* ctx, const MemRef& input, const MemRef& kernel,
                  const Strides& strides);

MemRef _and_pp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _and_sp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _and_ss(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _and_vv(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _and_vp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _and_sv(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _xor_pp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _xor_sp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _xor_ss(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _xor_vv(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _xor_vp(SPUContext* ctx, const MemRef& x, const MemRef& y);
MemRef _xor_sv(SPUContext* ctx, const MemRef& x, const MemRef& y);

MemRef _bitrev_p(SPUContext* ctx, const MemRef& in, size_t start, size_t end);
MemRef _bitrev_s(SPUContext* ctx, const MemRef& in, size_t start, size_t end);
MemRef _bitrev_v(SPUContext* ctx, const MemRef& in, size_t start, size_t end);

MemRef _make_p(SPUContext* ctx, uint128_t init, SemanticType type,
               const Shape& shape);

// MemRef _rand_p(SPUContext* ctx, const Shape& shape);
MemRef _rand_s(SPUContext* ctx, SemanticType type, const Shape& shape);

MemRef _ring_cast_p(SPUContext* ctx, const MemRef& in, SemanticType to_type);
MemRef _ring_cast_s(SPUContext* ctx, const MemRef& in, SemanticType to_type);
MemRef _ring_cast_v(SPUContext* ctx, const MemRef& in, SemanticType to_type);

// FIXME: temporary API, formalize later
MemRef _rand_perm_s(SPUContext* ctx, const Shape& shape);
MemRef _perm_ss(SPUContext* ctx, const MemRef& x, const MemRef& perm);
MemRef _perm_sp(SPUContext* ctx, const MemRef& x, const MemRef& perm);
MemRef _perm_pp(SPUContext* ctx, const MemRef& x, const MemRef& perm);
MemRef _perm_vv(SPUContext* ctx, const MemRef& x, const MemRef& perm);
MemRef _inv_perm_ss(SPUContext* ctx, const MemRef& x, const MemRef& perm);
MemRef _inv_perm_sp(SPUContext* ctx, const MemRef& x, const MemRef& perm);
MemRef _inv_perm_sv(SPUContext* ctx, const MemRef& x, const MemRef& perm);
MemRef _inv_perm_pp(SPUContext* ctx, const MemRef& x, const MemRef& perm);
MemRef _inv_perm_vv(SPUContext* ctx, const MemRef& x, const MemRef& perm);

MemRef _gen_inv_perm_p(SPUContext* ctx, const MemRef& x, bool is_ascending);
MemRef _gen_inv_perm_v(SPUContext* ctx, const MemRef& x, bool is_ascending);
MemRef _merge_keys_p(SPUContext* ctx, absl::Span<MemRef const> inputs,
                     bool is_ascending);
MemRef _merge_keys_v(SPUContext* ctx, absl::Span<MemRef const> inputs,
                     bool is_ascending);

// Shape ops
MemRef _broadcast(SPUContext* ctx, const MemRef& in, const Shape& to_shape,
                  const Axes& in_dims);
MemRef _reshape(SPUContext* ctx, const MemRef& in, const Shape& to_shape);
MemRef _extract_slice(SPUContext* ctx, const MemRef& in, const Index& offsets,
                      const Shape& sizes, const Strides& strides);
MemRef _insert_slice(SPUContext* ctx, const MemRef& in, const MemRef& update,
                     const Index& offsets, const Strides& strides,
                     bool prefer_in_place);
MemRef _transpose(SPUContext* ctx, const MemRef& in,
                  const Axes& permutation = {});
MemRef _reverse(SPUContext* ctx, const MemRef& in, const Axes& dimensions);
MemRef _fill(SPUContext* ctx, const MemRef& in, const Shape& to_shape);
MemRef _pad(SPUContext* ctx, const MemRef& in, const MemRef& padding_MemRef,
            const Sizes& edge_padding_low, const Sizes& edge_padding_high);
MemRef _concatenate(SPUContext* ctx, const std::vector<MemRef>& MemRefs,
                    int64_t axis);

// secret database, secret start_indice
std::optional<MemRef> _oramonehot_ss(SPUContext* ctx, const MemRef& x,
                                     int64_t db_size);
MemRef _oramread_ss(SPUContext* ctx, const MemRef& x, const MemRef& y,
                    int64_t offset);
// public database, secret start_indice
std::optional<MemRef> _oramonehot_sp(SPUContext* ctx, const MemRef& x,
                                     int64_t db_size);
MemRef _oramread_sp(SPUContext* ctx, const MemRef& x, const MemRef& y,
                    int64_t offset);

// NOLINTEND(readability-identifier-naming)

}  // namespace spu::kernel::hal
