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

#include "libspu/kernel/hal/ring.h"

#include <cmath>
#include <vector>

#include "libspu/core/bit_utils.h"
#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::kernel::hal {

Type _common_type(SPUContext* ctx, const Type& a, const Type& b) {
  if (a.isa<Secret>() && b.isa<Secret>()) {
    return _common_type_s(ctx, a, b);
  } else if (a.isa<Private>() && b.isa<Private>()) {
    return _common_type_v(ctx, a, b);
  } else if (a.isa<Secret>()) {
    return a;
  } else if (b.isa<Secret>()) {
    return b;
  } else if (a.isa<Private>()) {
    return a;
  } else if (b.isa<Private>()) {
    return b;
  } else {
    SPU_ENFORCE(a.isa<Public>() && b.isa<Public>());
    return a;
  }
}

MemRef _cast_type(SPUContext* ctx, const MemRef& x, const Type& to) {
  if (x.eltype() == to) {
    return x;
  }
  if (x.isPublic() && to.isa<Public>()) {
    return x;
  } else if (x.isPublic() && to.isa<Secret>()) {
    // FIXME: casting to BShare semantic is wrong.
    return _p2s(ctx, x);
  } else if (x.isPublic() && to.isa<Private>()) {
    return _p2v(ctx, x, to.as<Private>()->owner());
  } else if (x.isPrivate() && to.isa<Secret>()) {
    return _v2s(ctx, x);
  } else if (x.isSecret() && to.isa<Secret>()) {
    return _cast_type_s(ctx, x, to);
  } else {
    SPU_THROW("should not be here x={}, to={}", x, to);
  }
}

#define IMPL_UNARY_OP(Name)                                 \
  MemRef Name(SPUContext* ctx, const MemRef& in) {          \
    SPU_TRACE_HAL_LEAF(ctx, in);                            \
    if (in.isPublic()) {                                    \
      return Name##_p(ctx, in);                             \
    } else if (in.isSecret()) {                             \
      return Name##_s(ctx, in);                             \
    } else if (in.isPrivate()) {                            \
      return Name##_v(ctx, in);                             \
    } else {                                                \
      SPU_THROW("unsupport unary op={} for {}", #Name, in); \
    }                                                       \
  }
IMPL_UNARY_OP(_not)
IMPL_UNARY_OP(_negate)
IMPL_UNARY_OP(_msb)
IMPL_UNARY_OP(_square)

#undef IMPL_UNARY_OP

#define IMPL_SHIFT_OP(Name)                                           \
  MemRef Name(SPUContext* ctx, const MemRef& in, const Sizes& bits) { \
    SPU_TRACE_HAL_LEAF(ctx, in, bits);                                \
    if (in.isPublic()) {                                              \
      return Name##_p(ctx, in, bits);                                 \
    } else if (in.isSecret()) {                                       \
      return Name##_s(ctx, in, bits);                                 \
    } else if (in.isPrivate()) {                                      \
      return Name##_v(ctx, in, bits);                                 \
    } else {                                                          \
      SPU_THROW("unsupport unary op={} for {}", #Name, in);           \
    }                                                                 \
  }

IMPL_SHIFT_OP(_lshift)
IMPL_SHIFT_OP(_rshift)
IMPL_SHIFT_OP(_arshift)

#undef IMPL_SHIFT_OP

#define IMPL_COMMUTATIVE_BINARY_OP(Name)                           \
  MemRef Name(SPUContext* ctx, const MemRef& x, const MemRef& y) { \
    SPU_TRACE_HAL_LEAF(ctx, x, y);                                 \
    if (x.isPublic() && y.isPublic()) { /*PP*/                     \
      return Name##_pp(ctx, x, y);                                 \
    } else if (x.isPrivate() && y.isPrivate()) { /*VV*/            \
      return Name##_vv(ctx, x, y);                                 \
    } else if (x.isSecret() && y.isSecret()) { /*SS*/              \
      return Name##_ss(ctx, y, x);                                 \
    } else if (x.isSecret() && y.isPublic()) { /*SP*/              \
      return Name##_sp(ctx, x, y);                                 \
    } else if (x.isPublic() && y.isSecret()) { /*PS*/              \
      /* commutative, swap args */                                 \
      return Name##_sp(ctx, y, x);                                 \
    } else if (x.isPrivate() && y.isPublic()) { /*VP*/             \
      return Name##_vp(ctx, x, y);                                 \
    } else if (x.isPublic() && y.isPrivate()) { /*PV*/             \
      /* commutative, swap args */                                 \
      return Name##_vp(ctx, y, x);                                 \
    } else if (x.isPrivate() && y.isSecret()) { /*VS*/             \
      return Name##_sv(ctx, y, x);                                 \
    } else if (x.isSecret() && y.isPrivate()) { /*SV*/             \
      /* commutative, swap args */                                 \
      return Name##_sv(ctx, x, y);                                 \
    } else {                                                       \
      SPU_THROW("unsupported op {} for x={}, y={}", #Name, x, y);  \
    }                                                              \
  }

IMPL_COMMUTATIVE_BINARY_OP(_add)
IMPL_COMMUTATIVE_BINARY_OP(_mul)
IMPL_COMMUTATIVE_BINARY_OP(_and)
IMPL_COMMUTATIVE_BINARY_OP(_xor)

#undef IMPL_COMMUTATIVE_BINARY_OP

static OptionalAPI<MemRef> _equal_impl(SPUContext* ctx, const MemRef& x,
                                       const MemRef& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  if (x.isPublic() && y.isPublic()) {
    return _equal_pp(ctx, x, y);
  } else if (x.isSecret() && y.isPublic()) {
    return _equal_sp(ctx, x, y);
  } else if (x.isPublic() && y.isSecret()) { /* commutative, swap args */
    return _equal_sp(ctx, y, x);
  } else if (x.isSecret() && y.isSecret()) {
    return _equal_ss(ctx, y, x);
  }

  return NotAvailable;
}

MemRef _conv2d(SPUContext* ctx, const MemRef& input, const MemRef& kernel,
               const Strides& window_strides) {
  SPU_TRACE_HAL_DISP(ctx, input, kernel, window_strides);

  // TODO: assume s*p and p*p should call `dot`
  SPU_ENFORCE(input.isSecret() && kernel.isSecret());
  return _conv2d_ss(ctx, input, kernel, window_strides);
}

static MemRef _mmul_impl(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  if (x.eltype().semantic_type() != y.eltype().semantic_type()) {
    auto ret_st =
        std::max(x.eltype().semantic_type(), y.eltype().semantic_type());
    return _mmul(ctx, _ring_cast(ctx, x, ret_st), _ring_cast(ctx, y, ret_st));
  }

  if (x.isPublic() && y.isPublic()) {  // PP
    return _mmul_pp(ctx, x, y);
  } else if (x.isSecret() && y.isSecret()) {  // SS
    return _mmul_ss(ctx, x, y);
  } else if (x.isPrivate() && y.isPrivate()) {  // VV
    return _mmul_vv(ctx, x, y);
  } else if (x.isSecret() && y.isPublic()) {  // SP
    return _mmul_sp(ctx, x, y);
  } else if (x.isPublic() && y.isSecret()) {  // PS
    return _transpose(ctx,
                      _mmul_sp(ctx, _transpose(ctx, y), _transpose(ctx, x)));
  } else if (x.isPrivate() && y.isPublic()) {  // VP
    return _mmul_vp(ctx, x, y);
  } else if (x.isPublic() && y.isPrivate()) {  // PV
    return _transpose(ctx,
                      _mmul_vp(ctx, _transpose(ctx, y), _transpose(ctx, x)));
  } else if (x.isSecret() && y.isPrivate()) {  // SV
    return _mmul_sv(ctx, x, y);
  } else if (x.isPrivate() && y.isSecret()) {  // VS
    return _transpose(ctx,
                      _mmul_sv(ctx, _transpose(ctx, y), _transpose(ctx, x)));
  } else {
    SPU_THROW("unsupported op {} for x={}, y={}", "_matmul", x, y);
  }
};

MemRef _trunc(SPUContext* ctx, const MemRef& x, size_t bits, SignType sign) {
  SPU_TRACE_HAL_LEAF(ctx, x, bits);
  bits = (bits == 0) ? ctx->getFxpBits() : bits;

  if (x.isPublic()) {
    return _trunc_p(ctx, x, bits, sign);
  } else if (x.isSecret()) {
    return _trunc_s(ctx, x, bits, sign);
  } else if (x.isPrivate()) {
    return _trunc_v(ctx, x, bits, sign);
  } else {
    SPU_THROW("unsupport unary op={} for {}", __func__, x);
  }
}

// swap bits of [start, end)
MemRef _bitrev(SPUContext* ctx, const MemRef& x, size_t start, size_t end) {
  SPU_TRACE_HAL_LEAF(ctx, x, start, end);

  if (x.isPublic()) {
    return _bitrev_p(ctx, x, start, end);
  } else if (x.isSecret()) {
    return _bitrev_s(ctx, x, start, end);
  } else if (x.isPrivate()) {
    return _bitrev_v(ctx, x, start, end);
  }

  SPU_THROW("unsupport op={} for {}", "_bitrev", x);
}

namespace {

std::tuple<int64_t, int64_t, int64_t> deduceMmulArgs(const Shape& lhs,
                                                     const Shape& rhs) {
  SPU_ENFORCE(lhs.ndim() > 0 && lhs.ndim() <= 2, "{}", lhs);
  SPU_ENFORCE(rhs.ndim() > 0 && rhs.ndim() <= 2, "{}", rhs);

  if (lhs.size() == 1 && rhs.size() == 1) {
    SPU_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, 1, rhs[0]);
  }
  if (lhs.size() == 1 && rhs.size() == 2) {
    SPU_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, rhs[1], rhs[0]);
  }
  if (lhs.size() == 2 && rhs.size() == 1) {
    SPU_ENFORCE(lhs[1] == rhs[0]);
    return std::make_tuple(lhs[0], 1, rhs[0]);
  }
  SPU_ENFORCE(lhs[1] == rhs[0], "lhs[1] = {}, rhs[0] = {}", lhs[1], rhs[0]);
  return std::make_tuple(lhs[0], rhs[1], rhs[0]);
}

std::tuple<int64_t, int64_t, int64_t> calcMmulTilingSize(int64_t m, int64_t n,
                                                         int64_t k,
                                                         size_t elsize,
                                                         size_t mem_limit) {
  if (m == 0 || n == 0 || k == 0) {
    return {m, n, k};
  }
  if ((m * k + k * n) * elsize < mem_limit) {
    return {m, n, k};
  }

  const double elnum_limit = mem_limit / elsize;
  int64_t k_step;
  int64_t expected_mn_step;

  if (k > (m + n) * 8) {
    // for "tall and skinny", only split large dimensions.
    expected_mn_step = m + n;
    k_step = std::max<int64_t>(1, std::ceil(elnum_limit / expected_mn_step));
  } else if ((m + n) > k * 8) {
    // for "tall and skinny", only split large dimensions.
    k_step = k;
    expected_mn_step = std::max<int64_t>(1, std::ceil(elnum_limit / k_step));
  } else {
    // Solving equations:
    // k_step * mn_step == elnum_limit
    // k_step / mn_step == k / (m+n)
    double k_mn_radio = static_cast<double>(k) / static_cast<double>(m + n);
    double mn_step = std::sqrt(elnum_limit / k_mn_radio);
    k_step = std::max<int64_t>(1, std::ceil(elnum_limit / mn_step));
    expected_mn_step = std::max<int64_t>(1, std::ceil(mn_step));
  }

  // split expected_mn_step into m/n by radio
  const int64_t m_step = std::max<int64_t>(expected_mn_step * m / (m + n), 1);
  const int64_t n_step = std::max<int64_t>(expected_mn_step * n / (m + n), 1);

  return {m_step, n_step, k_step};
}

}  // namespace

MemRef _sub(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  return _add(ctx, x, _negate(ctx, y));
}

MemRef _mmul(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  auto [m, n, k] = deduceMmulArgs(x.shape(), y.shape());

  // Enforce no vector
  if (x.shape() != Shape{m, k} || y.shape() != Shape{k, n}) {
    return _mmul(ctx, _reshape(ctx, x, {m, k}), _reshape(ctx, y, {k, n}));
  }
  auto [m_step, n_step, k_step] =
      calcMmulTilingSize(m, n, k, x.elsize(), 256UL * 1024 * 1024);

  if (ctx->config().experimental_disable_mmul_split() ||
      (m_step == m && n_step == n && k_step == k)) {
    // no split
    return _mmul_impl(ctx, x, y);
  }

  int64_t m_blocks = (m + m_step - 1) / m_step;
  int64_t n_blocks = (n + n_step - 1) / n_step;
  int64_t k_blocks = (k + k_step - 1) / k_step;

  std::vector<std::vector<MemRef>> ret_blocks(m_blocks,
                                              std::vector<MemRef>(n_blocks));

  for (int64_t r = 0; r < m_blocks; r++) {
    for (int64_t c = 0; c < n_blocks; c++) {
      for (int64_t i = 0; i < k_blocks; i++) {
        auto m_start = r * m_step;
        auto n_start = c * n_step;
        auto k_start = i * k_step;
        auto m_end = std::min(m, m_start + m_step);
        auto n_end = std::min(n, n_start + n_step);
        auto k_end = std::min(k, k_start + k_step);

        MemRef x_block;
        if (x.shape().size() == 1) {
          SPU_ENFORCE(m_start == 0 && m_end == 1);
          x_block = _extract_slice(ctx, x, {k_start}, {k_end}, {});
        } else {
          x_block =
              _extract_slice(ctx, x, {m_start, k_start}, {m_end, k_end}, {});
        }

        MemRef y_block;
        if (y.shape().size() == 1) {
          SPU_ENFORCE(n_start == 0 && n_end == 1);
          y_block = _extract_slice(ctx, y, {k_start}, {k_end}, {});
        } else {
          y_block =
              _extract_slice(ctx, y, {k_start, n_start}, {k_end, n_end}, {});
        }

        auto mmul_ret = _mmul_impl(ctx, x_block, y_block);
        if (i == 0) {
          ret_blocks[r][c] = std::move(mmul_ret);
        } else {
          ret_blocks[r][c] = _add(ctx, ret_blocks[r][c], mmul_ret);
        }
      }
    }
  }

  // merge blocks.
  const auto& eltype = ret_blocks[0][0].eltype();
  MemRef ret(MemRef(eltype, {m, n}));

  for (int64_t r = 0; r < static_cast<int64_t>(ret_blocks.size()); r++) {
    const auto& row_blocks = ret_blocks[r];
    for (int64_t c = 0; c < static_cast<int64_t>(row_blocks.size()); c++) {
      const auto& block = row_blocks[c];
      const int64_t block_rows = block.shape()[0];
      const int64_t block_cols = block.shape()[1];
      if (block.isCompact()) {
        if (n_blocks == 1) {
          SPU_ENFORCE(row_blocks.size() == 1);
          SPU_ENFORCE(block_cols == n);
          char* dst = &ret.at<char>({r * m_step, 0});
          const char* src = &block.at<char>({0, 0});
          size_t cp_len = block.elsize() * block.numel();
          std::memcpy(dst, src, cp_len);
        } else {
          for (int64_t i = 0; i < block_rows; i++) {
            char* dst = &ret.at<char>({r * m_step + i, c * n_step});
            const char* src = &block.at<char>({i, 0});
            size_t cp_len = block.elsize() * block_cols;
            std::memcpy(dst, src, cp_len);
          }
        }
      } else {
        for (int64_t i = 0; i < block_rows; i++) {
          for (int64_t j = 0; j < block_cols; j++) {
            char* dst = &ret.at<char>({r * m_step + i, c * n_step + j});
            const char* src = &block.at<char>({i, j});
            std::memcpy(dst, src, block.elsize());
          }
        }
      }
    }
  }

  return ret;
}

MemRef _or(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  // X or Y = X xor Y xor (X and Y)
  return _xor(ctx, x, _xor(ctx, y, _and(ctx, x, y)));
}

MemRef _equal(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  // First try use equal kernel, i.e. for 2PC , equal can be done with the same
  // cost of half MSB.
  //      x0 + x1 = y0 + y1 mod 2^k
  // <=>  x0 - y0 = y1 - x1 mod 2^k
  // <=>  [1{x = y}]_B <- EQ(x0 - y0, y1 - x1) where EQ is a 2PC protocol.
  auto z = _equal_impl(ctx, x, y);
  if (z.has_value()) {
    return z.value();
  }

  // Note: With optimized msb kernel, A2B+PreOr is slower than 2*MSB
  // eq(x, y) = !lt(x, y) & !lt(y, x)
  //          = xor(a, 1) & xor(b, 1)  // let a = lt(x, y), b = lt(y, x)
  const auto _k1 = _constant(ctx, 1, SE_1, x.shape());
  return _and(ctx, _xor(ctx, _less(ctx, x, y), _k1),
              _xor(ctx, _less(ctx, y, x), _k1));
}

MemRef _sign(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // is_negative = x < 0 ? 1 : 0;
  MemRef is_negative = _msb(ctx, x);

  // promote when necessary
  is_negative = _ring_cast(ctx, is_negative, x.eltype().semantic_type());

  // sign = 1 - 2 * is_negative
  //      = +1 ,if x >= 0
  //      = -1 ,if x < 0
  const auto one =
      _constant(ctx, 1, x.eltype().semantic_type(), is_negative.shape());
  const auto two =
      _constant(ctx, 2, x.eltype().semantic_type(), is_negative.shape());

  //
  return _sub(ctx, one, _mul(ctx, two, is_negative));
}

MemRef _less(SPUContext* ctx, const MemRef& x, const MemRef& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  SPU_ENFORCE(isUnsigned(x.eltype().semantic_type()) ==
                  isUnsigned(y.eltype().semantic_type()),
              "lhs = {}, rhs ={}", x.eltype(), y.eltype());

  // promote x, y to larger signed type
  auto promoted_type = promoteToNextSignedType(x.eltype().semantic_type());

  auto x_promoted = _ring_cast(ctx, x, promoted_type);
  auto y_promoted = _ring_cast(ctx, y, promoted_type);

  // Note: the impl assume inputs are signed with two's complement encoding.
  // test msb(x-y) == 1
  return _msb(ctx, _sub(ctx, x_promoted, y_promoted));
}

MemRef _mux(SPUContext* ctx, const MemRef& pred, const MemRef& a,
            const MemRef& b) {
  SPU_TRACE_HAL_LEAF(ctx, pred, a, b);

  // b + pred*(a-b)
  return _add(ctx, b, _mul(ctx, pred, _sub(ctx, a, b)));
}

MemRef _clamp(SPUContext* ctx, const MemRef& x, const MemRef& minv,
              const MemRef& maxv) {
  SPU_TRACE_HAL_LEAF(ctx, x, minv, maxv);

  // clamp lower bound, res = x < minv ? minv : x
  auto res = _mux(ctx, _less(ctx, x, minv), minv, x);

  // clamp upper bound, res = res < maxv ? res, maxv
  return _mux(ctx, _less(ctx, res, maxv), res, maxv);
}

MemRef _constant(SPUContext* ctx, uint128_t init, SemanticType type,
                 const Shape& shape) {
  return _make_p(ctx, init, type, shape);
}

MemRef _bit_parity(SPUContext* ctx, const MemRef& x, size_t bits) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(absl::has_single_bit(bits), "currently only support power of 2");
  auto ret = _prefer_b(ctx, x);
  while (bits > 1) {
    ret = _xor(ctx, ret, _rshift(ctx, ret, {static_cast<int64_t>(bits / 2)}));
    bits /= 2;
  }

  ret = _and(ctx, ret,
             _constant(ctx, 1, ret.eltype().semantic_type(), x.shape()));

  return _ring_cast(ctx, ret, SE_1);
}

// TODO(jint): OPTIMIZE ME, this impl seems to be super slow.
MemRef _popcount(SPUContext* ctx, const MemRef& x, size_t bits) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  if (x.shape().isEmpty()) {
    return x;
  }

  auto xb = _prefer_b(ctx, x);

  std::vector<MemRef> vs;
  vs.reserve(bits);
  for (size_t idx = 0; idx < bits; idx++) {
    auto x_ = _rshift(ctx, xb, {static_cast<int64_t>(idx)});
    x_ = _and(ctx, x_,
              _constant(ctx, 1U, x_.eltype().semantic_type(), x.shape()));

    if (x_.eltype().isa<BoolShare>()) {
      const_cast<Type&>(x_.eltype()).as<BaseRingType>()->set_valid_bits(1);
    }
    vs.push_back(std::move(x_));
  }

  return vreduce(vs.begin(), vs.end(), [&](const MemRef& a, const MemRef& b) {
    return _add(ctx, a, b);
  });
}

// Fill all bits after msb to 1.
//
// Algorithm, lets consider one bit, in each iteration we fill
// [msb-2^k, msb) to 1.
//   x0:  010000000   ; x0
//   x1:  011000000   ; x0 | (x0>>1)
//   x2:  011110000   ; x1 | (x1>>2)
//   x3:  011111111   ; x2 | (x2>>4)
//
MemRef _prefix_or(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  auto b0 = _prefer_b(ctx, x);
  const size_t bit_width = SizeOf(b0.eltype().storage_type()) * 8;
  for (int idx = 0; idx < absl::bit_width(bit_width) - 1; idx++) {
    const int64_t offset = 1L << idx;
    auto b1 = _rshift(ctx, b0, {offset});
    b0 = _or(ctx, b0, b1);
  }
  return b0;
}

MemRef _bitdeintl(SPUContext* ctx, const MemRef& in) {
  SPU_TRACE_HAL_LEAF(ctx, in);

  // algorithm:
  //      0101010101010101
  // swap  ^^  ^^  ^^  ^^
  //      0011001100110011
  // swap   ^^^^    ^^^^
  //      0000111100001111
  // swap     ^^^^^^^^
  //      0000000011111111
  MemRef out = in;
  for (int64_t idx = 0; idx + 1 < Log2Ceil(in.eltype().size() * 8); idx++) {
    auto keep = _constant(ctx, detail::kBitIntlKeepMasks[idx],
                          in.eltype().semantic_type(), in.shape());
    auto move = _constant(ctx, detail::kBitIntlSwapMasks[idx],
                          in.eltype().semantic_type(), in.shape());
    int64_t shift = 1 << idx;
    // out = (out & keep) ^ ((out >> shift) & move) ^ ((out & move) << shift);
    out = _xor(ctx,
               _xor(ctx, _and(ctx, out, keep),
                    _and(ctx, _rshift(ctx, out, {shift}), move)),
               _lshift(ctx, _and(ctx, out, move), {shift}));
  }
  return out;
}

MemRef _prefer_a(SPUContext* ctx, const MemRef& x) {
  if (x.eltype().isa<BoolShare>()) {
    // B2A
    return _add(ctx, x,
                _constant(ctx, 0, x.eltype().semantic_type(), x.shape()));
  }

  return x;
}

// Assumption
//   let v : ashare
//   y1 = v >> c1
//   y2 = v ^ c2
// When a variable is followed by multiple binary operation, it's more efficient
// to convert it to boolean share first.
MemRef _prefer_b(SPUContext* ctx, const MemRef& x) {
  if (x.eltype().isa<ArithShare>()) {
    const auto k0 = _constant(ctx, 0U, x.eltype().semantic_type(), x.shape());
    return _xor(ctx, x, k0);  // noop, to bshare
  }

  return x;
}

MemRef _encode_fp(SPUContext* ctx, PtBufferView bv, int64_t fxp_bits,
                  SemanticType type) {
  const PtType pt_type = bv.pt_type;
  const size_t numel = bv.shape.numel();
  MemRef dst(makeType<mpc::Pub2kTy>(type), bv.shape);

  DISPATCH_FLOAT_PT_TYPES(pt_type, [&]() {
    using Float = ScalarT;
    DISPATCH_ALL_STORAGE_TYPES(dst.eltype().storage_type(), [&]() {
      using T = std::make_signed_t<ScalarT>;

      // Reference: https://eprint.iacr.org/2019/599.pdf
      // To make `msb based comparison` work, the safe range is
      // [-2^(k-2), 2^(k-2))
      const size_t k = sizeof(T) * 8;
      const T kScale = T(1) << fxp_bits;
      const T kFxpLower = -(T)std::pow(2, k - 2);
      const T kFxpUpper = (T)std::pow(2, k - 2) - 1;
      const auto kFlpUpper =
          static_cast<Float>(static_cast<double>(kFxpUpper) / kScale);
      const auto kFlpLower =
          static_cast<Float>(static_cast<double>(kFxpLower) / kScale);

      auto _dst = MemRefView<T>(dst);

      pforeach(0, numel, [&](int64_t idx) {
        auto src_value = bv.get<Float>(idx);
        if (std::isnan(src_value)) {
          // see numpy.nan_to_num
          // note(jint) I dont know why nan could be
          // encoded as zero..
          _dst[idx] = 0;
        } else if (src_value >= kFlpUpper) {
          _dst[idx] = kFxpUpper;
        } else if (src_value <= kFlpLower) {
          _dst[idx] = kFxpLower;
        } else {
          _dst[idx] = static_cast<T>(src_value * kScale);
        }
      });
    });
  });

  return MemRef(dst);
}

MemRef _copy_fp(SPUContext* ctx, PtBufferView bv) {
  SemanticType type;
  switch (bv.pt_type) {
    case PT_F16: {
      type = SE_I16;
      break;
    }
    case PT_F32: {
      type = SE_I32;
      break;
    }
    case PT_F64: {
      type = SE_I64;
      break;
    }
    default: {
      SPU_THROW("Unhandled fp type = {}", bv.pt_type);
    }
  }

  const PtType pt_type = bv.pt_type;
  const size_t numel = bv.shape.numel();
  MemRef dst(makeType<mpc::Pub2kTy>(type), bv.shape);

  DISPATCH_FLOAT_PT_TYPES(pt_type, [&]() {
    using Float = ScalarT;
    DISPATCH_ALL_STORAGE_TYPES(dst.eltype().storage_type(), [&]() {
      using T = std::make_signed_t<ScalarT>;
      auto _dst = MemRefView<T>(dst);
      pforeach(0, numel, [&](int64_t idx) {
        auto src_value = bv.get<Float>(idx);
        std::memcpy(&_dst[idx], &src_value, sizeof(Float));
      });
    });
  });

  return MemRef(dst);
}

MemRef _encode_int(SPUContext* ctx, PtBufferView bv, SemanticType type) {
  const PtType pt_type = bv.pt_type;
  const size_t numel = bv.shape.numel();
  MemRef dst(makeType<mpc::Pub2kTy>(type), bv.shape);

  // handle integer & boolean
  DISPATCH_INT_PT_TYPES(pt_type, [&]() {
    using Integer = ScalarT;
    DISPATCH_ALL_STORAGE_TYPES(dst.eltype().storage_type(), [&]() {
      using T = std::make_signed_t<ScalarT>;

      auto _dst = MemRefView<T>(dst);
      // TODO: encoding integer in range [-2^(k-2),2^(k-2))
      pforeach(0, numel, [&](int64_t idx) {
        auto src_value = bv.get<Integer>(idx);
        _dst[idx] = static_cast<T>(src_value);  // NOLINT
      });
    });
  });

  return dst;
}

void _decode_int(SPUContext* ctx, const MemRef& encoded, PtBufferView* bv) {
  const Type& encoded_type = encoded.eltype();
  const PtType pt_type = bv->pt_type;
  const size_t numel = encoded.numel();

  SPU_ENFORCE(encoded_type.isa<RingTy>(), "source must be ring_type, got={}",
              encoded_type);

  DISPATCH_ALL_STORAGE_TYPES(encoded_type.storage_type(), [&]() {
    using T = std::make_signed_t<ScalarT>;

    DISPATCH_ALL_PT_TYPES(pt_type, [&]() {
      auto _src = MemRefView<T>(encoded);

      if (pt_type == PT_I1) {
        pforeach(0, numel, [&](int64_t idx) {
          bool value = !((_src[idx] & 0x1) == 0);
          bv->set<bool>(idx, value);
        });
      } else {
        pforeach(0, numel, [&](int64_t idx) {
          auto value = static_cast<ScalarT>(_src[idx]);
          bv->set<ScalarT>(idx, value);
        });
      }
    });
  });
}

void _decode_fp(SPUContext* ctx, const MemRef& encoded, PtBufferView* bv,
                int64_t fxp_bits) {
  const Type& encoded_type = encoded.eltype();
  const PtType pt_type = bv->pt_type;
  const size_t numel = encoded.numel();

  SPU_ENFORCE(encoded_type.isa<RingTy>(), "source must be ring_type, got={}",
              encoded_type);

  DISPATCH_ALL_STORAGE_TYPES(encoded_type.storage_type(), [&]() {
    using T = std::make_signed_t<ScalarT>;
    DISPATCH_ALL_PT_TYPES(pt_type, [&]() {
      auto _src = MemRefView<T>(encoded);

      const T kScale = T(1) << fxp_bits;
      pforeach(0, numel, [&](int64_t idx) {
        auto value =
            static_cast<ScalarT>(static_cast<double>(_src[idx]) / kScale);
        bv->set<ScalarT>(idx, value);
      });
    });
  });
}

MemRef _iota(SPUContext* ctx, PtType pt_type, int64_t numel, int64_t fxp_bits) {
  return DISPATCH_ALL_NONE_BOOL_PT_TYPES(pt_type, [&]() {
    std::vector<ScalarT> arr(numel);
    std::iota(arr.begin(), arr.end(), 0);
    MemRef ret;
    if (pt_type == PT_F16 || pt_type == PT_F32 || pt_type == PT_F64) {
      ret = _encode_fp(ctx, arr, fxp_bits, GetEncodedType(pt_type));
    } else {
      ret = _encode_int(ctx, arr, GetEncodedType(pt_type));
    }
    return _reshape(ctx, ret, {numel});
  });
}

std::optional<MemRef> _oramonehot(SPUContext* ctx, const MemRef& x,
                                  int64_t db_size, bool db_is_public) {
  std::optional<MemRef> ret;
  if (db_is_public) {
    ret = _oramonehot_sp(ctx, x, db_size);
  } else {
    if (x.isPrivate()) {
      ret = _oramonehot_ss(ctx, _v2s(ctx, x), db_size);
    } else {
      ret = _oramonehot_ss(ctx, x, db_size);
    }
  }

  if (!ret.has_value()) {
    return std::nullopt;
  }

  return ret;
}

MemRef _oramread(SPUContext* ctx, const MemRef& x, const MemRef& y,
                 int64_t offset) {
  SPU_ENFORCE(x.isSecret(), "onehot should be secret shared");
  auto reshaped_x = x.reshape({1, x.numel()});
  auto reshaped_y = y;
  if (y.shape().size() == 1) {
    reshaped_y = y.reshape({y.numel(), 1});
  }

  MemRef ret;
  if (y.isSecret()) {
    ret = _oramread_ss(ctx, reshaped_x, reshaped_y, offset);
  } else if (y.isPublic()) {
    ret = _oramread_sp(ctx, reshaped_x, reshaped_y, offset);
  } else if (y.isPrivate()) {
    ret = _oramread_ss(ctx, reshaped_x, _v2s(ctx, reshaped_y), offset);
  } else {
    SPU_THROW("unexpected vtype, got onehot {}, database {}.", x.vtype(),
              y.vtype());
  }

  return ret;
}

MemRef _ring_cast(SPUContext* ctx, const MemRef& x, SemanticType to_type) {
  if (x.eltype().semantic_type() == to_type) {
    return x;
  }

  if (x.isPublic()) {
    return _ring_cast_p(ctx, x, to_type);
  } else if (x.isSecret()) {
    return _ring_cast_s(ctx, x, to_type);
  } else if (x.isPrivate()) {
    return _ring_cast_v(ctx, x, to_type);
  }

  SPU_THROW("unexpected vtype, got {}.", x.vtype());
}

}  // namespace spu::kernel::hal
