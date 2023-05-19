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

#include <array>
#include <cmath>

#include "libspu/core/bit_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/shape_util.h"
#include "libspu/kernel/hal/prot_wrapper.h"

namespace spu::kernel::hal {
namespace {

std::tuple<int64_t, int64_t, int64_t> deduceMmulArgs(
    const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
  SPU_ENFORCE(!lhs.empty() && lhs.size() <= 2);
  SPU_ENFORCE(!rhs.empty() && rhs.size() <= 2);

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
  SPU_ENFORCE(lhs[1] == rhs[0]);
  return std::make_tuple(lhs[0], rhs[1], rhs[0]);
}

std::tuple<int64_t, int64_t, int64_t> calcMmulTilingSize(int64_t m, int64_t n,
                                                         int64_t k,
                                                         size_t elsize,
                                                         size_t mem_limit) {
  if (m == 0 || n == 0 || k == 0) {
    return {m, n, k};
  }
  const auto elnum_limit = static_cast<int64_t>(mem_limit / elsize);
  const int64_t expected_step = std::ceil(std::sqrt(elnum_limit));

  const int64_t expected_mn_step = std::min((m + n), expected_step);
  const int64_t k_step = std::max(std::min(k, elnum_limit / expected_mn_step),
                                  static_cast<int64_t>(1));

  // split expected_mn_step into m/n by radio
  const int64_t m_step =
      std::max(expected_mn_step * m / (m + n), static_cast<int64_t>(1));
  const int64_t n_step =
      std::max(expected_mn_step * n / (m + n), static_cast<int64_t>(1));

  return {m_step, n_step, k_step};
}

// FIXME: the bellow two functions are copied from shape_ops because of the
// following dependency problem:
//
// shape_ops -> type_cast : concat/pad requires _common_type
// ring -> shape_ops      : tiled_mmul(transpose/slice)
// type_cast -> ring      : _p2s/arshift/shift
Value transpose(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  return Value(in.data().transpose({}), in.dtype());
}

Value slice(SPUContext* ctx, const Value& in,
            absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> end_indices,
            absl::Span<const int64_t> strides) {
  SPU_TRACE_HAL_DISP(ctx, in, start_indices, end_indices, strides);

  return Value(in.data().slice(start_indices, end_indices, strides),
               in.dtype());
}

}  // namespace

Type _common_type(SPUContext* ctx, const Type& a, const Type& b) {
  if (a.isa<Secret>() && b.isa<Secret>()) {
    return _common_type_s(ctx, a, b);
  } else if (a.isa<Secret>()) {
    return a;
  } else if (b.isa<Secret>()) {
    return b;
  } else {
    SPU_ENFORCE(a.isa<Public>() && b.isa<Public>());
    return a;
  }
}

Value _cast_type(SPUContext* ctx, const Value& x, const Type& to) {
  if (x.isPublic() && to.isa<Public>()) {
    return x;
  } else if (x.isPublic() && to.isa<Secret>()) {
    // FIXME: casting to BShare semantic is wrong.
    return _p2s(ctx, x);
  } else if (x.isSecret() && to.isa<Secret>()) {
    return _cast_type_s(ctx, x, to);
  } else {
    SPU_THROW("show not be here x={}, to={}", x, to);
  }
}

#define IMPL_UNARY_OP(Name, FnP, FnS)                       \
  Value Name(SPUContext* ctx, const Value& in) {            \
    SPU_TRACE_HAL_LEAF(ctx, in);                            \
    if (in.isPublic()) {                                    \
      return FnP(ctx, in);                                  \
    } else if (in.isSecret()) {                             \
      return FnS(ctx, in);                                  \
    } else {                                                \
      SPU_THROW("unsupport unary op={} for {}", #Name, in); \
    }                                                       \
  }

#define IMPL_SHIFT_OP(Name, FnP, FnS)                         \
  Value Name(SPUContext* ctx, const Value& in, size_t bits) { \
    SPU_TRACE_HAL_LEAF(ctx, in, bits);                        \
    if (in.isPublic()) {                                      \
      return FnP(ctx, in, bits);                              \
    } else if (in.isSecret()) {                               \
      return FnS(ctx, in, bits);                              \
    } else {                                                  \
      SPU_THROW("unsupport unary op={} for {}", #Name, in);   \
    }                                                         \
  }

#define IMPL_COMMUTATIVE_BINARY_OP(Name, FnPP, FnSP, FnSS)        \
  Value Name(SPUContext* ctx, const Value& x, const Value& y) {   \
    SPU_TRACE_HAL_LEAF(ctx, x, y);                                \
    if (x.isPublic() && y.isPublic()) {                           \
      return FnPP(ctx, x, y);                                     \
    } else if (x.isSecret() && y.isPublic()) {                    \
      return FnSP(ctx, x, y);                                     \
    } else if (x.isPublic() && y.isSecret()) {                    \
      /* commutative, swap args */                                \
      return FnSP(ctx, y, x);                                     \
    } else if (x.isSecret() && y.isSecret()) {                    \
      return FnSS(ctx, y, x);                                     \
    } else {                                                      \
      SPU_THROW("unsupported op {} for x={}, y={}", #Name, x, y); \
    }                                                             \
  }

IMPL_UNARY_OP(_not, _not_p, _not_s)
IMPL_UNARY_OP(_msb, _msb_p, _msb_s)

IMPL_SHIFT_OP(_lshift, _lshift_p, _lshift_s)
IMPL_SHIFT_OP(_rshift, _rshift_p, _rshift_s)
IMPL_SHIFT_OP(_arshift, _arshift_p, _arshift_s)

IMPL_COMMUTATIVE_BINARY_OP(_add, _add_pp, _add_sp, _add_ss)
IMPL_COMMUTATIVE_BINARY_OP(_mul, _mul_pp, _mul_sp, _mul_ss)
IMPL_COMMUTATIVE_BINARY_OP(_and, _and_pp, _and_sp, _and_ss)
IMPL_COMMUTATIVE_BINARY_OP(_xor, _xor_pp, _xor_sp, _xor_ss)

Value _sub(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  return _add(ctx, x, _negate(ctx, y));
}

// TODO: remove this kernel, the algorithm could be used for boolean equal test.
[[maybe_unused]] Value _eqz(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // eqz(x) = not(lsb(pre_or(x)))
  // all equal to zero means lsb equals to zero
  auto _k1 = _constant(ctx, 1U, x.shape());
  auto res = _xor(ctx, _and(ctx, _prefix_or(ctx, x), _k1), _k1);

  // FIXME(jint): see hintNumberOfBits
  if (res.storage_type().isa<BShare>()) {
    const_cast<Type&>(res.storage_type()).as<BShare>()->setNbits(1);
  }

  return res;
}

Value _conv2d(SPUContext* ctx, const Value& x, const Value& y,
              absl::Span<const int64_t> window_strides,
              absl::Span<const int64_t> result_shape) {
  // s*p and p*p should call `dot`
  SPU_ENFORCE(x.isSecret() && y.isSecret());
  return _conv2d_ss(ctx, x, y, window_strides, result_shape);
}

static Value _mmul_impl(SPUContext* ctx, const Value& x, const Value& y) {
  if (x.isPublic() && y.isPublic()) {
    return _mmul_pp(ctx, x, y);
  } else if (x.isSecret() && y.isPublic()) {
    return _mmul_sp(ctx, x, y);
  } else if (x.isPublic() && y.isSecret()) {
    return transpose(ctx, _mmul_sp(ctx, transpose(ctx, y), transpose(ctx, x)));
  } else if (x.isSecret() && y.isSecret()) {
    return _mmul_ss(ctx, x, y);
  } else {
    SPU_THROW("unsupported op {} for x={}, y={}", "_matmul", x, y);
  }
};

Value _mmul(SPUContext* ctx, const Value& x, const Value& y) {
  auto [m, n, k] = deduceMmulArgs(x.shape(), y.shape());
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

  std::vector<std::vector<Value>> ret_blocks(m_blocks,
                                             std::vector<Value>(n_blocks));

  for (int64_t r = 0; r < m_blocks; r++) {
    for (int64_t c = 0; c < n_blocks; c++) {
      for (int64_t i = 0; i < k_blocks; i++) {
        auto m_start = r * m_step;
        auto n_start = c * n_step;
        auto k_start = i * k_step;
        auto m_end = std::min(m, m_start + m_step);
        auto n_end = std::min(n, n_start + n_step);
        auto k_end = std::min(k, k_start + k_step);

        Value x_block;
        if (x.shape().size() == 1) {
          SPU_ENFORCE(m_start == 0 && m_end == 1);
          x_block = slice(ctx, x, {k_start}, {k_end}, {});
        } else {
          x_block = slice(ctx, x, {m_start, k_start}, {m_end, k_end}, {});
        }

        Value y_block;
        if (y.shape().size() == 1) {
          SPU_ENFORCE(n_start == 0 && n_end == 1);
          y_block = slice(ctx, y, {k_start}, {k_end}, {});
        } else {
          y_block = slice(ctx, y, {k_start, n_start}, {k_end, n_end}, {});
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
  const auto& eltype = ret_blocks[0][0].data().eltype();
  const auto& dtype = ret_blocks[0][0].dtype();
  Value ret(NdArrayRef(eltype, {m, n}), dtype);

  for (int64_t r = 0; r < static_cast<int64_t>(ret_blocks.size()); r++) {
    const auto& row_blocks = ret_blocks[r];
    for (int64_t c = 0; c < static_cast<int64_t>(row_blocks.size()); c++) {
      const auto& block = row_blocks[c];
      SPU_ENFORCE(block.data().isCompact());
      const int64_t block_rows = block.shape()[0];
      const int64_t block_cols = block.shape()[1];
      if (n_blocks == 1) {
        SPU_ENFORCE(row_blocks.size() == 1);
        SPU_ENFORCE(block_cols == n);
        char* dst = &ret.data().at<char>({r * m_step, 0});
        const char* src = &block.data().at<char>({0, 0});
        size_t cp_len = block.elsize() * block.numel();
        std::memcpy(dst, src, cp_len);
      } else {
        for (int64_t i = 0; i < block_rows; i++) {
          char* dst = &ret.data().at<char>({r * m_step + i, c * n_step});
          const char* src = &block.data().at<char>({i, 0});
          size_t cp_len = block.elsize() * block_cols;
          std::memcpy(dst, src, cp_len);
        }
      }
    }
  }

  return ret;
}

Value _or(SPUContext* ctx, const Value& x, const Value& y) {
  // X or Y = X xor Y xor (X and Y)
  return _xor(ctx, x, _xor(ctx, y, _and(ctx, x, y)));
}

static std::optional<Value> _equal_impl(SPUContext* ctx, const Value& x,
                                        const Value& y) {
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

  return std::nullopt;
}

Value _equal(SPUContext* ctx, const Value& x, const Value& y) {
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
  const auto _k1 = _constant(ctx, 1, x.shape());
  return _and(ctx, _xor(ctx, _less(ctx, x, y), _k1),
              _xor(ctx, _less(ctx, y, x), _k1));
}

Value _trunc_with_sign(SPUContext* ctx, const Value& x, size_t bits,
                       bool is_positive) {
  SPU_TRACE_HAL_LEAF(ctx, x, bits);
  bits = (bits == 0) ? ctx->getFxpBits() : bits;

  if (x.isPublic()) {
    return _trunc_p_with_sign(ctx, x, bits, is_positive);
  } else if (x.isSecret()) {
    return _trunc_s_with_sign(ctx, x, bits, is_positive);
  } else {
    SPU_THROW("unsupport unary op={} for {}", "_trunc_with_sign", x);
  }
}

Value _trunc(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_LEAF(ctx, x, bits);
  bits = (bits == 0) ? ctx->getFxpBits() : bits;

  if (x.isPublic()) {
    return _trunc_p(ctx, x, bits);
  } else if (x.isSecret()) {
    return _trunc_s(ctx, x, bits);
  } else {
    SPU_THROW("unsupport unary op={} for {}", "_rshift", x);
  }
}

Value _negate(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // negate(x) = not(x) + 1
  return _add(ctx, _not(ctx, x), _constant(ctx, 1, x.shape()));
}

Value _sign(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // is_negative = x < 0 ? 1 : 0;
  const Value is_negative = _msb(ctx, x);

  // sign = 1 - 2 * is_negative
  //      = +1 ,if x >= 0
  //      = -1 ,if x < 0
  const auto one = _constant(ctx, 1, is_negative.shape());
  const auto two = _constant(ctx, 2, is_negative.shape());

  //
  return _sub(ctx, one, _mul(ctx, two, is_negative));
}

Value _less(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  // Note: the impl assume inputs are signed with two's complement encoding.
  // test msb(x-y) == 1
  return _msb(ctx, _sub(ctx, x, y));
}

// swap bits of [start, end)
Value _bitrev(SPUContext* ctx, const Value& x, size_t start, size_t end) {
  SPU_TRACE_HAL_LEAF(ctx, x, start, end);

  if (x.isPublic()) {
    return _bitrev_p(ctx, x, start, end);
  } else if (x.isSecret()) {
    return _bitrev_s(ctx, x, start, end);
  }

  SPU_THROW("unsupport op={} for {}", "_bitrev", x);
}

Value _mux(SPUContext* ctx, const Value& pred, const Value& a, const Value& b) {
  SPU_TRACE_HAL_LEAF(ctx, pred, a, b);

  // b + pred*(a-b)
  return _add(ctx, b, _mul(ctx, pred, _sub(ctx, a, b)));
}

Value _clamp(SPUContext* ctx, const Value& x, const Value& minv,
             const Value& maxv) {
  SPU_TRACE_HAL_LEAF(ctx, x, minv, maxv);

  // clamp lower bound, res = x < minv ? minv : x
  auto res = _mux(ctx, _less(ctx, x, minv), minv, x);

  // clamp upper bound, res = res < maxv ? res, maxv
  return _mux(ctx, _less(ctx, res, maxv), res, maxv);
}

Value _constant(SPUContext* ctx, uint128_t init,
                absl::Span<const int64_t> shape) {
  return _make_p(ctx, init, shape);
}

Value _bit_parity(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(absl::has_single_bit(bits), "currently only support power of 2");
  auto ret = _prefer_b(ctx, x);
  while (bits > 1) {
    ret = _xor(ctx, ret, _rshift(ctx, ret, bits / 2));
    bits /= 2;
  }

  ret = _and(ctx, ret, _constant(ctx, 1, x.shape()));
  return ret;
}

// TODO(jint): OPTIMIZE ME, this impl seems to be super slow.
Value _popcount(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  Value ret = _constant(ctx, 0, x.shape());
  // TODO:
  // 1. x's dtype may not be set at the moment.
  // 2. x's stype could be dynamic, especial for variadic boolean shares.
  const auto k1 = _constant(ctx, 1, x.shape());

  for (size_t idx = 0; idx < bits; idx++) {
    auto x_ = _rshift(ctx, x, idx);
    ret = _add(ctx, ret, _and(ctx, x_, k1));
  }

  return ret;
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
Value _prefix_or(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  auto b0 = _prefer_b(ctx, x);
  const size_t bit_width = SizeOf(ctx->getField()) * 8;
  for (int idx = 0; idx < absl::bit_width(bit_width) - 1; idx++) {
    const size_t offset = 1UL << idx;
    auto b1 = _rshift(ctx, b0, offset);
    b0 = _or(ctx, b0, b1);
  }
  return b0;
}

Value _bitdeintl(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_LEAF(ctx, in);

  // algorithm:
  //      0101010101010101
  // swap  ^^  ^^  ^^  ^^
  //      0011001100110011
  // swap   ^^^^    ^^^^
  //      0000111100001111
  // swap     ^^^^^^^^
  //      0000000011111111
  Value out = in;
  const size_t k = SizeOf(ctx->getField()) * 8;
  for (int64_t idx = 0; idx + 1 < Log2Ceil(k); idx++) {
    auto keep = _constant(ctx, detail::kBitIntlKeepMasks[idx], in.shape());
    auto move = _constant(ctx, detail::kBitIntlSwapMasks[idx], in.shape());
    int64_t shift = 1 << idx;
    // out = (out & keep) ^ ((out >> shift) & move) ^ ((out & move) << shift);
    out = _xor(ctx,
               _xor(ctx, _and(ctx, out, keep),
                    _and(ctx, _rshift(ctx, out, shift), move)),
               _lshift(ctx, _and(ctx, out, move), shift));
  }
  return out;
}

// Assumption:
//   let p: bshare
//   z1 = select(p, x1, y1)
//   z2 = select(p, x2, y2)
// where
//   select(p, x, y) = mul(p, y-x) + x
//                   = mul(b2a(p), y-x) + x     (1)
//                or = mula1b(p, y-x) + x       (2)
// when the cost of
// - b2a+2*mul < 2*mula2b, we prefer to convert p to ashare to avoid 2*b2a.
// - b2a+2*mul > 2*mula2b, we prefer to leave p as bshare.
//
// Cheetah is the later case.
Value _prefer_a(SPUContext* ctx, const Value& x) {
  if (x.storage_type().isa<BShare>()) {
    if (ctx->config().protocol() == ProtocolKind::CHEETAH &&
        x.storage_type().as<BShare>()->nbits() == 1) {
      return x;
    }

    // B2A
    return _add(ctx, x, _constant(ctx, 0, x.shape())).setDtype(x.dtype());
  }

  return x;
}

// Assumption
//   let v : ashare
//   y1 = v >> c1
//   y2 = v ^ c2
// When a variable is followed by multiple binary operation, it's more efficient
// to convert it to boolean share first.
Value _prefer_b(SPUContext* ctx, const Value& x) {
  if (x.storage_type().isa<AShare>()) {
    const auto k0 = _constant(ctx, 0U, x.shape());
    return _xor(ctx, x, k0).setDtype(x.dtype());  // noop, to bshare
  }

  return x;
}

}  // namespace spu::kernel::hal
