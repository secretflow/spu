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

Value _cast_type(SPUContext* ctx, const Value& x, const Type& to) {
  if (x.storage_type() == to) {
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
  Value Name(SPUContext* ctx, const Value& in) {            \
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

#define IMPL_SHIFT_OP(Name)                                         \
  Value Name(SPUContext* ctx, const Value& in, const Sizes& bits) { \
    SPU_TRACE_HAL_LEAF(ctx, in, bits);                              \
    if (in.isPublic()) {                                            \
      return Name##_p(ctx, in, bits);                               \
    } else if (in.isSecret()) {                                     \
      return Name##_s(ctx, in, bits);                               \
    } else if (in.isPrivate()) {                                    \
      return Name##_v(ctx, in, bits);                               \
    } else {                                                        \
      SPU_THROW("unsupport unary op={} for {}", #Name, in);         \
    }                                                               \
  }

IMPL_SHIFT_OP(_lshift)
IMPL_SHIFT_OP(_rshift)
IMPL_SHIFT_OP(_arshift)

#undef IMPL_SHIFT_OP

#define IMPL_COMMUTATIVE_BINARY_OP(Name)                          \
  Value Name(SPUContext* ctx, const Value& x, const Value& y) {   \
    SPU_TRACE_HAL_LEAF(ctx, x, y);                                \
    if (x.isPublic() && y.isPublic()) { /*PP*/                    \
      return Name##_pp(ctx, x, y);                                \
    } else if (x.isPrivate() && y.isPrivate()) { /*VV*/           \
      return Name##_vv(ctx, x, y);                                \
    } else if (x.isSecret() && y.isSecret()) { /*SS*/             \
      return Name##_ss(ctx, y, x);                                \
    } else if (x.isSecret() && y.isPublic()) { /*SP*/             \
      return Name##_sp(ctx, x, y);                                \
    } else if (x.isPublic() && y.isSecret()) { /*PS*/             \
      /* commutative, swap args */                                \
      return Name##_sp(ctx, y, x);                                \
    } else if (x.isPrivate() && y.isPublic()) { /*VP*/            \
      return Name##_vp(ctx, x, y);                                \
    } else if (x.isPublic() && y.isPrivate()) { /*PV*/            \
      /* commutative, swap args */                                \
      return Name##_vp(ctx, y, x);                                \
    } else if (x.isPrivate() && y.isSecret()) { /*VS*/            \
      return Name##_sv(ctx, y, x);                                \
    } else if (x.isSecret() && y.isPrivate()) { /*SV*/            \
      /* commutative, swap args */                                \
      return Name##_sv(ctx, x, y);                                \
    } else {                                                      \
      SPU_THROW("unsupported op {} for x={}, y={}", #Name, x, y); \
    }                                                             \
  }

IMPL_COMMUTATIVE_BINARY_OP(_add)
IMPL_COMMUTATIVE_BINARY_OP(_mul)
IMPL_COMMUTATIVE_BINARY_OP(_and)
IMPL_COMMUTATIVE_BINARY_OP(_xor)

#undef IMPL_COMMUTATIVE_BINARY_OP

static OptionalAPI<Value> _equal_impl(SPUContext* ctx, const Value& x,
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

  return NotAvailable;
}

Value _conv2d(SPUContext* ctx, const Value& input, const Value& kernel,
              const Strides& window_strides) {
  SPU_TRACE_HAL_DISP(ctx, input, kernel, window_strides);

  // TODO: assume s*p and p*p should call `dot`
  SPU_ENFORCE(input.isSecret() && kernel.isSecret());
  return _conv2d_ss(ctx, input, kernel, window_strides);
}

static Value _mmul_impl(SPUContext* ctx, const Value& x, const Value& y) {
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

Value _trunc(SPUContext* ctx, const Value& x, size_t bits, SignType sign) {
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
Value _bitrev(SPUContext* ctx, const Value& x, size_t start, size_t end) {
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
  SPU_ENFORCE(lhs.ndim() > 0 && lhs.ndim() <= 2);
  SPU_ENFORCE(rhs.ndim() > 0 && rhs.ndim() <= 2);

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

Value _mmul(SPUContext* ctx, const Value& x, const Value& y) {
  auto [m, n, k] = deduceMmulArgs(x.shape(), y.shape());

  // Enforce no vector
  if (x.shape() != Shape{m, k} || y.shape() != Shape{k, n}) {
    return _mmul(ctx, Value(x.data().reshape({m, k}), x.dtype()),
                 Value(y.data().reshape({k, n}), y.dtype()));
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
          x_block = _extract_slice(ctx, x, {k_start}, {k_end}, {});
        } else {
          x_block =
              _extract_slice(ctx, x, {m_start, k_start}, {m_end, k_end}, {});
        }

        Value y_block;
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
  const auto& eltype = ret_blocks[0][0].data().eltype();
  const auto& dtype = ret_blocks[0][0].dtype();
  Value ret(NdArrayRef(eltype, {m, n}), dtype);

  for (int64_t r = 0; r < static_cast<int64_t>(ret_blocks.size()); r++) {
    const auto& row_blocks = ret_blocks[r];
    for (int64_t c = 0; c < static_cast<int64_t>(row_blocks.size()); c++) {
      const auto& block = row_blocks[c];
      const int64_t block_rows = block.shape()[0];
      const int64_t block_cols = block.shape()[1];
      if (block.data().isCompact()) {
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
      } else {
        for (int64_t i = 0; i < block_rows; i++) {
          for (int64_t j = 0; j < block_cols; j++) {
            char* dst = &ret.data().at<char>({r * m_step + i, c * n_step + j});
            const char* src = &block.data().at<char>({i, j});
            std::memcpy(dst, src, block.elsize());
          }
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

// TODO: refactor polymorphic, and may use select functions in polymorphic
Value _clamp_lower(SPUContext* ctx, const Value& x, const Value& minv) {
  SPU_TRACE_HAL_LEAF(ctx, x, minv);
  // clamp lower bound, res = x < minv ? minv : x
  return _mux(ctx, _less(ctx, x, minv), minv, x);
}

Value _clamp_upper(SPUContext* ctx, const Value& x, const Value& maxv) {
  SPU_TRACE_HAL_LEAF(ctx, x, maxv);
  // clamp upper bound, x = x < maxv ? x, maxv
  return _mux(ctx, _less(ctx, x, maxv), x, maxv);
}

Value _constant(SPUContext* ctx, uint128_t init, const Shape& shape) {
  return _make_p(ctx, init, shape);
}

Value _bit_parity(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  SPU_ENFORCE(absl::has_single_bit(bits), "currently only support power of 2");
  auto ret = _prefer_b(ctx, x);
  while (bits > 1) {
    ret = _xor(ctx, ret, _rshift(ctx, ret, {static_cast<int64_t>(bits / 2)}));
    bits /= 2;
  }

  ret = _and(ctx, ret, _constant(ctx, 1, x.shape()));
  return ret;
}

// TODO(jint): OPTIMIZE ME, this impl seems to be super slow.
Value _popcount(SPUContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  if (x.shape().isEmpty()) {
    return x;
  }

  auto xb = _prefer_b(ctx, x);

  std::vector<Value> vs;
  vs.reserve(bits);
  for (size_t idx = 0; idx < bits; idx++) {
    auto x_ = _rshift(ctx, xb, {static_cast<int64_t>(idx)});
    x_ = _and(ctx, x_, _constant(ctx, 1U, x.shape()));

    if (x_.storage_type().isa<BShare>()) {
      const_cast<Type&>(x_.storage_type()).as<BShare>()->setNbits(1);
    }
    vs.push_back(std::move(x_));
  }

  return vreduce(vs.begin(), vs.end(), [&](const Value& a, const Value& b) {
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
Value _prefix_or(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  auto b0 = _prefer_b(ctx, x);
  const size_t bit_width = SizeOf(ctx->getField()) * 8;
  for (int idx = 0; idx < absl::bit_width(bit_width) - 1; idx++) {
    const int64_t offset = 1L << idx;
    auto b1 = _rshift(ctx, b0, {offset});
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
                    _and(ctx, _rshift(ctx, out, {shift}), move)),
               _lshift(ctx, _and(ctx, out, move), {shift}));
  }
  return out;
}

Value _prefer_a(SPUContext* ctx, const Value& x) {
  if (x.storage_type().isa<BShare>()) {
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

namespace {

// Example:
// in  = {1, 3}, n = 5
// res = {1, 3, 0, 2, 4}
Index buildFullIndex(const Index& in, int64_t n) {
  Index out = in;
  out.reserve(n);
  for (int64_t dim = 0; dim < n; dim++) {
    SPU_ENFORCE_LT(dim, n, "dim={} out of bound={}", dim, n);
    if (std::find(in.begin(), in.end(), dim) == in.end()) {
      out.push_back(dim);
    }
  }
  return out;
}

template <typename Itr>
inline int64_t product(Itr first, Itr last) {
  return std::accumulate(first, last, 1, std::multiplies<>());
}

}  // namespace

// TODO: test me.
Value _tensordot(SPUContext* ctx, const Value& x, const Value& y,
                 const Index& ix, const Index& iy) {
  SPU_ENFORCE(ix.size() == iy.size());

  // number of dims to contract.
  const size_t nc = ix.size();

  Index perm_x = buildFullIndex(ix, x.shape().ndim());  //
  Index perm_y = buildFullIndex(iy, y.shape().ndim());
  std::rotate(perm_x.begin(), perm_x.begin() + nc, perm_x.end());

  // convert to mmul shape.
  auto xx = _transpose(ctx, x, Axes(perm_x));
  Shape xxs = xx.shape();
  xx = _reshape(ctx, xx,
                {product(xxs.begin(), xxs.end() - nc),
                 product(xxs.end() - nc, xxs.end())});

  auto yy = _transpose(ctx, y, Axes(perm_y));
  Shape yys = yy.shape();
  yy = _reshape(ctx, yy,
                {product(yys.begin(), yys.begin() + nc),
                 product(yys.begin() + nc, yys.end())});

  // do matrix multiplication.
  auto zz = _mmul(ctx, xx, yy);

  // decompose shape back.
  Shape res_shape(xxs.begin(), xxs.end() - nc);
  res_shape.insert(res_shape.end(), yys.begin() + nc, yys.end());

  return _reshape(ctx, zz, res_shape);
}

std::optional<Value> _oramonehot(SPUContext* ctx, const Value& x,
                                 int64_t db_size, bool db_is_public) {
  std::optional<Value> ret;
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

Value _oramread(SPUContext* ctx, const Value& x, const Value& y,
                int64_t offset) {
  SPU_ENFORCE(x.isSecret(), "onehot should be secret shared");
  auto reshaped_x = Value(x.data().reshape({1, x.numel()}), x.dtype());
  auto reshaped_y = y;
  if (y.shape().size() == 1) {
    reshaped_y = Value(y.data().reshape({y.numel(), 1}), y.dtype());
  }

  Value ret;
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

}  // namespace spu::kernel::hal
