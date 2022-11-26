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

#include "spu/kernel/hal/ring.h"

#include <array>
#include <cmath>

#include "yacl/base/exception.h"

#include "spu/kernel/hal/constants.h"
#include "spu/kernel/hal/prot_wrapper.h"
#include "spu/kernel/hal/shape_ops.h"

namespace spu::kernel::hal {

namespace {

std::tuple<int64_t, int64_t, int64_t> deduceMmulArgs(
    const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
  YACL_ENFORCE(!lhs.empty() && lhs.size() <= 2);
  YACL_ENFORCE(!rhs.empty() && rhs.size() <= 2);

  if (lhs.size() == 1 && rhs.size() == 1) {
    YACL_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, 1, rhs[0]);
  }
  if (lhs.size() == 1 && rhs.size() == 2) {
    YACL_ENFORCE(lhs[0] == rhs[0]);
    return std::make_tuple(1, rhs[1], rhs[0]);
  }
  if (lhs.size() == 2 && rhs.size() == 1) {
    YACL_ENFORCE(lhs[1] == rhs[0]);
    return std::make_tuple(lhs[0], 1, rhs[0]);
  }
  YACL_ENFORCE(lhs[1] == rhs[0]);
  return std::make_tuple(lhs[0], rhs[1], rhs[0]);
}

std::tuple<int64_t, int64_t, int64_t> calcMmulTilingSize(int64_t m, int64_t n,
                                                         int64_t k,
                                                         size_t elsize,
                                                         size_t mem_limit) {
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

}  // namespace

Type _common_type(HalContext* ctx, const Type& a, const Type& b) {
  if (a.isa<Secret>() && b.isa<Secret>()) {
    return _common_type_s(ctx, a, b);
  } else if (a.isa<Secret>()) {
    return a;
  } else if (b.isa<Secret>()) {
    return b;
  } else {
    YACL_ENFORCE(a.isa<Public>() && b.isa<Public>());
    return a;
  }
}

Value _cast_type(HalContext* ctx, const Value& x, const Type& to) {
  if (x.isPublic() && to.isa<Public>()) {
    return x;
  } else if (x.isPublic() && to.isa<Secret>()) {
    // FIXME: casting to BShare semantic is wrong.
    return _p2s(ctx, x);
  } else if (x.isSecret() && to.isa<Secret>()) {
    return _cast_type_s(ctx, x, to);
  } else {
    YACL_THROW("show not be here x={}, to={}", x, to);
  }
}

#define IMPL_UNARY_OP(Name, FnP, FnS)                        \
  Value Name(HalContext* ctx, const Value& in) {             \
    SPU_TRACE_HAL_LEAF(ctx, in);                             \
    if (in.isPublic()) {                                     \
      return FnP(ctx, in);                                   \
    } else if (in.isSecret()) {                              \
      return FnS(ctx, in);                                   \
    } else {                                                 \
      YACL_THROW("unsupport unary op={} for {}", #Name, in); \
    }                                                        \
  }

#define IMPL_SHIFT_OP(Name, FnP, FnS)                         \
  Value Name(HalContext* ctx, const Value& in, size_t bits) { \
    SPU_TRACE_HAL_LEAF(ctx, in, bits);                        \
    if (in.isPublic()) {                                      \
      return FnP(ctx, in, bits);                              \
    } else if (in.isSecret()) {                               \
      return FnS(ctx, in, bits);                              \
    } else {                                                  \
      YACL_THROW("unsupport unary op={} for {}", #Name, in);  \
    }                                                         \
  }

#define IMPL_COMMUTATIVE_BINARY_OP(Name, FnPP, FnSP, FnSS)         \
  Value Name(HalContext* ctx, const Value& x, const Value& y) {    \
    SPU_TRACE_HAL_LEAF(ctx, x, y);                                 \
    if (x.isPublic() && y.isPublic()) {                            \
      return FnPP(ctx, x, y);                                      \
    } else if (x.isSecret() && y.isPublic()) {                     \
      return FnSP(ctx, x, y);                                      \
    } else if (x.isPublic() && y.isSecret()) {                     \
      /* commutative, swap args */                                 \
      return FnSP(ctx, y, x);                                      \
    } else if (x.isSecret() && y.isSecret()) {                     \
      return FnSS(ctx, y, x);                                      \
    } else {                                                       \
      YACL_THROW("unsupported op {} for x={}, y={}", #Name, x, y); \
    }                                                              \
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

Value _sub(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);
  return _add(ctx, x, _negate(ctx, y));
}

Value _eqz(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // eqz(x) = not(lsb(pre_or(x)))
  // all equal to zero means lsb equals to zero
  auto _k1 = constant(ctx, 1U, x.shape());
  auto res = _xor(ctx, _and(ctx, _prefix_or(ctx, x), _k1), _k1);

  // FIXME(jint): see hintNumberOfBits
  if (res.storage_type().isa<BShare>()) {
    const_cast<Type&>(res.storage_type()).as<BShare>()->setNbits(1);
  }

  return res;
}

Value _mmul(HalContext* ctx, const Value& x, const Value& y) {
  auto [m, n, k] = deduceMmulArgs(x.shape(), y.shape());
  auto [m_step, n_step, k_step] =
      calcMmulTilingSize(m, n, k, x.elsize(), 256UL * 1024 * 1024);

  auto mmul_impl = [&](const Value& x, const Value& y) {
    if (x.isPublic() && y.isPublic()) {
      return _mmul_pp(ctx, x, y);
    } else if (x.isSecret() && y.isPublic()) {
      return _mmul_sp(ctx, x, y);
    } else if (x.isPublic() && y.isSecret()) {
      return transpose(ctx,
                       _mmul_sp(ctx, transpose(ctx, y), transpose(ctx, x)));
    } else if (x.isSecret() && y.isSecret()) {
      return _mmul_ss(ctx, x, y);
    } else {
      YACL_THROW("unsupported op {} for x={}, y={}", "_matmul", x, y);
    }
  };

  if (ctx->rt_config().experimental_disable_mmul_split() ||
      (m_step == m && n_step == n && k_step == k)) {
    // no split
    return mmul_impl(x, y);
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
          YACL_ENFORCE(m_start == 0 && m_end == 1);
          x_block = slice(ctx, x, {k_start}, {k_end}, {});
        } else {
          x_block = slice(ctx, x, {m_start, k_start}, {m_end, k_end}, {});
        }

        Value y_block;
        if (y.shape().size() == 1) {
          YACL_ENFORCE(n_start == 0 && n_end == 1);
          y_block = slice(ctx, y, {k_start}, {k_end}, {});
        } else {
          y_block = slice(ctx, y, {k_start, n_start}, {k_end, n_end}, {});
        }

        auto mmul_ret = mmul_impl(x_block, y_block);
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
      YACL_ENFORCE(block.data().isCompact());
      const int64_t block_rows = block.shape()[0];
      const int64_t block_cols = block.shape()[1];
      if (n_blocks == 1) {
        YACL_ENFORCE(row_blocks.size() == 1);
        YACL_ENFORCE(block_cols == n);
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

Value _or(HalContext* ctx, const Value& x, const Value& y) {
  // X or Y = X xor Y xor (X and Y)
  return _xor(ctx, x, _xor(ctx, y, _and(ctx, x, y)));
}

Value _trunc(HalContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL_LEAF(ctx, x, bits);
  bits = (bits == 0) ? ctx->getFxpBits() : bits;

  if (x.isPublic()) {
    return _arshift_p(ctx, x, bits);
  } else if (x.isSecret()) {
    return _truncpr_s(ctx, x, bits);
  } else {
    YACL_THROW("unsupport unary op={} for {}", "_rshift", x);
  }
}

Value _negate(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // negate(x) = not(x) + 1
  return _add(ctx, _not(ctx, x), constant(ctx, 1, x.shape()));
}

Value _sign(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  // is_negative = x < 0 ? 1 : 0;
  const Value is_negative = _msb(ctx, x);

  // sign = 1 - 2 * is_negative
  //      = +1 ,if x >= 0
  //      = -1 ,if x < 0
  const auto one = constant(ctx, 1, is_negative.shape());
  const auto two = constant(ctx, 2, is_negative.shape());

  //
  return _sub(ctx, one, _mul(ctx, two, is_negative));
}

Value _less(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_LEAF(ctx, x, y);

  // test msb(x-y) == 1
  return _msb(ctx, _sub(ctx, x, y));
}

// swap bits of [start, end)
Value _bitrev(HalContext* ctx, const Value& x, size_t start, size_t end) {
  SPU_TRACE_HAL_LEAF(ctx, x, start, end);

  if (x.isPublic()) {
    return _bitrev_p(ctx, x, start, end);
  } else if (x.isSecret()) {
    return _bitrev_s(ctx, x, start, end);
  }

  YACL_THROW("unsupport op={} for {}", "_bitrev", x);
}

Value _mux(HalContext* ctx, const Value& pred, const Value& a, const Value& b) {
  SPU_TRACE_HAL_LEAF(ctx, pred, a, b);

  // b + pred*(a-b)
  return _add(ctx, b, _mul(ctx, pred, _sub(ctx, a, b)));
}

// TODO(junfeng): OPTIMIZE ME
// TODO: test me.
Value _popcount(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  Value ret = constant(ctx, 0, x.shape());
  // TODO:
  // 1. x's dtype may not be set at the moment.
  // 2. x's stype could be dynamic, especial for variadic boolean shares.
  const size_t bit_width = SizeOf(ctx->rt_config().field()) * 8;
  const auto one = constant(ctx, 1, x.shape());

  for (size_t idx = 0; idx < bit_width; idx++) {
    auto x_ = _rshift(ctx, x, idx);
    ret = _add(ctx, ret, _and(ctx, x_, one));
  }

  return ret;
}

// Fill all bits after msb to 1.
//
// Algorithm, lets consider the msb only, in each iteration we fill
// [msb-2^k, msb) to 1.
//   x0:  010000000   ; x0
//   x1:  011000000   ; x0 | (x0>>1)
//   x2:  011110000   ; x1 | (x1>>2)
//   x3:  011111111   ; x2 | (x2>>4)
//
Value _prefix_or(HalContext* ctx, const Value& x) {
  auto b0 = _xor(ctx, x, constant(ctx, 0, x.shape()));  // nop, cast to bshr.
  const size_t bit_width = SizeOf(ctx->getField()) * 8;
  for (size_t idx = 0; idx < absl::bit_width(bit_width) - 1; idx++) {
    const size_t offset = 1UL << idx;
    auto b1 = _rshift(ctx, b0, offset);
    b0 = _or(ctx, b0, b1);
  }
  return b0;
}

}  // namespace spu::kernel::hal
