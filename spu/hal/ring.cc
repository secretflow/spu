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

#include "spu/hal/ring.h"

#include <array>

#include "xtensor/xoperation.hpp"

#include "spu/hal/constants.h"
#include "spu/hal/prot_wrapper.h"
#include "spu/hal/shape_ops.h"

namespace spu::hal {

#define IMPL_UNARY_OP(Name, FnP, FnS)                        \
  Value Name(HalContext* ctx, const Value& in) {             \
    SPU_TRACE_HAL(ctx, in);                                  \
    if (in.isPublic()) {                                     \
      return FnP(ctx, in);                                   \
    } else if (in.isSecret()) {                              \
      return FnS(ctx, in);                                   \
    } else {                                                 \
      YASL_THROW("unsupport unary op={} for {}", #Name, in); \
    }                                                        \
  }

#define IMPL_SHIFT_OP(Name, FnP, FnS)                         \
  Value Name(HalContext* ctx, const Value& in, size_t bits) { \
    SPU_TRACE_HAL(ctx, in, bits);                             \
    if (in.isPublic()) {                                      \
      return FnP(ctx, in, bits);                              \
    } else if (in.isSecret()) {                               \
      return FnS(ctx, in, bits);                              \
    } else {                                                  \
      YASL_THROW("unsupport unary op={} for {}", #Name, in);  \
    }                                                         \
  }

#define IMPL_COMMUTATIVE_BINARY_OP(Name, FnPP, FnSP, FnSS)         \
  Value Name(HalContext* ctx, const Value& x, const Value& y) {    \
    SPU_TRACE_HAL(ctx, x, y);                                      \
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
      YASL_THROW("unsupported op {} for x={}, y={}", #Name, x, y); \
    }                                                              \
  }

IMPL_UNARY_OP(_not, _not_p, _not_s)
IMPL_UNARY_OP(_msb, _msb_p, _msb_s)
IMPL_UNARY_OP(_eqz, _eqz_p, _eqz_s)

IMPL_SHIFT_OP(_lshift, _lshift_p, _lshift_s)
IMPL_SHIFT_OP(_rshift, _rshift_p, _rshift_s)
IMPL_SHIFT_OP(_arshift, _arshift_p, _arshift_s)

IMPL_COMMUTATIVE_BINARY_OP(_add, _add_pp, _add_sp, _add_ss)
IMPL_COMMUTATIVE_BINARY_OP(_mul, _mul_pp, _mul_sp, _mul_ss)
IMPL_COMMUTATIVE_BINARY_OP(_and, _and_pp, _and_sp, _and_ss)
IMPL_COMMUTATIVE_BINARY_OP(_xor, _xor_pp, _xor_sp, _xor_ss)

Value _sub(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL(ctx, x, y);
  return _add(ctx, x, _negate(ctx, y));
}

Value _mmul(HalContext* ctx, const Value& x, const Value& y) {
  if (x.isPublic() && y.isPublic()) {
    return _mmul_pp(ctx, x, y);
  } else if (x.isSecret() && y.isPublic()) {
    return _mmul_sp(ctx, x, y);
  } else if (x.isPublic() && y.isSecret()) {
    return transpose(ctx, _mmul_sp(ctx, transpose(ctx, y), transpose(ctx, x)));
  } else if (x.isSecret() && y.isSecret()) {
    return _mmul_ss(ctx, x, y);
  } else {
    YASL_THROW("unsupported op {} for x={}, y={}", "_matmul", x, y);
  }
}

Value _or(HalContext* ctx, const Value& x, const Value& y) {
  // X or Y = X xor Y xor (X and Y)
  return _xor(ctx, x, _xor(ctx, y, _and(ctx, x, y)));
}

Value _trunc(HalContext* ctx, const Value& x, size_t bits) {
  SPU_TRACE_HAL(ctx, x, bits);
  bits = (bits == 0) ? ctx->getFxpBits() : bits;

  if (x.isPublic()) {
    return _arshift_p(ctx, x, bits);
  } else if (x.isSecret()) {
    return _truncpr_s(ctx, x, bits);
  } else {
    YASL_THROW("unsupport unary op={} for {}", "_rshift", x);
  }
}

Value _negate(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL(ctx, x);

  // negate(x) = not(x) + 1
  return _add(ctx, _not(ctx, x), constant(ctx, 1, x.shape()));
}

Value _sign(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL(ctx, x);

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
  SPU_TRACE_HAL(ctx, x, y);

  // test msb(x-y) == 1
  return _msb(ctx, _sub(ctx, x, y));
}

// swap bits of [start, end)
Value _bitrev(HalContext* ctx, const Value& x, size_t start, size_t end) {
  SPU_TRACE_HAL(ctx, x, start, end);

  if (x.isPublic()) {
    return _bitrev_p(ctx, x, start, end);
  } else if (x.isSecret()) {
    return _bitrev_s(ctx, x, start, end);
  }

  YASL_THROW("unsupport op={} for {}", "_bitrev", x);
}

Value _mux(HalContext* ctx, const Value& pred, const Value& a, const Value& b) {
  SPU_TRACE_HLO(ctx, pred, a, b);

  // b + pred*(a-b)
  return _add(ctx, b, _mul(ctx, pred, _sub(ctx, a, b)));
}

// TODO(junfeng): OPTIMIZE ME
// TODO: test me.
Value _popcount(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL(ctx, x);

  Value ret = constant(ctx, 0, x.shape());
  const size_t bit_width = SizeOf(x.storage_type().as<Ring2k>()->field()) * 8;
  const auto one = constant(ctx, 1, x.shape());

  for (size_t idx = 0; idx < bit_width; idx++) {
    auto x_ = _rshift(ctx, x, idx);
    ret = _add(ctx, ret, _and(ctx, x_, one));
  }

  return ret;
}

}  // namespace spu::hal
