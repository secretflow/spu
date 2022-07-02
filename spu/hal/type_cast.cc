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

#include "spu/hal/type_cast.h"

#include "spu/core/type_util.h"
#include "spu/hal/constants.h"
#include "spu/hal/prot_wrapper.h"  // vtype_cast
#include "spu/hal/ring.h"

namespace spu::hal {
namespace {

Value _expand_boolean(HalContext* ctx, const Value& x) {
  if (getWidth(x.dtype()) != 1) {
    return x;
  }
  const size_t bit_width = SizeOf(x.storage_type().as<Ring2k>()->field()) * 8;
  const size_t length = bit_width - getWidth(x.dtype());
  return _rshift(ctx, _lshift(ctx, x, length), length);
}

Value int2fxp(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL(ctx, x);
  YASL_ENFORCE(x.isInt(), "expect integer, got {}", x.dtype());

  return _lshift(ctx, _expand_boolean(ctx, x), ctx->getFxpBits()).asFxp();
}

// Casting fxp to integer.
// note: casting truncates toward, floor truncates toward negative infinite.
//
// fxp2int(x) =
//   floor(x)                  if x >= 0
//   floor(x+1-fxp.epsilon)    else
//
// e.g.
//   fxp2int(0.5) = floor(0.5) = 0
//   fxp2int(1.0) = floor(1.0) = 1
//   fxp2int(1.2) = floor(1.2) = 1
//   fxp2int(-0.5) = floor(-0.5+0.999999) = 0
//   fxp2int(-1.0) = floor(-1+0.999999) = -1
//   fxp2int(-1.2) = floor(-1.2+0.9999999) = -1
//
Value fxp2int(HalContext* ctx, const Value& x, DataType to_type) {
  SPU_TRACE_HAL(ctx, x);
  YASL_ENFORCE(x.dtype() == DataType::DT_FXP);

  const size_t fxp_bits = ctx->getFxpBits();
  const Value kOneMinusEps = constant(ctx, (1 << fxp_bits) - 1, x.shape());

  // (x + 0.99 * (x < 0)) >> fxp_bits
  return _arshift(ctx, _add(ctx, x, _mul(ctx, kOneMinusEps, _msb(ctx, x))),
                  fxp_bits)
      .setDtype(to_type);
}

}  // namespace

// TODO: move p2s/reveal into a new header file.
Value p2s(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL(ctx, x);
  return _p2s(ctx, x).setDtype(x.dtype());
}

Value reveal(HalContext* ctx, const Value& x) {
  SPU_TRACE_HAL(ctx, x);
  return _s2p(ctx, x).setDtype(x.dtype());
}

Value dtype_cast(HalContext* ctx, const Value& in, DataType to_type) {
  SPU_TRACE_HAL(ctx, in, to_type);

  if (to_type == in.dtype()) {
    return in;
  }

  if (in.isInt()) {
    if (isInteger(to_type)) {
      // both integer, since we always use the whole ring for calculation, so we
      // can directly set to the new dtype for now.
      // once we start to optimize according to bit length, we may involve
      // carry-out calculation here.
      return Value(_expand_boolean(ctx, in).data(), to_type);
    } else {
      YASL_ENFORCE(isFixedPoint(to_type));
      return int2fxp(ctx, in);
    }
  } else {
    if (isInteger(to_type)) {
      return fxp2int(ctx, in, to_type);
    } else {
      YASL_ENFORCE(to_type == DT_FXP, "expect to_type FXP, got {}", to_type);
      YASL_ENFORCE(in.dtype() == DT_FXP, "expect in type FXP, got {}", to_type);
      // we only support one FXP type, do nothing.
      return in;
    }
  }

  YASL_THROW("should not be here");
}

}  // namespace spu::hal
