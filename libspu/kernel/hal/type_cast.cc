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

#include "libspu/kernel/hal/type_cast.h"

#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/prot_wrapper.h"  // vtype_cast
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hal {
namespace {

Value int2fxp(SPUContext* ctx, const Value& x, DataType to_type) {
  SPU_TRACE_HAL_LEAF(ctx, x);
  SPU_ENFORCE(x.isInt(), "expect integer, got {}", x.dtype());

  return _lshift(ctx, x, ctx->getFxpBits()).setDtype(to_type);
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
Value fxp2int(SPUContext* ctx, const Value& x, DataType to_type) {
  SPU_TRACE_HAL_LEAF(ctx, x);
  SPU_ENFORCE(x.isFxp());

  const size_t fxp_bits = ctx->getFxpBits();
  const Value kOneMinusEps = _constant(ctx, (1 << fxp_bits) - 1, x.shape());

  // (x + 0.99 * (x < 0)) >> fxp_bits
  return _arshift(ctx, _add(ctx, x, _mul(ctx, kOneMinusEps, _msb(ctx, x))),
                  fxp_bits)
      .setDtype(to_type);
}

}  // namespace

// TODO: move seal/reveal into a new header file.
Value seal(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);
  return _p2s(ctx, x).setDtype(x.dtype());
}

Value reveal(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);
  return _s2p(ctx, x).setDtype(x.dtype());
}

Value dtype_cast(SPUContext* ctx, const Value& in, DataType to_type) {
  SPU_TRACE_HAL_DISP(ctx, in, to_type);

  if (to_type == in.dtype()) {
    return in;
  }

  if (in.isInt()) {
    if (isInteger(to_type)) {
      // both integer, since we always use the whole ring for calculation, so we
      // can directly set to the new dtype for now.
      // once we start to optimize according to bit length, we may involve
      // carry-out calculation here.
      return Value(in.data(), to_type);
    } else {
      SPU_ENFORCE(isFixedPoint(to_type));
      return int2fxp(ctx, in, to_type);
    }
  } else {
    if (isInteger(to_type)) {
      return fxp2int(ctx, in, to_type);
    } else {
      SPU_ENFORCE(to_type == DT_F32 || to_type == DT_F64,
                  "expect to_type FXP, got {}", to_type);
      SPU_ENFORCE(in.isFxp(), "expect in type FXP, got {}", in.dtype());
      return Value(in.data(), to_type);
    }
  }

  SPU_THROW("should not be here");
}

Value stype_cast(SPUContext* ctx, const Value& in, const Type& to) {
  if (in.storage_type() == to) {
    return in;
  }
  return _cast_type(ctx, in, to).setDtype(in.dtype());
}

}  // namespace spu::kernel::hal
