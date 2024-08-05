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

#include "libspu/core/field_type_mapping.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/prot_wrapper.h"  // vtype_cast
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hal {
namespace {

Value int2fxp(SPUContext* ctx, const Value& x, DataType to_type) {
  SPU_TRACE_HAL_LEAF(ctx, x);
  SPU_ENFORCE(x.isInt(), "expect integer, got {}", x.dtype());
  auto to_field = getFieldTypeFromDataType(to_type);

  return _lshift(ctx, _cast_ring(ctx, x, to_field).setDtype(x.dtype()),
                 ctx->getFxpBits(to_field))
      .setDtype(to_type, true);
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
  auto from_field = getFieldTypeFromDataType(x.dtype());
  auto to_field = getFieldTypeFromDataType(to_type);

  const size_t fxp_bits = ctx->getFxpBits(from_field);
  const Value kOneMinusEps =
      _constant(ctx, (1 << fxp_bits) - 1, x.shape(), x.dtype());

  // (x + 0.99 * (x < 0)) >> fxp_bits
  auto tmp = _add(ctx, x, _mul(ctx, kOneMinusEps, _msb(ctx, x)));
  // convert to a share
  auto i = _add(ctx, _arshift(ctx, tmp, fxp_bits),
                _constant(ctx, 0, tmp.shape(), tmp.dtype()));
  return _cast_ring(ctx, i, to_field).setDtype(to_type, true);
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

  auto from_field = getFieldTypeFromDataType(in.dtype());
  auto to_field = getFieldTypeFromDataType(to_type);

  if (in.isInt() && isInteger(to_type)) {
    // Int to int
    if (from_field == to_field) {
      // both integer, since we always use the whole ring for calculation, so
      // we can directly set to the new dtype for now. once we start to
      // optimize according to bit length, we may involve carry-out
      // calculation here.
      return Value(in.data(), to_type);
    } else {
      // Cast ring...
      return _cast_ring(ctx, in, to_field).setDtype(to_type, true);
    }
  }

  if (in.isFxp() && isFixedPoint(to_type)) {
    auto in_fxp_bits = ctx->getFxpBits(from_field);
    auto to_fxp_bits = ctx->getFxpBits(to_field);
    if (in_fxp_bits > to_fxp_bits) {
      auto reduced =
          _trunc(ctx, in, in_fxp_bits - to_fxp_bits).setDtype(in.dtype());
      return _cast_ring(ctx, reduced, to_field).setDtype(to_type, true);
      // NOTE: the truncation in down cast for floating-point numbers can be
      // optimized.
      // return _cast_ring(ctx, in, to_field).setDtype(to_type, true);
    } else {
      auto up_casted = _cast_ring(ctx, in, to_field).setDtype(to_type, true);
      return _lshift(ctx, up_casted, to_fxp_bits - in_fxp_bits)
          .setDtype(to_type, true);
    }
  }

  if (in.isInt() && isFixedPoint(to_type)) {
    return int2fxp(ctx, in, to_type);
  }

  if (in.isFxp() && isInteger(to_type)) {
    return fxp2int(ctx, in, to_type);
  }

  SPU_THROW("should not be here");
}

}  // namespace spu::kernel::hal
