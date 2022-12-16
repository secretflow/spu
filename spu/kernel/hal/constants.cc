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

#include "spu/kernel/hal/constants.h"

#include "spu/core/encoding.h"
#include "spu/core/ndarray_ref.h"
#include "spu/core/shape_util.h"
#include "spu/core/xt_helper.h"
#include "spu/kernel/hal/prot_wrapper.h"
#include "spu/kernel/hal/shape_ops.h"
#include "spu/mpc/common/pub2k.h"

namespace spu::kernel::hal {
namespace {

// make a public typed value.
//
// Note: there is a abraction leakage, we should NOT touch Ring2kPubTy directly.
Value make_pub2k(HalContext* ctx, PtBufferView bv) {
  SPU_TRACE_HAL_DISP(ctx, bv);

  NdArrayRef raw = xt_to_ndarray(bv);

  DataType dtype;
  NdArrayRef encoded =
      encodeToRing(raw, ctx->getField(), ctx->getFxpBits(), &dtype);

  return Value(encoded.as(makeType<mpc::Pub2kTy>(ctx->getField())), dtype);
}

}  // namespace

Value constant(HalContext* ctx, PtBufferView bv,
               absl::Span<const int64_t> shape) {
  SPU_TRACE_HAL_DISP(ctx, bv, shape);

  if (isEmpty(shape)) {
    auto pt = make_pub2k(ctx, bv);
    return Value(NdArrayRef(nullptr, pt.storage_type(), shape), pt.dtype());
  }

  // If view shape is same as destination shape, just make public
  if (shape.empty() || shape == bv.shape) {
    return make_pub2k(ctx, bv);
  }

  // Same calcNumel but shape is different, do a reshape
  if (calcNumel(bv.shape) == calcNumel(shape)) {
    return reshape(ctx, make_pub2k(ctx, bv), shape);
  }

  // Other, do a broadcast, let broadcast handles the sanity check
  YACL_ENFORCE(calcNumel(bv.shape) <= calcNumel(shape));
  return broadcast_to(ctx, make_pub2k(ctx, bv), shape);
}

Value const_secret(HalContext* ctx, PtBufferView bv,
                   absl::Span<const int64_t> shape) {
  SPU_TRACE_HAL_DISP(ctx, bv);

  auto pv = constant(ctx, bv, shape);
  return _p2s(ctx, pv).setDtype(pv.dtype());
}

NdArrayRef dump_public(HalContext* ctx, const Value& v) {
  SPU_TRACE_HAL_DISP(ctx, v);
  YACL_ENFORCE(v.storage_type().isa<mpc::Pub2kTy>(), "got {}",
               v.storage_type());
  const auto field = v.storage_type().as<Ring2k>()->field();
  auto encoded = v.data().as(makeType<RingTy>(field));

  return decodeFromRing(encoded, v.dtype(), ctx->getFxpBits());
}

Value make_value(HalContext* ctx, Visibility vtype, PtBufferView bv) {
  switch (vtype) {
    case VIS_PUBLIC:
      return make_pub2k(ctx, bv);
    case VIS_SECRET:
      return const_secret(ctx, bv);
    default:
      YACL_THROW("not support vtype={}", vtype);
  }
}

spu::Value zeros(HalContext* ctx, Visibility vis, DataType dtype,
                 absl::Span<const int64_t> shape) {
  spu::Value scalar;
  switch (dtype) {
    case DT_I1: {
      scalar = hal::make_value(ctx, vis, false);
      break;
    }
    case DT_I8: {
      scalar = hal::make_value(ctx, vis, static_cast<std::int8_t>(0));
      break;
    }
    case DT_U8: {
      scalar = hal::make_value(ctx, vis, static_cast<std::uint8_t>(0));
      break;
    }
    case DT_I16: {
      scalar = hal::make_value(ctx, vis, static_cast<std::int16_t>(0));
      break;
    }
    case DT_U16: {
      scalar = hal::make_value(ctx, vis, static_cast<std::uint16_t>(0));
      break;
    }
    case DT_I32: {
      scalar = hal::make_value(ctx, vis, static_cast<std::int32_t>(0));
      break;
    }
    case DT_U32: {
      scalar = hal::make_value(ctx, vis, static_cast<std::uint32_t>(0));
      break;
    }
    case DT_I64: {
      scalar = hal::make_value(ctx, vis, static_cast<std::int64_t>(0));
      break;
    }
    case DT_U64: {
      scalar = hal::make_value(ctx, vis, static_cast<std::uint64_t>(0));
      break;
    }
    case DT_FXP: {
      scalar = hal::make_value(ctx, vis, 0.0F);
      break;
    }
    default: {
      YACL_THROW("Should not hit, dtype = {}", dtype);
    }
  }

  if (shape.empty()) {
    return scalar;
  } else {
    return hal::expand(ctx, scalar, shape);
  }
}

}  // namespace spu::kernel::hal
