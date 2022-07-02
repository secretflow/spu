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

#include "spu/hal/public_intrinsic.h"

#include "spu/core/encoding.h"
#include "spu/core/xt_helper.h"

namespace spu::hal {
namespace {

Value applyFloatingPointFn(
    HalContext* ctx, const Value& in,
    const std::function<NdArrayRef(const xt::xarray<float>&)>& fn) {
  SPU_TRACE_HAL(ctx, in);
  YASL_ENFORCE(in.isPublic(), "float intrinsic, expected public, got {}",
               in.storage_type());
  YASL_ENFORCE(in.dtype() == DT_FXP, "expected fxp, got={}", in.dtype());

  const size_t fxp_bits = ctx->getFxpBits();
  const auto field = in.storage_type().as<Ring2k>()->field();
  const Type ring_ty = makeType<RingTy>(field);

  // decode to floating point
  const auto raw = decodeFromRing(in.data().as(ring_ty), in.dtype(), fxp_bits);

  DataType dtype;
  const auto out =
      encodeToRing(fn(xt_adapt<float>(raw)), field, fxp_bits, &dtype);
  YASL_ENFORCE(dtype == DT_FXP, "sanity failed");
  return Value(out.as(in.storage_type()), dtype);
}

Value applyFloatingPointFn(
    HalContext* ctx, const Value& x, const Value& y,
    const std::function<NdArrayRef(const xt::xarray<float>&,
                                   const xt::xarray<float>&)>& fn) {
  SPU_TRACE_HAL(ctx, x, y);
  YASL_ENFORCE(x.isPublic() && y.isPublic());
  YASL_ENFORCE((x.dtype() == DT_FXP) && (y.dtype() == DT_FXP));
  YASL_ENFORCE(x.dtype() == DT_FXP, "expected fxp, got={}", x.dtype());

  const auto field = x.storage_type().as<Ring2k>()->field();
  const size_t fxp_bits = ctx->getFxpBits();
  const Type ring_ty = makeType<RingTy>(field);

  // decode to floating point
  const auto flp_x = decodeFromRing(x.data().as(ring_ty), x.dtype(), fxp_bits);
  const auto flp_y = decodeFromRing(y.data().as(ring_ty), y.dtype(), fxp_bits);

  DataType dtype;
  const auto out =
      encodeToRing(fn(xt_adapt<float>(flp_x), xt_adapt<float>(flp_y)), field,
                   fxp_bits, &dtype);
  YASL_ENFORCE(dtype == DT_FXP, "sanity failed");
  return Value(out.as(x.storage_type()), dtype);
}

}  // namespace

Value f_reciprocal_p(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL(ctx, in);

  return applyFloatingPointFn(ctx, in, [&](const xt::xarray<float>& farr) {
    return xt_to_ndarray(1.0f / farr);
  });
}

Value f_log_p(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL(ctx, in);
  return applyFloatingPointFn(ctx, in, [&](const xt::xarray<float>& farr) {
    return xt_to_ndarray(xt::log(farr));
  });
}

Value f_exp_p(HalContext* ctx, const Value& in) {
  SPU_TRACE_HAL(ctx, in);
  return applyFloatingPointFn(ctx, in, [&](const xt::xarray<float>& farr) {
    return xt_to_ndarray(xt::exp(farr));
  });
}

Value f_div_p(HalContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL(ctx, x, y);
  return applyFloatingPointFn(
      ctx, x, y, [&](const xt::xarray<float>& a, const xt::xarray<float>& b) {
        return xt_to_ndarray(a / b);
      });
}

}  // namespace spu::hal
