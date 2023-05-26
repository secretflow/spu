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

#include "libspu/kernel/hal/fxp_cleartext.h"

#include <cmath>

#include "libspu/core/encoding.h"

namespace spu::kernel::hal {
namespace {

template <typename FN>
Value applyFloatingPointFn(SPUContext* ctx, const Value& in, FN&& fn) {
  SPU_TRACE_HAL_DISP(ctx, in);
  SPU_ENFORCE(in.isPublic(), "expected public, got {}", in.storage_type());
  SPU_ENFORCE(in.isFxp(), "expected fxp, got={}", in.dtype());

  const size_t fxp_bits = ctx->getFxpBits();
  const auto field = in.storage_type().as<Ring2k>()->field();
  const Type ring_ty = makeType<RingTy>(field);

  // decode to floating point
  auto f32_arr = decodeFromRing(in.data().as(ring_ty), in.dtype(), fxp_bits);

  for (auto iter = f32_arr.begin(); iter != f32_arr.end(); ++iter) {
    auto* ptr = reinterpret_cast<float*>(iter.getRawPtr());
    *ptr = fn(*ptr);
  }

  DataType dtype;
  const auto out = encodeToRing(f32_arr, field, fxp_bits, &dtype);

  SPU_ENFORCE(dtype == DT_F32 || dtype == DT_F64, "sanity failed");
  return Value(out.as(in.storage_type()), dtype);
}

template <typename FN>
Value applyFloatingPointFn(SPUContext* ctx, const Value& x, const Value& y,
                           FN&& fn) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  SPU_ENFORCE(x.isPublic() && y.isPublic(), "expect public, got {}, {}",
              x.vtype(), y.vtype());
  SPU_ENFORCE((x.isFxp()) && (y.isFxp()), "expected fxp, got={} {}", x.dtype(),
              y.dtype());
  SPU_ENFORCE(x.shape() == y.shape());

  const auto field = x.storage_type().as<Ring2k>()->field();
  const size_t fxp_bits = ctx->getFxpBits();
  const Type ring_ty = makeType<RingTy>(field);

  // decode to floating point
  auto flp_x = decodeFromRing(x.data().as(ring_ty), x.dtype(), fxp_bits);
  auto flp_y = decodeFromRing(y.data().as(ring_ty), y.dtype(), fxp_bits);

  for (auto itr_x = flp_x.begin(), itr_y = flp_y.begin(); itr_x != flp_x.end();
       itr_x++, itr_y++) {
    auto* ptr_x = reinterpret_cast<float*>(itr_x.getRawPtr());
    auto* ptr_y = reinterpret_cast<float*>(itr_y.getRawPtr());
    *ptr_x = fn(*ptr_x, *ptr_y);
  }

  DataType dtype;
  const auto out = encodeToRing(flp_x, field, fxp_bits, &dtype);
  SPU_ENFORCE(dtype == DT_F32 || dtype == DT_F64, "sanity failed");
  return Value(out.as(x.storage_type()), dtype);
}

}  // namespace

Value f_reciprocal_p(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);

  return applyFloatingPointFn(ctx, in, [](float x) { return 1.0 / x; });
}

Value f_log_p(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  return applyFloatingPointFn(ctx, in, [](float x) { return std::log(x); });
}

Value f_exp_p(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  return applyFloatingPointFn(ctx, in, [](float x) { return std::exp(x); });
}

Value f_div_p(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  return applyFloatingPointFn(ctx, x, y,
                              [](float a, float b) { return a / b; });
}

}  // namespace spu::kernel::hal
