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

#include "libspu/core/context.h"
#include "libspu/core/encoding.h"
#include "libspu/core/trace.h"

namespace spu::kernel::hal {
namespace {

NdArrayRef encodeToRing(const NdArrayRef& src, FieldType field, size_t fxp_bits,
                        DataType* out_type) {
  SPU_ENFORCE(src.eltype().isa<PtTy>(), "expect PtType, got={}", src.eltype());
  const PtType pt_type = src.eltype().as<PtTy>()->pt_type();
  PtBufferView pv(static_cast<const void*>(src.data()), pt_type, src.shape(),
                  src.strides());
  return encodeToRing(pv, field, fxp_bits, out_type);
}

NdArrayRef decodeFromRing(const NdArrayRef& src, DataType in_dtype,
                          size_t fxp_bits) {
  const PtType pt_type = getDecodeType(in_dtype);
  NdArrayRef dst(makePtType(pt_type), src.shape());
  PtBufferView pv(static_cast<void*>(dst.data()), pt_type, dst.shape(),
                  dst.strides());
  decodeFromRing(src, in_dtype, fxp_bits, &pv, nullptr);
  return dst;
}

template <typename FN>
Value applyFloatingPointFn(SPUContext* ctx, const Value& in, FN&& fn) {
  SPU_TRACE_HAL_DISP(ctx, in);
  SPU_ENFORCE(in.isPublic(), "expected public, got {}", in.storage_type());
  SPU_ENFORCE(in.isFxp(), "expected fxp, got={}", in.dtype());

  const size_t fxp_bits = ctx->getFxpBits();
  const auto field = in.storage_type().as<Ring2k>()->field();
  const Type ring_ty = makeType<RingTy>(field);

  // decode to floating point
  auto fp_arr = decodeFromRing(in.data().as(ring_ty), in.dtype(), fxp_bits);
  auto pt_type = getDecodeType(in.dtype());

  for (auto iter = fp_arr.begin(); iter != fp_arr.end(); ++iter) {
    DISPATCH_FLOAT_PT_TYPES(pt_type, "pt_type", [&]() {
      auto* ptr = reinterpret_cast<ScalarT*>(&*iter);
      *ptr = fn(*ptr);
    });
  }

  DataType dtype;
  const auto out = encodeToRing(fp_arr, field, fxp_bits, &dtype);
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
  auto x_pt_type = getDecodeType(x.dtype());
  auto y_pt_type = getDecodeType(y.dtype());

  for (auto itr_x = flp_x.begin(), itr_y = flp_y.begin(); itr_x != flp_x.end();
       itr_x++, itr_y++) {
    DISPATCH_FLOAT_PT_TYPES(x_pt_type, "x_pt_type", [&]() {
      auto* ptr_x = reinterpret_cast<ScalarT*>(&*itr_x);
      DISPATCH_FLOAT_PT_TYPES(y_pt_type, "y_pt_type", [&]() {
        auto* ptr_y = reinterpret_cast<ScalarT*>(&*itr_y);
        *ptr_x = fn(*ptr_x, *ptr_y);
      });
    });
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

Value f_sine_p(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  return applyFloatingPointFn(ctx, in, [](float x) { return std::sin(x); });
}

Value f_cosine_p(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  return applyFloatingPointFn(ctx, in, [](float x) { return std::cos(x); });
}

Value f_erf_p(SPUContext* ctx, const Value& in) {
  SPU_TRACE_HAL_DISP(ctx, in);
  return applyFloatingPointFn(ctx, in, [](float x) { return std::erf(x); });
}

Value f_pow_p(SPUContext* ctx, const Value& x, const Value& y) {
  SPU_TRACE_HAL_DISP(ctx, x, y);
  return applyFloatingPointFn(ctx, x, y,
                              [](float a, float b) { return std::pow(a, b); });
}

}  // namespace spu::kernel::hal
