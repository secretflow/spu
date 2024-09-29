// Copyright 2023 Ant Group Co., Ltd.
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

#pragma once

#include "libspu/core/context.h"
#include "libspu/core/memref.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/xt_helper.h"
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hal {

template <typename T>
xt::xarray<T> dump_public_as(SPUContext* ctx, const MemRef& in,
                             int64_t fxp_bits = 0) {
  std::vector<size_t> shape(in.shape().begin(), in.shape().end());
  xt::xarray<T> ret(shape, static_cast<T>(0));
  PtBufferView pt_ret(ret);
  auto pt_type = PtTypeToEnum<T>::value;
  if (pt_type == PT_F16 || pt_type == PT_F32 || pt_type == PT_F64) {
    kernel::hal::_decode_fp(ctx, in, &pt_ret, fxp_bits);
  } else {
    kernel::hal::_decode_int(ctx, in, &pt_ret);
  }
  return ret;
}

template <typename T>
std::vector<T> dump_public_as_vec(SPUContext* ctx, const MemRef& in,
                                  int64_t fxp_bits = 0) {
  std::vector<T> ret(in.numel(), static_cast<T>(0));
  PtBufferView pt_ret(ret);
  auto pt_type = PtTypeToEnum<T>::value;
  if (pt_type == PT_F16 || pt_type == PT_F32 || pt_type == PT_F64) {
    kernel::hal::_decode_fp(ctx, in, &pt_ret, fxp_bits);
  } else {
    kernel::hal::_decode_int(ctx, in, &pt_ret);
  }
  return ret;
}

template <typename T>
T getScalarValue(SPUContext* ctx, const MemRef& value) {
  SPU_ENFORCE(value.numel() == 1, "{} is not a scalar tensor.", value);
  SPU_ENFORCE(value.isPublic(), "{} is not a public value", value);

  const auto pvar =
      kernel::hal::dump_public_as<T>(ctx, value, ctx->getFxpBits());
  return pvar.front();
}

}  // namespace spu::kernel::hal
