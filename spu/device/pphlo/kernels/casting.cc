// Copyright 2022 Ant Group Co., Ltd.
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

#include "spu/device/pphlo/kernels/casting.h"

#include "spu/hal/polymorphic.h"
#include "spu/hal/type_cast.h"

namespace spu::device::pphlo::kernel {

hal::Value Cast(HalContext *ctx, const hal::Value &in, Visibility dst_vtype,
                DataType dst_dtype) {
  hal::Value ret = in;
  if (ret.vtype() != dst_vtype) {
    if (dst_vtype == VIS_PUBLIC) {
      ret = hal::reveal(ctx, ret);
    } else {
      ret = hal::p2s(ctx, ret);
    }
  }
  if (ret.dtype() != dst_dtype) {
    ret = hal::dtype_cast(ctx, ret, dst_dtype);
  }
  return ret;
}

hal::Value Bitcast(HalContext *ctx, const hal::Value &in, DataType dst_dtype,
                   size_t elsize) {
  return hal::bitcast(ctx, in, dst_dtype, elsize);
}

hal::Value Reveal(HalContext *ctx, const hal::Value &in) {
  return hal::reveal(ctx, in);
}

hal::Value Seal(HalContext *ctx, const hal::Value &in) {
  return hal::p2s(ctx, in);
}

} // namespace spu::device::pphlo::kernel