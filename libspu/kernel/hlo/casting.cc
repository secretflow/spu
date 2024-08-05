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

#include "libspu/kernel/hlo/casting.h"

#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/type_cast.h"

namespace spu::kernel::hlo {

spu::Value Cast(SPUContext *ctx, const spu::Value &in, Visibility dst_vtype,
                DataType dst_dtype) {
  spu::Value ret = in;
  if (ret.vtype() != dst_vtype) {
    if (dst_vtype == VIS_PUBLIC) {
      ret = hal::reveal(ctx, ret);
    } else {
      ret = hal::seal(ctx, ret);
    }
  }
  if (ret.dtype() != dst_dtype) {
    ret = hal::dtype_cast(ctx, ret, dst_dtype);
  }
  return ret;
}

spu::Value Bitcast(SPUContext *ctx, const spu::Value &in, DataType dst_dtype) {
  return hal::bitcast(ctx, in, dst_dtype);
}

spu::Value Reveal(SPUContext *ctx, const spu::Value &in) {
  return hal::reveal(ctx, in);
}

spu::Value Seal(SPUContext *ctx, const spu::Value &in) {
  return hal::seal(ctx, in);
}

}  // namespace spu::kernel::hlo
