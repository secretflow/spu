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

#include "libspu/core/context.h"
#include "libspu/core/trace.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/prot_wrapper.h"  // vtype_cast
#include "libspu/kernel/hal/ring.h"

namespace spu::kernel::hal {

// TODO: move seal/reveal into a new header file.
MemRef seal(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);
  if (x.isPrivate()) {
    return _v2s(ctx, x);
  }
  return _p2s(ctx, x);
}

MemRef reveal(SPUContext* ctx, const MemRef& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);
  if (x.isPrivate()) {
    return _v2p(ctx, x);
  }
  return _s2p(ctx, x);
}

MemRef vtype_cast(SPUContext* ctx, const MemRef& in, Visibility to_vis) {
  SPU_TRACE_HAL_DISP(ctx, in, to_vis);
  MemRef ret = in;
  if (in.vtype() != to_vis) {
    if (to_vis == VIS_PUBLIC) {
      if (in.isPrivate()) {
        return _v2p(ctx, in);
      }
      return _s2p(ctx, in);
    } else {
      if (in.isPrivate()) {
        return _v2s(ctx, in);
      }
      return _p2s(ctx, in);
    }
  }
  return ret;
}

}  // namespace spu::kernel::hal
