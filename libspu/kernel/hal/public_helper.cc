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

#include "libspu/kernel/hal/public_helper.h"

#include "libspu/core/encoding.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::kernel::hal {

NdArrayRef dump_public(SPUContext* ctx, const Value& v) {
  SPU_TRACE_HAL_DISP(ctx, v);
  SPU_ENFORCE(v.storage_type().isa<mpc::Pub2kTy>(), "got {}", v.storage_type());
  const auto field = v.storage_type().as<Ring2k>()->field();
  auto encoded = v.data().as(makeType<RingTy>(field));

  return decodeFromRing(encoded, v.dtype(), ctx->getFxpBits());
}

}  // namespace spu::kernel::hal
