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

#include "libspu/core/context.h"
#include "libspu/core/encoding.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/common/pv_gfmp.h"

namespace spu::kernel::hal {

NdArrayRef dump_public(SPUContext* ctx, const Value& v) {
  SPU_TRACE_HAL_DISP(ctx, v);

  if (ctx->config().protocol == ProtocolKind::SHAMIR) {
    SPU_ENFORCE(v.storage_type().isa<mpc::PubGfmpTy>(), "got {}",
                v.storage_type());

    const auto field = v.storage_type().as<GfmpTy>()->field();
    auto encoded = v.data().as(makeType<GfmpTy>(field));

    const PtType pt_type = getDecodeType(v.dtype());
    NdArrayRef dst(makePtType(pt_type), v.shape());
    PtBufferView pv(static_cast<void*>(dst.data()), pt_type, dst.shape(),
                    dst.strides());
    decodeFromGfmp(encoded, v.dtype(), ctx->getFxpBits(), &pv);
    return dst;
  } else {
    SPU_ENFORCE(v.storage_type().isa<mpc::Pub2kTy>(), "got {}",
                v.storage_type());

    const auto field = v.storage_type().as<Ring2k>()->field();
    auto encoded = v.data().as(makeType<RingTy>(field));

    const PtType pt_type = getDecodeType(v.dtype());
    NdArrayRef dst(makePtType(pt_type), v.shape());
    PtBufferView pv(static_cast<void*>(dst.data()), pt_type, dst.shape(),
                    dst.strides());

    decodeFromRing(encoded, v.dtype(), ctx->getFxpBits(), &pv);
    return dst;
  }
}

bool getBooleanValue(SPUContext* ctx, const spu::Value& value) {
  SPU_ENFORCE(value.numel() == 1, "Condition value must be a scalar tensor.");
  SPU_ENFORCE(value.dtype() == DT_I1, "Expect bool, got {}", value.dtype());
  SPU_ENFORCE(value.isPublic(), "Expect public value");

  const auto public_val = kernel::hal::dump_public_as<bool>(ctx, value);
  return public_val.front();
}

}  // namespace spu::kernel::hal
