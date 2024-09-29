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

#include "libspu/kernel/hal/constants.h"

#include "libspu/core/encoding.h"
#include "libspu/core/memref.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/trace.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::kernel::hal {
namespace {

// make a public typed value.
//
// FIXME: this is a abstraction leakage, we should NOT invoke Pub2kTy directly.
MemRef make_pub2k(SPUContext* ctx, const PtBufferView& bv) {
  SPU_TRACE_HAL_DISP(ctx, bv);

  const auto fxp_bits = ctx->getFxpBits();

  MemRef encoded(makeType<mpc::Pub2kTy>(GetEncodedType(bv.pt_type)), bv.shape);

  encodeToRing(bv, encoded, fxp_bits);

  return encoded;
}

}  // namespace

MemRef constant(SPUContext* ctx, PtBufferView init, const Shape& shape) {
  SPU_TRACE_HAL_DISP(ctx, init, shape);

  auto result = make_pub2k(ctx, init);

  if (shape.numel() == 0) {
    return MemRef(nullptr, result.eltype(), shape);
  }

  // If view shape is same as destination shape, just make public
  if (shape.isScalar() || shape == init.shape) {
    return result;
  }

  // Same calcNumel but shape is different, do a reshape
  if (init.shape.numel() == shape.numel()) {
    return result.reshape(shape);
  }

  // Other, do a broadcast, let broadcast handles the sanity check
  SPU_ENFORCE(init.shape.numel() <= shape.numel());
  return result.broadcast_to(shape, {});
}

MemRef zeros(SPUContext* ctx, PtType pt_type, const Shape& shape) {
  if (pt_type == PT_F32 || pt_type == PT_F64) {
    return constant(ctx, 0.0F, shape);
  } else {
    return constant(ctx, static_cast<uint8_t>(0), shape);
  }
}

MemRef iota(SPUContext* ctx, PtType pt_type, int64_t numel) {
  return DISPATCH_ALL_NONE_BOOL_PT_TYPES(pt_type, [&]() {
    std::vector<ScalarT> arr(numel);
    std::iota(arr.begin(), arr.end(), 0);
    return constant(ctx, arr, {numel});
  });
}

}  // namespace spu::kernel::hal
