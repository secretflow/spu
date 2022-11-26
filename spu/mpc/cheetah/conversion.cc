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

#include "spu/mpc/cheetah/conversion.h"

#include "spu/core/trace.h"
#include "spu/core/vectorize.h"
#include "spu/mpc/api.h"
#include "spu/mpc/cheetah/object.h"
#include "spu/mpc/cheetah/utils.h"
#include "spu/mpc/semi2k/type.h"
#include "spu/mpc/util/circuits.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::cheetah {

ArrayRef B2A::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_TRACE_MPC_LEAF(ctx, x);

  auto primitives =
      ctx->caller()->getState<CheetahState>()->beaver()->OTPrimitives();
  auto shareType = x.eltype().as<semi2k::BShrTy>();
  const auto field = x.eltype().as<Ring2k>()->field();
  size_t size = x.numel();
  ArrayRef y(makeType<RingTy>(field), size);

  if (shareType->nbits() == 1) {
    DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
      using U = ring2k_t;
      auto x_buf = x.getOrCreateCompactBuf();
      auto y_buf = y.getOrCreateCompactBuf();
      yacl::Buffer buf(size);
      cast(buf.data<uint8_t>(), x_buf->data<U>(), size);

      primitives->nonlinear()->b2a(y_buf->data<U>(), buf.data<uint8_t>(), size,
                                   sizeof(U) * 8);
      primitives->nonlinear()->flush();
    });
  } else {
    DISPATCH_ALL_FIELDS(field, kBindName, [&]() {
      using U = ring2k_t;
      auto x_buf = x.getOrCreateCompactBuf();
      auto y_buf = y.getOrCreateCompactBuf();

      primitives->nonlinear()->b2a_full(y_buf->data<U>(), x_buf->data<U>(),
                                        size, shareType->nbits());
      primitives->nonlinear()->flush();
    });
  }
  return y.as(makeType<semi2k::AShrTy>(field));
}

}  // namespace spu::mpc::cheetah
