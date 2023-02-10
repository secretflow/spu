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

#include "libspu/mpc/cheetah/conversion.h"

#include "libspu/core/trace.h"
#include "libspu/mpc/cheetah/object.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/semi2k/type.h"  // TODO: use cheetah type

namespace spu::mpc::cheetah {

ArrayRef B2A::proc(KernelEvalContext* ctx, const ArrayRef& x) const {
  SPU_TRACE_MPC_LEAF(ctx, x);
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto ty = makeType<semi2k::AShrTy>(field);
  return ctx->getState<CheetahOTState>()->get()->B2A(x).as(ty);
}

}  // namespace spu::mpc::cheetah
