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

#include "libspu/mpc/cheetah/boolean.h"

#include "libspu/core/memref.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"

namespace spu::mpc::cheetah {

MemRef AndBB::proc(KernelEvalContext* ctx, const MemRef& lhs,
                   const MemRef& rhs) const {
  SPU_ENFORCE_EQ(lhs.shape(), rhs.shape());

  int64_t numel = lhs.numel();
  if (numel == 0) {
    return MemRef(lhs.eltype(), lhs.shape());
  }

  return TiledDispatchOTFunc(
             ctx, lhs, rhs,
             [&](const MemRef& input0, const MemRef& input1,
                 const std::shared_ptr<BasicOTProtocols>& base_ot) {
               return base_ot->BitwiseAnd(input0, input1);
             })
      .as(lhs.eltype());
}

}  // namespace spu::mpc::cheetah
