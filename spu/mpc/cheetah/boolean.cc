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

#include "spu/mpc/cheetah/boolean.h"

#include "spu/core/profile.h"
#include "spu/mpc/cheetah/object.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/interfaces.h"
#include "spu/mpc/kernel.h"
#include "spu/mpc/semi2k/type.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::cheetah {

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<CheetahState>()->beaver();

  // generate beaver and triple.
  auto [a, b, c] = beaver->And(field, lhs.numel());

  // open x^a, y^b
  auto res =
      vectorize({ring_xor(lhs, a), ring_xor(rhs, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::XOR, s, kBindName);
      });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci ^ ((X ^ A) & Bi) ^ ((Y ^ B) & Ai) ^ <(X ^ A) & (Y ^ B)>
  auto z = ring_xor(ring_xor(ring_and(x_a, b), ring_and(y_b, a)), c);
  if (comm->getRank() == 0) {
    ring_xor_(z, ring_and(x_a, y_b));
  }

  return z.as(lhs.eltype());
}

}  // namespace spu::mpc::cheetah
