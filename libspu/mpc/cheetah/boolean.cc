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

#include "libspu/core/trace.h"
#include "libspu/mpc/cheetah/object.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
  SPU_ENFORCE_EQ(lhs.numel(), rhs.numel());

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* const shareType = lhs.eltype().as<semi2k::BShrTy>();
  const size_t size = lhs.numel();

  auto* comm = ctx->getState<Communicator>();
  auto ot_prot = ctx->getState<CheetahOTState>()->get();
  // Create `size` AND triples, each contains shareType->nbits() bits
  // and stored in the low-end bits of the ring element.
  auto [a, b, c] = ot_prot->AndTriple(field, size, shareType->nbits());
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
