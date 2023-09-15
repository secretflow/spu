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

#include "yacl/utils/parallel.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  std::vector<NdArrayRef> bshrs;
  const auto bty = makeType<BShrTy>(field);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto [r0, r1] =
        prg_state->genPrssPair(field, x.shape(), PrgState::GenPrssCtrl::Both);
    auto b = ring_xor(r0, r1).as(bty);

    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }
    bshrs.push_back(b.as(bty));
  }

  NdArrayRef res = vreduce(bshrs.begin(), bshrs.end(),
                           [&](const NdArrayRef& xx, const NdArrayRef& yy) {
                             return wrap_add_bb(ctx->sctx(), xx, yy);
                           });
  return res.as(bty);
}

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  size_t n = x.numel();
  size_t nworker = InitOTState(ctx, n);
  size_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto flatten_x = x.reshape({static_cast<int64_t>(n)});
  NdArrayRef out(x.eltype(), x.shape());

  yacl::parallel_for(0, nworker, 1, [&](int64_t bgn, int64_t end) {
    for (int64_t job = bgn; job < end; ++job) {
      auto slice_bgn = std::min<int64_t>(n, job * work_load);
      auto slice_end = std::min<int64_t>(n, slice_bgn + work_load);
      if (slice_bgn == slice_end) {
        break;
      }
      auto out_slice = ctx->getState<CheetahOTState>()->get(job)->B2A(
          flatten_x.slice({slice_bgn}, {slice_end}, {1}));
      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.elsize() * out_slice.numel());
    }
  });

  return out.as(makeType<AShrTy>(field));
}

}  // namespace spu::mpc::cheetah
