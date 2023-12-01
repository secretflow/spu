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

#include "libspu/mpc/cheetah/state.h"

namespace spu::mpc::cheetah {

NdArrayRef AndBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE_EQ(lhs.shape(), rhs.shape());

  int64_t numel = lhs.numel();
  NdArrayRef out(lhs.eltype(), lhs.shape());
  if (numel == 0) {
    return out;
  }

  int64_t nworker = InitOTState(ctx, numel);
  int64_t work_load = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  auto flat_lhs = lhs.reshape({lhs.numel()});
  auto flat_rhs = rhs.reshape({rhs.numel()});
  TiledDispatch(ctx, nworker, [&](int64_t job) {
    int64_t slice_bgn = std::min(numel, job * work_load);
    int64_t slice_end = std::min(numel, slice_bgn + work_load);
    if (slice_bgn == slice_end) {
      return;
    }

    auto out_slice = ctx->getState<CheetahOTState>()->get(job)->BitwiseAnd(
        flat_lhs.slice({slice_bgn}, {slice_end}, {1}),
        flat_rhs.slice({slice_bgn}, {slice_end}, {1}));
    std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                out_slice.elsize() * out_slice.numel());
  });

  return out;
}

}  // namespace spu::mpc::cheetah
