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
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
constexpr size_t kMinWorkSize = 5000;

NdArrayRef AndBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE_EQ(lhs.shape(), rhs.shape());

  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  size_t n = lhs.numel();
  size_t nworker =
      std::min(ot_state->parallel_size(), CeilDiv(n, kMinWorkSize));
  size_t work_load = nworker == 0 ? 0 : CeilDiv(n, nworker);
  for (size_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }

  NdArrayRef z(lhs.eltype(), lhs.shape());
  auto flat_lhs = flatten(lhs);
  auto flat_rhs = flatten(rhs);
  yacl::parallel_for(0, nworker, 1, [&](size_t bgn, size_t end) {
    for (size_t job = bgn; job < end; ++job) {
      size_t slice_bgn = std::min(n, job * work_load);
      size_t slice_end = std::min(n, slice_bgn + work_load);
      if (slice_bgn == slice_end) {
        break;
      }
      auto out_slice =
          ot_state->get(job)->BitwiseAnd(flat_lhs.slice(slice_bgn, slice_end),
                                         flat_rhs.slice(slice_bgn, slice_end));
      std::memcpy(&z.at(slice_bgn), &out_slice.at(0),
                  out_slice.elsize() * out_slice.numel());
    }
  });
  return z.as(lhs.eltype());
}

}  // namespace spu::mpc::cheetah
