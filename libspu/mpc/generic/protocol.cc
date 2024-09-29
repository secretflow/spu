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

#include "libspu/mpc/generic/protocol.h"

#include "libspu/core/context.h"
#include "libspu/mpc/generic/kernels.h"

namespace spu::mpc {

void regGenericKernels(SPUContext* ctx) {
  ctx->prot()->regKernel<generic::Broadcast>();
  ctx->prot()->regKernel<generic::Reshape>();
  ctx->prot()->regKernel<generic::ExtractSlice>();
  ctx->prot()->regKernel<generic::InsertSlice>();
  ctx->prot()->regKernel<generic::Transpose>();
  ctx->prot()->regKernel<generic::Fill>();
  ctx->prot()->regKernel<generic::Pad>();
  ctx->prot()->regKernel<generic::Concate>();
  ctx->prot()->regKernel<generic::Reverse>();
  ctx->prot()->regKernel<generic::LShift>();
  ctx->prot()->regKernel<generic::RShift>();
  ctx->prot()->regKernel<generic::ARShift>();
  ctx->prot()->regKernel<generic::BitDeintl>();
  ctx->prot()->regKernel<generic::BitIntl>();
  ctx->prot()->regKernel<generic::Bitrev>();
}

}  // namespace spu::mpc
