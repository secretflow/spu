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

#include "libspu/mpc/standard_shape/protocol.h"

#include "libspu/core/context.h"
#include "libspu/mpc/standard_shape/kernels.h"

namespace spu::mpc {

void regStandardShapeOps(SPUContext* ctx) {
  ctx->prot()->regKernel<standard_shape::Broadcast>();
  ctx->prot()->regKernel<standard_shape::Reshape>();
  ctx->prot()->regKernel<standard_shape::ExtractSlice>();
  ctx->prot()->regKernel<standard_shape::UpdateSlice>();
  ctx->prot()->regKernel<standard_shape::Transpose>();
  ctx->prot()->regKernel<standard_shape::Fill>();
  ctx->prot()->regKernel<standard_shape::Pad>();
  ctx->prot()->regKernel<standard_shape::Concate>();
  ctx->prot()->regKernel<standard_shape::Reverse>();
}

}  // namespace spu::mpc
