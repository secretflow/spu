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

#include "libspu/mpc/semi2k/protocol.h"

#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/arithmetic.h"
#include "libspu/mpc/semi2k/boolean.h"
#include "libspu/mpc/semi2k/conversion.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"

namespace spu::mpc {

void regSemi2kProtocol(SPUContext* ctx,
                       const std::shared_ptr<yacl::link::Context>& lctx) {
  semi2k::registerTypes();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // add Z2k state.
  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // register arithmetic & binary kernels
  ctx->prot()->addState<Semi2kState>(ctx->config(), lctx);
  ctx->prot()->regKernel<semi2k::P2A>();
  ctx->prot()->regKernel<semi2k::A2P>();
  ctx->prot()->regKernel<semi2k::A2V>();
  ctx->prot()->regKernel<semi2k::V2A>();
  ctx->prot()->regKernel<semi2k::NotA>();
  ctx->prot()->regKernel<semi2k::AddAP>();
  ctx->prot()->regKernel<semi2k::AddAA>();
  ctx->prot()->regKernel<semi2k::MulAP>();
  ctx->prot()->regKernel<semi2k::MulAA>();
  ctx->prot()->regKernel<semi2k::MatMulAP>();
  ctx->prot()->regKernel<semi2k::MatMulAA>();
  ctx->prot()->regKernel<semi2k::LShiftA>();
  if (ctx->config().trunc_allow_msb_error()) {
    ctx->prot()->regKernel<semi2k::TruncA>();
  } else {
    ctx->prot()->regKernel<semi2k::TruncAPr>();
  }

  ctx->prot()->regKernel<semi2k::CommonTypeB>();
  ctx->prot()->regKernel<semi2k::CastTypeB>();
  ctx->prot()->regKernel<semi2k::B2P>();
  ctx->prot()->regKernel<semi2k::P2B>();
  ctx->prot()->regKernel<semi2k::A2B>();

  if (lctx->WorldSize() == 2) {
    ctx->prot()->regKernel<semi2k::MsbA2B>();
  }
  // ctx->prot()->regKernel<semi2k::B2A>();
  ctx->prot()->regKernel<semi2k::B2A_Randbit>();
  ctx->prot()->regKernel<semi2k::AndBP>();
  ctx->prot()->regKernel<semi2k::AndBB>();
  ctx->prot()->regKernel<semi2k::XorBP>();
  ctx->prot()->regKernel<semi2k::XorBB>();
  ctx->prot()->regKernel<semi2k::LShiftB>();
  ctx->prot()->regKernel<semi2k::RShiftB>();
  ctx->prot()->regKernel<semi2k::ARShiftB>();
  ctx->prot()->regKernel<semi2k::BitrevB>();
  ctx->prot()->regKernel<semi2k::BitIntlB>();
  ctx->prot()->regKernel<semi2k::BitDeintlB>();
  ctx->prot()->regKernel<semi2k::RandA>();
}

std::unique_ptr<SPUContext> makeSemi2kProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  semi2k::registerTypes();

  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regSemi2kProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
