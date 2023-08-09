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

#include "libspu/mpc/securenn/protocol.h"

#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/securenn/arithmetic.h"
#include "libspu/mpc/securenn/boolean.h"
#include "libspu/mpc/securenn/conversion.h"
#include "libspu/mpc/securenn/state.h"
#include "libspu/mpc/securenn/type.h"

namespace spu::mpc {

void regSecurennProtocol(SPUContext* ctx,
                         const std::shared_ptr<yacl::link::Context>& lctx) {
  securenn::registerTypes();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // add Z2k state.
  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // register arithmetic & binary kernels
  // ctx->prot()->addState<SecurennState>();
  ctx->prot()->regKernel<securenn::P2A>();
  ctx->prot()->regKernel<securenn::A2P>();
  ctx->prot()->regKernel<securenn::A2V>();
  ctx->prot()->regKernel<securenn::V2A>();
  ctx->prot()->regKernel<securenn::NotA>();
  ctx->prot()->regKernel<securenn::AddAP>();
  ctx->prot()->regKernel<securenn::AddAA>();
  ctx->prot()->regKernel<securenn::MulAP>();
  ctx->prot()->regKernel<securenn::MulAA>();
  ctx->prot()->regKernel<securenn::MatMulAP>();
  ctx->prot()->regKernel<securenn::MatMulAA>();
  ctx->prot()->regKernel<securenn::MatMulAA_simple>();
  ctx->prot()->regKernel<securenn::LShiftA>();
  ctx->prot()->regKernel<securenn::Msb>();
  ctx->prot()->regKernel<securenn::Msb_opt>();
  ctx->prot()->regKernel<securenn::TruncAPr>();

  ctx->prot()->regKernel<securenn::CommonTypeB>();
  ctx->prot()->regKernel<securenn::CastTypeB>();
  ctx->prot()->regKernel<securenn::B2P>();
  ctx->prot()->regKernel<securenn::P2B>();
  ctx->prot()->regKernel<securenn::A2B>();

  ctx->prot()->regKernel<securenn::Msb_a2b>();
  // ctx->prot()->regKernel<securenn::B2A>();
  ctx->prot()->regKernel<securenn::B2A_Randbit>();
  ctx->prot()->regKernel<securenn::AndBP>();
  ctx->prot()->regKernel<securenn::AndBB>();
  ctx->prot()->regKernel<securenn::XorBP>();
  ctx->prot()->regKernel<securenn::XorBB>();
  ctx->prot()->regKernel<securenn::LShiftB>();
  ctx->prot()->regKernel<securenn::RShiftB>();
  ctx->prot()->regKernel<securenn::ARShiftB>();
  ctx->prot()->regKernel<securenn::BitrevB>();
  ctx->prot()->regKernel<securenn::BitIntlB>();
  ctx->prot()->regKernel<securenn::BitDeintlB>();
  ctx->prot()->regKernel<securenn::RandA>();
}

std::unique_ptr<SPUContext> makeSecurennProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  securenn::registerTypes();

  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regSecurennProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
