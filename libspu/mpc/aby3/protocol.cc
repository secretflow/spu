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

#include "libspu/mpc/aby3/protocol.h"

#include "libspu/mpc/aby3/arithmetic.h"
#include "libspu/mpc/aby3/boolean.h"
#include "libspu/mpc/aby3/conversion.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc {

void regAby3Protocol(SPUContext* ctx,
                     const std::shared_ptr<yacl::link::Context>& lctx) {
  aby3::registerTypes();

  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // register arithmetic & binary kernels
  ctx->prot()->regKernel<aby3::P2A>();
  ctx->prot()->regKernel<aby3::V2A>();
  ctx->prot()->regKernel<aby3::A2P>();
  ctx->prot()->regKernel<aby3::A2V>();
  ctx->prot()->regKernel<aby3::NotA>();
  ctx->prot()->regKernel<aby3::AddAP>();
  ctx->prot()->regKernel<aby3::AddAA>();
  ctx->prot()->regKernel<aby3::MulAP>();
  ctx->prot()->regKernel<aby3::MulAA>();
  ctx->prot()->regKernel<aby3::MulA1B>();
  ctx->prot()->regKernel<aby3::MatMulAP>();
  ctx->prot()->regKernel<aby3::MatMulAA>();
  ctx->prot()->regKernel<aby3::LShiftA>();

#define ENABLE_PRECISE_ABY3_TRUNCPR
#ifdef ENABLE_PRECISE_ABY3_TRUNCPR
  ctx->prot()->regKernel<aby3::TruncAPr>();
#else
  ctx->prot()->regKernel<aby3::TruncA>();
#endif

  ctx->prot()->regKernel<aby3::MsbA2B>();

  ctx->prot()->regKernel<aby3::CommonTypeB>();
  ctx->prot()->regKernel<aby3::CastTypeB>();
  ctx->prot()->regKernel<aby3::B2P>();
  ctx->prot()->regKernel<aby3::P2B>();
  ctx->prot()->regKernel<aby3::A2B>();
  ctx->prot()->regKernel<aby3::B2ASelector>();
  // ctx->prot()->regKernel<aby3::B2AByOT>();
  // ctx->prot()->regKernel<aby3::B2AByPPA>();
  ctx->prot()->regKernel<aby3::AndBP>();
  ctx->prot()->regKernel<aby3::AndBB>();
  ctx->prot()->regKernel<aby3::XorBP>();
  ctx->prot()->regKernel<aby3::XorBB>();
  ctx->prot()->regKernel<aby3::LShiftB>();
  ctx->prot()->regKernel<aby3::RShiftB>();
  ctx->prot()->regKernel<aby3::ARShiftB>();
  ctx->prot()->regKernel<aby3::BitrevB>();
  ctx->prot()->regKernel<aby3::BitIntlB>();
  ctx->prot()->regKernel<aby3::BitDeintlB>();
  ctx->prot()->regKernel<aby3::RandA>();
}

std::unique_ptr<SPUContext> makeAby3Protocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regAby3Protocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
