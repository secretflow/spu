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

#include "libspu/mpc/spdz2k/protocol.h"

#include "libspu/core/context.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdz2k/arithmetic.h"
#include "libspu/mpc/spdz2k/boolean.h"
#include "libspu/mpc/spdz2k/conversion.h"
#include "libspu/mpc/spdz2k/state.h"
#include "libspu/mpc/spdz2k/type.h"

namespace spu::mpc {

void regSpdz2kProtocol(SPUContext* ctx,
                       const std::shared_ptr<yacl::link::Context>& lctx) {
  spdz2k::registerTypes();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // register arithmetic kernels
  ctx->prot()->addState<Spdz2kState>(ctx->config(), lctx);
  ctx->prot()->regKernel<spdz2k::P2A>();
  ctx->prot()->regKernel<spdz2k::A2P>();
  ctx->prot()->regKernel<spdz2k::A2V>();
  ctx->prot()->regKernel<spdz2k::V2A>();
  ctx->prot()->regKernel<spdz2k::NotA>();
  ctx->prot()->regKernel<spdz2k::AddAP>();
  ctx->prot()->regKernel<spdz2k::AddAA>();
  ctx->prot()->regKernel<spdz2k::MulAP>();
  ctx->prot()->regKernel<spdz2k::MulAA>();
  ctx->prot()->regKernel<spdz2k::MatMulAP>();
  ctx->prot()->regKernel<spdz2k::MatMulAA>();
  ctx->prot()->regKernel<spdz2k::LShiftA>();
  ctx->prot()->regKernel<spdz2k::TruncA>();
  ctx->prot()->regKernel<spdz2k::RandA>();

  // register boolean kernels
  ctx->prot()->regKernel<spdz2k::CommonTypeB>();
  ctx->prot()->regKernel<spdz2k::B2P>();
  ctx->prot()->regKernel<spdz2k::P2B>();

  ctx->prot()->regKernel<spdz2k::NotB>();
  ctx->prot()->regKernel<spdz2k::BitrevB>();
  ctx->prot()->regKernel<spdz2k::XorBB>();
  ctx->prot()->regKernel<spdz2k::XorBP>();
  ctx->prot()->regKernel<spdz2k::AndBB>();
  ctx->prot()->regKernel<spdz2k::AndBP>();
  ctx->prot()->regKernel<spdz2k::LShiftB>();
  ctx->prot()->regKernel<spdz2k::RShiftB>();
  ctx->prot()->regKernel<spdz2k::ARShiftB>();
  ctx->prot()->regKernel<spdz2k::BitIntlB>();
  ctx->prot()->regKernel<spdz2k::BitDeintlB>();

  // register conversion kernels
  ctx->prot()->regKernel<spdz2k::AddBB>();
  ctx->prot()->regKernel<spdz2k::AddBP>();
  ctx->prot()->regKernel<spdz2k::BitLTBB>();
  ctx->prot()->regKernel<spdz2k::BitLEBB>();
  ctx->prot()->regKernel<spdz2k::A2Bit>();
  ctx->prot()->regKernel<spdz2k::Bit2A>();
  ctx->prot()->regKernel<spdz2k::MSB>();
  ctx->prot()->regKernel<spdz2k::A2B>();
  ctx->prot()->regKernel<spdz2k::B2A>();
}

std::unique_ptr<SPUContext> makeSpdz2kProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regSpdz2kProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
