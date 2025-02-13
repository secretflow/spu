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
#include "libspu/mpc/standard_shape/protocol.h"

namespace spu::mpc {

void regSpdz2kProtocol(SPUContext* ctx,
                       const std::shared_ptr<yacl::link::Context>& lctx) {
  spdz2k::registerTypes();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // add Z2k state.
  ctx->prot()->addState<Z2kState>(ctx->config().field);

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic kernels
  ctx->prot()->addState<Spdz2kState>(ctx->config(), lctx);
  ctx->prot()
      ->regKernel<spdz2k::P2A, spdz2k::A2P, spdz2k::A2V, spdz2k::V2A,
                  spdz2k::NegateA, spdz2k::AddAP, spdz2k::AddAA, spdz2k::MulAP,
                  spdz2k::MulAA, spdz2k::MatMulAP, spdz2k::MatMulAA,
                  spdz2k::LShiftA, spdz2k::TruncA, spdz2k::RandA>();

  // register boolean kernels
  ctx->prot()
      ->regKernel<spdz2k::CommonTypeB, spdz2k::B2P, spdz2k::P2B, spdz2k::NotB,
                  spdz2k::BitrevB, spdz2k::XorBB, spdz2k::XorBP, spdz2k::AndBB,
                  spdz2k::AndBP, spdz2k::LShiftB, spdz2k::RShiftB,
                  spdz2k::ARShiftB, spdz2k::BitIntlB, spdz2k::BitDeintlB>();

  // register conversion kernels
  ctx->prot()
      ->regKernel<spdz2k::AddBB, spdz2k::AddBP, spdz2k::BitLTBB,
                  spdz2k::BitLEBB, spdz2k::A2Bit, spdz2k::Bit2A, spdz2k::MSB,
                  spdz2k::A2B, spdz2k::B2A>();
}

std::unique_ptr<SPUContext> makeSpdz2kProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regSpdz2kProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
