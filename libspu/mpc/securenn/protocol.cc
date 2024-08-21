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

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/securenn/arithmetic.h"
#include "libspu/mpc/securenn/boolean.h"
#include "libspu/mpc/securenn/conversion.h"
#include "libspu/mpc/securenn/type.h"
#include "libspu/mpc/standard_shape/protocol.h"

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

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic & binary kernels
  // ctx->prot()->addState<SecurennState>();

  ctx->prot()
      ->regKernel<
          securenn::P2A, securenn::A2P, securenn::A2V, securenn::V2A,         //
          securenn::NegateA,                                                  //
          securenn::AddAP, securenn::AddAA,                                   //
          securenn::MulAP, securenn::MulAA,                                   //
          securenn::MatMulAP, securenn::MatMulAA, securenn::MatMulAA_simple,  //
          securenn::LShiftA, securenn::LShiftB, securenn::RShiftB,
          securenn::ARShiftB,                //
          securenn::Msb, securenn::Msb_opt,  //
          securenn::TruncAPr,                //
          securenn::CommonTypeB, securenn::CommonTypeV, securenn::CastTypeB,
          securenn::B2P, securenn::P2B, securenn::A2B, securenn::Msb_a2b,
          /*securenn::B2A,*/ securenn::B2A_Randbit,  //
          securenn::AndBP, securenn::AndBB,          //
          securenn::XorBP, securenn::XorBB,          //
          securenn::BitrevB, securenn::BitIntlB, securenn::BitDeintlB,
          securenn::RandA>();
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
