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

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/arithmetic.h"
#include "libspu/mpc/semi2k/boolean.h"
#include "libspu/mpc/semi2k/conversion.h"
#include "libspu/mpc/semi2k/exp.h"
#include "libspu/mpc/semi2k/permute.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/standard_shape/protocol.h"

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

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic & binary kernels
  ctx->prot()->addState<Semi2kState>(ctx->config(), lctx);
  ctx->prot()
      ->regKernel<
          semi2k::P2A, semi2k::A2P, semi2k::A2V, semi2k::V2A,                //
          semi2k::NegateA,                                                   //
          semi2k::AddAP, semi2k::AddAA,                                      //
          semi2k::MulAP, semi2k::MulAA, semi2k::SquareA,                     //
          semi2k::MatMulAP, semi2k::MatMulAA,                                //
          semi2k::LShiftA, semi2k::LShiftB, semi2k::RShiftB,                 //
          semi2k::ARShiftB,                                                  //
          semi2k::CommonTypeB, semi2k::CommonTypeV, semi2k::CastTypeB,       //
          semi2k::B2P, semi2k::P2B,                                          //
          semi2k::A2B, semi2k::B2A_Randbit, semi2k::B2A_Disassemble,         //
          semi2k::AndBP, semi2k::AndBB, semi2k::XorBP, semi2k::XorBB,        //
          semi2k::BitrevB,                                                   //
          semi2k::BitIntlB, semi2k::BitDeintlB,                              //
          semi2k::RandA, semi2k::RandPermM, semi2k::PermAM, semi2k::PermAP,  //
          semi2k::InvPermAM, semi2k::InvPermAP, semi2k::InvPermAV,           //
          semi2k::EqualAA, semi2k::EqualAP,                                  //
          semi2k::BeaverCacheKernel>();

  if (ctx->config().trunc_allow_msb_error()) {
    ctx->prot()->regKernel<semi2k::TruncA>();
  } else {
    ctx->prot()->regKernel<semi2k::TruncAPr>();
  }

  if (lctx->WorldSize() == 2) {
    ctx->prot()->regKernel<semi2k::MsbA2B>();

    // only supports 2pc fm128 for now
    if (ctx->getField() == FieldType::FM128 &&
        ctx->config().experimental_enable_exp_prime()) {
      ctx->prot()->regKernel<semi2k::ExpA>();
    }
  }
  // ctx->prot()->regKernel<semi2k::B2A>();
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
