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

#include "libspu/mpc/cheetah/protocol.h"

// FIXME: both emp-tools & openssl defines AES_KEY, hack the include order to
// avoid compiler error.
#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/common/prg_state.h"
//

#include "libspu/mpc/cheetah/arithmetic.h"
#include "libspu/mpc/cheetah/boolean.h"
#include "libspu/mpc/cheetah/conversion.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/standard_shape/protocol.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc {

void regCheetahProtocol(SPUContext* ctx,
                        const std::shared_ptr<yacl::link::Context>& lctx) {
  cheetah::registerTypes();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // add Z2k state.
  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // add Cheetah states
  ctx->prot()->addState<cheetah::CheetahMulState>(
      lctx, ctx->config().cheetah_2pc_config().enable_mul_lsb_error());
  ctx->prot()->addState<cheetah::CheetahDotState>(
      lctx, ctx->config().cheetah_2pc_config().disable_matmul_pack());
  ctx->prot()->addState<cheetah::CheetahOTState>(
      ctx->getClusterLevelMaxConcurrency(),
      ctx->config().cheetah_2pc_config().ot_kind());

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic & binary kernels
  ctx->prot()
      ->regKernel<cheetah::P2A, cheetah::A2P, cheetah::V2A, cheetah::A2V,   //
                  cheetah::B2P, cheetah::P2B, cheetah::A2B, cheetah::B2A,   //
                  cheetah::NegateA,                                         //
                  cheetah::AddAP, cheetah::AddAA,                           //
                  cheetah::MulAP, cheetah::MulAA, cheetah::MulAV,           //
                  cheetah::SquareA,                                         //
                  cheetah::MulA1B, cheetah::MulA1BV,                        //
                  cheetah::EqualAA, cheetah::EqualAP,                       //
                  cheetah::MatMulAP, cheetah::MatMulAA, cheetah::MatMulAV,  //
                  cheetah::MatMulVVS,                                       //
                  cheetah::LShiftA, cheetah::ARShiftB, cheetah::LShiftB,    //
                  cheetah::RShiftB,                                         //
                  cheetah::BitrevB,                                         //
                  cheetah::TruncA,                                          //
                  cheetah::MsbA2B,                                          //
                  cheetah::CommonTypeB, cheetah::CommonTypeV,               //
                  cheetah::CastTypeB, cheetah::AndBP, cheetah::AndBB,       //
                  cheetah::XorBP, cheetah::XorBB,                           //
                  cheetah::RandA>();
}

std::unique_ptr<SPUContext> makeCheetahProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  cheetah::registerTypes();

  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regCheetahProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
