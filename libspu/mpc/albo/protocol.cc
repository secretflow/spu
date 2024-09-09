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

#include "libspu/mpc/albo/protocol.h"

#include "libspu/mpc/albo/arithmetic.h"
#include "libspu/mpc/albo/boolean.h"
#include "libspu/mpc/albo/conversion.h"
#include "libspu/mpc/albo/permute.h"
#include "libspu/mpc/albo/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/standard_shape/protocol.h"

#define ENABLE_PRECISE_ALBO_TRUNCPR

namespace spu::mpc {

void regAlboProtocol(SPUContext* ctx,
                     const std::shared_ptr<yacl::link::Context>& lctx) {
  albo::registerTypes();

  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic & binary kernels
  ctx->prot()
      ->regKernel<                                              //
          albo::P2A, albo::V2A, albo::A2P, albo::A2V,           // Conversions
          albo::B2P, albo::P2B, albo::A2B,                      // Conversion2
          albo::B2ASelector, /*albo::B2AByOT, albo::B2AByPPA*/  // B2A
          albo::CastTypeB,                                      // Cast
          albo::NotA,                                           // Not
          albo::AddAP, albo::AddAA,                             // Add
          albo::MulAP, albo::MulAA, albo::MulA1B,               // Mul
          albo::MatMulAP, albo::MatMulAA,                       // MatMul
          albo::LShiftA, albo::LShiftB,                         // LShift
          albo::RShiftB, albo::ARShiftB,                        // (A)Rshift
          albo::MsbA2B,                                         // MSB
          albo::EqualAA, albo::EqualAP,                         // Equal
          albo::CommonTypeB, albo::CommonTypeV,                 // CommonType
          albo::AndBP, albo::AndBB,                             // And
          albo::XorBP, albo::XorBB,                             // Xor
          albo::BitrevB,                                        // bitreverse
          albo::BitIntlB, albo::BitDeintlB,  // bit(de)interleave
          albo::RandA,                       // rand
#ifdef ENABLE_PRECISE_ALBO_TRUNCPR
          albo::TruncAPr,  // Trunc
#else
          albo::TruncA,
#endif
          albo::RandPermM, albo::PermAM, albo::PermAP, albo::InvPermAM,  // perm
          albo::InvPermAP                                                // perm
          >();
}

std::unique_ptr<SPUContext> makeAlboProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regAlboProtocol(ctx.get(), lctx);

  if (ctx->getState<Communicator>()->getRank() == 0) std::cout << "make albo protocol." << std::endl;

  return ctx;
}

}  // namespace spu::mpc
