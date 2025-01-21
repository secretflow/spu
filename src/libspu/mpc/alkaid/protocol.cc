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

#include "libspu/mpc/alkaid/protocol.h"

#include "libspu/mpc/alkaid/arithmetic.h"
#include "libspu/mpc/alkaid/boolean.h"
#include "libspu/mpc/alkaid/conversion.h"
#include "libspu/mpc/alkaid/oram.h"
#include "libspu/mpc/alkaid/permute.h"
#include "libspu/mpc/alkaid/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/standard_shape/protocol.h"

#define ENABLE_PRECISE_ALKAID_TRUNCPR

namespace spu::mpc {

void regAlkaidProtocol(SPUContext* ctx,
                     const std::shared_ptr<yacl::link::Context>& lctx) {
  alkaid::registerTypes();

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
          alkaid::P2A, alkaid::V2A, alkaid::A2P, alkaid::A2V,         // Conversions
          alkaid::B2P, alkaid::P2B, alkaid::A2B,                      // Conversion2
          alkaid::B2ASelector, /*alkaid::B2AByOT, alkaid::B2AByPPA*/  // B2A
          alkaid::CastTypeB,                                          // Cast
          alkaid::NegateA,                                            // Negate
          alkaid::AddAP, alkaid::AddAA,                               // Add
          alkaid::MulAP, alkaid::MulAA, alkaid::MulA1B,               // Mul
          alkaid::MatMulAP, alkaid::MatMulAA,                         // MatMul
          alkaid::LShiftA, alkaid::LShiftB,                           // LShift
          alkaid::RShiftB, alkaid::ARShiftB,                          // (A)Rshift
          alkaid::MsbA2B,                                             // MSB
          alkaid::EqualAA, alkaid::EqualAP,                           // Equal
          alkaid::CommonTypeB, alkaid::CommonTypeV,                   // CommonType
          alkaid::AndBP, alkaid::AndBB,                               // And
          alkaid::XorBP, alkaid::XorBB,                               // Xor
          alkaid::BitrevB,                                            // bitreverse
          alkaid::BitIntlB, alkaid::BitDeintlB,                       // bit(de)interleave
          alkaid::RandA,                                              // rand
#ifdef ENABLE_PRECISE_ALKAID_TRUNCPR
          alkaid::TruncAPr,  // Trunc
#else
          alkaid::TruncA,
#endif
          alkaid::OramOneHotAA, alkaid::OramOneHotAP, alkaid::OramReadOA,       // oram
          alkaid::OramReadOP,                                                   // oram
          alkaid::RandPermM, alkaid::PermAM, alkaid::PermAP, alkaid::InvPermAM, // perm
          alkaid::InvPermAP                                                     // perm                                             // perm
          >();
}

std::unique_ptr<SPUContext> makeAlkaidProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regAlkaidProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
