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

#include "libspu/mpc/alkaid_boolean/protocol.h"

#include "libspu/mpc/alkaid_boolean/arithmetic.h"
#include "libspu/mpc/alkaid_boolean/boolean.h"
#include "libspu/mpc/alkaid_boolean/conversion.h"
#include "libspu/mpc/alkaid_boolean/permute.h"
#include "libspu/mpc/alkaid_boolean/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/standard_shape/protocol.h"

#define ENABLE_PRECISE_ALKAID_TRUNCPR

namespace spu::mpc {

void regAlkaidBooleanProtocol(SPUContext* ctx,
                     const std::shared_ptr<yacl::link::Context>& lctx) {
  alkaid_boolean::registerTypes();

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
          alkaid_boolean::P2A, alkaid_boolean::V2A, alkaid_boolean::A2P, alkaid_boolean::A2V,           // Conversions
          alkaid_boolean::B2P, alkaid_boolean::P2B, alkaid_boolean::A2B,                      // Conversion2
          alkaid_boolean::B2ASelector, /*alkaid_boolean::B2AByOT, alkaid_boolean::B2AByPPA*/  // B2A
          alkaid_boolean::CastTypeB,                                      // Cast
          alkaid_boolean::NotA,                                           // Not
          alkaid_boolean::AddAP, alkaid_boolean::AddAA,                             // Add
          alkaid_boolean::MulAP, alkaid_boolean::MulAA, alkaid_boolean::MulA1B,               // Mul
          alkaid_boolean::MatMulAP, alkaid_boolean::MatMulAA,                       // MatMul
          alkaid_boolean::LShiftA, alkaid_boolean::LShiftB,                         // LShift
          alkaid_boolean::RShiftB, alkaid_boolean::ARShiftB,                        // (A)Rshift
          alkaid_boolean::MsbA2B,                                         // MSB
          alkaid_boolean::EqualAA, alkaid_boolean::EqualAP,                         // Equal
          alkaid_boolean::CommonTypeB, alkaid_boolean::CommonTypeV,                 // CommonType
          alkaid_boolean::AndBP, alkaid_boolean::AndBB,                             // And
          alkaid_boolean::XorBP, alkaid_boolean::XorBB,                             // Xor
          alkaid_boolean::BitrevB,                                        // bitreverse
          alkaid_boolean::BitIntlB, alkaid_boolean::BitDeintlB,  // bit(de)interleave
          alkaid_boolean::RandA,                       // rand
#ifdef ENABLE_PRECISE_ALKAID_TRUNCPR
          alkaid_boolean::TruncAPr,  // Trunc
#else
          alkaid_boolean::TruncA,
#endif
          alkaid_boolean::RandPermM, alkaid_boolean::PermAM, alkaid_boolean::PermAP, alkaid_boolean::InvPermAM,  // perm
          alkaid_boolean::InvPermAP                                                // perm
          >();
}

std::unique_ptr<SPUContext> makeAlkaidBooleanProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regAlkaidBooleanProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
