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
#include "libspu/mpc/aby3/oram.h"
#include "libspu/mpc/aby3/permute.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/standard_shape/protocol.h"

#define ENABLE_PRECISE_ABY3_TRUNCPR

namespace spu::mpc {

void regAby3Protocol(SPUContext* ctx,
                     const std::shared_ptr<yacl::link::Context>& lctx) {
  aby3::registerTypes();

  ctx->prot()->addState<Z2kState>(ctx->config().field);

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
          aby3::P2A, aby3::V2A, aby3::A2P, aby3::A2V,           // Conversions
          aby3::B2P, aby3::P2B, aby3::A2B, aby3::A2B_Bits,      // Conversion2
          aby3::B2ASelector, /*aby3::B2AByOT, aby3::B2AByPPA*/  // B2A
          aby3::CastTypeB,                                      // Cast
          aby3::NegateA,                                        // Negate
          aby3::AddAP, aby3::AddAA,                             // Add
          aby3::MulAP, aby3::MulAA, aby3::MulA1B,               // Mul
          aby3::MatMulAP, aby3::MatMulAA,                       // MatMul
          aby3::LShiftA, aby3::LShiftB,                         // LShift
          aby3::RShiftB, aby3::ARShiftB,                        // (A)Rshift
          aby3::MsbA2B,                                         // MSB
          aby3::EqualAA, aby3::EqualAP,                         // Equal
          aby3::CommonTypeB, aby3::CommonTypeV,                 // CommonType
          aby3::AndBP, aby3::AndBB,                             // And
          aby3::XorBP, aby3::XorBB,                             // Xor
          aby3::BitrevB,                                        // bitreverse
          aby3::BitIntlB, aby3::BitDeintlB,  // bit(de)interleave
          aby3::RandA, aby3::RandB,          // rand
#ifdef ENABLE_PRECISE_ABY3_TRUNCPR
          // aby3::TruncAPr,  // Trunc
          aby3::TruncAPr2,  // Trunc
#else
          aby3::TruncA,
#endif
          aby3::OramOneHotAA, aby3::OramOneHotAP, aby3::OramReadOA,      // oram
          aby3::OramReadOP,                                              // oram
          aby3::RandPermM, aby3::PermAM, aby3::PermAP, aby3::InvPermAM,  // perm
          aby3::InvPermAP,                                               // perm
          aby3::RingCastDownA                                            // cast
          >();
}

std::unique_ptr<SPUContext> makeAby3Protocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regAby3Protocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
