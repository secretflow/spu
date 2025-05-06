// Copyright 2025 Ant Group Co., Ltd.
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

#include "libspu/mpc/experimental/fantastic4/protocol.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/experimental/fantastic4/arithmetic.h"
#include "libspu/mpc/experimental/fantastic4/boolean.h"
#include "libspu/mpc/experimental/fantastic4/conversion.h"
#include "libspu/mpc/experimental/fantastic4/state.h"
#include "libspu/mpc/experimental/fantastic4/type.h"
#include "libspu/mpc/standard_shape/protocol.h"

namespace spu::mpc {

//  For Rep4 / Fantastic Four
//    Secret is split into 4 shares x_0, x_1, x_2, x_3
//    Differently from the paper, we let Party i (i in {0, 1, 2, 3}) holds x_i,
//    x_i+1, x_i+2 (mod 4) Similarly in prg_state.h, PRG keys are k_0, k_1, k_2,
//    k_3, we let Party i holds k_i--self, k_i+1 --next, k_i+2--next next Each
//    x_i, k_i is unknown to next party P_i+1

//  If use optimized protocol, define OPTIMIZED_F4 in jmp.h

void regFantastic4Protocol(SPUContext* ctx,
                           const std::shared_ptr<yacl::link::Context>& lctx) {
  fantastic4::registerTypes();

  ctx->prot()->addState<Z2kState>(ctx->config().field);

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // add MAC state
  ctx->prot()->addState<spu::mpc::fantastic4::Fantastic4MacState>(lctx);

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic & binary kernels
  ctx->prot()
      ->regKernel<
          fantastic4::P2A, fantastic4::V2A, fantastic4::A2P, fantastic4::A2V,
          fantastic4::AddAA, fantastic4::AddAP, fantastic4::NegateA,
          fantastic4::MulAP, fantastic4::MulAA, fantastic4::MatMulAP,
          fantastic4::MatMulAA, fantastic4::LShiftA, fantastic4::TruncAPr,
          fantastic4::CommonTypeB, fantastic4::CastTypeB, fantastic4::B2P,
          fantastic4::P2B, fantastic4::XorBB, fantastic4::XorBP,
          fantastic4::AndBP, fantastic4::AndBB, fantastic4::LShiftB,
          fantastic4::RShiftB, fantastic4::ARShiftB, fantastic4::BitrevB,
          fantastic4::A2B, fantastic4::B2A, fantastic4::MsbA2B,
          fantastic4::RandA, fantastic4::EqualAP, fantastic4::EqualAA>();
}

std::unique_ptr<SPUContext> makeFantastic4Protocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regFantastic4Protocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
