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

#include "libspu/mpc/fantastic4/protocol.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/fantastic4/arithmetic.h"
#include "libspu/mpc/fantastic4/boolean.h"
#include "libspu/mpc/fantastic4/conversion.h"
#include "libspu/mpc/fantastic4/type.h"
#include "libspu/mpc/standard_shape/protocol.h"

#include "libspu/mpc/fantastic4/state.h"

namespace spu::mpc {

void regFantastic4Protocol(SPUContext* ctx,
                           const std::shared_ptr<yacl::link::Context>& lctx) {
  fantastic4::registerTypes();

  ctx->prot()->addState<Z2kState>(ctx->config().field());

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
          fantastic4::P2A, fantastic4::V2A, fantastic4::A2P, fantastic4::A2V,fantastic4::AddAA, fantastic4::AddAP, fantastic4::NegateA,fantastic4::MulAP, fantastic4::MulAA, fantastic4::MatMulAP, fantastic4::MatMulAA, fantastic4::LShiftA, fantastic4::TruncAPr,
          fantastic4::CommonTypeB, fantastic4::CastTypeB, fantastic4::B2P, fantastic4::P2B, fantastic4::XorBB, fantastic4::XorBP, fantastic4::AndBP, fantastic4::AndBB,
          fantastic4::LShiftB, fantastic4::RShiftB, fantastic4::ARShiftB, fantastic4::BitrevB, fantastic4::A2B, fantastic4::B2A, fantastic4::MsbA2B, fantastic4::RandA, fantastic4::EqualAP, fantastic4::EqualAA
          >();
}

std::unique_ptr<SPUContext> makeFantastic4Protocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regFantastic4Protocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
