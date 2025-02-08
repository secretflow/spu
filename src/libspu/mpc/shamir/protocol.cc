// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/shamir/protocol.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/common/pv_gfmp.h"
#include "libspu/mpc/shamir/arithmetic.h"
#include "libspu/mpc/shamir/boolean.h"
#include "libspu/mpc/shamir/conversion.h"
#include "libspu/mpc/shamir/type.h"
#include "libspu/mpc/standard_shape/protocol.h"

namespace spu::mpc {

void regShamirProtocol(SPUContext* ctx,
                       const std::shared_ptr<yacl::link::Context>& lctx) {
  shamir::registerTypes();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // add Z2k state.
  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // register public kernels.
  regPVGfmpKernels(ctx->prot());

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic & binary kernels
  ctx->prot()
      ->regKernel<shamir::P2A, shamir::A2P, shamir::A2V, shamir::V2A,
                  shamir::RandA,                 //
                  shamir::NegateA,               //
                  shamir::AddAP, shamir::AddAA,  //
                  shamir::MulAP, shamir::MulAA, shamir::MulAAP,
                  shamir::MulAAA,                      //
                  shamir::MatMulAP, shamir::MatMulAA,  //
                  shamir::LShiftB, shamir::RShiftB, shamir::ARShiftB,
                  shamir::CommonTypeB, shamir::CastTypeB,         //
                  shamir::CommonTypeV, shamir::A2B, shamir::B2A,  //
                  shamir::AndBP, shamir::XorBP,                   //
                  shamir::P2B, shamir::B2P, shamir::B2V, shamir::XorBB,
                  shamir::AndBB, shamir::BitrevB, shamir::TruncA,
                  shamir::MulAATrunc, shamir::MsbA, shamir::ReLU,  //
                  shamir::BitIntlB, shamir::BitDeintlB>();
}

std::unique_ptr<SPUContext> makeShamirProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  shamir::registerTypes();

  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regShamirProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
