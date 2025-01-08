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

#include "libspu/mpc/standard_shape/protocol.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/swift/arithmetic.h"
#include "libspu/mpc/swift/boolean.h"
#include "libspu/mpc/swift/conversion.h"
#include "libspu/mpc/swift/protocol.h"
#include "libspu/mpc/swift/type.h"
#include "libspu/mpc/swift/value.h"

namespace spu::mpc {

void regSwiftProtocol(SPUContext* ctx,
                      const std::shared_ptr<yacl::link::Context>& lctx) {
  swift::registerTypes();

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

  ctx->prot()
      ->regKernel<
          swift::P2A, swift::A2P, swift::UnaryTest1, swift::NegateA, swift::V2A,
          swift::A2V, swift::RandA, swift::AddAP, swift::AddAA, swift::MulAP,
          swift::MulAA_semi, swift::MulAA, swift::MatMulAP, swift::MatMulAA,
          swift::LShiftA, swift::TruncA, swift::P2B, swift::B2P, swift::XorBP,
          swift::XorBB, swift::AndBP, swift::AndBB, swift::LShiftB,
          swift::RShiftB, swift::ARShiftB, swift::BitrevB, swift::BitIntlB,
          swift::BitDeintlB, swift::A2B, swift::MsbA2B, swift::B2A>();

  // if (!(ctx->getField() == FieldType::FM32 ||
  //       ctx->getField() == FieldType::FM64)) {
  //   SPU_THROW("The malicious protocol for MulAA only support FM32 and FM64");
  // }
}

std::unique_ptr<SPUContext> makeSwiftProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  swift::registerTypes();

  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regSwiftProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc