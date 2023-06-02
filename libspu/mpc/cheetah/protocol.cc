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
#include "libspu/mpc/common/prg_state.h"
//

#include "libspu/mpc/cheetah/arithmetic.h"
#include "libspu/mpc/cheetah/boolean.h"
#include "libspu/mpc/cheetah/conversion.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/state.h"
#include "libspu/mpc/semi2k/type.h"

namespace spu::mpc {

void regCheetahProtocol(SPUContext* ctx,
                        const std::shared_ptr<yacl::link::Context>& lctx) {
  semi2k::registerTypes();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // add Z2k state.
  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // add Cheetah states
  ctx->prot()->addState<cheetah::CheetahMulState>(lctx);
  ctx->prot()->addState<cheetah::CheetahDotState>(lctx);
  ctx->prot()->addState<cheetah::CheetahOTState>();

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // register arithmetic & binary kernels
  ctx->prot()->regKernel<cheetah::P2A>();
  ctx->prot()->regKernel<cheetah::A2P>();
  ctx->prot()->regKernel<cheetah::V2A>();
  ctx->prot()->regKernel<cheetah::A2V>();
  ctx->prot()->regKernel<cheetah::NotA>();
  ctx->prot()->regKernel<cheetah::AddAP>();
  ctx->prot()->regKernel<cheetah::AddAA>();
  ctx->prot()->regKernel<cheetah::MulAP>();
  ctx->prot()->regKernel<cheetah::MulAA>();
  ctx->prot()->regKernel<cheetah::MulA1B>();
  ctx->prot()->regKernel<cheetah::EqualAA>();
  ctx->prot()->regKernel<cheetah::EqualAP>();
  ctx->prot()->regKernel<cheetah::MatMulAP>();
  ctx->prot()->regKernel<cheetah::MatMulAA>();
  ctx->prot()->regKernel<cheetah::Conv2DAA>();
  ctx->prot()->regKernel<cheetah::LShiftA>();
  ctx->prot()->regKernel<cheetah::TruncA>();
  ctx->prot()->regKernel<cheetah::TruncAWithSign>();
  ctx->prot()->regKernel<cheetah::MsbA2B>();

  ctx->prot()->regKernel<cheetah::CommonTypeB>();
  ctx->prot()->regKernel<cheetah::CastTypeB>();
  ctx->prot()->regKernel<cheetah::B2P>();
  ctx->prot()->regKernel<cheetah::P2B>();
  ctx->prot()->regKernel<cheetah::A2B>();
  ctx->prot()->regKernel<cheetah::B2A>();

  ctx->prot()->regKernel<cheetah::AndBP>();
  ctx->prot()->regKernel<cheetah::AndBB>();
  ctx->prot()->regKernel<cheetah::XorBP>();
  ctx->prot()->regKernel<cheetah::XorBB>();
  ctx->prot()->regKernel<cheetah::LShiftB>();
  ctx->prot()->regKernel<cheetah::RShiftB>();
  ctx->prot()->regKernel<cheetah::ARShiftB>();
  ctx->prot()->regKernel<cheetah::BitrevB>();
  ctx->prot()->regKernel<cheetah::RandA>();
}

std::unique_ptr<SPUContext> makeCheetahProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  semi2k::registerTypes();

  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regCheetahProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
