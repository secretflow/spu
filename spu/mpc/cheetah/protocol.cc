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

#include "spu/mpc/cheetah/protocol.h"

#include "spu/mpc/cheetah/arithmetic.h"
#include "spu/mpc/cheetah/boolean.h"
#include "spu/mpc/cheetah/conversion.h"
#include "spu/mpc/cheetah/object.h"
#include "spu/mpc/common/abprotocol.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/object.h"
#include "spu/mpc/semi2k/object.h"
#include "spu/mpc/semi2k/type.h"

namespace spu::mpc {

std::unique_ptr<Object> makeCheetahProtocol(
    const std::shared_ptr<yasl::link::Context>& lctx) {
  semi2k::registerTypes();

  auto obj = std::make_unique<Object>();

  // add communicator
  obj->addState<Communicator>(lctx);

  // register random states & kernels.
  obj->addState<PrgState>(lctx);

  // register public kernels.
  regPub2kKernels(obj.get());

  // register compute kernels
  obj->addState<ABProtState>();
  obj->regKernel<ABProtP2S>();
  obj->regKernel<ABProtS2P>();
  obj->regKernel<ABProtNotS>();
  obj->regKernel<ABProtAddSP>();
  obj->regKernel<ABProtAddSS>();
  obj->regKernel<ABProtMulSP>();
  obj->regKernel<ABProtMulSS>();
  obj->regKernel<ABProtMatMulSP>();
  obj->regKernel<ABProtMatMulSS>();
  obj->regKernel<ABProtAndSP>();
  obj->regKernel<ABProtAndSS>();
  obj->regKernel<ABProtXorSP>();
  obj->regKernel<ABProtXorSS>();
  obj->regKernel<ABProtEqzS>();
  obj->regKernel<ABProtLShiftS>();
  obj->regKernel<ABProtRShiftS>();
  obj->regKernel<ABProtARShiftS>();
  obj->regKernel<ABProtTruncPrS>();
  obj->regKernel<ABProtBitrevS>();
  obj->regKernel<ABProtMsbS>();

  // register arithmetic & binary kernels
  obj->addState<CheetahState>(lctx);
  obj->regKernel<cheetah::ZeroA>();
  obj->regKernel<cheetah::P2A>();
  obj->regKernel<cheetah::A2P>();
  obj->regKernel<cheetah::NotA>();
  obj->regKernel<cheetah::AddAP>();
  obj->regKernel<cheetah::AddAA>();
  obj->regKernel<cheetah::MulAP>();
  obj->regKernel<cheetah::MulAA>();
  obj->regKernel<cheetah::MatMulAP>();
  obj->regKernel<cheetah::MatMulAA>();
  obj->regKernel<cheetah::LShiftA>();
  obj->regKernel<cheetah::TruncPrA>();
  obj->regKernel<cheetah::MsbA>();

  obj->regKernel<cheetah::ZeroB>();
  obj->regKernel<cheetah::B2P>();
  obj->regKernel<cheetah::P2B>();
  obj->regKernel<cheetah::AddBB>();
  obj->regKernel<cheetah::A2B>();
  obj->regKernel<cheetah::B2A>();
  // obj->regKernel<cheetah::B2A_Randbit>();
  obj->regKernel<cheetah::AndBP>();
  obj->regKernel<cheetah::AndBB>();
  obj->regKernel<cheetah::XorBP>();
  obj->regKernel<cheetah::XorBB>();
  obj->regKernel<cheetah::LShiftB>();
  obj->regKernel<cheetah::RShiftB>();
  obj->regKernel<cheetah::ARShiftB>();
  obj->regKernel<cheetah::BitrevB>();

  return obj;
}

}  // namespace spu::mpc
