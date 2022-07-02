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

#include "spu/mpc/aby3/protocol.h"

#include "spu/mpc/aby3/arithmetic.h"
#include "spu/mpc/aby3/boolean.h"
#include "spu/mpc/aby3/conversion.h"
#include "spu/mpc/aby3/type.h"
#include "spu/mpc/common/abprotocol.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/object.h"
#include "spu/mpc/util/communicator.h"

namespace spu::mpc {

std::unique_ptr<Object> makeAby3Protocol(
    const std::shared_ptr<yasl::link::Context>& lctx) {
  aby3::registerTypes();

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
  obj->regKernel<aby3::P2A>();
  obj->regKernel<aby3::A2P>();
  obj->regKernel<aby3::NotA>();
  obj->regKernel<aby3::AddAP>();
  obj->regKernel<aby3::AddAA>();
  obj->regKernel<aby3::MulAP>();
  obj->regKernel<aby3::MulAA>();
  obj->regKernel<aby3::MatMulAP>();
  obj->regKernel<aby3::MatMulAA>();
  obj->regKernel<aby3::LShiftA>();

#define ENABLE_PRECISE_ABY3_TRUNCPR
#ifdef ENABLE_PRECISE_ABY3_TRUNCPR
  obj->regKernel<aby3::TruncPrAPrecise>();
#else
  obj->regKernel<aby3::TruncPrA>();
#endif

  obj->regKernel<aby3::B2P>();
  obj->regKernel<aby3::P2B>();
  obj->regKernel<aby3::AddBB>();
  obj->regKernel<aby3::A2B>();
  obj->regKernel<aby3::B2AByOT>();
  obj->regKernel<aby3::AndBP>();
  obj->regKernel<aby3::AndBB>();
  obj->regKernel<aby3::XorBP>();
  obj->regKernel<aby3::XorBB>();
  obj->regKernel<aby3::LShiftB>();
  obj->regKernel<aby3::RShiftB>();
  obj->regKernel<aby3::ARShiftB>();
  obj->regKernel<aby3::BitrevB>();

  return obj;
}

}  // namespace spu::mpc
