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

#include "spu/mpc/semi2k/protocol.h"

#include "spu/mpc/common/abprotocol.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/object.h"
#include "spu/mpc/semi2k/arithmetic.h"
#include "spu/mpc/semi2k/boolean.h"
#include "spu/mpc/semi2k/conversion.h"
#include "spu/mpc/semi2k/object.h"
#include "spu/mpc/semi2k/type.h"

namespace spu::mpc {

std::unique_ptr<Object> makeSemi2kProtocol(
    const RuntimeConfig& conf,
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
  regABKernels(obj.get());

  // register arithmetic & binary kernels
  obj->addState<Semi2kState>(lctx);
  obj->regKernel<semi2k::ZeroA>();
  obj->regKernel<semi2k::P2A>();
  obj->regKernel<semi2k::A2P>();
  obj->regKernel<semi2k::NotA>();
  obj->regKernel<semi2k::AddAP>();
  obj->regKernel<semi2k::AddAA>();
  obj->regKernel<semi2k::MulAP>();
  obj->regKernel<semi2k::MulAA>();
  obj->regKernel<semi2k::MatMulAP>();
  obj->regKernel<semi2k::MatMulAA>();
  obj->regKernel<semi2k::LShiftA>();
  obj->regKernel<semi2k::TruncPrA>();

  obj->regKernel<semi2k::CommonTypeB>();
  obj->regKernel<semi2k::CastTypeB>();
  obj->regKernel<semi2k::ZeroB>();
  obj->regKernel<semi2k::B2P>();
  obj->regKernel<semi2k::P2B>();
  obj->regKernel<semi2k::AddBB>();
  obj->regKernel<semi2k::A2B>();
  // obj->regKernel<semi2k::B2A>();
  obj->regKernel<semi2k::B2A_Randbit>();
  obj->regKernel<semi2k::AndBP>();
  obj->regKernel<semi2k::AndBB>();
  obj->regKernel<semi2k::XorBP>();
  obj->regKernel<semi2k::XorBB>();
  obj->regKernel<semi2k::LShiftB>();
  obj->regKernel<semi2k::RShiftB>();
  obj->regKernel<semi2k::ARShiftB>();
  obj->regKernel<semi2k::BitrevB>();

  return obj;
}

}  // namespace spu::mpc
