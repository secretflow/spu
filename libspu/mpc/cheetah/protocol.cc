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
#include "libspu/mpc/cheetah/object.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/ab_kernels.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/object.h"
#include "libspu/mpc/semi2k/object.h"
#include "libspu/mpc/semi2k/type.h"

namespace spu::mpc {

std::unique_ptr<Object> makeCheetahProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  semi2k::registerTypes();

  auto obj = std::make_unique<Object>("CHEETAH");

  // add communicator
  obj->addState<Communicator>(lctx);

  // register random states & kernels.
  obj->addState<PrgState>(lctx);

  // add Z2k state.
  obj->addState<Z2kState>(conf.field());

  // register public kernels.
  regPub2kKernels(obj.get());

  // register compute kernels
  regABKernels(obj.get());

  // register arithmetic & binary kernels
  obj->addState<cheetah::CheetahMulState>(lctx);
  obj->addState<cheetah::CheetahDotState>(lctx);
  obj->addState<cheetah::CheetahOTState>(lctx);
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
  obj->regKernel<cheetah::TruncA>();
  obj->regKernel<cheetah::MsbA2B>();

  obj->regKernel<common::AddBB>();
  obj->regKernel<common::BitIntlB>();
  obj->regKernel<common::BitDeintlB>();
  obj->regKernel<cheetah::CommonTypeB>();
  obj->regKernel<cheetah::CastTypeB>();
  obj->regKernel<cheetah::ZeroB>();
  obj->regKernel<cheetah::B2P>();
  obj->regKernel<cheetah::P2B>();
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
  obj->regKernel<cheetah::RandA>();

  return obj;
}

}  // namespace spu::mpc
