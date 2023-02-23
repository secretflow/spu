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

#include "libspu/mpc/spdzwisefield/protocol.h"

#include "libspu/mpc/spdzwisefield/arithmetic.h"
#include "libspu/mpc/spdzwisefield/type.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/ab_kernels.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"

namespace spu::mpc {

std::unique_ptr<Object> makeSpdzWiseFieldProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  
  spdzwisefield::registerTypes();

  auto obj = std::make_unique<Object>("SPDZWISEFIELD");

  obj->addState<Z2kState>(conf.field());

  // add communicator
  obj->addState<Communicator>(lctx);

  // register random states & kernels.
  obj->addState<PrgState>(lctx);

  // register public kernels.
  regPub2kKernels(obj.get());

  // register api kernels
  regABKernels(obj.get());

  // register arithmetic & binary kernels
  obj->regKernel<spdzwisefield::P2A>();
  obj->regKernel<spdzwisefield::A2P>();

}


}  // namespace spu::mpc::spdzwisefield