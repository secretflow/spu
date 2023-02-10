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

#include "libspu/mpc/spdz2k/protocol.h"

#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/object.h"
#include "libspu/mpc/spdz2k/arithmetic.h"
#include "libspu/mpc/spdz2k/object.h"
#include "libspu/mpc/spdz2k/type.h"

namespace spu::mpc {

std::unique_ptr<Object> makeSpdz2kProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  spdz2k::registerTypes();

  auto obj = std::make_unique<Object>("SPDZ2K");

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

  // register arithmetic kernels
  obj->addState<Spdz2kState>(lctx);
  obj->regKernel<spdz2k::ZeroA>();
  obj->regKernel<spdz2k::P2A>();
  obj->regKernel<spdz2k::A2P>();
  obj->regKernel<spdz2k::NotA>();
  obj->regKernel<spdz2k::AddAP>();
  obj->regKernel<spdz2k::AddAA>();
  obj->regKernel<spdz2k::MulAP>();
  obj->regKernel<spdz2k::MulAA>();
  obj->regKernel<spdz2k::MatMulAP>();
  obj->regKernel<spdz2k::MatMulAA>();
  obj->regKernel<spdz2k::LShiftA>();
  obj->regKernel<spdz2k::TruncA>();
  obj->regKernel<spdz2k::RandA>();

  return obj;
}

}  // namespace spu::mpc
