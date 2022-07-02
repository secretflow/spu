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

#include "spu/hal/context.h"

#include "spu/mpc/factory.h"

namespace spu {

HalContext::HalContext(RuntimeConfig config,
                       std::shared_ptr<yasl::link::Context> lctx)
    : rt_config_(config),
      lctx_(lctx),
      prot_(mpc::Factory::CreateCompute(config.protocol(), lctx)),
      rand_engine_(config.public_random_seed()) {
  setTracingEnabled(rt_config_.enable_action_trace());
  prot()->setTracingEnabled(rt_config_.enable_action_trace());
  setProfilingEnabled(rt_config_.enable_hal_profile());
  // TODO: expose `enable_mpc_profile`
  prot()->setProfilingEnabled(rt_config_.enable_hal_profile());
}

}  // namespace spu
