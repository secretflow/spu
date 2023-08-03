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

#pragma once

#include <memory>

#include "yacl/link/link.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/securenn/beaver/beaver_interface.h"
#include "libspu/mpc/securenn/beaver/beaver_tfp.h"
#include "libspu/mpc/securenn/beaver/beaver_ttp.h"

namespace spu::mpc {

class SecurennState : public State {
  std::unique_ptr<securenn::Beaver> beaver_;

 private:
  SecurennState() = default;

 public:
  static constexpr char kBindName[] = "SecurennState";

  explicit SecurennState(const RuntimeConfig& conf,
                         const std::shared_ptr<yacl::link::Context>& lctx) {
    if (conf.beaver_type() == RuntimeConfig_BeaverType_TrustedFirstParty) {
      beaver_ = std::make_unique<securenn::BeaverTfpUnsafe>(lctx);
    } else if (conf.beaver_type() ==
               RuntimeConfig_BeaverType_TrustedThirdParty) {
      securenn::BeaverTtp::Options ops;
      ops.server_host = conf.ttp_beaver_config().server_host();
      ops.adjust_rank = conf.ttp_beaver_config().adjust_rank();
      const auto& sid = conf.ttp_beaver_config().session_id();
      ops.session_id = sid.empty() ? lctx->Id() : sid;
      beaver_ = std::make_unique<securenn::BeaverTtp>(lctx, std::move(ops));
    } else {
      SPU_THROW("unsupported beaver type {}", conf.beaver_type());
    }
  }

  securenn::Beaver* beaver() { return beaver_.get(); }

  std::unique_ptr<State> fork() override {
    auto ret = std::unique_ptr<SecurennState>(new SecurennState);
    ret->beaver_ = beaver_->Spawn();
    return ret;
  }
};

}  // namespace spu::mpc
