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

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/semi2k/beaver/beaver_interface.h"
#include "libspu/mpc/semi2k/beaver/beaver_tfp.h"
#include "libspu/mpc/semi2k/beaver/beaver_ttp.h"

namespace spu::mpc {

// TODO(jint) split this into individual states.
class Semi2kState : public State {
  std::unique_ptr<semi2k::Beaver> beaver_;

 private:
  Semi2kState() = default;

 public:
  static constexpr char kBindName[] = "Semi2kState";

  explicit Semi2kState(const RuntimeConfig& conf,
                       const std::shared_ptr<yacl::link::Context>& lctx) {
    if (conf.beaver_type() == RuntimeConfig_BeaverType_TrustedFirstParty) {
      beaver_ = std::make_unique<semi2k::BeaverTfpUnsafe>(lctx);
    } else if (conf.beaver_type() ==
               RuntimeConfig_BeaverType_TrustedThirdParty) {
      semi2k::BeaverTtp::Options ops;
      ops.server_host = conf.ttp_beaver_config().server_host();
      ops.adjust_rank = conf.ttp_beaver_config().adjust_rank();
      const auto& sid = conf.ttp_beaver_config().session_id();
      ops.session_id = sid.empty() ? lctx->Id() : sid;
      // TODO: TLS & brpc options.
      beaver_ = std::make_unique<semi2k::BeaverTtp>(lctx, std::move(ops));
    } else {
      SPU_THROW("unsupported beaver type {}", conf.beaver_type());
    }
  }

  semi2k::Beaver* beaver() { return beaver_.get(); }

  std::unique_ptr<State> fork() override {
    auto ret = std::unique_ptr<Semi2kState>(new Semi2kState);
    ret->beaver_ = beaver_->Spawn();
    return ret;
  }
};

}  // namespace spu::mpc
