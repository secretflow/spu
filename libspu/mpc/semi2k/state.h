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

#include "libspu/core/object.h"
#include "libspu/mpc/semi2k/beaver/beaver_cache.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/beaver_tfp.h"
#include "libspu/mpc/semi2k/beaver/beaver_impl/beaver_ttp.h"
#include "libspu/mpc/semi2k/beaver/beaver_interface.h"

namespace spu::mpc {

// TODO(jint) split this into individual states.
class Semi2kState : public State {
  std::unique_ptr<semi2k::Beaver> beaver_;
  std::shared_ptr<semi2k::BeaverCache> beaver_cache_;

 private:
  Semi2kState() = default;

 public:
  static constexpr const char* kBindName() { return "Semi2kState"; }

  explicit Semi2kState(const RuntimeConfig& conf,
                       const std::shared_ptr<yacl::link::Context>& lctx) {
    auto beaver_type = BeaverType::TrustedFirstParty;
    if (conf.protocol().has_semi2k_config()) {
      beaver_type = conf.protocol().semi2k_config().beaver_type();
    }
    if (beaver_type == BeaverType::TrustedFirstParty) {
      beaver_ = std::make_unique<semi2k::BeaverTfpUnsafe>(lctx);
    } else if (beaver_type == BeaverType::TrustedThirdParty) {
      SPU_ENFORCE(conf.protocol().has_semi2k_config(),
                  " Semi2k protocol config not set");
      auto semi2k_cfg = conf.protocol().semi2k_config();
      semi2k::BeaverTtp::Options ops;
      SPU_ENFORCE(semi2k_cfg.has_ttp_beaver_config(),
                  " Semi2k protocol ttp config not set");

      ops.server_host = semi2k_cfg.ttp_beaver_config().server_host();
      ops.adjust_rank = semi2k_cfg.ttp_beaver_config().adjust_rank();
      ops.asym_crypto_schema =
          semi2k_cfg.ttp_beaver_config().asym_crypto_schema();
      {
        const auto& key = semi2k_cfg.ttp_beaver_config().server_public_key();
        ops.server_public_key = yacl::Buffer(key.data(), key.size());
      }
      // TODO: TLS & brpc options.
      beaver_ = std::make_unique<semi2k::BeaverTtp>(lctx, std::move(ops));
    } else {
      SPU_THROW("unsupported beaver type {}", beaver_type);
    }
    beaver_cache_ = std::make_unique<semi2k::BeaverCache>();
  }

  semi2k::Beaver* beaver() { return beaver_.get(); }
  semi2k::BeaverCache* beaver_cache() { return beaver_cache_.get(); }

  std::unique_ptr<State> fork() override {
    auto ret = std::unique_ptr<Semi2kState>(new Semi2kState);
    ret->beaver_ = beaver_->Spawn();
    ret->beaver_cache_ = beaver_cache_;
    return ret;
  }
};

}  // namespace spu::mpc
