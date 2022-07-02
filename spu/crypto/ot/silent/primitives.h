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

#include "nonlinear_protocols.h"
#include "silent_ot_pack.h"
#include "yasl/link/link.h"

namespace spu {

class CheetahPrimitives {
  int cheetah_party_;
  std::shared_ptr<SilentOTPack> silent_ot_pack_;
  std::unique_ptr<NonlinearProtocols> nonlinear_;

 public:
  explicit CheetahPrimitives(std::shared_ptr<yasl::link::Context> lctx) {
    // Map rank to party.
    cheetah_party_ = lctx->Rank() == 0 ? emp::ALICE : emp::BOB;
    // Setup silent ot.
    silent_ot_pack_ = std::make_shared<SilentOTPack>(
        cheetah_party_, std::make_unique<CheetahIo>(lctx));
    // Setup primitive protocols.
    nonlinear_ = std::make_unique<NonlinearProtocols>(silent_ot_pack_);
  }

  NonlinearProtocols* nonlinear() { return nonlinear_.get(); }
};

}  // namespace spu
